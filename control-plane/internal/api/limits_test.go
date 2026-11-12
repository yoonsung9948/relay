package api

import (
	"context"
	"errors"
	"github.com/yoonsung9948/relay/internal/config"
	"github.com/yoonsung9948/relay/internal/controlplane"
	"github.com/yoonsung9948/relay/internal/ratelimit"
	"github.com/yoonsung9948/relay/internal/request"
	"github.com/yoonsung9948/relay/internal/response"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestRequestRateLimitToggle(t *testing.T) {
	for _, enabled := range []bool{false, true} {
		handler := withRequestRateLimit(ratelimit.NewRequestLimiter(0.001, 1), enabled, func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(204) })
		for i := 0; i < 2; i++ {
			w := httptest.NewRecorder()
			handler(w, httptest.NewRequest("POST", "/generate", nil))
			if enabled && i == 1 {
				if w.Code != 429 || !strings.Contains(w.Body.String(), "request_rate_limited") || w.Header().Get("Retry-After") == "" {
					t.Fatalf("bad rejection: %d %s", w.Code, w.Body.String())
				}
			} else if w.Code != 204 {
				t.Fatal(w.Code)
			}
		}
	}
}
func TestConcurrencySlotReleasedOnErrorAndPanic(t *testing.T) {
	sem := make(chan struct{}, 1)
	for _, panicHandler := range []bool{false, true} {
		handler := withConcurrencyLimit(sem, func(w http.ResponseWriter, r *http.Request) {
			if panicHandler {
				panic("handler failed")
			}
			w.WriteHeader(500)
		})
		func() {
			defer func() { recover() }()
			handler(httptest.NewRecorder(), httptest.NewRequest("POST", "/generate", nil))
		}()
		if len(sem) != 0 {
			t.Fatal("slot leaked")
		}
	}
}

type blockingClient struct {
	streamClient
	entered chan struct{}
}

func (c *blockingClient) GenerateStream(ctx context.Context, _ request.GenerateRequest) (<-chan response.StreamResponse, error) {
	close(c.entered)
	return make(chan response.StreamResponse), nil
}
func TestSharedGenerationSlotAndCancellation(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	client := &blockingClient{entered: make(chan struct{})}
	cp := controlplane.NewControlPlane(ctx, config.ControlPlaneConfig{}, client)
	cp.Provider = &streamProvider{}
	if err := cp.StartEngine(ctx, config.GPUProviderConfig{}); err != nil {
		t.Fatal(err)
	}
	for cp.Status().State != controlplane.StateReady {
		select {
		case <-ctx.Done():
			t.Fatal("not ready")
		case <-time.After(time.Millisecond):
		}
	}
	s := NewServer(config.ServeConfig{DemoKey: "test", MaxConcurrentGenerations: 1}, cp)
	streamCtx, stopStream := context.WithCancel(ctx)
	defer stopStream()
	req := httptest.NewRequest("POST", "/generate", strings.NewReader(`{"prompt":"hi","stream":true}`)).WithContext(streamCtx)
	req.Header.Set("X-Demo-Key", "test")
	finished := make(chan struct{})
	go func() { defer close(finished); s.httpServer.Handler.ServeHTTP(httptest.NewRecorder(), req) }()
	select {
	case <-client.entered:
	case <-ctx.Done():
		t.Fatal("stream not admitted")
	}
	generate := func() *httptest.ResponseRecorder {
		req := httptest.NewRequest("POST", "/generate", strings.NewReader(`{"prompt":"hi","stream":false}`))
		req.Header.Set("X-Demo-Key", "test")
		w := httptest.NewRecorder()
		s.httpServer.Handler.ServeHTTP(w, req)
		return w
	}
	w := generate()
	if w.Code != 503 || !strings.Contains(w.Body.String(), "generation_capacity") || w.Header().Get("Retry-After") != "" {
		t.Fatalf("unexpected rejection: %d %s", w.Code, w.Body.String())
	}
	stopStream()
	select {
	case <-finished:
	case <-ctx.Done():
		t.Fatal("cancelled stream held its slot")
	}
	if w := generate(); w.Code != 200 {
		t.Fatalf("slot not released: %d %s", w.Code, w.Body.String())
	}
}
func TestStartupConflictResponse(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	cp := controlplane.NewControlPlane(ctx, config.ControlPlaneConfig{}, &streamClient{})
	cp.Provider = &streamProvider{}
	if err := cp.StartEngine(ctx, config.GPUProviderConfig{}); err != nil {
		t.Fatal(err)
	}
	var conflict *controlplane.EngineStateConflictError
	if err := cp.StartEngine(ctx, config.GPUProviderConfig{}); !errors.As(err, &conflict) {
		t.Fatal(err)
	}
	w := httptest.NewRecorder()
	(&Server{control: cp}).StartEngineHandler(w, httptest.NewRequest("POST", "/start_engine", nil))
	if w.Code != 409 || !strings.Contains(w.Body.String(), "engine_state_conflict") {
		t.Fatalf("unexpected conflict: %d %s", w.Code, w.Body.String())
	}
}

func TestAuthenticationDoesNotConsumeGenerationTokens(t *testing.T) {
	cp := controlplane.NewControlPlane(context.Background(), config.ControlPlaneConfig{RateLimitEnabled: true}, &streamClient{})
	s := NewServer(config.ServeConfig{DemoKey: "test", GenerationRequestsPerSecond: 0.001, GenerationBurst: 1}, cp)
	for i, want := range []int{401, 503, 429} {
		req := httptest.NewRequest("POST", "/generate", strings.NewReader(`{"prompt":"hello"}`))
		if i > 0 {
			req.Header.Set("X-Demo-Key", "test")
		}
		w := httptest.NewRecorder()
		s.httpServer.Handler.ServeHTTP(w, req)
		if w.Code != want {
			t.Fatalf("request %d: status=%d body=%s", i, w.Code, w.Body.String())
		}
	}
}
