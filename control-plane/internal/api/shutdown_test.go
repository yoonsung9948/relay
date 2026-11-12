package api

import (
	"context"
	"errors"
	"net"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/yoonsung9948/relay/internal/config"
	"github.com/yoonsung9948/relay/internal/controlplane"
	"github.com/yoonsung9948/relay/internal/controlplane/gpuprovider"
	"github.com/yoonsung9948/relay/internal/request"
	"github.com/yoonsung9948/relay/internal/response"
)

type cleanupProvider struct {
	mu         sync.Mutex
	name       string
	terminated bool
	calls      int
	fail       bool
}

func (p *cleanupProvider) Create(ctx context.Context, spec gpuprovider.CreateInstanceSpec) (*gpuprovider.Instance, error) {
	p.mu.Lock()
	p.name = spec.Name
	p.mu.Unlock()
	return &gpuprovider.Instance{ID: "owned", Name: spec.Name, Endpoint: "http://fake"}, nil
}
func (p *cleanupProvider) Get(context.Context, string) (*gpuprovider.Instance, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.terminated {
		return nil, gpuprovider.ErrNotFound
	}
	return &gpuprovider.Instance{ID: "owned", Name: p.name}, nil
}
func (p *cleanupProvider) Terminate(ctx context.Context, id string) error {
	p.mu.Lock()
	p.calls++
	fail := p.fail
	if !fail {
		p.terminated = true
	}
	p.mu.Unlock()
	if fail {
		<-ctx.Done()
		return ctx.Err()
	}
	return nil
}
func (*cleanupProvider) Start(context.Context, string) error { return errors.New("unexpected start") }
func (*cleanupProvider) Stop(context.Context, string) error  { return errors.New("unexpected stop") }

type cancelClient struct{ started, canceled chan struct{} }

func (*cancelClient) Health(context.Context) error { return nil }
func (*cancelClient) SetEndpoint(string)           {}
func (c *cancelClient) Generate(ctx context.Context, req request.GenerateRequest) (response.GenerateResponse, error) {
	close(c.started)
	<-ctx.Done()
	close(c.canceled)
	return response.GenerateResponse{}, ctx.Err()
}
func (c *cancelClient) GenerateStream(ctx context.Context, req request.GenerateRequest) (<-chan response.StreamResponse, error) {
	close(c.started)
	stream := make(chan response.StreamResponse)
	go func() { <-ctx.Done(); close(c.canceled); close(stream) }()
	return stream, nil
}
func readyCleanupControl(t *testing.T, p *cleanupProvider, client *cancelClient) *controlplane.ControlPlane {
	t.Helper()
	cp := controlplane.NewControlPlane(context.Background(), config.ControlPlaneConfig{CleanupTimeout: 100 * time.Millisecond, ShutdownTimeout: time.Second}, client)
	cp.Provider = p
	if err := cp.StartEngine(context.Background(), config.GPUProviderConfig{}); err != nil {
		t.Fatal(err)
	}
	for deadline := time.Now().Add(time.Second); cp.Status().State != controlplane.StateReady; {
		if time.Now().After(deadline) {
			t.Fatal("engine not ready")
		}
		time.Sleep(time.Millisecond)
	}
	return cp
}

func TestListenerBindFailureStillCleansWorker(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()
	provider := &cleanupProvider{}
	cp := readyCleanupControl(t, provider, &cancelClient{})
	server := NewServer(config.ServeConfig{DemoKey: "test"}, cp)
	server.httpServer.Addr = listener.Addr().String()
	if err := server.ServeAndListen(context.Background()); err == nil {
		t.Fatal("bind failure hidden")
	}
	provider.mu.Lock()
	calls := provider.calls
	provider.mu.Unlock()
	if calls != 1 || cp.Status().State != controlplane.StateOffline {
		t.Fatalf("calls=%d state=%+v", calls, cp.Status())
	}
}

func TestProcessShutdownCancelsGenerationAndCleansWorker(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run(strconv.FormatBool(stream), func(t *testing.T) {
			listener, err := net.Listen("tcp", "127.0.0.1:0")
			if err != nil {
				t.Fatal(err)
			}
			address := listener.Addr().String()
			listener.Close()
			provider := &cleanupProvider{}
			client := &cancelClient{started: make(chan struct{}), canceled: make(chan struct{})}
			cp := readyCleanupControl(t, provider, client)
			server := NewServer(config.ServeConfig{DemoKey: "test"}, cp)
			server.httpServer.Addr = address
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			serverDone := make(chan error, 1)
			go func() { serverDone <- server.ServeAndListen(ctx) }()
			httpClient := &http.Client{Timeout: time.Second}
			for deadline := time.Now().Add(time.Second); ; {
				result, err := httpClient.Get("http://" + address + "/health")
				if err == nil {
					result.Body.Close()
					break
				}
				if time.Now().After(deadline) {
					t.Fatal("server did not start", err)
				}
				time.Sleep(time.Millisecond)
			}
			requestDone := make(chan struct{})
			go func() {
				defer close(requestDone)
				req, _ := http.NewRequest("POST", "http://"+address+"/generate", strings.NewReader(`{"prompt":"hello","stream":`+strconv.FormatBool(stream)+`}`))
				req.Header.Set("X-Demo-Key", "test")
				resp, err := httpClient.Do(req)
				if err == nil {
					resp.Body.Close()
				}
			}()
			select {
			case <-client.started:
			case <-time.After(time.Second):
				t.Fatal("generation did not start")
			}
			cancel()
			select {
			case err := <-serverDone:
				if err != nil {
					t.Fatal(err)
				}
			case <-time.After(2 * time.Second):
				t.Fatal("unbounded shutdown")
			}
			<-client.canceled
			<-requestDone
			provider.mu.Lock()
			calls := provider.calls
			provider.mu.Unlock()
			if calls != 1 || cp.Status().ActiveRequests != 0 {
				t.Fatalf("calls=%d status=%+v", calls, cp.Status())
			}
		})
	}
}

func TestManualStopRequiresKeyAndReportsCleanupFailure(t *testing.T) {
	provider := &cleanupProvider{fail: true}
	cp := readyCleanupControl(t, provider, &cancelClient{})
	server := NewServer(config.ServeConfig{DemoKey: "test"}, cp)
	req := httptest.NewRequest("POST", "/stop_engine", nil)
	recorder := httptest.NewRecorder()
	server.httpServer.Handler.ServeHTTP(recorder, req)
	if recorder.Code != http.StatusUnauthorized {
		t.Fatal(recorder.Code)
	}
	req.Header.Set("X-Demo-Key", "test")
	recorder = httptest.NewRecorder()
	start := time.Now()
	server.httpServer.Handler.ServeHTTP(recorder, req)
	if recorder.Code != http.StatusServiceUnavailable || time.Since(start) > time.Second || cp.Status().InstanceID != "owned" {
		t.Fatal(recorder.Code, cp.Status())
	}
	provider.mu.Lock()
	provider.fail = false
	provider.mu.Unlock()
	recorder = httptest.NewRecorder()
	server.httpServer.Handler.ServeHTTP(recorder, req)
	if recorder.Code != http.StatusOK || cp.Status().InstanceID != "" {
		t.Fatal(recorder.Code, cp.Status())
	}
}
