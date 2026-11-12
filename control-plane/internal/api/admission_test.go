package api

import (
	"context"
	"github.com/yoonsung9948/relay/internal/config"
	"github.com/yoonsung9948/relay/internal/controlplane"
	"github.com/yoonsung9948/relay/internal/controlplane/gpuprovider"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestStartHandlerMapsCooldown(t *testing.T) {
	cp := controlplane.NewControlPlane(context.Background(), config.ControlPlaneConfig{RateLimitEnabled: true, ProvisionCooldown: time.Hour, ProvisionRetryBackoff: 15 * time.Second}, &streamClient{})
	// The first attempt reserves a cooldown, then receives confirmed no capacity.
	cp.Provider = capacityProvider{}
	s := &Server{control: cp}
	for _, status := range []int{503, 429} {
		w := httptest.NewRecorder()
		s.StartEngineHandler(w, httptest.NewRequest("POST", "/start_engine", nil))
		if delay, err := strconv.Atoi(w.Header().Get("Retry-After")); err != nil || delay < 1 || delay > 15 {
			t.Fatalf("missing retry header: %v", w.Header())
		}
		if w.Code != status {
			t.Fatalf("status=%d body=%s", w.Code, w.Body.String())
		}
	}
}
func TestRetryAfterRoundsUp(t *testing.T) {
	w := httptest.NewRecorder()
	writeRetryError(w, 429, "provision_cooldown", time.Now().Add(1500*time.Millisecond))
	seconds, err := strconv.Atoi(w.Header().Get("Retry-After"))
	if err != nil || seconds != 2 || !strings.Contains(w.Body.String(), "provision_cooldown") {
		t.Fatalf("headers=%v body=%s", w.Header(), w.Body.String())
	}
	w = httptest.NewRecorder()
	writeRetryError(w, 503, "no_gpu_available", time.Time{})
	if w.Header().Get("Retry-After") != "" {
		t.Fatal("disabled limiter advertised a retry delay")
	}
}

type capacityProvider struct{}

func (capacityProvider) Create(context.Context, gpuprovider.CreateInstanceSpec) (*gpuprovider.Instance, error) {
	return nil, gpuprovider.ErrNoCapacity
}
func (capacityProvider) Get(context.Context, string) (*gpuprovider.Instance, error) {
	panic("unexpected get")
}
func (capacityProvider) Stop(context.Context, string) error      { panic("unexpected stop") }
func (capacityProvider) Start(context.Context, string) error     { panic("unexpected start") }
func (capacityProvider) Terminate(context.Context, string) error { panic("unexpected terminate") }
