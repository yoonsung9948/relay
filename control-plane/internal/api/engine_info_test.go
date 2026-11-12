package api

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/yoonsung9948/relay/internal/config"
	"github.com/yoonsung9948/relay/internal/controlplane"
	"github.com/yoonsung9948/relay/internal/controlplane/gpuprovider"
	"github.com/yoonsung9948/relay/internal/engine"
)

type infoProvider struct{ endpoint string }

func (p infoProvider) Create(context.Context, gpuprovider.CreateInstanceSpec) (*gpuprovider.Instance, error) {
	return &gpuprovider.Instance{ID: "test", Endpoint: p.endpoint}, nil
}
func (p infoProvider) Get(context.Context, string) (*gpuprovider.Instance, error) {
	return &gpuprovider.Instance{ID: "test", Endpoint: p.endpoint}, nil
}

func (infoProvider) Stop(context.Context, string) error      { return nil }
func (infoProvider) Start(context.Context, string) error     { return nil }
func (infoProvider) Terminate(context.Context, string) error { return nil }

func TestEngineInfoAuthenticatedLiveAndSanitized(t *testing.T) {
	var slots atomic.Int32
	slots.Store(4)
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "GET" || r.URL.Path != "/health/ready" {
			t.Error("unexpected worker request")
		}
		json.NewEncoder(w).Encode(map[string]any{"status": "ok", "secret": "hidden-token", "detail": "private-detail",
			"capabilities": map[string]any{"hardware": map[string]any{"gpu_name": "Actual GPU", "visible_gpu_count": 1, "vram_bytes_per_gpu": int64(24) * 1024 * 1024 * 1024}, "decode_batching": true, "temperature_sampling": true, "pool_slots": slots.Load(), "config_path": "/app/docker/pool.yaml", "engine_env": map[string]string{"HF_TOKEN": "hidden-token"}}})
	}))
	defer backend.Close()
	cp := controlplane.NewControlPlane(context.Background(), config.ControlPlaneConfig{}, engine.NewHttpClient(backend.URL))
	cp.Provider = infoProvider{backend.URL}
	if err := cp.StartEngine(context.Background(), config.GPUProviderConfig{}); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(time.Second)
	for cp.Status().State != controlplane.StateReady && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	server := NewServer(config.ServeConfig{DemoKey: "key", MaxConcurrentGenerations: 4, MaxOutputTokens: 256}, cp)
	for _, authenticated := range []bool{false, true} {
		req := httptest.NewRequest("GET", "/engine/info", nil)
		if authenticated {
			req.Header.Set("X-Demo-Key", "key")
		}
		output := httptest.NewRecorder()
		server.httpServer.Handler.ServeHTTP(output, req)
		if !authenticated {
			if output.Code != http.StatusUnauthorized {
				t.Fatal(output.Code)
			}
			continue
		}
		if output.Code != http.StatusOK || strings.Contains(output.Body.String(), "hidden-token") || strings.Contains(output.Body.String(), "private-detail") || strings.Contains(output.Body.String(), "/app/docker/pool.yaml") {
			t.Fatal(output.Body.String())
		}
		var result engineInfoResponse
		if err := json.Unmarshal(output.Body.Bytes(), &result); err != nil {
			t.Fatal(err)
		}
		if result.Worker.Capabilities.Hardware == nil || *result.Worker.Capabilities.Hardware.GPUName != "Actual GPU" || result.Deployment.Region != nil {
			t.Fatal("hardware metadata lost or region fabricated")
		}
		if result.Worker.Capabilities.PoolSlots != 4 || result.Admission.MaxConcurrentGenerations != 4 || result.ControlPlane.MaxOutputTokens != 256 {
			t.Fatal(result)
		}
	}
	slots.Store(0)
	req := httptest.NewRequest("GET", "/engine/info", nil)
	req.Header.Set("X-Demo-Key", "key")
	output := httptest.NewRecorder()
	server.httpServer.Handler.ServeHTTP(output, req)
	var result engineInfoResponse
	json.Unmarshal(output.Body.Bytes(), &result)
	if result.Worker.Capabilities.PoolSlots != 0 {
		t.Fatal("cached stale capabilities")
	}
	backend.Close()
	output = httptest.NewRecorder()
	server.httpServer.Handler.ServeHTTP(output, req)
	if output.Code != http.StatusServiceUnavailable || strings.Contains(output.Body.String(), backend.URL) {
		t.Fatal(output.Body.String())
	}
}
