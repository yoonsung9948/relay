package api

import (
	"context"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/yoonsung9948/relay/internal/config"
	"github.com/yoonsung9948/relay/internal/controlplane"
)

func TestMetricsRouteAuthenticatedAndAdmissionObserved(t *testing.T) {
	cp := controlplane.NewControlPlane(context.Background(), config.ControlPlaneConfig{}, nil)
	server := NewServer(config.ServeConfig{DemoKey: "test-key", MaxConcurrentGenerations: 4}, cp)
	handler := server.httpServer.Handler
	denied := httptest.NewRecorder()
	handler.ServeHTTP(denied, httptest.NewRequest("GET", "/metrics", nil))
	if denied.Code != 401 {
		t.Fatal("unauthenticated metrics allowed")
	}
	handler.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest("POST", "/generate", strings.NewReader(`{"prompt":"hi"}`)))
	request := httptest.NewRequest("GET", "/metrics", nil)
	request.Header.Set("X-Demo-Key", "test-key")
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, request)
	for _, expected := range []string{
		`relay_control_plane_rejections_total{reason="unauthorized"} 1`,
		`relay_control_plane_http_requests_total{status="401",outcome="rejected"`,
		"relay_control_plane_generations_limit 4",
	} {
		if !strings.Contains(recorder.Body.String(), expected) {
			t.Fatal("missing", expected, recorder.Body.String())
		}
	}
	if recorder.Code != 200 {
		t.Fatal(recorder.Code)
	}
}
