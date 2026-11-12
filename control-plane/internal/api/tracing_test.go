package api

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/yoonsung9948/relay/internal/config"
	"github.com/yoonsung9948/relay/internal/controlplane"
	"github.com/yoonsung9948/relay/internal/observability"
	"github.com/yoonsung9948/relay/internal/response"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	traceapi "go.opentelemetry.io/otel/trace"
)

func TestStreamErrorMarksHTTP200TraceFailed(t *testing.T) {
	t.Setenv("RELAY_TRACING_ENABLED", "true")
	exporter := tracetest.NewInMemoryExporter()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
	previous := otel.GetTracerProvider()
	otel.SetTracerProvider(provider)
	defer otel.SetTracerProvider(previous)
	defer provider.Shutdown(context.Background())
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	id := "engine-request"
	cp := controlplane.NewControlPlane(ctx, config.ControlPlaneConfig{}, &streamClient{events: []response.StreamResponse{
		{Chunk: response.GenerateChunk{Type: "delta", RequestID: &id, Text: "hello"}},
		{Err: errors.New("private engine details")},
	}})
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
	server := &Server{control: cp}
	recorder := httptest.NewRecorder()
	metrics := observability.NewMetrics(make(chan struct{}, 2))
	observability.Handler(metrics.Handler(http.HandlerFunc(server.GenerateHandler))).ServeHTTP(recorder,
		httptest.NewRequest("POST", "/generate", strings.NewReader(`{"prompt":"hello","stream":true}`)).WithContext(ctx))
	if recorder.Code != 200 {
		t.Fatal("expected already-started HTTP 200 stream")
	}
	scrape := httptest.NewRecorder()
	metrics.ServeHTTP(scrape, httptest.NewRequest("GET", "/metrics", nil))
	if !strings.Contains(scrape.Body.String(), `status="200",outcome="failed"`) {
		t.Fatal("stream failure counted as HTTP success", scrape.Body.String())
	}
	spans := exporter.GetSpans()
	for _, span := range spans {
		if span.SpanKind == traceapi.SpanKindServer {
			if span.Status.Code != codes.Error {
				t.Fatal("stream failure was not marked")
			}
			return
		}
	}
	t.Fatal("server span missing")
}
