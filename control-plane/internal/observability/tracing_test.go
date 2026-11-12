package observability

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

func TestHTTPPropagationAndStreamingLifetime(t *testing.T) {
	t.Setenv("RELAY_TRACING_ENABLED", "true")
	exporter := tracetest.NewInMemoryExporter()
	provider := trace.NewTracerProvider(trace.WithSyncer(exporter))
	previous := otel.GetTracerProvider()
	otel.SetTracerProvider(provider)
	defer otel.SetTracerProvider(previous)
	defer provider.Shutdown(context.Background())

	engine := httptest.NewServer(Handler(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("traceparent") == "" {
			t.Error("missing propagated traceparent")
		}
		w.Write([]byte("token\n"))
		w.(http.Flusher).Flush()
	})))
	defer engine.Close()
	control := Handler(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		req, _ := http.NewRequestWithContext(r.Context(), "POST", engine.URL, nil)
		response, err := (&http.Client{Transport: Transport(http.DefaultTransport)}).Do(req)
		if err != nil {
			t.Error(err)
			return
		}
		// Client span must remain open until the response body is read/closed.
		for _, span := range exporter.GetSpans() {
			if span.Name == "engine.request" {
				t.Error("client span ended before body consumption")
			}
		}
		io.Copy(w, response.Body)
		response.Body.Close()
	}))
	request := httptest.NewRequest("POST", "/generate", nil)
	request.Header.Set("traceparent", "00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01")
	control.ServeHTTP(httptest.NewRecorder(), request)
	spans := exporter.GetSpans()
	if len(spans) != 3 {
		t.Fatalf("expected 3 spans, got %d", len(spans))
	}
	for _, span := range spans {
		if span.SpanContext.TraceID().String() != "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" {
			t.Error("trace ID was not preserved")
		}
	}
}

func TestSamplerConfiguration(t *testing.T) {
	for _, name := range []string{"", "parentbased_always_on", "parentbased_always_off", "always_on", "always_off", "traceidratio", "parentbased_traceidratio"} {
		t.Setenv("OTEL_TRACES_SAMPLER", name)
		t.Setenv("OTEL_TRACES_SAMPLER_ARG", "0.5")
		if _, err := samplerFromEnv(); err != nil {
			t.Fatal(err)
		}
	}
	t.Setenv("OTEL_TRACES_SAMPLER", "traceidratio")
	for _, value := range []string{"NaN", "-1", "1.1", "invalid"} {
		t.Setenv("OTEL_TRACES_SAMPLER_ARG", value)
		if _, err := samplerFromEnv(); err == nil {
			t.Errorf("accepted ratio %s", value)
		}
	}
}
