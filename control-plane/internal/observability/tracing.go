// Package observability provides opt-in tracing without benchmark-specific state.
package observability

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
)

func Enabled() bool {
	return strings.EqualFold(os.Getenv("RELAY_TRACING_ENABLED"), "true") &&
		!strings.EqualFold(os.Getenv("OTEL_SDK_DISABLED"), "true")
}

func samplerFromEnv() (sdktrace.Sampler, error) {
	name := os.Getenv("OTEL_TRACES_SAMPLER")
	ratio := 1.0
	if strings.Contains(name, "traceidratio") {
		if raw := os.Getenv("OTEL_TRACES_SAMPLER_ARG"); raw != "" {
			var err error
			ratio, err = strconv.ParseFloat(raw, 64)
			if err != nil || !(ratio >= 0 && ratio <= 1) {
				return nil, fmt.Errorf("OTEL_TRACES_SAMPLER_ARG must be between 0 and 1")
			}
		}
	}
	switch name {
	case "", "parentbased_always_on":
		return sdktrace.ParentBased(sdktrace.AlwaysSample()), nil
	case "parentbased_always_off":
		return sdktrace.ParentBased(sdktrace.NeverSample()), nil
	case "always_on":
		return sdktrace.AlwaysSample(), nil
	case "always_off":
		return sdktrace.NeverSample(), nil
	case "traceidratio":
		return sdktrace.TraceIDRatioBased(ratio), nil
	case "parentbased_traceidratio":
		return sdktrace.ParentBased(sdktrace.TraceIDRatioBased(ratio)), nil
	default:
		return nil, fmt.Errorf("unsupported OTEL_TRACES_SAMPLER: %s", name)
	}
}

func Init(ctx context.Context) (func(context.Context) error, error) {
	if !Enabled() {
		return func(context.Context) error { return nil }, nil
	}
	sampler, err := samplerFromEnv()
	if err != nil {
		return nil, err
	}
	instance := make([]byte, 16)
	if _, err := rand.Read(instance); err != nil {
		return nil, err
	}
	name := os.Getenv("OTEL_SERVICE_NAME")
	if name == "" {
		name = "relay-control-plane"
	}
	revision := os.Getenv("RELAY_REVISION")
	if revision == "" {
		revision = "unknown"
	}
	res, err := resource.New(ctx, resource.WithFromEnv(), resource.WithAttributes(
		attribute.String("service.name", name), attribute.String("service.version", revision),
		attribute.String("service.instance.id", hex.EncodeToString(instance)),
	))
	if err != nil {
		return nil, err
	}
	exporter, err := otlptracehttp.New(ctx, otlptracehttp.WithTimeout(2*time.Second))
	if err != nil {
		return nil, fmt.Errorf("initialize trace exporter: %w", err)
	}
	provider := sdktrace.NewTracerProvider(
		sdktrace.WithResource(res),
		sdktrace.WithSampler(sampler),
		sdktrace.WithBatcher(exporter,
			sdktrace.WithMaxQueueSize(1024),
			sdktrace.WithMaxExportBatchSize(128),
			sdktrace.WithBatchTimeout(time.Second),
			sdktrace.WithExportTimeout(3*time.Second),
		),
	)
	otel.SetTracerProvider(provider)
	otel.SetTextMapPropagator(propagation.TraceContext{})
	return provider.Shutdown, nil
}

func Handler(next http.Handler) http.Handler {
	if !Enabled() {
		return next
	}
	return otelhttp.NewHandler(next, "POST /generate", otelhttp.WithPropagators(propagation.TraceContext{}))
}

func Transport(base http.RoundTripper) http.RoundTripper {
	return otelhttp.NewTransport(base, otelhttp.WithPropagators(propagation.TraceContext{}),
		otelhttp.WithSpanNameFormatter(func(_ string, _ *http.Request) string { return "engine.request" }))
}
