package observability

import (
	"context"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"
)

type metricContextKey struct{}
type requestMeasurement struct {
	outcome, reason, mode string
	batching              bool
}

// These labels are fixed enums, never prompts, IDs, URLs or exception messages.
func SetExecution(ctx context.Context, mode string, batching bool) {
	if m, ok := ctx.Value(metricContextKey{}).(*requestMeasurement); ok {
		switch mode {
		case "none", "dynamic", "pool":
			m.mode = mode
		default:
			m.mode = "none"
		}
		m.batching = batching
	}
}
func SetOutcome(ctx context.Context, outcome string) {
	if m, ok := ctx.Value(metricContextKey{}).(*requestMeasurement); ok {
		if m.reason == "" {
			switch outcome {
			case "success", "failed", "cancelled":
				m.outcome = outcome
			}
		}
	}
}
func Rejected(ctx context.Context, reason string) {
	if m, ok := ctx.Value(metricContextKey{}).(*requestMeasurement); ok {
		switch reason {
		case "unauthorized", "generation_capacity", "request_rate_limited", "engine_capacity", "engine_invalid_request":
			m.outcome, m.reason = "rejected", reason
		}
	}
}

var durationBuckets = [...]float64{.001, .005, .01, .05, .1, .5, 1, 5, 15, 60, 180}

type httpLabels struct {
	status        int
	outcome, mode string
	batching      bool
}
type durationValues struct {
	count   uint64
	sum     float64
	buckets [len(durationBuckets)]uint64
}

// Metrics belongs to one server; its mutex also makes scrapes race-free.
type Metrics struct {
	mu       sync.Mutex
	inFlight int
	requests map[httpLabels]durationValues
	rejected map[string]uint64
	capacity chan struct{}
}

func NewMetrics(capacity chan struct{}) *Metrics {
	return &Metrics{requests: make(map[httpLabels]durationValues), rejected: make(map[string]uint64), capacity: capacity}
}

type measuredWriter struct {
	http.ResponseWriter
	status int
}

func (w *measuredWriter) Unwrap() http.ResponseWriter { return w.ResponseWriter }
func (w *measuredWriter) WriteHeader(status int) {
	if w.status == 0 {
		w.status = status
		w.ResponseWriter.WriteHeader(status)
	}
}
func (w *measuredWriter) Write(data []byte) (int, error) {
	if w.status == 0 {
		w.WriteHeader(http.StatusOK)
	}
	return w.ResponseWriter.Write(data)
}
func (w *measuredWriter) Flush() { _ = w.FlushError() }
func (w *measuredWriter) FlushError() error {
	if w.status == 0 {
		w.WriteHeader(http.StatusOK)
	}
	return http.NewResponseController(w.ResponseWriter).Flush()
}

func (m *Metrics) Handler(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		measurement := &requestMeasurement{outcome: "failed", mode: "unknown"}
		ctx := context.WithValue(r.Context(), metricContextKey{}, measurement)
		wrapped := &measuredWriter{ResponseWriter: w}
		started := time.Now()
		m.mu.Lock()
		m.inFlight++
		m.mu.Unlock()
		defer func() {
			outcome := measurement.outcome
			if ctx.Err() != nil && outcome != "success" {
				outcome = "cancelled"
			}
			status := wrapped.status
			if status == 0 {
				status = http.StatusInternalServerError
			}
			key := httpLabels{status, outcome, measurement.mode, measurement.batching}
			seconds := time.Since(started).Seconds()
			m.mu.Lock()
			defer m.mu.Unlock()
			m.inFlight--
			value := m.requests[key]
			value.count++
			value.sum += seconds
			for i, bound := range durationBuckets {
				if seconds <= bound {
					value.buckets[i]++
				}
			}
			m.requests[key] = value
			if measurement.reason != "" {
				m.rejected[measurement.reason]++
			}
		}()
		next.ServeHTTP(wrapped, r.WithContext(ctx))
	})
}

func (m *Metrics) ServeHTTP(w http.ResponseWriter, _ *http.Request) {
	var out strings.Builder
	m.mu.Lock()
	fmt.Fprintf(&out, "# TYPE relay_control_plane_http_in_flight gauge\nrelay_control_plane_http_in_flight %d\n", m.inFlight)
	fmt.Fprintf(&out, "# TYPE relay_control_plane_generations_active gauge\nrelay_control_plane_generations_active %d\n", len(m.capacity))
	fmt.Fprintf(&out, "# TYPE relay_control_plane_generations_limit gauge\nrelay_control_plane_generations_limit %d\n", cap(m.capacity))
	out.WriteString("# TYPE relay_control_plane_http_requests_total counter\n# TYPE relay_control_plane_http_duration_seconds histogram\n")
	for key, value := range m.requests {
		labels := fmt.Sprintf("status=\"%d\",outcome=%q,mode=%q,batching=%q", key.status, key.outcome, key.mode, strconv.FormatBool(key.batching))
		fmt.Fprintf(&out, "relay_control_plane_http_requests_total{%s} %d\n", labels, value.count)
		for i, bound := range durationBuckets {
			fmt.Fprintf(&out, "relay_control_plane_http_duration_seconds_bucket{%s,le=%q} %d\n", labels, strconv.FormatFloat(bound, 'g', -1, 64), value.buckets[i])
		}
		fmt.Fprintf(&out, "relay_control_plane_http_duration_seconds_bucket{%s,le=\"+Inf\"} %d\n", labels, value.count)
		fmt.Fprintf(&out, "relay_control_plane_http_duration_seconds_count{%s} %d\n", labels, value.count)
		fmt.Fprintf(&out, "relay_control_plane_http_duration_seconds_sum{%s} %g\n", labels, value.sum)
	}
	out.WriteString("# TYPE relay_control_plane_rejections_total counter\n")
	for reason, count := range m.rejected {
		fmt.Fprintf(&out, "relay_control_plane_rejections_total{reason=%q} %d\n", reason, count)
	}
	m.mu.Unlock()
	w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
	_, _ = w.Write([]byte(out.String()))
}
