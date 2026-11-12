package observability

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
)

func TestMetricsStreamOutcomeAndRejectedReason(t *testing.T) {
	sem := make(chan struct{}, 3)
	metrics := NewMetrics(sem)
	for _, outcome := range []string{"success", "failed", "cancelled", "rejected"} {
		handler := metrics.Handler(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			SetExecution(r.Context(), "pool", true)
			SetOutcome(r.Context(), outcome)
			if outcome == "rejected" {
				Rejected(r.Context(), "engine_capacity")
				// The handler's failure defer must not overwrite a rejection.
				SetOutcome(r.Context(), "failed")
				w.WriteHeader(503)
			} else {
				w.Write([]byte("data: {}\n\n"))
				if err := http.NewResponseController(w).Flush(); err != nil {
					t.Fatal(err)
				}
			}
		}))
		handler.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest("POST", "/generate", nil))
	}
	recorder := httptest.NewRecorder()
	metrics.ServeHTTP(recorder, httptest.NewRequest("GET", "/metrics", nil))
	text := recorder.Body.String()
	for _, expected := range []string{
		`status="200",outcome="failed",mode="pool",batching="true"`,
		`status="200",outcome="cancelled",mode="pool",batching="true"`,
		`status="503",outcome="rejected",mode="pool",batching="true"`,
		`relay_control_plane_rejections_total{reason="engine_capacity"} 1`,
		"relay_control_plane_http_in_flight 0",
		"relay_control_plane_generations_limit 3",
	} {
		if !strings.Contains(text, expected) {
			t.Fatalf("missing %s in %s", expected, text)
		}
	}
}

func TestMetricsConcurrentScrapeAndRequests(t *testing.T) {
	m := NewMetrics(make(chan struct{}, 2))
	handler := m.Handler(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		SetOutcome(r.Context(), "success")
		w.WriteHeader(200)
	}))
	var wg sync.WaitGroup
	for range 20 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			handler.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest("POST", "/generate", nil))
			m.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest("GET", "/metrics", nil))
		}()
	}
	wg.Wait()
	recorder := httptest.NewRecorder()
	m.ServeHTTP(recorder, httptest.NewRequest("GET", "/metrics", nil))
	if !strings.Contains(recorder.Body.String(), `relay_control_plane_http_requests_total{status="200",outcome="success",mode="unknown",batching="false"} 20`) {
		t.Fatal(recorder.Body.String())
	}
}
