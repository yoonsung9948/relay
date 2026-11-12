package gpuprovider

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/yoonsung9948/relay/internal/config"
)

func TestCreatePassesExplicitEngineEnvironment(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload runPodCreateRequest
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Error(err)
		}
		if payload.Env["RELAY_TRACING_ENABLED"] != "true" || payload.Env["OTEL_EXPORTER_OTLP_ENDPOINT"] != "http://collector:4318" {
			t.Error("worker environment not forwarded")
		}
		if payload.Env["RELAY_CONFIG_PATH"] != "/app/docker/pool.yaml" {
			t.Error("worker config selection not forwarded")
		}
		if payload.Env["HF_TOKEN"] != "dedicated-token" {
			t.Error("HF token configuration changed")
		}
		w.WriteHeader(http.StatusCreated)
		w.Write([]byte(`{"id":"test","status":"RUNNING"}`))
	}))
	defer server.Close()
	env := map[string]string{"RELAY_CONFIG_PATH": "/app/docker/pool.yaml", "RELAY_TRACING_ENABLED": "true", "OTEL_EXPORTER_OTLP_ENDPOINT": "http://collector:4318"}
	provider := NewRunPodProvider(&config.RunPodConfig{BaseURL: server.URL, Timeout: time.Second, EnginePort: 8000, HFToken: "dedicated-token", EngineEnv: env})
	if _, err := provider.Create(context.Background(), CreateInstanceSpec{}); err != nil {
		t.Fatal(err)
	}
	if _, ok := env["HF_TOKEN"]; ok {
		t.Error("configuration map was mutated")
	}
}

func TestCreateCapacityClassification(t *testing.T) {
	for _, tc := range []struct {
		status   int
		body     string
		capacity bool
	}{
		{400, `{"detail":"There are no longer any instances available"}`, true},
		{503, `{"detail":"There are no longer any instances available"}`, true},
		{500, `{"detail":"There are no longer any instances available"}`, false},
		{503, `{"detail":"upstream unavailable"}`, false},
	} {
		t.Run(http.StatusText(tc.status)+tc.body, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(tc.status); w.Write([]byte(tc.body)) }))
			defer server.Close()
			provider := NewRunPodProvider(&config.RunPodConfig{BaseURL: server.URL, Timeout: time.Second})
			_, err := provider.Create(context.Background(), CreateInstanceSpec{})
			if errors.Is(err, ErrNoCapacity) != tc.capacity {
				t.Fatalf("err=%v capacity=%v", err, tc.capacity)
			}
		})
	}
}
func TestCreateCanceledRequestDoesNotPanic(t *testing.T) {
	provider := NewRunPodProvider(&config.RunPodConfig{BaseURL: "http://127.0.0.1:1", Timeout: time.Second})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := provider.Create(ctx, CreateInstanceSpec{})
	if !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
}

func TestFindByNameFollowsPaginationAndMatchesExactly(t *testing.T) {
	var pages []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/pods" || r.URL.Query().Get("limit") != "1000" {
			t.Errorf("unexpected request: %s", r.URL)
		}
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Error("missing credentials")
		}
		cursor := r.URL.Query().Get("cursor")
		pages = append(pages, cursor)
		if cursor == "" {
			w.Write([]byte(`{"pods":[{"id":"unrelated","name":"relay-owned-other"}],"pagination":{"hasNextPage":true,"nextCursor":"next/+="}}`))
			return
		}
		if cursor != "next/+=" {
			t.Error("cursor not preserved")
		}
		w.Write([]byte(`{"pods":[{"id":"owned","name":"relay-owned","status":"RUNNING"}],"pagination":{"hasNextPage":false,"nextCursor":null}}`))
	}))
	defer server.Close()
	provider := NewRunPodProvider(&config.RunPodConfig{BaseURL: server.URL, APIKey: "test-key", EnginePort: 8000})
	matches, err := provider.FindByName(context.Background(), "relay-owned")
	if err != nil || len(matches) != 1 || matches[0].ID != "owned" || len(pages) != 2 {
		t.Fatalf("matches=%+v pages=%v err=%v", matches, pages, err)
	}
}

func TestFindByNameRejectsIncompleteOrLoopingPagination(t *testing.T) {
	for _, body := range []string{
		`{"pods":[]}`, `{"pods":[],"pagination":{"hasNextPage":true,"nextCursor":null}}`,
		`{"pods":[],"pagination":{"hasNextPage":true,"nextCursor":"again"}}`,
	} {
		t.Run(body, func(t *testing.T) {
			calls := 0
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { calls++; w.Write([]byte(body)) }))
			defer server.Close()
			provider := NewRunPodProvider(&config.RunPodConfig{BaseURL: server.URL})
			if _, err := provider.FindByName(context.Background(), "owned"); err == nil {
				t.Fatal("incomplete lookup accepted")
			}
			if calls > 2 {
				t.Fatalf("pagination loop: %d calls", calls)
			}
		})
	}
}

func TestTerminateUsesDeleteAndValidatesID(t *testing.T) {
	calls := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		if r.Method != http.MethodDelete || r.URL.Path != "/pods/pod-1" {
			t.Errorf("%s %s", r.Method, r.URL)
		}
		w.WriteHeader(http.StatusNoContent)
	}))
	defer server.Close()
	provider := NewRunPodProvider(&config.RunPodConfig{BaseURL: server.URL})
	if err := provider.Terminate(context.Background(), "pod-1"); err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"", "../other", "pod/other"} {
		if err := provider.Terminate(context.Background(), id); err == nil {
			t.Fatal("invalid ID accepted")
		}
		if _, err := provider.Get(context.Background(), id); err == nil {
			t.Fatal("invalid ID accepted")
		}
	}
	if calls != 1 {
		t.Fatalf("invalid IDs reached provider: %d calls", calls)
	}
}

func TestProviderHTTPFailureClassification(t *testing.T) {
	for _, status := range []int{401, 403, 404, 429, 500, 503} {
		t.Run(http.StatusText(status), func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(status) }))
			defer server.Close()
			provider := NewRunPodProvider(&config.RunPodConfig{BaseURL: server.URL})
			_, err := provider.Get(context.Background(), "pod")
			if status == 404 {
				if !errors.Is(err, ErrNotFound) {
					t.Fatal(err)
				}
				return
			}
			var failure *HTTPError
			if !errors.As(err, &failure) || failure.StatusCode != status {
				t.Fatal(err)
			}
			if Retryable(err) != (status == 429 || status >= 500) {
				t.Fatal("incorrect retry classification", err)
			}
		})
	}
}

func TestCreateMissingIDIsAmbiguousAndScopeProtectsCredentials(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(201); w.Write([]byte(`{"name":"owned"}`)) }))
	defer server.Close()
	provider := NewRunPodProvider(&config.RunPodConfig{BaseURL: server.URL, APIKey: "one"})
	if _, err := provider.Create(context.Background(), CreateInstanceSpec{}); err == nil || DefinitiveCreateFailure(err) {
		t.Fatal("missing ID did not require reconciliation", err)
	}
	other := NewRunPodProvider(&config.RunPodConfig{BaseURL: server.URL, APIKey: "two"})
	if provider.OwnershipScope() == other.OwnershipScope() {
		t.Fatal("credential rotation not detected")
	}
}
