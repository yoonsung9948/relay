package controlplane

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"testing/synctest"
	"time"

	"github.com/yoonsung9948/relay/internal/config"
	"github.com/yoonsung9948/relay/internal/controlplane/gpuprovider"
	"github.com/yoonsung9948/relay/internal/request"
	"github.com/yoonsung9948/relay/internal/response"
)

type fakeProvider struct {
	mu          sync.Mutex
	createFn    func(ctx context.Context, spec gpuprovider.CreateInstanceSpec) (*gpuprovider.Instance, error)
	terminateFn func(ctx context.Context, id string, call int) error // call is 1-based per id
	created     []gpuprovider.CreateInstanceSpec
	termCalls   map[string]int
}

func (f *fakeProvider) Create(ctx context.Context, s gpuprovider.CreateInstanceSpec) (*gpuprovider.Instance, error) {
	f.mu.Lock()
	f.created = append(f.created, s)
	f.mu.Unlock()
	if f.createFn != nil {
		return f.createFn(ctx, s)
	}
	return &gpuprovider.Instance{ID: "pod-" + s.Name, Endpoint: "http://fake"}, nil
}

func (f *fakeProvider) Terminate(ctx context.Context, id string) error {
	f.mu.Lock()
	if f.termCalls == nil {
		f.termCalls = map[string]int{}
	}
	f.termCalls[id]++
	n := f.termCalls[id]
	f.mu.Unlock()
	if f.terminateFn != nil {
		return f.terminateFn(ctx, id, n)
	}
	return nil
}

func (f *fakeProvider) Get(context.Context, string) (*gpuprovider.Instance, error) { return nil, nil }
func (f *fakeProvider) Stop(context.Context, string) error                         { return nil }
func (f *fakeProvider) Start(context.Context, string) error                        { return nil }

type fakeClient struct {
	healthFn   func(ctx context.Context) error
	generateFn func(ctx context.Context, r request.GenerateRequest) (response.GenerateResponse, error)
	mu         sync.Mutex
	endpoint   string
	genCalls   int
}

func (c *fakeClient) Health(ctx context.Context) error {
	if c.healthFn != nil {
		return c.healthFn(ctx)
	}
	return nil // healthy immediately
}

func (c *fakeClient) Generate(ctx context.Context, r request.GenerateRequest) (response.GenerateResponse, error) {
	c.mu.Lock()
	c.genCalls++
	c.mu.Unlock()
	if c.generateFn != nil {
		return c.generateFn(ctx, r)
	}
	return response.GenerateResponse{Text: "ok"}, nil
}

func (c *fakeClient) SetEndpoint(e string) {
	c.mu.Lock()
	c.endpoint = e
	c.mu.Unlock()
}

func newTestCP(t *testing.T, p *fakeProvider, c *fakeClient) *ControlPlane {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	cp := NewControlPlane(ctx, config.ControlPlaneConfig{}, c)
	cp.Provider = p
	return cp
}

func waitForState(t *testing.T, cp *ControlPlane, want EngineState) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if cp.getState() == want {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("state = %s, want %s", cp.getState(), want)
}

// ---------- extra helpers ----------

func podIDs(cp *ControlPlane) []string {
	cp.mu.RLock()
	defer cp.mu.RUnlock()
	return append([]string(nil), cp.instanceIDs...)
}

func terminateCalls(p *fakeProvider, id string) int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.termCalls[id]
}

// scripted returns a terminateFn that replays script[id] by call number
// (1-based). Calls past the end of the script succeed.
func scripted(script map[string][]error) func(context.Context, string, int) error {
	return func(_ context.Context, id string, call int) error {
		s := script[id]
		if call > len(s) {
			return nil
		}
		return s[call-1]
	}
}

func repeatErr(err error, n int) []error {
	out := make([]error, n)
	for i := range out {
		out[i] = err
	}
	return out
}

func TestStartEngine_CreateFailures(t *testing.T) {
	boom := errors.New("boom")

	tests := []struct {
		name      string
		createErr error
		wantState EngineState
	}{
		{"no capacity returns to offline", gpuprovider.ErrNoCapacity, StateOffline},
		{"other create error goes to error", boom, StateError},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p := &fakeProvider{
				createFn: func(context.Context, gpuprovider.CreateInstanceSpec) (*gpuprovider.Instance, error) {
					return nil, tc.createErr
				},
			}
			cp := newTestCP(t, p, &fakeClient{})

			err := cp.StartEngine(context.Background(), config.GPUProviderConfig{})

			if !errors.Is(err, tc.createErr) {
				t.Fatalf("err = %v, want it to wrap %v", err, tc.createErr)
			}
			if got := cp.getState(); got != tc.wantState {
				t.Errorf("state = %s, want %s", got, tc.wantState)
			}
			if ids := podIDs(cp); len(ids) != 0 {
				t.Errorf("recorded pods = %v, want none", ids)
			}
			// Only the error path surfaces a message in Status.
			if tc.wantState == StateError && cp.Status().Error == "" {
				t.Error("Status().Error is empty after a failed create")
			}
		})
	}
}

func TestShutdown_Outcomes(t *testing.T) {
	boom := errors.New("boom")

	tests := []struct {
		name          string
		script        map[string][]error // terminate results per pod, by call number
		wantState     EngineState
		wantErrSubstr string         // empty means Shutdown must succeed
		wantCalls     map[string]int // exact Terminate attempts per pod
		wantPodsLeft  int            // pods still tracked afterwards
	}{
		{
			name:         "all pods terminate cleanly",
			script:       nil,
			wantState:    StateOffline,
			wantCalls:    map[string]int{"a": 1, "b": 1, "c": 1},
			wantPodsLeft: 0,
		},
		{
			name:         "transient error is retried",
			script:       map[string][]error{"b": {boom}},
			wantState:    StateOffline,
			wantCalls:    map[string]int{"a": 1, "b": 2, "c": 1},
			wantPodsLeft: 0,
		},
		{
			name:         "not found counts as already gone",
			script:       map[string][]error{"a": {gpuprovider.ErrNotFound}},
			wantState:    StateOffline,
			wantCalls:    map[string]int{"a": 1, "b": 1, "c": 1}, // not retried
			wantPodsLeft: 0,
		},
		{
			name:          "one stuck pod is reported, others still terminated",
			script:        map[string][]error{"b": repeatErr(boom, 5)},
			wantState:     StateError,
			wantErrSubstr: "pod b",
			wantCalls:     map[string]int{"a": 1, "b": 5, "c": 1},
			wantPodsLeft:  3, // list is kept so a retry can find them
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			// synctest fakes the clock, so the retry backoff (1s, 2s, 4s, ...)
			// runs instantly instead of taking ~25s for the stuck-pod case.
			synctest.Test(t, func(t *testing.T) {
				p := &fakeProvider{terminateFn: scripted(tc.script)}
				cp := newTestCP(t, p, &fakeClient{})
				cp.instanceIDs = []string{"a", "b", "c"}

				err := cp.Shutdown(context.Background())

				if tc.wantErrSubstr == "" && err != nil {
					t.Fatalf("Shutdown error: %v", err)
				}
				if tc.wantErrSubstr != "" {
					if err == nil || !strings.Contains(err.Error(), tc.wantErrSubstr) {
						t.Fatalf("err = %v, want it to contain %q", err, tc.wantErrSubstr)
					}
				}
				if got := cp.getState(); got != tc.wantState {
					t.Errorf("state = %s, want %s", got, tc.wantState)
				}
				for id, want := range tc.wantCalls {
					if got := terminateCalls(p, id); got != want {
						t.Errorf("Terminate(%s) calls = %d, want %d", id, got, want)
					}
				}
				if got := len(podIDs(cp)); got != tc.wantPodsLeft {
					t.Errorf("pods still tracked = %d, want %d", got, tc.wantPodsLeft)
				}
			})
		})
	}
}

func TestStartEngine_ConcurrentCallsProvisionOnce(t *testing.T) {
	const callers = 5
	gate := make(chan struct{})
	p := &fakeProvider{
		createFn: func(context.Context, gpuprovider.CreateInstanceSpec) (*gpuprovider.Instance, error) {
			<-gate // hold the winner inside Create so the others overlap with it
			return &gpuprovider.Instance{ID: "pod-1", Endpoint: "http://fake"}, nil
		},
	}
	cp := newTestCP(t, p, &fakeClient{})

	start := make(chan struct{})
	var ready sync.WaitGroup
	ready.Add(callers)

	results := make(chan error, callers)
	for i := 0; i < callers; i++ {
		go func() {
			ready.Done()
			<-start
			results <- cp.StartEngine(context.Background(), config.GPUProviderConfig{})
		}()
	}
	ready.Wait()
	close(start)

	for i := 0; i < callers-1; i++ {
		select {
		case err := <-results:
			if err == nil {
				t.Fatal("a second caller succeeded while the first was still provisioning")
			}
		case <-time.After(2 * time.Second):
			t.Fatal("timed out waiting for rejected callers")
		}
	}

	close(gate)
	select {
	case err := <-results:
		if err != nil {
			t.Fatalf("winning caller failed: %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for the winning caller")
	}
	waitForState(t, cp, StateReady)

	p.mu.Lock()
	created := len(p.created)
	p.mu.Unlock()
	if created != 1 {
		t.Errorf("Create called %d times, want 1", created)
	}
	if ids := podIDs(cp); len(ids) != 1 {
		t.Errorf("tracked pods = %v, want exactly one", ids)
	}
}

// ---------- Workflow: hung health poll ----------

func TestStartEngine_HungHealthTimesOutToError(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		var healthCalls int
		c := &fakeClient{
			healthFn: func(context.Context) error {
				healthCalls++
				return errors.New("not ready")
			},
		}
		cp := newTestCP(t, &fakeProvider{}, c)

		if err := cp.StartEngine(context.Background(), config.GPUProviderConfig{}); err != nil {
			t.Fatalf("StartEngine: %v", err)
		}

		time.Sleep(16 * time.Minute) // past the 15m workflow timeout, in fake time
		synctest.Wait()              // let the workflow goroutine finish

		if got := cp.getState(); got != StateError {
			t.Errorf("state = %s, want %s", got, StateError)
		}
		if msg := cp.Status().Error; !strings.Contains(msg, "deadline exceeded") {
			t.Errorf("Status().Error = %q, want it to mention the deadline", msg)
		}
		if healthCalls < 2 {
			t.Errorf("health polled %d times, want it to keep retrying until the timeout", healthCalls)
		}
	})
}
