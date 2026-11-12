package controlplane

import (
	"context"
	"errors"
	"reflect"
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

// ---------- Fake GPU provider ----------

type fakeProvider struct {
	mu sync.Mutex

	createFn func(
		ctx context.Context,
		spec gpuprovider.CreateInstanceSpec,
	) (*gpuprovider.Instance, error)

	terminateFn func(
		ctx context.Context,
		id string,
		call int,
	) error

	created   []gpuprovider.CreateInstanceSpec
	termCalls map[string]int
	instances map[string]*gpuprovider.Instance
	deleted   map[string]bool
	getFn     func(context.Context, string) (*gpuprovider.Instance, error)
	findFn    func(context.Context, string) ([]*gpuprovider.Instance, error)
}

func (f *fakeProvider) Create(
	ctx context.Context,
	spec gpuprovider.CreateInstanceSpec,
) (*gpuprovider.Instance, error) {
	f.mu.Lock()
	f.created = append(f.created, spec)
	f.mu.Unlock()

	var instance *gpuprovider.Instance
	var err error
	if f.createFn != nil {
		instance, err = f.createFn(ctx, spec)
	} else {
		instance = &gpuprovider.Instance{ID: "pod-" + spec.Name, Endpoint: "http://fake"}
	}
	if instance != nil {
		f.mu.Lock()
		if f.instances == nil {
			f.instances = make(map[string]*gpuprovider.Instance)
		}
		copy := *instance
		if copy.Name == "" {
			copy.Name = spec.Name
		}
		f.instances[copy.ID] = &copy
		f.mu.Unlock()
	}
	return instance, err
}

func (f *fakeProvider) Terminate(
	ctx context.Context,
	id string,
) error {
	f.mu.Lock()

	if f.termCalls == nil {
		f.termCalls = make(map[string]int)
	}

	f.termCalls[id]++
	call := f.termCalls[id]

	f.mu.Unlock()

	var err error
	if f.terminateFn != nil {
		err = f.terminateFn(ctx, id, call)
	}
	if err == nil || errors.Is(err, gpuprovider.ErrNotFound) {
		f.mu.Lock()
		if f.deleted == nil {
			f.deleted = make(map[string]bool)
		}
		f.deleted[id] = true
		f.mu.Unlock()
	}
	return err
}
func (f *fakeProvider) Get(ctx context.Context, id string) (*gpuprovider.Instance, error) {
	if f.getFn != nil {
		return f.getFn(ctx, id)
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.deleted[id] || f.instances[id] == nil {
		return nil, gpuprovider.ErrNotFound
	}
	copy := *f.instances[id]
	return &copy, nil
}
func (f *fakeProvider) FindByName(ctx context.Context, name string) ([]*gpuprovider.Instance, error) {
	if f.findFn != nil {
		return f.findFn(ctx, name)
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	var matches []*gpuprovider.Instance
	for id, instance := range f.instances {
		if instance.Name == name && !f.deleted[id] {
			copy := *instance
			matches = append(matches, &copy)
		}
	}
	return matches, nil
}

func (f *fakeProvider) Stop(
	context.Context,
	string,
) error {
	return nil
}

func (f *fakeProvider) Start(
	context.Context,
	string,
) error {
	return nil
}

// ---------- Fake engine client ----------

type fakeClient struct {
	mu sync.Mutex

	healthFn func(
		ctx context.Context,
	) error

	generateFn func(
		ctx context.Context,
		req request.GenerateRequest,
	) (response.GenerateResponse, error)

	generateStreamFn func(
		ctx context.Context,
		req request.GenerateRequest,
	) (<-chan response.StreamResponse, error)

	endpoint string

	genCalls    int
	streamCalls int

	lastGenerateRequest request.GenerateRequest
	lastStreamRequest   request.GenerateRequest
}

func (c *fakeClient) Health(ctx context.Context) error {
	if c.healthFn != nil {
		return c.healthFn(ctx)
	}

	return nil
}

func (c *fakeClient) Generate(
	ctx context.Context,
	req request.GenerateRequest,
) (response.GenerateResponse, error) {
	c.mu.Lock()
	c.genCalls++
	c.lastGenerateRequest = req
	c.mu.Unlock()

	if c.generateFn != nil {
		return c.generateFn(ctx, req)
	}

	return response.GenerateResponse{
		Text: "ok",
	}, nil
}

func (c *fakeClient) GenerateStream(
	ctx context.Context,
	req request.GenerateRequest,
) (<-chan response.StreamResponse, error) {
	c.mu.Lock()
	c.streamCalls++
	c.lastStreamRequest = req
	c.mu.Unlock()

	if c.generateStreamFn != nil {
		return c.generateStreamFn(ctx, req)
	}

	// Default: successfully established, empty stream.
	results := make(chan response.StreamResponse)
	close(results)

	return results, nil
}

func (c *fakeClient) SetEndpoint(endpoint string) {
	c.mu.Lock()
	c.endpoint = endpoint
	c.mu.Unlock()
}

// ---------- Test helpers ----------

func newTestCP(
	t *testing.T,
	provider *fakeProvider,
	client *fakeClient,
) *ControlPlane {
	t.Helper()

	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)

	cp := NewControlPlane(
		ctx,
		config.ControlPlaneConfig{},
		client,
	)

	cp.Provider = provider

	return cp
}

func waitForState(
	t *testing.T,
	cp *ControlPlane,
	want EngineState,
) {
	t.Helper()

	deadline := time.Now().Add(2 * time.Second)

	for time.Now().Before(deadline) {
		if cp.getState() == want {
			return
		}

		time.Sleep(5 * time.Millisecond)
	}

	t.Fatalf(
		"state = %s, want %s",
		cp.getState(),
		want,
	)
}

func newReadyTestCP(
	t *testing.T,
	provider *fakeProvider,
	client *fakeClient,
) *ControlPlane {
	t.Helper()

	cp := newTestCP(t, provider, client)

	err := cp.StartEngine(
		context.Background(),
		config.GPUProviderConfig{},
	)
	if err != nil {
		t.Fatalf("StartEngine: %v", err)
	}

	waitForState(t, cp, StateReady)

	return cp
}

func podIDs(cp *ControlPlane) []string {
	cp.mu.RLock()
	defer cp.mu.RUnlock()

	if cp.worker == nil || cp.worker.InstanceID == "" {
		return nil
	}
	return []string{cp.worker.InstanceID}
}

func terminateCalls(
	provider *fakeProvider,
	id string,
) int {
	provider.mu.Lock()
	defer provider.mu.Unlock()

	return provider.termCalls[id]
}

// scripted returns a terminateFn that replays errors by call number.
// Calls past the end of the script succeed.

func scripted(
	script map[string][]error,
) func(context.Context, string, int) error {
	return func(
		_ context.Context,
		id string,
		call int,
	) error {
		results := script[id]

		if call > len(results) {
			return nil
		}

		return results[call-1]
	}
}

func repeatErr(err error, n int) []error {
	out := make([]error, n)

	for i := range out {
		out[i] = err
	}

	return out
}

// ---------- Engine creation tests ----------

func TestStartEngine_CreateFailures(t *testing.T) {
	boom := errors.New("boom")

	tests := []struct {
		name      string
		createErr error
		wantState EngineState
	}{
		{
			name:      "no capacity returns to offline",
			createErr: gpuprovider.ErrNoCapacity,
			wantState: StateOffline,
		},
		{
			name:      "other create error goes to error",
			createErr: boom,
			wantState: StateError,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			provider := &fakeProvider{
				createFn: func(
					context.Context,
					gpuprovider.CreateInstanceSpec,
				) (*gpuprovider.Instance, error) {
					return nil, tc.createErr
				},
			}

			cp := newTestCP(
				t,
				provider,
				&fakeClient{},
			)

			err := cp.StartEngine(
				context.Background(),
				config.GPUProviderConfig{},
			)

			if !errors.Is(err, tc.createErr) {
				t.Fatalf(
					"err = %v, want it to wrap %v",
					err,
					tc.createErr,
				)
			}

			if tc.wantState == StateError {
				_ = cp.StopEngine(context.Background(), "test_reconcile")
			}
			waitForState(t, cp, tc.wantState)
			if got := cp.getState(); got != tc.wantState {
				t.Errorf(
					"state = %s, want %s",
					got,
					tc.wantState,
				)
			}

			if ids := podIDs(cp); len(ids) != 0 {
				t.Errorf(
					"recorded pods = %v, want none",
					ids,
				)
			}

			if tc.wantState == StateError &&
				cp.Status().Error == "" {
				t.Error(
					"Status().Error is empty after failed create",
				)
			}
		})
	}
}

// ---------- Engine shutdown tests ----------

func TestShutdown_Outcomes(t *testing.T) {
	transient := &gpuprovider.HTTPError{StatusCode: 503, Operation: "terminate"}
	permanent := &gpuprovider.HTTPError{StatusCode: 401, Operation: "terminate"}
	for _, tc := range []struct {
		name   string
		errors []error
		calls  int
		fail   bool
	}{
		{"clean", nil, 1, false},
		{"transient retry", []error{transient}, 2, false},
		{"already gone", []error{gpuprovider.ErrNotFound}, 1, false},
		{"permanent retained", []error{permanent}, 1, true},
		{"retry exhausted retained", repeatErr(transient, 5), 5, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				provider := &fakeProvider{}
				cp := newReadyTestCP(t, provider, &fakeClient{})
				id := podIDs(cp)[0]
				provider.terminateFn = scripted(map[string][]error{id: tc.errors})
				err := cp.Shutdown(context.Background())
				if (err != nil) != tc.fail {
					t.Fatalf("Shutdown: %v", err)
				}
				if terminateCalls(provider, id) != tc.calls {
					t.Fatalf("calls=%d want=%d", terminateCalls(provider, id), tc.calls)
				}
				wantState, wantPods := StateOffline, 0
				if tc.fail {
					wantState, wantPods = StateError, 1
				}
				if cp.getState() != wantState || len(podIDs(cp)) != wantPods {
					t.Fatalf("status=%+v pods=%v", cp.Status(), podIDs(cp))
				}
			})
		})
	}
}

// ---------- Concurrent engine startup ----------

func TestStartEngine_ConcurrentCallsProvisionOnce(t *testing.T) {
	const callers = 5

	gate := make(chan struct{})

	provider := &fakeProvider{
		createFn: func(
			context.Context,
			gpuprovider.CreateInstanceSpec,
		) (*gpuprovider.Instance, error) {
			<-gate

			return &gpuprovider.Instance{
				ID:       "pod-1",
				Endpoint: "http://fake",
			}, nil
		},
	}

	cp := newTestCP(t, provider, &fakeClient{})

	start := make(chan struct{})

	var ready sync.WaitGroup
	ready.Add(callers)

	results := make(chan error, callers)

	for i := 0; i < callers; i++ {
		go func() {
			ready.Done()
			<-start

			results <- cp.StartEngine(
				context.Background(),
				config.GPUProviderConfig{},
			)
		}()
	}

	ready.Wait()
	close(start)

	for i := 0; i < callers-1; i++ {
		select {
		case err := <-results:
			if err == nil {
				t.Fatal(
					"a second caller succeeded while the first was provisioning",
				)
			}

		case <-time.After(2 * time.Second):
			t.Fatal(
				"timed out waiting for rejected callers",
			)
		}
	}

	close(gate)

	select {
	case err := <-results:
		if err != nil {
			t.Fatalf(
				"winning caller failed: %v",
				err,
			)
		}

	case <-time.After(2 * time.Second):
		t.Fatal(
			"timed out waiting for the winning caller",
		)
	}

	waitForState(t, cp, StateReady)

	provider.mu.Lock()
	created := len(provider.created)
	provider.mu.Unlock()

	if created != 1 {
		t.Errorf(
			"Create called %d times, want 1",
			created,
		)
	}

	if ids := podIDs(cp); len(ids) != 1 {
		t.Errorf(
			"tracked pods = %v, want exactly one",
			ids,
		)
	}
}

// ---------- Engine health polling ----------

func TestStartEngine_HungHealthTerminatesWorker(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		var healthCalls int

		client := &fakeClient{
			healthFn: func(context.Context) error {
				healthCalls++
				return errors.New("not ready")
			},
		}

		cp := newTestCP(
			t,
			&fakeProvider{},
			client,
		)

		if err := cp.StartEngine(
			context.Background(),
			config.GPUProviderConfig{},
		); err != nil {
			t.Fatalf(
				"StartEngine: %v",
				err,
			)
		}

		// Advance beyond the 15-minute workflow timeout.
		time.Sleep(16 * time.Minute)

		synctest.Wait()

		if got := cp.getState(); got != StateOffline {
			t.Errorf(
				"state = %s, want %s",
				got,
				StateOffline,
			)
		}

		if msg := cp.Status().Error; !strings.Contains(
			msg,
			"deadline exceeded",
		) {
			t.Errorf(
				"Status().Error = %q, want deadline exceeded",
				msg,
			)
		}

		if healthCalls < 2 {
			t.Errorf(
				"health polled %d times, want retries",
				healthCalls,
			)
		}
	})
}

// ---------- Streaming generation tests ----------

func TestGenerateStream_RejectsWhenEngineNotReady(t *testing.T) {
	client := &fakeClient{}

	cp := newTestCP(
		t,
		&fakeProvider{},
		client,
	)

	pending := request.PendingRequest{
		Payload: request.GenerateRequest{
			Prompt:    "hello",
			MaxTokens: 3,
			Stream:    true,
		},
	}

	stream, err := cp.GenerateStream(
		context.Background(),
		pending,
	)

	if err == nil {
		t.Fatal("expected error when engine is not ready")
	}

	if stream != nil {
		t.Fatal("expected nil stream when engine is not ready")
	}

	client.mu.Lock()
	calls := client.streamCalls
	client.mu.Unlock()

	if calls != 0 {
		t.Errorf(
			"GenerateStream called %d times, want 0",
			calls,
		)
	}
}

func TestGenerateStream_ForwardsResults(t *testing.T) {
	// Simulate two generated tokens followed by an engine error.
	//
	// The control plane should forward the stream without
	// consuming or modifying its contents.

	streamErr := errors.New("engine stream interrupted")

	want := []response.StreamResponse{
		{
			Chunk: response.GenerateChunk{
				RequestID: nil,
				Text:      "Hello",
			},
		},
		{
			Chunk: response.GenerateChunk{
				RequestID: nil,
				Text:      " world",
			},
		},
		{
			Err: streamErr,
		},
	}

	client := &fakeClient{
		generateStreamFn: func(
			ctx context.Context,
			req request.GenerateRequest,
		) (<-chan response.StreamResponse, error) {
			results := make(
				chan response.StreamResponse,
				len(want),
			)

			for _, result := range want {
				results <- result
			}

			close(results)

			return results, nil
		},
	}

	cp := newReadyTestCP(
		t,
		&fakeProvider{},
		client,
	)

	pending := request.PendingRequest{
		Payload: request.GenerateRequest{
			Prompt:    "hello",
			MaxTokens: 3,
			Stream:    true,
		},
	}

	stream, err := cp.GenerateStream(
		context.Background(),
		pending,
	)

	if err != nil {
		t.Fatalf(
			"GenerateStream: %v",
			err,
		)
	}

	var got []response.StreamResponse

	for result := range stream {
		got = append(got, result)
	}

	if !reflect.DeepEqual(got, want) {
		t.Errorf(
			"stream results = %+v, want %+v",
			got,
			want,
		)
	}

	client.mu.Lock()
	calls := client.streamCalls
	lastReq := client.lastStreamRequest
	client.mu.Unlock()

	if calls != 1 {
		t.Errorf(
			"GenerateStream called %d times, want 1",
			calls,
		)
	}

	if lastReq.Prompt != pending.Payload.Prompt {
		t.Errorf(
			"forwarded prompt = %q, want %q",
			lastReq.Prompt,
			pending.Payload.Prompt,
		)
	}

	if lastReq.MaxTokens != pending.Payload.MaxTokens {
		t.Errorf(
			"forwarded max tokens = %d, want %d",
			lastReq.MaxTokens,
			pending.Payload.MaxTokens,
		)
	}

	if !lastReq.Stream {
		t.Error("stream flag was not forwarded")
	}
}

func TestGenerateStream_PropagatesClientError(t *testing.T) {
	boom := errors.New("engine unavailable")

	client := &fakeClient{
		generateStreamFn: func(
			ctx context.Context,
			req request.GenerateRequest,
		) (<-chan response.StreamResponse, error) {
			return nil, boom
		},
	}

	cp := newReadyTestCP(
		t,
		&fakeProvider{},
		client,
	)

	pending := request.PendingRequest{
		Payload: request.GenerateRequest{
			Prompt:    "hello",
			MaxTokens: 3,
			Stream:    true,
		},
	}

	stream, err := cp.GenerateStream(
		context.Background(),
		pending,
	)

	if !errors.Is(err, boom) {
		t.Fatalf(
			"err = %v, want it to wrap %v",
			err,
			boom,
		)
	}

	if stream != nil {
		t.Fatal(
			"expected nil stream when client returns an error",
		)
	}
}
