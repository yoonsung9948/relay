package controlplane

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
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

func TestOwnershipJournalLockPersistenceAndValidation(t *testing.T) {
	dir := t.TempDir()
	journal, err := openJournal(dir)
	if err != nil {
		t.Fatal(err)
	}
	owner := journal.data.OwnerID
	attempt := uniqueID()
	record := &WorkerRecord{AttemptID: attempt, Name: workerName(owner, attempt), InstanceID: "pod-1", Phase: "loading", CreatedAt: time.Now().UTC()}
	if err := journal.save(record); err != nil {
		t.Fatal(err)
	}
	if second, err := openJournal(dir); err == nil {
		second.close()
		t.Fatal("second process acquired ownership lock")
	}
	info, err := os.Stat(filepath.Join(dir, "worker.json"))
	if err != nil || info.Mode().Perm() != 0600 {
		t.Fatalf("journal permissions: %v %v", info, err)
	}
	journal.close()
	journal, err = openJournal(dir)
	if err != nil {
		t.Fatal(err)
	}
	if journal.data.OwnerID != owner || journal.data.Worker.InstanceID != "pod-1" {
		t.Fatal("ownership lost on reopen")
	}
	if err := journal.save(nil); err != nil {
		t.Fatal(err)
	}
	journal.close()
	journal, err = openJournal(dir)
	if err != nil {
		t.Fatal(err)
	}
	if journal.data.Worker != nil || journal.data.OwnerID != owner {
		t.Fatal("confirmed cleanup did not persist")
	}
	journal.close()
	for _, content := range []string{`{}`, `{"version":1}`, `invalid`, `{"version":99,"owner_id":"` + owner + `","worker":null}`, `{"version":1,"owner_id":"` + owner + `","worker":null} {}`} {
		if err := os.WriteFile(filepath.Join(dir, "worker.json"), []byte(content), 0600); err != nil {
			t.Fatal(err)
		}
		if invalid, err := openJournal(dir); err == nil {
			invalid.close()
			t.Fatalf("invalid journal accepted: %s", content)
		}
	}
}

func TestCreationIntentAndIDPersistBeforeHealth(t *testing.T) {
	dir := t.TempDir()
	cp := NewControlPlane(context.Background(), config.ControlPlaneConfig{StateDirectory: dir}, &fakeClient{})
	defer cp.Close()
	readRecord := func() *WorkerRecord {
		t.Helper()
		raw, err := os.ReadFile(filepath.Join(dir, "worker.json"))
		if err != nil {
			t.Fatal(err)
		}
		var data journalData
		if err := json.Unmarshal(raw, &data); err != nil {
			t.Fatal(err)
		}
		return data.Worker
	}
	cp.Provider = &fakeProvider{createFn: func(ctx context.Context, spec gpuprovider.CreateInstanceSpec) (*gpuprovider.Instance, error) {
		record := readRecord()
		if record == nil || record.Name != spec.Name || record.InstanceID != "" || record.Phase != "creating" {
			t.Error("create preceded durable intent")
		}
		return &gpuprovider.Instance{ID: "pod-1", Endpoint: "http://fake"}, nil
	}}
	checked := make(chan struct{})
	cp.Client = &fakeClient{healthFn: func(context.Context) error {
		if record := readRecord(); record.InstanceID != "pod-1" || record.Phase != "loading" {
			t.Error("health preceded durable ID")
		}
		close(checked)
		return nil
	}}
	if err := cp.StartEngine(context.Background(), config.GPUProviderConfig{}); err != nil {
		t.Fatal(err)
	}
	<-checked
	waitForState(t, cp, StateReady)
	if err := cp.StopEngine(context.Background(), "test"); err != nil {
		t.Fatal(err)
	}
	if record := readRecord(); record != nil {
		t.Fatal("cleanup was not durable")
	}
}

func TestDisconnectedStartStillOwnsAndCleansWorker(t *testing.T) {
	started, release := make(chan struct{}), make(chan struct{})
	provider := &fakeProvider{createFn: func(ctx context.Context, spec gpuprovider.CreateInstanceSpec) (*gpuprovider.Instance, error) {
		close(started)
		<-release
		if ctx.Err() != nil {
			t.Error("HTTP disconnect canceled workflow")
		}
		return &gpuprovider.Instance{ID: "pod-late", Endpoint: "http://fake"}, nil
	}}
	cp := newTestCP(t, provider, &fakeClient{})
	ctx, cancel := context.WithCancel(context.Background())
	result := make(chan error, 1)
	go func() { result <- cp.StartEngine(ctx, config.GPUProviderConfig{}) }()
	<-started
	cancel()
	if err := <-result; !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	close(release)
	waitForState(t, cp, StateReady)
	if err := cp.StopEngine(context.Background(), "test"); err != nil {
		t.Fatal(err)
	}
	if terminateCalls(provider, "pod-late") != 1 {
		t.Fatal("late worker was abandoned")
	}
}

func TestStopDuringCreateRecordsAndTerminatesLateSuccess(t *testing.T) {
	started, canceled, release := make(chan struct{}), make(chan struct{}), make(chan struct{})
	provider := &fakeProvider{createFn: func(ctx context.Context, spec gpuprovider.CreateInstanceSpec) (*gpuprovider.Instance, error) {
		close(started)
		<-ctx.Done()
		close(canceled)
		<-release
		return &gpuprovider.Instance{ID: "pod-late", Endpoint: "http://fake"}, nil
	}}
	cp := newTestCP(t, provider, &fakeClient{})
	startDone, stopDone := make(chan error, 1), make(chan error, 1)
	go func() { startDone <- cp.StartEngine(context.Background(), config.GPUProviderConfig{}) }()
	<-started
	go func() { stopDone <- cp.StopEngine(context.Background(), "test") }()
	<-canceled
	if cp.Status().State != StateShuttingDown {
		t.Fatal(cp.Status())
	}
	close(release)
	if err := <-startDone; !errors.Is(err, context.Canceled) {
		t.Fatalf("startup=%v", err)
	}
	if err := <-stopDone; err != nil {
		t.Fatal(err)
	}
	if cp.Status().State != StateOffline || len(podIDs(cp)) != 0 || terminateCalls(provider, "pod-late") != 1 {
		t.Fatal(cp.Status())
	}
}

func TestConcurrentStopSharesCleanupAndIgnoresCallerCancellation(t *testing.T) {
	entered, release := make(chan struct{}), make(chan struct{})
	provider := &fakeProvider{terminateFn: func(ctx context.Context, id string, call int) error {
		if call != 1 {
			t.Error("duplicate termination")
		}
		close(entered)
		<-release
		if ctx.Err() != nil {
			t.Error("caller canceled cleanup")
		}
		return nil
	}}
	cp := newReadyTestCP(t, provider, &fakeClient{})
	ctx, cancel := context.WithCancel(context.Background())
	first := make(chan error, 1)
	go func() { first <- cp.StopEngine(ctx, "test") }()
	<-entered
	cancel()
	if err := <-first; !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	const callers = 8
	var wg sync.WaitGroup
	wg.Add(callers)
	for range callers {
		go func() {
			defer wg.Done()
			if err := cp.StopEngine(context.Background(), "test"); err != nil {
				t.Error(err)
			}
		}()
	}
	close(release)
	wg.Wait()
	if cp.Status().State != StateOffline || len(podIDs(cp)) != 0 {
		t.Fatal(cp.Status())
	}
}

func TestFailedCleanupRetainsOwnershipAndCanRetry(t *testing.T) {
	provider := &fakeProvider{terminateFn: func(context.Context, string, int) error {
		return &gpuprovider.HTTPError{StatusCode: 401, Operation: "terminate"}
	}}
	cp := newReadyTestCP(t, provider, &fakeClient{})
	id := podIDs(cp)[0]
	if err := cp.StopEngine(context.Background(), "test"); err == nil {
		t.Fatal("failure hidden")
	}
	if len(podIDs(cp)) != 1 || cp.Status().State != StateError {
		t.Fatal(cp.Status())
	}
	if err := cp.StartEngine(context.Background(), config.GPUProviderConfig{}); err == nil {
		t.Fatal("new worker admitted with unresolved ownership")
	}
	provider.terminateFn = nil
	if err := cp.StopEngine(context.Background(), "retry"); err != nil {
		t.Fatal(err)
	}
	if len(podIDs(cp)) != 0 || terminateCalls(provider, id) != 2 {
		t.Fatal(cp.Status())
	}
}

func TestTerminationMustBeConfirmed(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		provider := &fakeProvider{}
		cp := newReadyTestCP(t, provider, &fakeClient{})
		cp.Cfg.CleanupTimeout = 3 * time.Second
		id := podIDs(cp)[0]
		provider.mu.Lock()
		owned := *provider.instances[id]
		provider.mu.Unlock()
		provider.getFn = func(context.Context, string) (*gpuprovider.Instance, error) { return &owned, nil }
		if err := cp.StopEngine(context.Background(), "test"); err == nil {
			t.Fatal("accepted DELETE without confirmation")
		}
		if len(podIDs(cp)) != 1 {
			t.Fatal("ownership discarded while pod still visible")
		}
	})
}

func TestIdleTimeoutWaitsForGenerationAndRestartsAfterCompletion(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		started, release := make(chan struct{}), make(chan struct{})
		provider := &fakeProvider{}
		cp := newTestCP(t, provider, &fakeClient{generateFn: func(ctx context.Context, req request.GenerateRequest) (response.GenerateResponse, error) {
			close(started)
			<-release
			return response.GenerateResponse{}, nil
		}})
		cp.Cfg.IdleTimeout = time.Minute
		if err := cp.StartEngine(context.Background(), config.GPUProviderConfig{}); err != nil {
			t.Fatal(err)
		}
		waitForState(t, cp, StateReady)
		finished := make(chan error, 1)
		go func() { _, err := cp.Generate(context.Background(), request.PendingRequest{}); finished <- err }()
		<-started
		time.Sleep(2 * time.Minute)
		synctest.Wait()
		if cp.Status().State != StateReady || cp.Status().ActiveRequests != 1 {
			t.Fatal(cp.Status())
		}
		close(release)
		if err := <-finished; err != nil {
			t.Fatal(err)
		}
		time.Sleep(59 * time.Second)
		synctest.Wait()
		if cp.Status().State != StateReady {
			t.Fatal("idle measured from request start")
		}
		time.Sleep(time.Second)
		synctest.Wait()
		if cp.Status().State != StateOffline || len(podIDs(cp)) != 0 {
			t.Fatal(cp.Status())
		}
	})
}

func TestSessionTimeoutCancelsActiveStreamEvenWithLimitsDisabled(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		canceled := make(chan struct{})
		client := &fakeClient{generateStreamFn: func(ctx context.Context, req request.GenerateRequest) (<-chan response.StreamResponse, error) {
			source := make(chan response.StreamResponse)
			go func() { <-ctx.Done(); close(canceled); close(source) }()
			return source, nil
		}}
		cp := newTestCP(t, &fakeProvider{}, client)
		cp.Cfg.IdleTimeout = 10 * time.Second
		cp.Cfg.MaxSessionDuration = time.Minute
		cp.Cfg.RateLimitEnabled = false
		if err := cp.StartEngine(context.Background(), config.GPUProviderConfig{}); err != nil {
			t.Fatal(err)
		}
		waitForState(t, cp, StateReady)
		stream, err := cp.GenerateStream(context.Background(), request.PendingRequest{})
		if err != nil {
			t.Fatal(err)
		}
		time.Sleep(30 * time.Second)
		synctest.Wait()
		if cp.Status().State != StateReady {
			t.Fatal("idle timeout stopped active stream")
		}
		time.Sleep(30 * time.Second)
		synctest.Wait()
		<-canceled
		if _, open := <-stream; open {
			t.Fatal("stream remained open")
		}
		if cp.Status().State != StateOffline || cp.Status().ActiveRequests != 0 {
			t.Fatal(cp.Status())
		}
	})
}

func TestStaleAttemptCannotStopReplacement(t *testing.T) {
	cp := newReadyTestCP(t, &fakeProvider{}, &fakeClient{})
	cp.mu.RLock()
	attempt := cp.worker.AttemptID
	cp.mu.RUnlock()
	if err := cp.StopEngine(context.Background(), "test"); err != nil {
		t.Fatal(err)
	}
	if err := cp.StartEngine(context.Background(), config.GPUProviderConfig{}); err != nil {
		t.Fatal(err)
	}
	waitForState(t, cp, StateReady)
	cp.expireLease(attempt)
	cp.stopAttempt(attempt, "stale_failure")
	if cp.Status().State != StateReady {
		t.Fatal("stale operation stopped new worker")
	}
	if err := cp.StopEngine(context.Background(), "test"); err != nil {
		t.Fatal(err)
	}
}

func TestStartupRecoveryOnlyCleansExactRecordedWorker(t *testing.T) {
	for _, unknownID := range []bool{false, true} {
		t.Run(map[bool]string{false: "known ID", true: "lost create response"}[unknownID], func(t *testing.T) {
			dir := t.TempDir()
			provider := &fakeProvider{}
			original := NewControlPlane(context.Background(), config.ControlPlaneConfig{StateDirectory: dir}, &fakeClient{})
			original.Provider = provider
			if err := original.StartEngine(context.Background(), config.GPUProviderConfig{}); err != nil {
				t.Fatal(err)
			}
			waitForState(t, original, StateReady)
			id := podIDs(original)[0]
			original.mu.Lock()
			if unknownID {
				original.worker.InstanceID = ""
				original.worker.Phase = "creating"
				if err := original.saveLocked(); err != nil {
					t.Fatal(err)
				}
			}
			original.mu.Unlock()
			original.Close() // Simulate restart without normal Shutdown.
			recovered := NewControlPlane(context.Background(), config.ControlPlaneConfig{StateDirectory: dir}, &fakeClient{})
			defer recovered.Close()
			recovered.Provider = provider
			if err := recovered.StartEngine(context.Background(), config.GPUProviderConfig{}); err == nil {
				t.Fatal("startup recovery bypassed")
			}
			if err := recovered.Recover(context.Background()); err != nil {
				t.Fatal(err)
			}
			provider.mu.Lock()
			created := len(provider.created)
			provider.mu.Unlock()
			if created != 1 || terminateCalls(provider, id) != 1 || recovered.Status().State != StateOffline {
				t.Fatal(recovered.Status())
			}
		})
	}
}

func TestAmbiguousCreationWithoutExactMatchBlocksNewStarts(t *testing.T) {
	for _, count := range []int{0, 2} {
		t.Run(string(rune('0'+count)), func(t *testing.T) {
			provider := &fakeProvider{createFn: func(context.Context, gpuprovider.CreateInstanceSpec) (*gpuprovider.Instance, error) {
				return nil, errors.New("lost response")
			}, findFn: func(ctx context.Context, name string) ([]*gpuprovider.Instance, error) {
				matches := make([]*gpuprovider.Instance, count)
				for i := range matches {
					matches[i] = &gpuprovider.Instance{Name: name, ID: "pod"}
				}
				return matches, nil
			}}
			cp := newTestCP(t, provider, &fakeClient{})
			_ = cp.StartEngine(context.Background(), config.GPUProviderConfig{})
			if err := cp.StopEngine(context.Background(), "test"); err == nil {
				t.Fatal("ambiguous ownership cleared")
			}
			cp.mu.RLock()
			retained := cp.worker != nil
			cp.mu.RUnlock()
			if !retained {
				t.Fatal("attempt lost")
			}
			if err := cp.StartEngine(context.Background(), config.GPUProviderConfig{}); err == nil {
				t.Fatal("new worker admitted")
			}
			provider.mu.Lock()
			calls := len(provider.termCalls)
			provider.mu.Unlock()
			if calls != 0 {
				t.Fatal("terminated unverified worker")
			}
		})
	}
}

func TestCorruptJournalAndChangedProviderScopeBlockCleanup(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "worker.json"), []byte(`{}`), 0600); err != nil {
		t.Fatal(err)
	}
	cp := NewControlPlane(context.Background(), config.ControlPlaneConfig{StateDirectory: dir}, &fakeClient{})
	provider := &fakeProvider{}
	cp.Provider = provider
	if cp.Recover(context.Background()) == nil || cp.StartEngine(context.Background(), config.GPUProviderConfig{}) == nil {
		t.Fatal("corrupt journal ignored")
	}
	cp.Close()
	cp = newReadyTestCP(t, provider, &fakeClient{})
	cp.mu.Lock()
	cp.worker.Scope = "different-account"
	cp.mu.Unlock()
	if err := cp.StopEngine(context.Background(), "test"); err == nil || !strings.Contains(err.Error(), "credentials") {
		t.Fatal(err)
	}
	if len(podIDs(cp)) != 1 {
		t.Fatal("wrong-account response discarded ownership")
	}
}

func TestRetryStopsOnPermanentErrorAndPreservesCause(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		transient := &gpuprovider.HTTPError{StatusCode: 503, Operation: "test"}
		start := time.Now()
		calls := 0
		err := withRetry(context.Background(), func() error { calls++; return transient }, 3, time.Second)
		if !errors.Is(err, transient) || calls != 3 || time.Since(start) != 3*time.Second {
			t.Fatalf("calls=%d elapsed=%s err=%v", calls, time.Since(start), err)
		}
		permanent := &gpuprovider.HTTPError{StatusCode: 403, Operation: "test"}
		calls = 0
		err = withRetry(context.Background(), func() error { calls++; return permanent }, 5, time.Second)
		if calls != 1 || !errors.Is(err, permanent) {
			t.Fatal(err, calls)
		}
	})
}

func TestJournalWriteFailurePreventsCreateAndRetainsFailedClear(t *testing.T) {
	provider := &fakeProvider{}
	cp := newTestCP(t, provider, &fakeClient{})
	cp.journal.close()
	if err := cp.StartEngine(context.Background(), config.GPUProviderConfig{}); err == nil {
		t.Fatal("created without durable intent")
	}
	provider.mu.Lock()
	created := len(provider.created)
	provider.mu.Unlock()
	if created != 0 {
		t.Fatal("provider called after failed intent persistence")
	}
	cp = newReadyTestCP(t, provider, &fakeClient{})
	id := podIDs(cp)[0]
	cp.journal.close()
	if err := cp.StopEngine(context.Background(), "test"); err == nil {
		t.Fatal("failed durable clear hidden")
	}
	if len(podIDs(cp)) != 1 || terminateCalls(provider, id) != 1 {
		t.Fatal("ownership discarded before durable clear")
	}
}

func TestMismatchedProviderNameIsNeverTerminated(t *testing.T) {
	provider := &fakeProvider{}
	cp := newReadyTestCP(t, provider, &fakeClient{})
	id := podIDs(cp)[0]
	provider.getFn = func(context.Context, string) (*gpuprovider.Instance, error) {
		return &gpuprovider.Instance{ID: id, Name: "somebody-elses-pod"}, nil
	}
	if err := cp.StopEngine(context.Background(), "test"); err == nil {
		t.Fatal("ownership mismatch ignored")
	}
	if terminateCalls(provider, id) != 0 || len(podIDs(cp)) != 1 {
		t.Fatal("unowned pod terminated")
	}
}

func TestZeroLeaseTimeoutsDisableAutomaticStop(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		cp := newReadyTestCP(t, &fakeProvider{}, &fakeClient{})
		time.Sleep(24 * time.Hour)
		synctest.Wait()
		if cp.Status().State != StateReady {
			t.Fatal(cp.Status())
		}
		if err := cp.StopEngine(context.Background(), "test"); err != nil {
			t.Fatal(err)
		}
	})
}

func TestProvisionTimeoutIncludesCreateAndReadiness(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		provider := &fakeProvider{createFn: func(ctx context.Context, spec gpuprovider.CreateInstanceSpec) (*gpuprovider.Instance, error) {
			time.Sleep(8 * time.Second)
			return &gpuprovider.Instance{ID: "slow", Endpoint: "http://fake"}, nil
		}}
		cp := newTestCP(t, provider, &fakeClient{healthFn: func(ctx context.Context) error { <-ctx.Done(); return ctx.Err() }})
		cp.Cfg.ProvisionTimeout = 10 * time.Second
		start := time.Now()
		if err := cp.StartEngine(context.Background(), config.GPUProviderConfig{}); err != nil {
			t.Fatal(err)
		}
		if time.Since(start) != 8*time.Second {
			t.Fatal("create budget not exercised")
		}
		time.Sleep(2 * time.Second)
		synctest.Wait()
		if cp.Status().State != StateOffline || terminateCalls(provider, "slow") != 1 {
			t.Fatal(cp.Status())
		}
	})
}

func TestUncooperativeRequestCannotPreventPodTermination(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		entered, release := make(chan struct{}), make(chan struct{})
		provider := &fakeProvider{}
		cp := newReadyTestCP(t, provider, &fakeClient{generateFn: func(ctx context.Context, req request.GenerateRequest) (response.GenerateResponse, error) {
			close(entered)
			<-release
			return response.GenerateResponse{}, ctx.Err()
		}})
		done := make(chan struct{})
		go func() { defer close(done); _, _ = cp.Generate(context.Background(), request.PendingRequest{}) }()
		<-entered
		id := podIDs(cp)[0]
		if err := cp.StopEngine(context.Background(), "test"); err == nil {
			t.Fatal("request drain timeout hidden")
		}
		if terminateCalls(provider, id) != 1 || len(podIDs(cp)) != 0 {
			t.Fatal("stuck request prevented GPU cleanup")
		}
		close(release)
		<-done
		if err := cp.StopEngine(context.Background(), "retry"); err != nil {
			t.Fatal(err)
		}
	})
}

func TestCreateOutlivingCleanupRetainsIntentAndCleansLatePod(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		entered, release := make(chan struct{}), make(chan struct{})
		provider := &fakeProvider{createFn: func(ctx context.Context, spec gpuprovider.CreateInstanceSpec) (*gpuprovider.Instance, error) {
			close(entered)
			<-release
			return &gpuprovider.Instance{ID: "late", Endpoint: "http://fake"}, nil
		}}
		cp := newTestCP(t, provider, &fakeClient{})
		cp.Cfg.CleanupTimeout = 3 * time.Second
		startDone := make(chan error, 1)
		go func() { startDone <- cp.StartEngine(context.Background(), config.GPUProviderConfig{}) }()
		<-entered
		if err := cp.StopEngine(context.Background(), "test"); err == nil {
			t.Fatal("ignored startup timeout")
		}
		cp.mu.RLock()
		retained := cp.worker != nil
		cp.mu.RUnlock()
		if !retained {
			t.Fatal("unresolved creation intent discarded")
		}
		close(release)
		if err := <-startDone; !errors.Is(err, context.Canceled) {
			t.Fatal(err)
		}
		synctest.Wait()
		if cp.Status().State != StateOffline || terminateCalls(provider, "late") != 1 {
			t.Fatal(cp.Status())
		}
	})
}
