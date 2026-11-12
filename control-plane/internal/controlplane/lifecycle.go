package controlplane

import (
	"context"
	"errors"
	"fmt"
	"log"
	"time"

	"github.com/yoonsung9948/relay/internal/controlplane/gpuprovider"
)

func (c *ControlPlane) startWorker(ctx context.Context, attempt string, spec gpuprovider.CreateInstanceSpec, op *startupOperation, result chan<- error) {
	cleanup := false
	defer func() {
		op.cancel()
		c.mu.Lock()
		defer c.mu.Unlock()
		close(op.done)
		// Publish completion and schedule cleanup atomically. A manual stop that
		// already joined this workflow must not be followed by a delayed retry.
		if cleanup && c.worker != nil && c.worker.AttemptID == attempt && c.stop == nil {
			c.beginStopLocked("startup_failure")
		}
	}()
	instance, err := c.Provider.Create(ctx, spec)
	if err != nil {
		c.mu.Lock()
		if gpuprovider.DefinitiveCreateFailure(err) {
			old := c.worker
			c.worker.Phase = "not_created"
			if saveErr := c.journal.save(nil); saveErr != nil {
				err = errors.Join(err, saveErr)
				c.failLocked(err)
				cleanup = true
			} else {
				c.worker = nil
				if c.stop == nil {
					c.state = StateOffline
				}
				c.lastError = err
				if errors.Is(err, gpuprovider.ErrNoCapacity) {
					var retry time.Time
					if c.Cfg.RateLimitEnabled {
						retry = c.rateLimiter.NoCapacity()
					}
					err = &NoCapacityError{Cause: err, RetryAt: retry}
					log.Printf("provision category=no_capacity attempt=%s retry_at=%s", old.AttemptID, retry)
				}
			}
		} else {
			c.worker.Phase = "uncertain"
			err = errors.Join(err, c.saveLocked())
			c.lastError = err
			if c.stop == nil {
				c.state = StateError
			}
			cleanup = true
		}
		c.mu.Unlock()
		result <- err
		return
	}
	c.mu.Lock()
	if instance == nil || !instanceID.MatchString(instance.ID) {
		err = errors.New("provider returned no valid instance id; ownership requires reconciliation")
		c.worker.Phase = "uncertain"
		err = errors.Join(err, c.saveLocked())
		c.lastError = err
		if c.stop == nil {
			c.state = StateError
		}
		c.mu.Unlock()
		result <- err
		cleanup = true
		return
	}
	c.worker.InstanceID = instance.ID
	c.worker.Phase = "loading"
	err = c.saveLocked()
	if err == nil && instance.Endpoint == "" {
		err = errors.New("provider returned no engine endpoint")
	}
	if err == nil {
		err = ctx.Err()
	}
	if err != nil {
		c.lastError = err
		if c.stop == nil {
			c.state = StateError
		}
		c.mu.Unlock()
		result <- err
		cleanup = true
		return
	}
	if c.stop == nil {
		c.state = StateLoading
	}
	c.Client.SetEndpoint(instance.Endpoint)
	c.mu.Unlock()
	log.Printf("provision category=created attempt=%s instance_id=%s", attempt, instance.ID)
	result <- nil
	// Every probe has a short deadline, nested inside the overall startup budget.
	ticker := time.NewTicker(3 * time.Second)
	defer ticker.Stop()
	for {
		probeCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
		healthErr := c.Client.Health(probeCtx)
		cancel()
		if healthErr == nil && ctx.Err() == nil {
			c.mu.Lock()
			if c.stop != nil || c.closed || c.worker == nil || c.worker.AttemptID != attempt {
				c.mu.Unlock()
				cleanup = true
				return
			}
			c.worker.ReadyAt = time.Now().UTC()
			c.worker.LastActivity = c.worker.ReadyAt
			c.worker.Phase = "ready"
			if err = c.saveLocked(); err != nil {
				c.failLocked(err)
				c.mu.Unlock()
				cleanup = true
				return
			}
			c.state = StateReady
			c.lastError = nil
			c.armLeaseLocked()
			c.mu.Unlock()
			return
		}
		select {
		case <-ctx.Done():
			c.mu.Lock()
			c.lastError = fmt.Errorf("startup deadline/cancellation: %w (last health check: %v)", ctx.Err(), healthErr)
			if c.stop == nil {
				c.state = StateError
			}
			c.mu.Unlock()
			cleanup = true
			return
		case <-ticker.C:
		}
	}
}

// StopEngine is idempotent. All concurrent callers observe the same cleanup result.
// The operation has its own budget; a disconnected caller cannot cancel cleanup.
func (c *ControlPlane) StopEngine(ctx context.Context, reason string) error {
	c.mu.Lock()
	if c.initErr != nil {
		c.mu.Unlock()
		return c.initErr
	}
	operation := c.beginStopLocked(reason)
	c.mu.Unlock()
	select {
	case <-operation.done:
		return operation.err
	case <-ctx.Done():
		return ctx.Err()
	}
}

// beginStopLocked closes admission and starts exactly one cleanup operation.
func (c *ControlPlane) beginStopLocked(reason string) *stopOperation {
	if c.stop != nil {
		return c.stop
	}
	op := &stopOperation{done: make(chan struct{})}
	c.stop = op
	c.state = StateShuttingDown
	if c.leaseTimer != nil {
		c.leaseTimer.Stop()
		c.leaseTimer = nil
	}
	if c.startup != nil {
		c.startup.cancel()
	}
	for _, request := range c.requests {
		request.cancel()
	}
	go c.cleanupWorker(op, reason)
	return op
}
func (c *ControlPlane) stopAttempt(attempt, reason string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.worker == nil || c.worker.AttemptID != attempt || c.stop != nil {
		return
	}
	c.beginStopLocked(reason)
}
func (c *ControlPlane) cleanupWorker(operation *stopOperation, reason string) {
	ctx, cancel := context.WithTimeout(context.Background(), c.Cfg.CleanupTimeout)
	defer cancel()
	c.mu.RLock()
	startup := c.startup
	c.mu.RUnlock()
	var err error
	if startup != nil {
		select {
		case <-startup.done:
		case <-ctx.Done():
			err = fmt.Errorf("startup did not finish; ownership retained: %w", ctx.Err())
		}
	}
	// Cancellation is cooperative; don't let a stuck request prevent Pod termination.
	c.mu.RLock()
	requests := make([]trackedRequest, 0, len(c.requests))
	for _, request := range c.requests {
		requests = append(requests, request)
	}
	c.mu.RUnlock()
	drainCtx, drainCancel := context.WithTimeout(ctx, 5*time.Second)
	var drainErr error
	for _, request := range requests {
		select {
		case <-request.done:
		case <-drainCtx.Done():
			drainErr = drainCtx.Err()
		}
	}
	drainCancel()
	if err == nil {
		c.mu.Lock()
		if c.worker != nil {
			c.worker.Phase = "stopping"
			// Preserve the existing ownership record even if this diagnostic phase
			// write fails. Verified termination and the final durable clear still run.
			if saveErr := c.saveLocked(); saveErr != nil {
				log.Printf("persist cleanup phase: %v", saveErr)
			}
		}
		c.mu.Unlock()
		err = c.terminateOwnedWorker(ctx)
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	operation.err = errors.Join(err, drainErr)
	if operation.err != nil {
		c.failLocked(operation.err)
	} else {
		c.state = StateOffline
		if reason != "startup_failure" {
			c.lastError = nil
		}
		c.recovered = true
	}
	c.stop = nil
	c.startup = nil
	log.Printf("cleanup reason=%s state=%s error=%v", reason, c.state, operation.err)
	close(operation.done)
}
func (c *ControlPlane) terminateOwnedWorker(ctx context.Context) error {
	c.mu.RLock()
	var worker *WorkerRecord
	if c.worker != nil {
		copy := *c.worker
		worker = &copy
	}
	c.mu.RUnlock()
	if worker == nil {
		return c.clearWorker()
	}
	if c.Provider == nil {
		return errors.New("cannot clean up recorded worker: provider is unavailable")
	}
	if worker.Provider != string(c.Cfg.GPUProviderConfig.Name) || worker.Scope != c.scope() {
		return errors.New("provider credentials or endpoint changed; reconcile recorded ownership manually before provisioning")
	}
	if worker.Phase == "not_created" {
		return c.clearWorker()
	}
	if worker.InstanceID == "" {
		finder, ok := c.Provider.(gpuprovider.InstanceFinder)
		if !ok {
			return errors.New("unresolved creation: provider cannot look up the recorded attempt")
		}
		var matches []*gpuprovider.Instance
		if err := withRetry(ctx, func() error {
			var lookupErr error
			matches, lookupErr = finder.FindByName(ctx, worker.Name)
			return lookupErr
		}, 5, time.Second); err != nil {
			return err
		}
		if len(matches) != 1 {
			return fmt.Errorf("unresolved creation for attempt %s: found %d exact matches; ownership retained for reconciliation", worker.AttemptID, len(matches))
		}
		if matches[0] == nil || matches[0].Name != worker.Name || !instanceID.MatchString(matches[0].ID) {
			return errors.New("provider lookup did not establish exact ownership")
		}
		worker.InstanceID = matches[0].ID
		c.mu.Lock()
		c.worker.InstanceID = worker.InstanceID
		c.worker.Phase = "stopping"
		err := c.saveLocked()
		c.mu.Unlock()
		if err != nil {
			return err
		}
	}
	var instance *gpuprovider.Instance
	err := withRetry(ctx, func() error {
		var getErr error
		instance, getErr = c.Provider.Get(ctx, worker.InstanceID)
		return getErr
	}, 5, time.Second)
	if errors.Is(err, gpuprovider.ErrNotFound) {
		return c.clearWorker()
	}
	if err != nil {
		return fmt.Errorf("verify owned pod %s: %w", worker.InstanceID, err)
	}
	if instance == nil || instance.ID != worker.InstanceID || instance.Name != worker.Name {
		return errors.New("provider response does not match recorded worker ownership; refusing termination")
	}
	if instance.Status != "TERMINATED" {
		if err = withRetry(ctx, func() error { return c.Provider.Terminate(ctx, worker.InstanceID) }, 5, time.Second); err != nil && !errors.Is(err, gpuprovider.ErrNotFound) {
			return fmt.Errorf("terminate pod %s; ownership retained: %w", worker.InstanceID, err)
		}
		ticker := time.NewTicker(time.Second)
		defer ticker.Stop()
		for {
			instance, err = c.Provider.Get(ctx, worker.InstanceID)
			if errors.Is(err, gpuprovider.ErrNotFound) {
				break
			}
			if err == nil {
				if instance == nil || instance.ID != worker.InstanceID || instance.Name != worker.Name {
					return errors.New("termination confirmation did not match recorded ownership")
				}
				if instance.Status == "TERMINATED" {
					break
				}
			}
			if err != nil && !gpuprovider.Retryable(err) {
				return fmt.Errorf("confirm pod termination: %w", err)
			}
			select {
			case <-ctx.Done():
				return fmt.Errorf("pod %s termination not confirmed: %w", worker.InstanceID, ctx.Err())
			case <-ticker.C:
			}
		}
	}
	return c.clearWorker()
}
func (c *ControlPlane) clearWorker() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if err := c.journal.save(nil); err != nil {
		return fmt.Errorf("persist confirmed cleanup: %w", err)
	}
	c.worker = nil
	if c.Client != nil {
		c.Client.SetEndpoint("")
	}
	return nil
}
func withRetry(ctx context.Context, operation func() error, maxAttempts int, backoff time.Duration) error {
	var last error
	for attempt := 0; attempt < maxAttempts; attempt++ {
		if err := ctx.Err(); err != nil {
			return errors.Join(err, last)
		}
		last = operation()
		if last == nil {
			return nil
		}
		if !gpuprovider.Retryable(last) || attempt+1 == maxAttempts {
			return last
		}
		timer := time.NewTimer(backoff)
		select {
		case <-ctx.Done():
			timer.Stop()
			return errors.Join(ctx.Err(), last)
		case <-timer.C:
		}
		backoff = min(backoff*2, 10*time.Second)
	}
	return last
}
func (c *ControlPlane) armLeaseLocked() {
	if c.leaseTimer != nil {
		c.leaseTimer.Stop()
		c.leaseTimer = nil
	}
	if c.state != StateReady || c.worker == nil || c.stop != nil {
		return
	}
	var deadline time.Time
	if c.Cfg.MaxSessionDuration > 0 {
		deadline = c.worker.ReadyAt.Add(c.Cfg.MaxSessionDuration)
	}
	if c.Cfg.IdleTimeout > 0 && len(c.requests) == 0 {
		idle := c.worker.LastActivity.Add(c.Cfg.IdleTimeout)
		if deadline.IsZero() || idle.Before(deadline) {
			deadline = idle
		}
	}
	if deadline.IsZero() {
		return
	}
	attempt := c.worker.AttemptID
	c.leaseTimer = time.AfterFunc(max(0, time.Until(deadline)), func() { c.expireLease(attempt) })
}
func (c *ControlPlane) expireLease(attempt string) {
	c.mu.Lock()
	if c.worker == nil || c.worker.AttemptID != attempt || c.state != StateReady || c.stop != nil {
		c.mu.Unlock()
		return
	}
	now := time.Now()
	reason := ""
	if c.Cfg.MaxSessionDuration > 0 && !now.Before(c.worker.ReadyAt.Add(c.Cfg.MaxSessionDuration)) {
		reason = "session_expired"
	} else if c.Cfg.IdleTimeout > 0 && len(c.requests) == 0 && !now.Before(c.worker.LastActivity.Add(c.Cfg.IdleTimeout)) {
		reason = "idle_expired"
	}
	if reason == "" {
		c.armLeaseLocked()
		c.mu.Unlock()
		return
	}
	c.beginStopLocked(reason)
	c.mu.Unlock()
}
