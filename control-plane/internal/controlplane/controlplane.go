package controlplane

import (
	"context"
	"errors"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/yoonsung9948/relay/internal/config"
	"github.com/yoonsung9948/relay/internal/controlplane/gpuprovider"
	"github.com/yoonsung9948/relay/internal/engine"
	"github.com/yoonsung9948/relay/internal/ratelimit"
)

type EngineState string

const (
	StateOffline      EngineState = "offline"
	StateProvisioning EngineState = "provisioning"
	StateStarting     EngineState = "starting"
	StateLoading      EngineState = "loading_model"
	StateWarming      EngineState = "warming"
	StateReady        EngineState = "ready"
	StateError        EngineState = "error"
	StateShuttingDown EngineState = "shutting_down"
)

type startupOperation struct {
	cancel context.CancelFunc
	done   chan struct{}
}
type stopOperation struct {
	done chan struct{}
	err  error
}
type trackedRequest struct {
	cancel context.CancelFunc
	done   chan struct{}
}
type ControlPlane struct {
	mu          sync.RWMutex
	state       EngineState
	lastError   error
	ctx         context.Context
	Client      engine.Client
	Provider    gpuprovider.GPUProvider
	Cfg         config.ControlPlaneConfig
	rateLimiter *ratelimit.ProvisionLimiter
	worker      *WorkerRecord
	journal     *ownershipJournal
	initErr     error
	recovered   bool
	closed      bool
	startup     *startupOperation
	stop        *stopOperation
	leaseTimer  *time.Timer
	requests    map[uint64]trackedRequest
	nextRequest uint64
}

func NewControlPlane(ctx context.Context, cfg config.ControlPlaneConfig, client engine.Client) *ControlPlane {
	if cfg.ProvisionTimeout == 0 {
		cfg.ProvisionTimeout = config.DefaultProvisionTimeout
	}
	if cfg.CleanupTimeout == 0 {
		cfg.CleanupTimeout = config.DefaultCleanupTimeout
	}
	if cfg.ShutdownTimeout == 0 {
		cfg.ShutdownTimeout = config.DefaultShutdownTimeout
	}
	var provider gpuprovider.GPUProvider
	if cfg.GPUProviderConfig.VastAI != nil {
		provider = gpuprovider.NewVastAIProvider(cfg.GPUProviderConfig.VastAI)
	} else if cfg.GPUProviderConfig.RunPod != nil {
		provider = gpuprovider.NewRunPodProvider(cfg.GPUProviderConfig.RunPod)
	}
	journal, err := openJournal(cfg.StateDirectory)
	c := &ControlPlane{ctx: ctx, Cfg: cfg, Client: client, Provider: provider, state: StateOffline, journal: journal, initErr: err, requests: make(map[uint64]trackedRequest), rateLimiter: ratelimit.NewProvisionLimiter(cfg)}
	if err != nil {
		c.state = StateError
		c.lastError = err
		return c
	}
	if journal.data.Worker != nil {
		copy := *journal.data.Worker
		c.worker = &copy
	}
	c.recovered = c.worker == nil
	if !c.recovered {
		c.state = StateError
		c.lastError = errors.New("recorded worker requires startup reconciliation")
	}
	return c
}
func (c *ControlPlane) getState() EngineState { c.mu.RLock(); defer c.mu.RUnlock(); return c.state }

type Status struct {
	State          EngineState `json:"state"`
	Error          string      `json:"error,omitempty"`
	InstanceID     string      `json:"instance_id,omitempty"`
	ActiveRequests int         `json:"active_requests"`
}

func (c *ControlPlane) Status() Status {
	c.mu.RLock()
	defer c.mu.RUnlock()
	s := Status{State: c.state, ActiveRequests: len(c.requests)}
	if c.lastError != nil {
		s.Error = c.lastError.Error()
	}
	if c.worker != nil {
		s.InstanceID = c.worker.InstanceID
	}
	return s
}
func (c *ControlPlane) scope() string {
	if provider, ok := c.Provider.(gpuprovider.ScopedProvider); ok {
		return provider.OwnershipScope()
	}
	return string(c.Cfg.GPUProviderConfig.Name)
}
func (c *ControlPlane) saveLocked() error {
	if c.journal == nil {
		return c.initErr
	}
	return c.journal.save(c.worker)
}
func (c *ControlPlane) failLocked(err error) {
	c.state = StateError
	c.lastError = err
	log.Printf("engine lifecycle error: %v", err)
}

// Creation/readiness belong to the control plane, never to an HTTP connection.
// Wait for the creation result to preserve the API's confirmed no-capacity response.
// A caller disconnect returns early without abandoning ownership or canceling startup.
func (c *ControlPlane) StartEngine(ctx context.Context, cfg config.GPUProviderConfig) error {
	c.mu.Lock()
	if c.initErr != nil {
		c.mu.Unlock()
		return c.initErr
	}
	if c.closed || !c.recovered || c.state != StateOffline || c.worker != nil || c.stop != nil {
		state := c.state
		c.mu.Unlock()
		return &EngineStateConflictError{State: state}
	}
	if err := ctx.Err(); err != nil {
		c.mu.Unlock()
		return err
	}
	if err := c.ctx.Err(); err != nil {
		c.mu.Unlock()
		return err
	}
	if c.Provider == nil || c.Client == nil {
		c.mu.Unlock()
		return errors.New("provider and engine client must be configured")
	}
	if c.Cfg.RateLimitEnabled {
		if err := c.rateLimiter.Allow(); err != nil {
			c.mu.Unlock()
			return err
		}
	}
	attempt := uniqueID()
	c.worker = &WorkerRecord{AttemptID: attempt, Name: workerName(c.journal.data.OwnerID, attempt), Provider: string(c.Cfg.GPUProviderConfig.Name), Scope: c.scope(), Phase: "creating", CreatedAt: time.Now().UTC()}
	if err := c.saveLocked(); err != nil {
		c.failLocked(err)
		c.mu.Unlock()
		return err
	}
	startupCtx, cancel := context.WithTimeout(context.Background(), c.Cfg.ProvisionTimeout)
	operation := &startupOperation{cancel: cancel, done: make(chan struct{})}
	c.startup = operation
	c.state = StateProvisioning
	c.lastError = nil
	name := c.worker.Name
	c.mu.Unlock()
	result := make(chan error, 1)
	spec := gpuprovider.CreateInstanceSpec{Image: cfg.InstanceConfig.Image, GPU: cfg.InstanceConfig.GPU, DiskGB: cfg.InstanceConfig.DiskGB, Name: name}
	go c.startWorker(startupCtx, attempt, spec, operation, result)
	select {
	case err := <-result:
		return err
	case <-ctx.Done():
		return ctx.Err()
	}
}

// Recover never rents a GPU. The demo's restart policy cleans up its previous worker.
func (c *ControlPlane) Recover(ctx context.Context) error {
	c.mu.RLock()
	err := c.initErr
	recovered := c.recovered
	c.mu.RUnlock()
	if err != nil {
		return err
	}
	if recovered {
		return nil
	}
	return c.StopEngine(ctx, "startup_recovery")
}

// Close releases the journal's process lock; call after Shutdown or before process exit.
func (c *ControlPlane) Close() error { c.mu.Lock(); defer c.mu.Unlock(); return c.journal.close() }
func (c *ControlPlane) Shutdown(ctx context.Context) error {
	c.mu.Lock()
	c.closed = true
	c.mu.Unlock()
	return c.StopEngine(ctx, "process_shutdown")
}

type NoCapacityError struct {
	Cause   error
	RetryAt time.Time
}

func (e *NoCapacityError) Error() string { return e.Cause.Error() }
func (e *NoCapacityError) Unwrap() error { return e.Cause }

type EngineStateConflictError struct{ State EngineState }

func (e *EngineStateConflictError) Error() string {
	return fmt.Sprintf("engine cannot start from state %s", e.State)
}
