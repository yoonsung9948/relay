package controlplane

import (
	"context"
	"errors"
	"fmt"
	"log"
	"slices"
	"sync"
	"sync/atomic"
	"time"

	"github.com/yoonsung9948/relay/internal/config"
	"github.com/yoonsung9948/relay/internal/controlplane/gpuprovider"
	"github.com/yoonsung9948/relay/internal/engine"
	"github.com/yoonsung9948/relay/internal/request"
	"github.com/yoonsung9948/relay/internal/response"
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

type ControlPlane struct {
	mu              sync.RWMutex
	state           EngineState
	lastError       error
	ctx             context.Context
	RequestManager  *request.RequestManager
	Client          engine.Client
	Provider        gpuprovider.GPUProvider
	instanceIDs     []string
	activeInstances map[string]gpuprovider.Instance
	instanceCounter atomic.Int64
	Cfg             config.ControlPlaneConfig
}

func (c *ControlPlane) setState(state EngineState) {
	c.mu.Lock()
	defer c.mu.Unlock()
	log.Printf("state: %s -> %s", c.state, state)
	c.state = state
}

func (c *ControlPlane) getState() EngineState {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.state
}

func (c *ControlPlane) StartEngine(ctx context.Context, cfg config.GPUProviderConfig) error {
	c.mu.Lock()
	if c.state != StateOffline {
		c.mu.Unlock()
		return errors.New("engine is not offline")
	}
	c.state = StateProvisioning
	c.mu.Unlock()

	prefix := "relay-instance"

	instanceSpec := gpuprovider.CreateInstanceSpec{
		Image:  cfg.InstanceConfig.Image,
		GPU:    cfg.InstanceConfig.GPU,
		DiskGB: cfg.InstanceConfig.DiskGB,
		Name:   prefix + fmt.Sprint(c.instanceCounter.Load()),
	}
	c.instanceCounter.Add(1)

	c.setState(StateStarting)
	instance, err := c.Provider.Create(ctx, instanceSpec)
	if err != nil {
		if errors.Is(err, gpuprovider.ErrNoCapacity) {
			c.setState(StateOffline)
			return gpuprovider.ErrNoCapacity
		}
		c.fail(err)
		return fmt.Errorf("failed to create instance: %w", err)
	}

	log.Printf("instance created: id=%s endpoint=%s", instance.ID, instance.Endpoint)
	c.Client.SetEndpoint(instance.Endpoint)
	c.addInstance(ctx, *instance)

	go c.startEngineWorkflow(c.ctx, instanceSpec, instance)
	return nil
}

func (c *ControlPlane) fail(err error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	log.Printf("state: %s -> %s (error: %v)", c.state, StateError, err)
	c.state = StateError
	c.lastError = err
}

type Status struct {
	State EngineState `json:"state"`
	Error string      `json:"error,omitempty"`
}

func (c *ControlPlane) Status() Status {
	c.mu.RLock()
	defer c.mu.RUnlock()

	status := Status{
		State: c.state,
	}

	if c.lastError != nil {
		status.Error = c.lastError.Error()
	}

	return status
}

func (c *ControlPlane) addInstance(ctx context.Context, instance gpuprovider.Instance) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.instanceIDs = append(c.instanceIDs, instance.ID)
	c.activeInstances[instance.ID] = instance
	return nil
}

// context is request context
func (c *ControlPlane) startEngineWorkflow(ctx context.Context, spec gpuprovider.CreateInstanceSpec, instance *gpuprovider.Instance) {
	// GPU exists, but inference engine isn't necessarily ready yet.
	c.setState(StateLoading)

	timeoutCtx, cancel := context.WithTimeout(ctx, 15*time.Minute)
	defer cancel()

	if err := c.waitForInstanceRunning(timeoutCtx, instance.ID, 3*time.Second); err != nil {
		c.fail(err)
		return
	}

	c.setState(StateReady)
}

func (c *ControlPlane) waitForInstanceRunning(
	ctx context.Context,
	instanceID string,
	pollInterval time.Duration,
) error {
	ticker := time.NewTicker(pollInterval)
	defer ticker.Stop()
	for {
		if err := c.Client.Health(ctx); err == nil {
			return nil
		}

		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
		}
	}
}

func (c *ControlPlane) Generate(
	ctx context.Context,
	pending request.PendingRequest,
) (response.GenerateResponse, error) {
	if c.getState() != StateReady {
		return response.GenerateResponse{}, errors.New("engine not ready")
	}
	resp, err := c.Client.Generate(ctx, pending.Payload)
	if err != nil {
		return response.GenerateResponse{}, fmt.Errorf("engine generation failed: %w", err)
	}
	return resp, nil
}

func NewControlPlane(
	ctx context.Context,
	cfg config.ControlPlaneConfig,
	client engine.Client,
) *ControlPlane {

	var provider gpuprovider.GPUProvider

	if cfg.GPUProviderConfig.VastAI != nil {
		provider = gpuprovider.NewVastAIProvider(cfg.GPUProviderConfig.VastAI)
	} else if cfg.GPUProviderConfig.RunPod != nil {
		provider = gpuprovider.NewRunPodProvider(cfg.GPUProviderConfig.RunPod)
	}

	return &ControlPlane{
		state:           StateOffline,
		lastError:       nil,
		ctx:             ctx,
		RequestManager:  request.NewRequestManager(cfg.RequestManagerConfig),
		Client:          client,
		Provider:        provider,
		Cfg:             cfg,
		activeInstances: make(map[string]gpuprovider.Instance),
		instanceIDs:     make([]string, 0),
	}
}

func withRetry(ctx context.Context, operation func() error, maxRetries int, initialBackoff time.Duration) error {
	backoff := initialBackoff
	for i := 0; i < maxRetries; i++ {
		err := operation()
		if err == nil {
			return nil
		}
		log.Printf("operation failed, retrying: %v", err)

		select {
		case <-ctx.Done():
			return fmt.Errorf("%w (last error: %v)", ctx.Err(), err)
		case <-time.After(backoff):
		}
		backoff = min(backoff*2, 10*time.Second)
	}
	return fmt.Errorf("operation failed after %d retries", maxRetries)
}

func (c *ControlPlane) Shutdown(ctx context.Context) error {
	c.setState(StateShuttingDown)

	ctx, cancel := context.WithTimeout(ctx, 60*time.Second)
	defer cancel()

	c.mu.Lock()
	ids := slices.Clone(c.instanceIDs)
	c.mu.Unlock()

	var (
		wg   sync.WaitGroup
		mu   sync.Mutex
		errs []error
	)
	for _, id := range ids {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := withRetry(ctx, func() error {
				terr := c.Provider.Terminate(ctx, id)
				if errors.Is(terr, gpuprovider.ErrNotFound) {
					return nil
				}
				return terr
			}, 5, time.Second); err != nil {
				mu.Lock()
				errs = append(errs, fmt.Errorf("pod %s still running, terminate it manually: %w", id, err))
				mu.Unlock()
			}
		}()
	}
	wg.Wait()

	if err := errors.Join(errs...); err != nil {
		c.fail(err)
		return err
	}

	c.mu.Lock()
	c.instanceIDs = nil
	c.mu.Unlock()
	c.setState(StateOffline)
	log.Println("Control plane shut down successfully.")
	return nil
}
