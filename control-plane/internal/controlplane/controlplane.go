package controlplane

import (
	"context"
	"fmt"

	"github.com/yoonsung9948/relay/internal/config"
	"github.com/yoonsung9948/relay/internal/controlplane/backends"
	"github.com/yoonsung9948/relay/internal/engine"
	"github.com/yoonsung9948/relay/internal/request"
	"github.com/yoonsung9948/relay/internal/response"
)

type ControlPlane struct {
	ctx            context.Context
	RequestManager *request.RequestManager
	Client         engine.Client
}

func (c *ControlPlane) Start(ctx context.Context) error {
	fmt.Println("Starting control plane...")
	return nil
}

// context is request context
func (c *ControlPlane) StartEngine(ctx context.Context) error {
	if err := backends.StartVastAI(c.ctx); err != nil {
		return fmt.Errorf("failed to start VastAI backend: %w", err)
	}

	return nil
}

func (c *ControlPlane) Generate(
	ctx context.Context,
	pending request.PendingRequest,
) (response.GenerateResponse, error) {
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

	return &ControlPlane{
		ctx:            ctx,
		RequestManager: request.NewRequestManager(cfg.RequestManagerConfig),
		Client:         client,
	}
}

func (c *ControlPlane) Shutdown(ctx context.Context) error {
	// stop accepting new work
	// stop scheduler
	// stop engines
	// drain or cancel requests
	return nil
}
