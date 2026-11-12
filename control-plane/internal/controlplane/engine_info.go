package controlplane

import (
	"context"
	"errors"

	"github.com/yoonsung9948/relay/internal/engine"
)

// EngineInfo fetches live readiness; it never starts or reconfigures a worker.
func (c *ControlPlane) EngineInfo(ctx context.Context) (engine.HealthResponse, error) {
	if c.getState() != StateReady {
		return engine.HealthResponse{}, errors.New("engine not ready")
	}
	client, ok := c.Client.(interface {
		Readiness(context.Context) (engine.HealthResponse, error)
	})
	if !ok {
		return engine.HealthResponse{}, errors.New("engine readiness information unavailable")
	}
	return client.Readiness(ctx)
}
