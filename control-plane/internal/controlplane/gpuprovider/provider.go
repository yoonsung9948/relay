package gpuprovider

import (
	"context"
	"errors"
)

type CreateInstanceSpec struct {
	Image  string
	GPU    string
	DiskGB int
	Name   string
}

type GPUProvider interface {
	Create(
		ctx context.Context,
		spec CreateInstanceSpec,
	) (*Instance, error)
	Get(ctx context.Context, id string) (*Instance, error)
	Stop(ctx context.Context, id string) error
	Start(ctx context.Context, id string) error
	Terminate(ctx context.Context, id string) error
}

type Instance struct {
	ID       string
	Running  bool
	Status   string
	Endpoint string
}

var (
	ErrNotFound = errors.New("instance not found")
)
