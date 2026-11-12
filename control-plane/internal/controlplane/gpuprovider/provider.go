package gpuprovider

import (
	"context"
	"errors"
	"fmt"
	"net"
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
	Name     string
	ID       string
	Running  bool
	Status   string
	Endpoint string
}

var (
	ErrNotFound = errors.New("instance not found")
)

// Optional provider capabilities needed to reconcile lost creation responses.
type InstanceFinder interface {
	FindByName(context.Context, string) ([]*Instance, error)
}
type ScopedProvider interface{ OwnershipScope() string }

var ErrUnsupported = errors.New("GPU provider is not implemented")

type HTTPError struct {
	StatusCode int
	Operation  string
}

func (e *HTTPError) Error() string {
	return fmt.Sprintf("provider %s: HTTP %d", e.Operation, e.StatusCode)
}
func Retryable(err error) bool {
	if errors.Is(err, context.Canceled) {
		return false
	}
	var response *HTTPError
	if errors.As(err, &response) {
		return response.StatusCode == 429 || response.StatusCode >= 500
	}
	var network net.Error
	return errors.As(err, &network)
}
func DefinitiveCreateFailure(err error) bool {
	if errors.Is(err, ErrNoCapacity) || errors.Is(err, ErrUnsupported) {
		return true
	}
	var response *HTTPError
	if errors.As(err, &response) {
		switch response.StatusCode {
		case 400, 401, 402, 403, 404, 422, 429:
			return true
		}
	}
	return false
}
