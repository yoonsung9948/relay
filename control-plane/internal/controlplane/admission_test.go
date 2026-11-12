package controlplane

import (
	"context"
	"errors"
	"fmt"
	"github.com/yoonsung9948/relay/internal/config"
	"github.com/yoonsung9948/relay/internal/controlplane/gpuprovider"
	"github.com/yoonsung9948/relay/internal/ratelimit"
	"testing"
	"time"
)

func admissionCP(t *testing.T, enabled bool, createErr error) *ControlPlane {
	cp := newTestCP(t, &fakeProvider{createFn: func(context.Context, gpuprovider.CreateInstanceSpec) (*gpuprovider.Instance, error) {
		return nil, createErr
	}}, &fakeClient{})
	cp.Cfg.RateLimitEnabled = enabled
	cp.rateLimiter = ratelimit.NewProvisionLimiter(config.ControlPlaneConfig{ProvisionCooldown: time.Hour, ProvisionRetryBackoff: 15 * time.Second})
	return cp
}
func TestCapacityRetryAdmission(t *testing.T) {
	cp := admissionCP(t, true, fmt.Errorf("provider rejection: %w", gpuprovider.ErrNoCapacity))
	err := cp.StartEngine(context.Background(), config.GPUProviderConfig{})
	var capacity *NoCapacityError
	if !errors.As(err, &capacity) || !errors.Is(err, gpuprovider.ErrNoCapacity) {
		t.Fatalf("capacity cause lost: %v", err)
	}
	if until := time.Until(capacity.RetryAt); until <= 0 || until > 15*time.Second {
		t.Fatalf("wrong backoff %s", until)
	}
	var cooldown *ratelimit.CooldownError
	if err := cp.StartEngine(context.Background(), config.GPUProviderConfig{}); !errors.As(err, &cooldown) {
		t.Fatalf("expected cooldown: %v", err)
	}
	if cp.Status().State != StateOffline {
		t.Fatal("cooldown changed lifecycle state")
	}
}

func TestDisabledLimitsDoNotBlockCapacityRetry(t *testing.T) {
	cp := admissionCP(t, false, gpuprovider.ErrNoCapacity)
	for i := 0; i < 2; i++ {
		var capacity *NoCapacityError
		if err := cp.StartEngine(context.Background(), config.GPUProviderConfig{}); !errors.As(err, &capacity) || !capacity.RetryAt.IsZero() {
			t.Fatalf("unexpected admission: %v", err)
		}
	}
}
func TestAmbiguousCreateRetainsCooldown(t *testing.T) {
	cp := admissionCP(t, true, context.DeadlineExceeded)
	if err := cp.StartEngine(context.Background(), config.GPUProviderConfig{}); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatal(err)
	}
	if err := cp.StopEngine(context.Background(), "test_reconcile"); err == nil {
		t.Fatal("ambiguous creation was cleared")
	}
	if cp.Status().State != StateError {
		t.Fatal("ambiguous failure must require reconciliation")
	}
	if time.Until(cp.rateLimiter.NextAllowed()) < 59*time.Minute {
		t.Fatal("ambiguous failure released cooldown")
	}
}
