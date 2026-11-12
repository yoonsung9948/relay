package ratelimit

import (
	"fmt"
	"sync"
	"time"

	"github.com/yoonsung9948/relay/internal/config"
)

// CooldownError describes an admission rejection, independent of HTTP.
type CooldownError struct{ RetryAt time.Time }

func (e *CooldownError) Error() string {
	return fmt.Sprintf("GPU provisioning rate limited until %s", e.RetryAt.Format(time.RFC3339))
}

type ProvisionLimiter struct {
	mu           sync.Mutex
	cooldown     time.Duration
	retryBackoff time.Duration
	nextAllowed  time.Time
}

func NewProvisionLimiter(cfg config.ControlPlaneConfig) *ProvisionLimiter {
	return &ProvisionLimiter{cooldown: cfg.ProvisionCooldown, retryBackoff: cfg.ProvisionRetryBackoff}
}

// Allow atomically reserves the normal cooldown for an admitted attempt.
func (l *ProvisionLimiter) Allow() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	now := time.Now()
	if now.Before(l.nextAllowed) {
		return &CooldownError{RetryAt: l.nextAllowed}
	}
	l.nextAllowed = now.Add(l.cooldown)
	return nil
}

// NoCapacity replaces the reservation after a confirmed rejection with no Pod.
// The caller must serialize attempts so this cannot release another attempt's slot.
func (l *ProvisionLimiter) NoCapacity() time.Time {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.nextAllowed = time.Now().Add(l.retryBackoff)
	return l.nextAllowed
}
func (l *ProvisionLimiter) NextAllowed() time.Time {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.nextAllowed
}
