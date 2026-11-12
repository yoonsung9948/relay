package ratelimit

import (
	"errors"
	"github.com/yoonsung9948/relay/internal/config"
	"testing"
	"testing/synctest"
	"time"
)

func TestCooldownAndCapacityBackoff(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		l := NewProvisionLimiter(config.ControlPlaneConfig{ProvisionCooldown: time.Hour, ProvisionRetryBackoff: 15 * time.Second})
		if err := l.Allow(); err != nil {
			t.Fatal(err)
		}
		original := l.NextAllowed()
		var cooldown *CooldownError
		if err := l.Allow(); !errors.As(err, &cooldown) || !cooldown.RetryAt.Equal(original) {
			t.Fatalf("unexpected cooldown: %v", err)
		}
		retry := l.NoCapacity()
		if !retry.Before(original) {
			t.Fatal("capacity rejection did not shorten cooldown")
		}
		time.Sleep(14 * time.Second)
		if err := l.Allow(); err == nil {
			t.Fatal("backoff not enforced")
		}
		time.Sleep(time.Second)
		if err := l.Allow(); err != nil {
			t.Fatal(err)
		}
		if !l.NextAllowed().Equal(time.Now().Add(time.Hour)) {
			t.Fatal("normal cooldown not restored")
		}
	})
}
