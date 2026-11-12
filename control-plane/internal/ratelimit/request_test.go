package ratelimit

import (
	"sync"
	"sync/atomic"
	"testing"
	"testing/synctest"
	"time"
)

func TestRequestBurstAndRefill(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		l := NewRequestLimiter(2, 2)
		if !l.Allow().IsZero() || !l.Allow().IsZero() {
			t.Fatal("initial burst rejected")
		}
		deadline := l.Allow()
		if !deadline.Equal(time.Now().Add(500 * time.Millisecond)) {
			t.Fatal("incorrect retry time")
		}
		time.Sleep(250 * time.Millisecond)
		if !l.Allow().Equal(deadline) {
			t.Fatal("rejected request extended the deadline")
		}
		time.Sleep(250 * time.Millisecond)
		if !l.Allow().IsZero() {
			t.Fatal("token did not refill")
		}
		time.Sleep(time.Hour)
		if !l.Allow().IsZero() || !l.Allow().IsZero() || l.Allow().IsZero() {
			t.Fatal("refill exceeded burst")
		}
	})
}
func TestRequestBucketConcurrentAdmission(t *testing.T) {
	l := NewRequestLimiter(0.000001, 3)
	var admitted atomic.Int32
	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if l.Allow().IsZero() {
				admitted.Add(1)
			}
		}()
	}
	wg.Wait()
	if admitted.Load() != 3 {
		t.Fatalf("admitted %d", admitted.Load())
	}
}
