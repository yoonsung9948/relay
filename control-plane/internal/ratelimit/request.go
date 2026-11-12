package ratelimit

import (
	"math"
	"sync"
	"time"
)

// RequestLimiter is one global token bucket for this Go process.
// It controls request frequency; it does not count in-flight generations.
type RequestLimiter struct {
	mu        sync.Mutex
	rate      float64
	burst     float64
	tokens    float64
	updatedAt time.Time
}

func NewRequestLimiter(requestsPerSecond float64, burst int) *RequestLimiter {
	if requestsPerSecond <= 0 || math.IsNaN(requestsPerSecond) || math.IsInf(requestsPerSecond, 0) || burst < 1 {
		panic("invalid request limiter configuration")
	}
	return &RequestLimiter{rate: requestsPerSecond, burst: float64(burst), tokens: float64(burst), updatedAt: time.Now()}
}

// Allow consumes a token or returns the earliest time another token is available.
// Rejected attempts do not extend the deadline.
func (l *RequestLimiter) Allow() (retryAt time.Time) {
	l.mu.Lock()
	defer l.mu.Unlock()
	now := time.Now()
	l.tokens = min(l.burst, l.tokens+now.Sub(l.updatedAt).Seconds()*l.rate)
	l.updatedAt = now
	if l.tokens >= 1 {
		l.tokens--
		return time.Time{}
	}
	waitSeconds := (1 - l.tokens) / l.rate
	// Clamp extremely low configured rates instead of overflowing time.Duration.
	waitNanos := math.Min(math.Ceil(waitSeconds*float64(time.Second)), float64(math.MaxInt64-1024))
	return now.Add(time.Duration(waitNanos))
}
