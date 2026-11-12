package api

import (
	"context"
	"crypto/subtle"
	"github.com/yoonsung9948/relay/internal/observability"
	"github.com/yoonsung9948/relay/internal/ratelimit"
	"log"
	"net/http"
	"time"
)

func withDemoKeyCheck(key string, next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		got := r.Header.Get("X-Demo-Key")
		if subtle.ConstantTimeCompare([]byte(got), []byte(key)) != 1 {
			observability.Rejected(r.Context(), "unauthorized")
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		next(w, r)
	}
}

func withConcurrencyLimit(sem chan struct{}, next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Context().Err() != nil {
			return
		}
		select {
		case sem <- struct{}{}: // acquire
			defer func() { <-sem }() // release
			next(w, r)
		default: // full
			observability.Rejected(r.Context(), "generation_capacity")
			log.Printf("admission category=generation_capacity")
			writeRetryError(w, http.StatusServiceUnavailable, "generation_capacity", time.Time{})
		}
	}
}

func withTimeout(d time.Duration, next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		ctx, cancel := context.WithTimeout(r.Context(), d)
		defer cancel()
		next(w, r.WithContext(ctx))
	}
}

func withRequestRateLimit(limiter *ratelimit.RequestLimiter, enabled bool, next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Context().Err() != nil {
			return
		}
		if enabled {
			if retryAt := limiter.Allow(); !retryAt.IsZero() {
				observability.Rejected(r.Context(), "request_rate_limited")
				log.Printf("admission category=request_rate_limited retry_at=%s", retryAt.Format(time.RFC3339Nano))
				writeRetryError(w, http.StatusTooManyRequests, "request_rate_limited", retryAt)
				return
			}
		}
		next(w, r)
	}
}
