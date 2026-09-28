package api

import (
	"context"
	"crypto/subtle"
	"net/http"
	"time"
)

func withDemoKeyCheck(key string, next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		got := r.Header.Get("X-Demo-Key")
		if subtle.ConstantTimeCompare([]byte(got), []byte(key)) != 1 {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		next(w, r)
	}
}

func withConcurrencyLimit(sem chan struct{}, next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		select {
		case sem <- struct{}{}: // acquire
			defer func() { <-sem }() // release
			next(w, r)
		default: // full
			w.Header().Set("Retry-After", "5")
			http.Error(w, "at capacity", http.StatusTooManyRequests)
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
