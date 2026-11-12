package engine

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/yoonsung9948/relay/internal/request"
)

func TestGenerateStreamDeliversBeforeCompletion(t *testing.T) {
	release := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		io.WriteString(w, "{\"type\":\"delta\",\"request_id\":\"abc\",\"text\":\"hello\"}\n")
		w.(http.Flusher).Flush()
		select {
		case <-release:
		case <-r.Context().Done():
			return
		}
		io.WriteString(w, `{"type":"done","request_id":"abc","metrics":{"version":1,"source":"engine","input_tokens":2,"output_tokens":3,"ttft_ms":0,"decode_tokens_per_second":null,"total_latency_ms":12}}`+"\n")
	}))
	defer server.Close()
	defer close(release)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	stream, err := NewHttpClient(server.URL).GenerateStream(ctx, request.GenerateRequest{})
	if err != nil {
		t.Fatal(err)
	}
	select {
	case result := <-stream:
		if result.Err != nil || result.Chunk.Text != "hello" || *result.Chunk.RequestID != "abc" {
			t.Fatalf("bad chunk: %+v", result)
		}
	case <-ctx.Done():
		t.Fatal("first chunk waited for stream completion")
	}
	release <- struct{}{}
	terminal := <-stream
	if terminal.Chunk.Type != "done" || terminal.Err != nil || terminal.Chunk.Metrics == nil || terminal.Chunk.Metrics.OutputTokens != 3 {
		t.Fatalf("bad completion: %+v", terminal)
	}
	if _, ok := <-stream; ok {
		t.Fatal("channel did not close")
	}
}

func TestGenerateStreamErrors(t *testing.T) {
	for _, tc := range []struct {
		name, body    string
		unexpectedEOF bool
	}{
		{"upstream error", "{\"type\":\"error\",\"error\":\"inference failed\"}\n", false},
		{"missing done", "{\"type\":\"delta\",\"text\":\"partial\"}\n", true},
		{"malformed", "not json\n", false},
		{"unknown type", "{\"type\":\"mystery\"}\n", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { io.WriteString(w, tc.body) }))
			defer server.Close()
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()
			stream, err := NewHttpClient(server.URL).GenerateStream(ctx, request.GenerateRequest{})
			if err != nil {
				t.Fatal(err)
			}
			var got error
			for result := range stream {
				if result.Err != nil {
					got = result.Err
				}
			}
			if got == nil {
				t.Fatal("stream failure reported as success")
			}
			if tc.unexpectedEOF && !errors.Is(got, io.ErrUnexpectedEOF) {
				t.Fatalf("got %v", got)
			}
		})
	}
}

func TestGenerateStreamCancellationClosesUpstream(t *testing.T) {
	disconnected := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.(http.Flusher).Flush()
		<-r.Context().Done()
		close(disconnected)
	}))
	defer server.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	stream, err := NewHttpClient(server.URL).GenerateStream(ctx, request.GenerateRequest{})
	if err != nil {
		t.Fatal(err)
	}
	cancel()
	select {
	case <-disconnected:
	case <-time.After(2 * time.Second):
		t.Fatal("upstream stayed connected")
	}
	select {
	case <-stream:
	case <-time.After(2 * time.Second):
		t.Fatal("decoder stayed blocked")
	}
}

type trackingBody struct {
	io.Reader
	closed bool
}

func (b *trackingBody) Close() error { b.closed = true; return nil }

type transportFunc func(*http.Request) (*http.Response, error)

func (f transportFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func TestErrorPathsCloseBodies(t *testing.T) {
	for _, status := range []int{http.StatusOK, http.StatusServiceUnavailable} {
		t.Run(http.StatusText(status), func(t *testing.T) {
			body := &trackingBody{Reader: strings.NewReader("invalid json")}
			original := http.DefaultClient
			http.DefaultClient = &http.Client{Transport: transportFunc(func(r *http.Request) (*http.Response, error) {
				return &http.Response{StatusCode: status, Status: http.StatusText(status), Body: body, Header: make(http.Header)}, nil
			})}
			defer func() { http.DefaultClient = original }()
			_, err := NewHttpClient("http://engine").Generate(context.Background(), request.GenerateRequest{})
			if err == nil || !body.closed {
				t.Fatalf("err=%v closed=%v", err, body.closed)
			}
		})
	}
}
