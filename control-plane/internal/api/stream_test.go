package api

import (
	"context"
	"errors"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/yoonsung9948/relay/internal/config"
	"github.com/yoonsung9948/relay/internal/controlplane"
	"github.com/yoonsung9948/relay/internal/controlplane/gpuprovider"
	"github.com/yoonsung9948/relay/internal/request"
	"github.com/yoonsung9948/relay/internal/response"
)

type streamClient struct{ events []response.StreamResponse }

func (*streamClient) Health(context.Context) error { return nil }
func (*streamClient) SetEndpoint(string)           {}
func (*streamClient) Generate(context.Context, request.GenerateRequest) (response.GenerateResponse, error) {
	return response.GenerateResponse{}, nil
}
func (c *streamClient) GenerateStream(context.Context, request.GenerateRequest) (<-chan response.StreamResponse, error) {
	results := make(chan response.StreamResponse, len(c.events))
	for _, event := range c.events {
		results <- event
	}
	close(results)
	return results, nil
}

type streamProvider struct{}

func (*streamProvider) Create(context.Context, gpuprovider.CreateInstanceSpec) (*gpuprovider.Instance, error) {
	return &gpuprovider.Instance{ID: "test", Endpoint: "http://fake"}, nil
}
func (*streamProvider) Get(context.Context, string) (*gpuprovider.Instance, error) {
	return &gpuprovider.Instance{ID: "test", Endpoint: "http://fake"}, nil
}
func (*streamProvider) Stop(context.Context, string) error      { return nil }
func (*streamProvider) Start(context.Context, string) error     { return nil }
func (*streamProvider) Terminate(context.Context, string) error { return nil }

func TestGenerateHandlerStreamTerminalEvents(t *testing.T) {
	for _, fail := range []bool{false, true} {
		name := "done"
		if fail {
			name = "error"
		}
		t.Run(name, func(t *testing.T) {
			id := "request-1"
			events := []response.StreamResponse{{Chunk: response.GenerateChunk{Type: "delta", RequestID: &id, Text: "hello"}}}
			if fail {
				events = append(events, response.StreamResponse{Err: errors.New("upstream failure")})
			} else {
				events = append(events, response.StreamResponse{Chunk: response.GenerateChunk{Type: "done", RequestID: &id, Metrics: &response.GenerationMetrics{Version: 1, Source: "engine", OutputTokens: 3}}})
			}
			ctx, cancel := context.WithTimeout(context.Background(), time.Second)
			defer cancel()
			cp := controlplane.NewControlPlane(ctx, config.ControlPlaneConfig{}, &streamClient{events: events})
			cp.Provider = &streamProvider{}
			if err := cp.StartEngine(ctx, config.GPUProviderConfig{}); err != nil {
				t.Fatal(err)
			}
			for cp.Status().State != controlplane.StateReady {
				select {
				case <-ctx.Done():
					t.Fatal("not ready")
				case <-time.After(time.Millisecond):
				}
			}
			server := &Server{control: cp}
			req := httptest.NewRequest("POST", "/generate", strings.NewReader(`{"prompt":"hi","stream":true}`)).WithContext(ctx)
			recorder := httptest.NewRecorder()
			server.GenerateHandler(recorder, req)
			body := recorder.Body.String()
			if recorder.Header().Get("Content-Type") != "text/event-stream" {
				t.Fatal("wrong content type")
			}
			if !strings.Contains(body, `"text":"hello"`) {
				t.Fatalf("missing delta: %s", body)
			}
			if !strings.Contains(body, "event: "+name+"\n") {
				t.Fatalf("missing terminal event: %s", body)
			}
			if !fail && !strings.Contains(body, `"output_tokens":3`) {
				t.Fatalf("metrics not forwarded: %s", body)
			}
			if !recorder.Flushed {
				t.Fatal("chunks not flushed")
			}
		})
	}
}
