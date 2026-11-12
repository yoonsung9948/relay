package engine

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"

	"github.com/yoonsung9948/relay/internal/observability"
	"github.com/yoonsung9948/relay/internal/request"
	"github.com/yoonsung9948/relay/internal/response"
)

type Client interface {
	Health(ctx context.Context) error
	Generate(ctx context.Context, request request.GenerateRequest) (response.GenerateResponse, error)
	SetEndpoint(endpoint string)
	GenerateStream(ctx context.Context, request request.GenerateRequest) (<-chan response.StreamResponse, error)
}

type HTTPError struct {
	StatusCode      int
	RetryAfter      string
	RejectionReason string
}

func (e *HTTPError) Error() string { return fmt.Sprintf("engine returned HTTP %d", e.StatusCode) }

type HttpClient struct {
	endpointMu sync.RWMutex
	Endpoint   string
	httpClient *http.Client
}

func (c *HttpClient) generationClient() *http.Client {
	if c.httpClient != nil {
		return c.httpClient
	}
	return http.DefaultClient
}

type Identity struct {
	EngineRevision      string `json:"engine_revision"`
	EngineSourceSHA256  string `json:"engine_source_sha256"`
	EngineDirty         *bool  `json:"engine_dirty"`
	ModelRevision       string `json:"model_revision"`
	ConfigSHA256        string `json:"config_sha256"`
	DeviceName          string `json:"device_name"`
	TorchVersion        string `json:"torch_version"`
	TransformersVersion string `json:"transformers_version"`
}

type Instrumentation struct {
	TracingEnabled  bool   `json:"tracing_enabled"`
	Sampler         string `json:"sampler"`
	SamplerName     string `json:"sampler_name"`
	TraceSteps      bool   `json:"trace_steps"`
	MemoryEnabled   bool   `json:"memory_enabled"`
	MemorySupported bool   `json:"memory_supported"`
}

type Hardware struct {
	GPUName         *string `json:"gpu_name"`
	VisibleGPUCount *int    `json:"visible_gpu_count"`
	VRAMBytesPerGPU *int64  `json:"vram_bytes_per_gpu"`
	CUDAVersion     *string `json:"cuda_version"`
}

type Capabilities struct {
	Hardware            *Hardware        `json:"hardware,omitempty"`
	TemperatureSampling *bool            `json:"temperature_sampling,omitempty"`
	DecodeBatching      *bool            `json:"decode_batching,omitempty"`
	MaxSeqLen           *int             `json:"max_seq_len,omitempty"`
	Identity            *Identity        `json:"identity,omitempty"`
	Instrumentation     *Instrumentation `json:"instrumentation,omitempty"`
	InFlightRequests    int              `json:"in_flight_requests"`
	ActiveRequests      int              `json:"active_requests"`
	PendingRequests     int              `json:"pending_requests"`
	MaxPendingRequests  int              `json:"max_pending_requests"`
	PoolMaxSeqLen       int              `json:"pool_max_seq_len"`
	ConfigPath          *string          `json:"config_path"`
	Model               *string          `json:"model"`
	Backend             *string          `json:"backend"`
	Device              *string          `json:"device"`
	Dtype               *string          `json:"dtype"`
	KVCacheModes        []string         `json:"kv_cache_modes"`
	PoolSlots           int              `json:"pool_slots"`
	PoolAllocatedBytes  int64            `json:"pool_allocated_bytes"`
	MaxActiveRequests   int              `json:"max_active_requests"`
	MaxDecodeBatchSize  int              `json:"max_decode_batch_size"`
}

type HealthResponse struct {
	Capabilities *Capabilities `json:"capabilities,omitempty"`
	Status       string        `json:"status"`
	Stage        string        `json:"stage,omitempty"`
	Detail       string        `json:"detail,omitempty"`
}

// Readiness returns a fresh worker description; callers must not cache capabilities.
func (c *HttpClient) Readiness(ctx context.Context) (HealthResponse, error) {
	var health HealthResponse
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.endpoint()+"/health/ready", nil)
	if err != nil {
		return health, fmt.Errorf("create readiness request: %w", err)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return health, fmt.Errorf("request readiness: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return health, fmt.Errorf("worker readiness HTTP %d", resp.StatusCode)
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 64<<10)).Decode(&health); err != nil {
		return health, fmt.Errorf("decode readiness: %w", err)
	}
	if health.Status != "ok" {
		return health, fmt.Errorf("engine not healthy: %s", health.Status)
	}
	return health, nil
}

func (c *HttpClient) Health(ctx context.Context) error {
	_, err := c.Readiness(ctx)
	return err
}

func NewHttpClient(endpoint string) *HttpClient {
	client := &HttpClient{Endpoint: endpoint}
	if observability.Enabled() {
		copy := *http.DefaultClient
		base := copy.Transport
		if base == nil {
			base = http.DefaultTransport
		}
		copy.Transport = observability.Transport(base)
		client.httpClient = &copy
	}
	return client
}

func (c *HttpClient) Generate(
	ctx context.Context,
	request request.GenerateRequest,
) (response.GenerateResponse, error) {
	resp, err := c.send(ctx, request)
	if err != nil {
		return response.GenerateResponse{}, fmt.Errorf("send http request: %w", err)
	}
	defer resp.Body.Close()
	generateResp, err := BuildGenerateResponse(resp)
	if err != nil {
		return response.GenerateResponse{}, fmt.Errorf("build generate response: %w", err)
	}
	return generateResp, nil
}

func (c *HttpClient) GenerateStream(
	ctx context.Context,
	request request.GenerateRequest,
) (<-chan response.StreamResponse, error) {
	results := make(chan response.StreamResponse)
	request.Stream = true
	resp, err := c.send(ctx, request)
	if err != nil {
		return nil, fmt.Errorf("send http request: %w", err)
	}

	go func() {
		defer close(results)
		defer resp.Body.Close()

		decoder := json.NewDecoder(resp.Body)
		send := func(result response.StreamResponse) bool {
			select {
			case results <- result:
				return true
			case <-ctx.Done():
				return false
			}
		}
		for {
			var chunk response.GenerateChunk
			if err := decoder.Decode(&chunk); err != nil {
				if err == io.EOF {
					err = fmt.Errorf("engine stream ended without a done event: %w", io.ErrUnexpectedEOF)
				}
				send(response.StreamResponse{Err: err})
				return
			}
			switch chunk.Type {
			case "delta":
				if !send(response.StreamResponse{Chunk: chunk}) {
					return
				}
			case "done":
				send(response.StreamResponse{Chunk: chunk})
				return
			case "error":
				send(response.StreamResponse{Chunk: chunk, Err: fmt.Errorf("engine stream failed: %s", chunk.Error)})
				return
			default:
				send(response.StreamResponse{Err: fmt.Errorf("unknown engine stream event: %q", chunk.Type)})
				return
			}
		}
	}()

	return results, nil
}

func (c *HttpClient) send(ctx context.Context, request request.GenerateRequest) (*http.Response, error) {
	req, err := buildGenerateRequest(ctx, c.endpoint(), request)
	if err != nil {
		return nil, fmt.Errorf("build HTTP request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.generationClient().Do(req)
	if err != nil {
		return nil, fmt.Errorf("send HTTP request: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		return nil, &HTTPError{StatusCode: resp.StatusCode, RetryAfter: resp.Header.Get("Retry-After"), RejectionReason: resp.Header.Get("X-Relay-Rejection")}
	}
	return resp, nil
}

func (c *HttpClient) SetEndpoint(endpoint string) {
	c.endpointMu.Lock()
	defer c.endpointMu.Unlock()
	c.Endpoint = endpoint
}

func buildGenerateRequest(ctx context.Context, endpoint string, request request.GenerateRequest) (*http.Request, error) {
	data, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}
	req, err := http.NewRequestWithContext(
		ctx,
		http.MethodPost,
		endpoint+"/generate",
		bytes.NewBuffer(data),
	)
	if err != nil {
		return nil, fmt.Errorf("create HTTP request: %w", err)
	}
	return req, nil
}

func BuildGenerateResponse(resp *http.Response) (response.GenerateResponse, error) {
	var genResp response.GenerateResponse
	if err := json.NewDecoder(resp.Body).Decode(&genResp); err != nil {
		return response.GenerateResponse{}, fmt.Errorf("failed to decode response: %w", err)
	}
	return genResp, nil
}

func (c *HttpClient) endpoint() string {
	c.endpointMu.RLock()
	defer c.endpointMu.RUnlock()
	return c.Endpoint
}
