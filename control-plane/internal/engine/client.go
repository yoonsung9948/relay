package engine

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/yoonsung9948/relay/internal/request"
	"github.com/yoonsung9948/relay/internal/response"
)

type Client interface {
	Health(ctx context.Context) error
	Generate(ctx context.Context, request request.GenerateRequest) (response.GenerateResponse, error)
	SetEndpoint(endpoint string)
}

type HttpClient struct {
	Endpoint string
}

type HealthResponse struct {
	Status string `json:"status"`
	Stage  string `json:"stage,omitempty"`
	Detail string `json:"detail,omitempty"`
}

func (c *HttpClient) Health(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.Endpoint+"/health/ready", nil)
	if err != nil {
		return fmt.Errorf("failed to create HTTP request: %w", err)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to send HTTP request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("received non-OK HTTP status: %s", resp.Status)
	}
	var health struct {
		Status string `json:"status"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&health); err != nil {
		return fmt.Errorf("decode health response: %w", err)
	}
	if health.Status != "ok" {
		return fmt.Errorf("engine not healthy: %s", health.Status)
	}

	return nil
}

func NewHttpClient(endpoint string) *HttpClient {
	return &HttpClient{
		Endpoint: endpoint,
	}
}

func (c *HttpClient) Generate(
	ctx context.Context,
	request request.GenerateRequest,
) (response.GenerateResponse, error) {
	data, err := json.Marshal(request)
	if err != nil {
		return response.GenerateResponse{}, fmt.Errorf("failed to marshal request: %w", err)
	}
	req, err := http.NewRequestWithContext(
		ctx,
		http.MethodPost,
		c.Endpoint+"/generate",
		bytes.NewBuffer(data),
	)
	if err != nil {
		return response.GenerateResponse{}, fmt.Errorf("failed to create HTTP request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return response.GenerateResponse{}, fmt.Errorf("failed to send HTTP request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return response.GenerateResponse{}, fmt.Errorf("received non-OK HTTP status: %s", resp.Status)
	}

	generatedResp, err := BuildGenerateResponse(resp)
	if err != nil {
		return response.GenerateResponse{}, fmt.Errorf("failed to build generate response: %w", err)
	}
	return generatedResp, nil
}

func (c *HttpClient) SetEndpoint(endpoint string) {
	c.Endpoint = endpoint
}

func BuildGenerateResponse(resp *http.Response) (response.GenerateResponse, error) {
	var genResp response.GenerateResponse
	if err := json.NewDecoder(resp.Body).Decode(&genResp); err != nil {
		return response.GenerateResponse{}, fmt.Errorf("failed to decode response: %w", err)
	}
	return genResp, nil
}
