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
	Generate(ctx context.Context, request request.GenerateRequest) (response.GenerateResponse, error)
}

type HttpClient struct {
	endpoint string
}

func NewHttpClient(endpoint string) *HttpClient {
	return &HttpClient{
		endpoint: endpoint,
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
		c.endpoint,
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

	if resp.StatusCode != http.StatusOK {
		return response.GenerateResponse{}, fmt.Errorf("received non-OK HTTP status: %s", resp.Status)
	}

	defer resp.Body.Close()
	generatedResp, err := BuildGenerateResponse(resp)
	if err != nil {
		return response.GenerateResponse{}, fmt.Errorf("failed to build generate response: %w", err)
	}
	return generatedResp, nil
}

func BuildGenerateResponse(resp *http.Response) (response.GenerateResponse, error) {
	var genResp response.GenerateResponse
	if err := json.NewDecoder(resp.Body).Decode(&genResp); err != nil {
		return response.GenerateResponse{}, fmt.Errorf("failed to decode response: %w", err)
	}
	return genResp, nil
}
