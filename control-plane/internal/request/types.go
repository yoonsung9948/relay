package request

import (
	"encoding/json"
	"net/http"
)

type GenerateRequest struct {
	Prompt      string  `json:"prompt"`
	MaxTokens   int     `json:"max_tokens"`
	Temperature float64 `json:"temperature"`
}

type PendingRequest struct {
	Meta    RequestMetadata
	Payload GenerateRequest
}

type RequestMetadata struct {
}

func BuildPendingRequest(r *http.Request) (PendingRequest, error) {
	var payload GenerateRequest
	if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
		return PendingRequest{}, err
	}

	// limit max tokens for now
	payload.MaxTokens = max(1, min(payload.MaxTokens, 64))
	payload.Temperature = max(0.0, min(payload.Temperature, 1.0))

	return PendingRequest{
		Meta:    RequestMetadata{},
		Payload: payload,
	}, nil
}
