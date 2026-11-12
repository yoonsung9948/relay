package request

import (
	"bytes"
	"encoding/json"
	"errors"
	"math"
	"net/http"
	"strings"
)

type GenerateRequest struct {
	KVCacheMode    string          `json:"kv_cache_mode,omitempty"`
	DecodeBatching bool            `json:"decode_batching"`
	Prompt         string          `json:"prompt"`
	MaxTokens      int             `json:"max_tokens"`
	Temperature    float64         `json:"temperature"`
	Meta           RequestMetadata `json:"meta,omitempty"`
	Stream         bool            `json:"stream"`
	UseKVCache     bool            `json:"use_kv_cache,omitempty"`
}

// Execution mode is normalized during decoding; the legacy cache flag remains supported.
func (r *GenerateRequest) UnmarshalJSON(data []byte) error {
	type wire GenerateRequest
	value := wire(*r)
	if err := json.Unmarshal(data, &value); err != nil {
		return err
	}
	var fields struct {
		Use         *bool           `json:"use_kv_cache"`
		Temperature json.RawMessage `json:"temperature"`
	}
	if err := json.Unmarshal(data, &fields); err != nil {
		return err
	}
	if bytes.Equal(bytes.TrimSpace(fields.Temperature), []byte("null")) ||
		math.IsNaN(value.Temperature) || math.IsInf(value.Temperature, 0) || value.Temperature < 0 {
		return errors.New("temperature must be finite and nonnegative")
	}
	mode := value.KVCacheMode
	if mode == "" {
		mode = "none"
		if fields.Use != nil && *fields.Use {
			mode = "dynamic"
		}
	}
	if mode != "none" && mode != "dynamic" && mode != "pool" {
		return errors.New("unknown kv_cache_mode")
	}
	if fields.Use != nil && *fields.Use != (mode != "none") {
		return errors.New("conflicting cache options")
	}
	if value.DecodeBatching && mode != "pool" {
		return errors.New("decode_batching requires pool")
	}
	value.KVCacheMode = mode
	value.UseKVCache = mode != "none"
	*r = GenerateRequest(value)
	return nil
}

type PendingRequest struct {
	Meta    RequestMetadata
	Payload GenerateRequest
}

type RequestMetadata struct {
}

func BuildPendingRequest(r *http.Request) (PendingRequest, error) {
	return BuildPendingRequestWithLimit(r, 64)
}

func BuildPendingRequestWithLimit(r *http.Request, maxOutputTokens int) (PendingRequest, error) {
	if maxOutputTokens < 1 {
		return PendingRequest{}, errors.New("output token limit must be positive")
	}
	payload := GenerateRequest{MaxTokens: 64}

	if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
		return PendingRequest{}, err
	}

	if payload.MaxTokens < 0 {
		return PendingRequest{}, errors.New("max_tokens must be nonnegative")
	}
	if strings.TrimSpace(payload.Prompt) == "" {
		return PendingRequest{}, errors.New("prompt must not be empty")
	}
	// Admission policy caps work independently of each request's desired output length.
	payload.MaxTokens = min(payload.MaxTokens, maxOutputTokens)

	return PendingRequest{
		Meta:    payload.Meta,
		Payload: payload,
	}, nil
}
