package response

type Response struct {
	Metadata ResponseMetadata
	Payload  GenerateResponse
}

type ResponseMetadata struct {
}

type GenerateResponse struct {
	RequestID       *string            `json:"request_id"`
	Text            string             `json:"text"`
	GeneratedTokens int                `json:"generated_tokens"`
	Metrics         *GenerationMetrics `json:"metrics,omitempty"`
	FinishReason    string             `json:"finish_reason,omitempty"`
	Generation      map[string]any     `json:"generation,omitempty"`
}

// GenerationMetrics contains Python-local durations, not proxy or browser timings.
// Nullable values preserve the distinction between zero and unavailable.
type GenerationMetrics struct {
	PrefillMs             *float64 `json:"prefill_ms"`
	DecodeSteps           *int     `json:"decode_steps"`
	DecodeBatchSizeMax    int      `json:"decode_batch_size_max"`
	DecodeBatchSizeMean   *float64 `json:"decode_batch_size_mean"`
	Version               int      `json:"version"`
	Source                string   `json:"source"`
	InputTokens           int      `json:"input_tokens"`
	OutputTokens          int      `json:"output_tokens"`
	TTFTMs                *float64 `json:"ttft_ms"`
	DecodeDurationMs      *float64 `json:"decode_duration_ms"`
	DecodeTokensPerSecond *float64 `json:"decode_tokens_per_second"`
	TotalLatencyMs        float64  `json:"total_latency_ms"`
	InitialQueueMs        *float64 `json:"initial_queue_ms"`
}

type GenerateChunk struct {
	Type         string             `json:"type"`
	RequestID    *string            `json:"request_id"`
	Text         string             `json:"text"`
	Error        string             `json:"error,omitempty"`
	Metrics      *GenerationMetrics `json:"metrics,omitempty"`
	FinishReason string             `json:"finish_reason,omitempty"`
	Generation   map[string]any     `json:"generation,omitempty"`
}

type StreamResponse struct {
	Chunk GenerateChunk
	Err   error
}
