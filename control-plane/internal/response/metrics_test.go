package response

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestMetricsPreserveZeroAndUnavailable(t *testing.T) {
	zero := 0.0
	original := GenerateChunk{Type: "done", Metrics: &GenerationMetrics{
		Version: 1, Source: "engine", TTFTMs: &zero, OutputTokens: 1,
	}, Generation: map[string]any{"sampling": "greedy", "max_tokens": 64}}
	raw, err := json.Marshal(original)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(raw), `"ttft_ms":0`) || !strings.Contains(string(raw), `"decode_tokens_per_second":null`) {
		t.Fatalf("metrics lost zero/null distinction: %s", raw)
	}
	var decoded GenerateChunk
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.Metrics == nil || decoded.Metrics.TTFTMs == nil || *decoded.Metrics.TTFTMs != 0 || decoded.Metrics.DecodeTokensPerSecond != nil {
		t.Fatalf("bad round trip: %+v", decoded.Metrics)
	}
}

func TestPoolMetricsRoundTrip(t *testing.T) {
	raw := `{"type":"done","metrics":{"version":1,"source":"engine","prefill_ms":0,"decode_steps":3,"decode_batch_size_max":4,"decode_batch_size_mean":2.5},"generation":{"kv_cache_mode":"pool","decode_batching":true}}`
	var chunk GenerateChunk
	if err := json.Unmarshal([]byte(raw), &chunk); err != nil {
		t.Fatal(err)
	}
	encoded, err := json.Marshal(chunk)
	if err != nil {
		t.Fatal(err)
	}
	for _, field := range []string{`"prefill_ms":0`, `"decode_steps":3`, `"decode_batch_size_max":4`, `"decode_batch_size_mean":2.5`, `"kv_cache_mode":"pool"`, `"decode_batching":true`} {
		if !strings.Contains(string(encoded), field) {
			t.Fatalf("lost %s: %s", field, encoded)
		}
	}
}
