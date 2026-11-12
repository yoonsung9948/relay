package request

import (
	"encoding/json"
	"testing"
)

func TestExecutionOptions(t *testing.T) {
	for _, tc := range []struct {
		body, mode string
		invalid    bool
	}{
		{`{"prompt":"hi"}`, "none", false},
		{`{"use_kv_cache":true}`, "dynamic", false},
		{`{"kv_cache_mode":"pool","decode_batching":true}`, "pool", false},
		{`{"kv_cache_mode":"pool","use_kv_cache":false}`, "", true},
		{`{"kv_cache_mode":"dynamic","decode_batching":true}`, "", true},
	} {
		var req GenerateRequest
		err := json.Unmarshal([]byte(tc.body), &req)
		if (err != nil) != tc.invalid {
			t.Fatalf("%s: %v", tc.body, err)
		}
		if err == nil && req.KVCacheMode != tc.mode {
			t.Fatal(req)
		}
	}
}
