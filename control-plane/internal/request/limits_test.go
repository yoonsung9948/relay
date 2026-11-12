package request

import (
	"fmt"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestEffectiveGenerationLimits(t *testing.T) {
	for _, tc := range []struct {
		body        string
		want        int
		temperature float64
		invalid     bool
	}{
		{`{"prompt":"hi","max_tokens":0}`, 0, 0, false},
		{`{"prompt":"hi"}`, 64, 0, false},
		{`{"prompt":"hi","max_tokens":256,"temperature":0.7}`, 64, 0.7, false},
		{`{"prompt":"hi","max_tokens":-1}`, 0, 0, true},
	} {
		got, err := BuildPendingRequest(httptest.NewRequest("POST", "/generate", strings.NewReader(tc.body)))
		if (err != nil) != tc.invalid {
			t.Fatalf("err=%v", err)
		}
		if err == nil && (got.Payload.MaxTokens != tc.want || got.Payload.Temperature != tc.temperature) {
			t.Fatalf("unexpected effective request: %+v", got.Payload)
		}
	}
}

func TestTemperatureValidationAndForwarding(t *testing.T) {
	for _, tc := range []struct {
		body    string
		want    float64
		invalid bool
	}{
		{`{"prompt":"hi"}`, 0, false},
		{`{"prompt":"hi","temperature":0}`, 0, false},
		{`{"prompt":"hi","temperature":0.7}`, .7, false},
		{`{"prompt":"hi","temperature":1e-300}`, 1e-300, false},
		{`{"prompt":"hi","temperature":1e300}`, 1e300, false},
		{`{"prompt":"hi","temperature":-0.1}`, 0, true},
		{`{"prompt":"hi","temperature":null}`, 0, true},
		{`{"prompt":"hi","temperature":"0.7"}`, 0, true},
		{`{"prompt":"hi","temperature":true}`, 0, true},
		{`{"prompt":"hi","temperature":1e309}`, 0, true},
		{`{"prompt":"hi","temperature":NaN}`, 0, true},
	} {
		pending, err := BuildPendingRequest(httptest.NewRequest("POST", "/generate", strings.NewReader(tc.body)))
		if (err != nil) != tc.invalid {
			t.Fatalf("body=%s error=%v", tc.body, err)
		}
		if err == nil && pending.Payload.Temperature != tc.want {
			t.Fatalf("temperature changed: %+v", pending.Payload)
		}
	}
}

func TestConfiguredOutputCap(t *testing.T) {
	for _, tc := range []struct{ requested, limit, want int }{{256, 256, 256}, {1024, 256, 256}, {32, 256, 32}} {
		body := fmt.Sprintf(`{"prompt":"hello","max_tokens":%d}`, tc.requested)
		pending, err := BuildPendingRequestWithLimit(httptest.NewRequest("POST", "/generate", strings.NewReader(body)), tc.limit)
		if err != nil || pending.Payload.MaxTokens != tc.want {
			t.Fatalf("configured output cap: %+v %v", pending, err)
		}
	}
}
