package engine

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/yoonsung9948/relay/internal/request"
)

func TestCacheToggleReachesPythonAndMetadataReturns(t *testing.T) {
	for _, enabled := range []bool{false, true} {
		for _, stream := range []bool{false, true} {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var payload request.GenerateRequest
				if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
					t.Error(err)
					return
				}
				if payload.UseKVCache != enabled {
					t.Errorf("cache=%v want %v", payload.UseKVCache, enabled)
				}
				metadata := map[string]any{"use_kv_cache": enabled}
				if stream {
					json.NewEncoder(w).Encode(map[string]any{"type": "done", "generation": metadata})
				} else {
					json.NewEncoder(w).Encode(map[string]any{"text": "hello", "generation": metadata})
				}
			}))
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			client := NewHttpClient(server.URL)
			payload := request.GenerateRequest{Prompt: "hi", UseKVCache: enabled, Stream: stream}
			if stream {
				results, err := client.GenerateStream(ctx, payload)
				if err != nil {
					t.Fatal(err)
				}
				result := <-results
				if result.Err != nil || result.Chunk.Generation["use_kv_cache"] != enabled {
					t.Fatalf("bad result: %+v", result)
				}
				for range results {
				}
			} else {
				result, err := client.Generate(ctx, payload)
				if err != nil || result.Generation["use_kv_cache"] != enabled {
					t.Fatalf("bad result: %+v %v", result, err)
				}
			}
			cancel()
			server.Close()
		}
	}
}

func TestPublicRequestPreservesCacheToggle(t *testing.T) {
	for _, enabled := range []bool{false, true} {
		reader, writer := io.Pipe()
		go func() {
			defer writer.Close()
			json.NewEncoder(writer).Encode(map[string]any{"prompt": "hi", "use_kv_cache": enabled})
		}()
		req := httptest.NewRequest("POST", "/generate", reader)
		pending, err := request.BuildPendingRequest(req)
		reader.Close()
		if err != nil || pending.Payload.UseKVCache != enabled {
			t.Fatalf("request=%+v err=%v", pending, err)
		}
	}
}

func TestTemperatureReachesPythonAndMetadataReturns(t *testing.T) {
	for _, stream := range []bool{false, true} {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var payload request.GenerateRequest
			if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
				t.Error(err)
				return
			}
			if payload.Temperature != .7 {
				t.Errorf("temperature=%v", payload.Temperature)
			}
			metadata := map[string]any{"temperature": payload.Temperature, "sampling": "temperature"}
			if stream {
				json.NewEncoder(w).Encode(map[string]any{"type": "done", "generation": metadata})
			} else {
				json.NewEncoder(w).Encode(map[string]any{"text": "hello", "generation": metadata})
			}
		}))
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		client := NewHttpClient(server.URL)
		pending, err := request.BuildPendingRequest(httptest.NewRequest("POST", "/generate",
			strings.NewReader(`{"prompt":"hi","temperature":0.7}`)))
		if err != nil {
			t.Fatal(err)
		}
		if stream {
			results, err := client.GenerateStream(ctx, pending.Payload)
			if err != nil {
				t.Fatal(err)
			}
			result := <-results
			if result.Err != nil || result.Chunk.Generation["temperature"] != .7 || result.Chunk.Generation["sampling"] != "temperature" {
				t.Fatalf("bad result: %+v", result)
			}
			for range results {
			}
		} else {
			result, err := client.Generate(ctx, pending.Payload)
			if err != nil || result.Generation["temperature"] != .7 || result.Generation["sampling"] != "temperature" {
				t.Fatalf("bad result: %+v %v", result, err)
			}
		}
		cancel()
		server.Close()
	}
}
