package api

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"runtime/debug"

	"github.com/yoonsung9948/relay/internal/config"
	"github.com/yoonsung9948/relay/internal/observability"
)

type servingInfo struct {
	MaxOutputTokens       int           `json:"max_output_tokens"`
	Revision              string        `json:"revision"`
	Dirty                 *bool         `json:"dirty"`
	BinarySHA256          string        `json:"binary_sha256"`
	ConfigSHA256          string        `json:"config_sha256"`
	Admission             admissionInfo `json:"admission"`
	TracingEnabled        bool          `json:"tracing_enabled"`
	Sampler               string        `json:"sampler"`
	RequestTimeoutSeconds float64       `json:"request_timeout_seconds"`
}

func newServingInfo(cfg config.ServeConfig, limited bool) servingInfo {
	info := servingInfo{MaxOutputTokens: cfg.MaxOutputTokens, Revision: "unreported", TracingEnabled: observability.Enabled(),
		Sampler: "disabled", RequestTimeoutSeconds: cfg.RequestTimeout.Seconds(),
		Admission: admissionInfo{MaxConcurrentGenerations: cfg.MaxConcurrentGenerations, GenerationRequestsPerSecond: cfg.GenerationRequestsPerSecond, GenerationBurst: cfg.GenerationBurst, RateLimitEnabled: limited}}
	if info.TracingEnabled {
		info.Sampler = os.Getenv("OTEL_TRACES_SAMPLER")
		if info.Sampler == "" {
			info.Sampler = "parentbased_always_on"
		}
	}
	if build, ok := debug.ReadBuildInfo(); ok {
		for _, item := range build.Settings {
			if item.Key == "vcs.revision" {
				info.Revision = item.Value
			}
			if item.Key == "vcs.modified" {
				value := item.Value == "true"
				info.Dirty = &value
			}
		}
	}
	if path, err := os.Executable(); err == nil {
		if data, err := os.ReadFile(path); err == nil {
			sum := sha256.Sum256(data)
			info.BinarySHA256 = hex.EncodeToString(sum[:])
		}
	}
	// Hash effective serving behavior, excluding the demo key and infrastructure addresses.
	data, _ := json.Marshal(struct {
		Admission       admissionInfo
		Timeout         float64
		MaxOutputTokens int
	}{info.Admission, cfg.RequestTimeout.Seconds(), cfg.MaxOutputTokens})
	sum := sha256.Sum256(data)
	info.ConfigSHA256 = hex.EncodeToString(sum[:])
	return info
}
