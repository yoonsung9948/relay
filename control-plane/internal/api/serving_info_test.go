package api

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/yoonsung9948/relay/internal/config"
)

func TestServingIdentityExcludesSecretsAndTracksEffectiveLimits(t *testing.T) {
	cfg := config.ServeConfig{DemoKey: "private-key", MaxConcurrentGenerations: 32, GenerationRequestsPerSecond: 100, GenerationBurst: 32, RequestTimeout: time.Minute}
	first := newServingInfo(cfg, true)
	cfg.DemoKey = "other-private-key"
	second := newServingInfo(cfg, true)
	if first.ConfigSHA256 != second.ConfigSHA256 || len(first.BinarySHA256) != 64 {
		t.Fatal("invalid public identity")
	}
	cfg.MaxConcurrentGenerations++
	if first.ConfigSHA256 == newServingInfo(cfg, true).ConfigSHA256 {
		t.Fatal("limit change not reflected")
	}
	data, err := json.Marshal(first)
	if err != nil || strings.Contains(string(data), "private-key") {
		t.Fatal("identity leaks configuration secrets")
	}
	if first.Admission.GenerationBurst != 32 || first.Admission.MaxConcurrentGenerations != 32 {
		t.Fatal("effective limits missing")
	}
}
