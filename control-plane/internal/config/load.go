package config

import (
	"errors"
	"fmt"
	"math"
	"os"
	"strconv"
	"time"

	"go.yaml.in/yaml/v3"
)

const ConfigPath = "/etc/relay/config.yaml"

// const ConfigPath = "examples/config.yaml"

func LoadConfig() (*Config, error) {
	return LoadConfigFile(ConfigPath)
}

func LoadConfigFile(path string) (*Config, error) {
	stateDirectory := os.Getenv("STATE_DIRECTORY")
	if stateDirectory == "" {
		stateDirectory = "state"
	}
	cfg := &Config{ServeConfig: ServeConfig{
		MaxConcurrentGenerations:    DefaultMaxConcurrentGenerations,
		MaxOutputTokens:             DefaultMaxOutputTokens,
		GenerationRequestsPerSecond: DefaultGenerationRequestsPerSecond,
		GenerationBurst:             DefaultGenerationBurst,
	}, ControlPlaneConfig: ControlPlaneConfig{
		StateDirectory:        stateDirectory,
		ProvisionTimeout:      DefaultProvisionTimeout,
		CleanupTimeout:        DefaultCleanupTimeout,
		ShutdownTimeout:       DefaultShutdownTimeout,
		IdleTimeout:           DefaultIdleTimeout,
		MaxSessionDuration:    DefaultMaxSessionDuration,
		ProvisionCooldown:     time.Hour,
		ProvisionRetryBackoff: 15 * time.Second,
	}}

	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open config file: %w", err)
	}
	defer file.Close()

	if err := yaml.NewDecoder(file).Decode(cfg); err != nil {
		return nil, fmt.Errorf("failed to decode config: %w", err)
	}

	lifecycle := cfg.ControlPlaneConfig
	switch {
	case lifecycle.StateDirectory == "":
		return nil, errors.New("state_directory must not be empty")
	case lifecycle.ProvisionTimeout <= 0 || lifecycle.CleanupTimeout <= 0:
		return nil, errors.New("provision_timeout and cleanup_timeout must be positive")
	case lifecycle.ShutdownTimeout < lifecycle.CleanupTimeout:
		return nil, errors.New("shutdown_timeout must cover cleanup_timeout")
	case lifecycle.IdleTimeout < 0 || lifecycle.MaxSessionDuration < 0:
		return nil, errors.New("idle_timeout and max_session_duration must be nonnegative; zero disables them")
	}
	if cfg.ServeConfig.MaxOutputTokens < 1 {
		return nil, errors.New("max_output_tokens must be positive")
	}
	if cfg.ServeConfig.MaxConcurrentGenerations < 1 || cfg.ServeConfig.GenerationBurst < 1 || cfg.ServeConfig.GenerationRequestsPerSecond <= 0 || math.IsNaN(cfg.ServeConfig.GenerationRequestsPerSecond) || math.IsInf(cfg.ServeConfig.GenerationRequestsPerSecond, 0) {
		return nil, errors.New("generation concurrency, request rate, and burst must be positive and finite")
	}
	if cfg.ControlPlaneConfig.ProvisionCooldown <= 0 || cfg.ControlPlaneConfig.ProvisionRetryBackoff <= 0 {
		return nil, errors.New("provision_cooldown and provision_retry_backoff must be positive")
	}
	cfg.ControlPlaneConfig.RateLimitEnabled = true
	limited := os.Getenv("RATE_LIMITED")
	if limited != "" {
		if limitedBool, err := strconv.ParseBool(limited); err == nil {
			cfg.ControlPlaneConfig.RateLimitEnabled = limitedBool
		}
	}

	if rp := cfg.ControlPlaneConfig.GPUProviderConfig.RunPod; rp != nil {
		if key := os.Getenv("RUNPOD_API_KEY"); key != "" {
			rp.APIKey = key
		}
		if hf := os.Getenv("HF_TOKEN"); hf != "" {
			rp.HFToken = hf
		}
		if rp.APIKey == "" {
			return nil, errors.New("runpod api key is empty: set RUNPOD_API_KEY")
		}
	}

	return cfg, nil
}
