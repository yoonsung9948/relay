package config

import (
	"errors"
	"fmt"
	"os"

	"go.yaml.in/yaml/v3"
)

const ConfigPath = "/etc/relay/config.yaml"

func LoadConfig() (*Config, error) {
	cfg := &Config{}

	file, err := os.Open(ConfigPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open config file: %w", err)
	}
	defer file.Close()

	if err := yaml.NewDecoder(file).Decode(cfg); err != nil {
		return nil, fmt.Errorf("failed to decode config: %w", err)
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
