package config

import (
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
	return cfg, nil
}
