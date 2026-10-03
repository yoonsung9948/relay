package config

import "time"

type ServeConfig struct {
	Host           string        `yaml:"host"`
	Port           int           `yaml:"port"`
	DemoKey        string        `yaml:"demo_key"`
	RequestTimeout time.Duration `yaml:"request_timeout"`
}

type ControlPlaneConfig struct {
	RequestManagerConfig RequestManagerConfig `yaml:"request_manager_config"`
	GPUProviderConfig    GPUProviderConfig    `yaml:"gpu_provider_config"`
}

type RequestManagerConfig struct {
}

type GPUProviderName string

const (
	GPUProviderRunPod GPUProviderName = "runpod"
	GPUProviderVastAI GPUProviderName = "vastai"
)

type GPUProviderConfig struct {
	Name           GPUProviderName `yaml:"name"`
	RunPod         *RunPodConfig   `yaml:"runpod,omitempty"`
	VastAI         *VastAIConfig   `yaml:"vastai,omitempty"`
	InstanceConfig InstanceConfig  `yaml:"instance_config"`
}

type InstanceConfig struct {
	GPU    string `yaml:"gpu"`
	DiskGB int    `yaml:"disk_gb"`
	Image  string `yaml:"image"`
}

type RunPodConfig struct {
	APIKey     string        `yaml:"api_key"`
	Image      string        `yaml:"image"`
	Timeout    time.Duration `yaml:"timeout"`
	BaseURL    string        `yaml:"base_url"`
	EnginePort int           `yaml:"engine_port"`
	HFToken    string        `yaml:"hf_token"`
}

type VastAIConfig struct {
	APIKey     string        `yaml:"api_key"`
	Image      string        `yaml:"image"`
	Timeout    time.Duration `yaml:"timeout"`
	BaseURL    string        `yaml:"base_url"`
	EnginePort int           `yaml:"engine_port"`
}

type Config struct {
	ServeConfig        ServeConfig        `yaml:"serve_config"`
	ControlPlaneConfig ControlPlaneConfig `yaml:"control_plane_config"`
	EngineConfig       EngineConfig       `yaml:"engine_config"`
}

type EngineConfig struct {
	Endpoint string `yaml:"endpoint"`
}
