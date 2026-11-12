package config

import "time"

const (
	DefaultMaxConcurrentGenerations    = 2
	DefaultGenerationRequestsPerSecond = 1.0
	DefaultGenerationBurst             = 4
	DefaultMaxOutputTokens             = 64
	DefaultProvisionTimeout            = 15 * time.Minute
	DefaultCleanupTimeout              = time.Minute
	DefaultShutdownTimeout             = 80 * time.Second
	DefaultIdleTimeout                 = 5 * time.Minute
	DefaultMaxSessionDuration          = 30 * time.Minute
)

type ServeConfig struct {
	MaxOutputTokens             int           `yaml:"max_output_tokens"`
	MaxConcurrentGenerations    int           `yaml:"max_concurrent_generations"`
	GenerationRequestsPerSecond float64       `yaml:"generation_requests_per_second"`
	GenerationBurst             int           `yaml:"generation_burst"`
	Host                        string        `yaml:"host"`
	Port                        int           `yaml:"port"`
	DemoKey                     string        `yaml:"demo_key"`
	RequestTimeout              time.Duration `yaml:"request_timeout"`
}

type ControlPlaneConfig struct {
	StateDirectory        string               `yaml:"state_directory"`
	CleanupTimeout        time.Duration        `yaml:"cleanup_timeout"`
	ShutdownTimeout       time.Duration        `yaml:"shutdown_timeout"`
	RequestManagerConfig  RequestManagerConfig `yaml:"request_manager_config"`
	GPUProviderConfig     GPUProviderConfig    `yaml:"gpu_provider_config"`
	RateLimitEnabled      bool                 `yaml:"-"`
	IdleTimeout           time.Duration        `yaml:"idle_timeout"`
	ProvisionRetryBackoff time.Duration        `yaml:"provision_retry_backoff"`
	ProvisionCooldown     time.Duration        `yaml:"provision_cooldown"`
	ProvisionTimeout      time.Duration        `yaml:"provision_timeout"`
	MaxSessionDuration    time.Duration        `yaml:"max_session_duration"`
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
	APIKey     string            `yaml:"api_key"`
	Image      string            `yaml:"image"`
	Timeout    time.Duration     `yaml:"timeout"`
	BaseURL    string            `yaml:"base_url"`
	EnginePort int               `yaml:"engine_port"`
	HFToken    string            `yaml:"hf_token"`
	EngineEnv  map[string]string `yaml:"engine_env,omitempty"`
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
