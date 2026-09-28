package config

type ServeConfig struct {
	Host    string `yaml:"host"`
	Port    int    `yaml:"port"`
	DemoKey string `yaml:"demo_key"`
}

type ControlPlaneConfig struct {
	RequestManagerConfig RequestManagerConfig `yaml:"request_manager_config"`
}

type RequestManagerConfig struct {
}

type Config struct {
	ServeConfig        ServeConfig        `yaml:"serve_config"`
	ControlPlaneConfig ControlPlaneConfig `yaml:"control_plane_config"`
	EngineConfig       EngineConfig       `yaml:"engine_config"`
}

type EngineConfig struct {
	Endpoint string `yaml:"endpoint"`
}
