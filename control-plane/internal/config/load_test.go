package config

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestRateLimitEnvironmentDefaults(t *testing.T) {
	for _, tc := range []struct {
		value   string
		enabled bool
	}{{"", true}, {"invalid", true}, {"false", false}, {"true", true}} {
		t.Run(tc.value, func(t *testing.T) {
			dir := t.TempDir()
			t.Chdir(dir)
			if err := os.MkdirAll(filepath.Join(dir, "examples"), 0700); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(dir, "examples/config.yaml"), []byte("control_plane_config:\n  rate_limit_enabled: false\n"), 0600); err != nil {
				t.Fatal(err)
			}
			t.Setenv("RATE_LIMITED", tc.value)
			cfg, err := LoadConfigFile("examples/config.yaml")
			if err != nil {
				t.Fatal(err)
			}
			if cfg.ControlPlaneConfig.RateLimitEnabled != tc.enabled {
				t.Fatal("environment policy not respected")
			}
			if cfg.ControlPlaneConfig.ProvisionCooldown != time.Hour || cfg.ControlPlaneConfig.ProvisionRetryBackoff != 15*time.Second {
				t.Fatal("missing default admission durations")
			}
		})
	}
}

func TestGenerationLimitConfiguration(t *testing.T) {
	for _, tc := range []struct {
		name, settings string
		invalid        bool
	}{
		{"defaults", "", false},
		{"custom", "  max_concurrent_generations: 3\n  generation_requests_per_second: 0.5\n  generation_burst: 2\n", false},
		{"zero concurrency", "  max_concurrent_generations: 0\n", true},
		{"negative burst", "  generation_burst: -1\n", true},
		{"invalid rate", "  generation_requests_per_second: .nan\n", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			t.Chdir(dir)
			t.Setenv("RATE_LIMITED", "false")
			if err := os.Mkdir("examples", 0700); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile("examples/config.yaml", []byte("serve_config:\n"+tc.settings), 0600); err != nil {
				t.Fatal(err)
			}
			cfg, err := LoadConfigFile("examples/config.yaml")
			if tc.invalid {
				if err == nil {
					t.Fatal("invalid limits accepted")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if tc.name == "custom" && (cfg.ServeConfig.MaxConcurrentGenerations != 3 || cfg.ServeConfig.GenerationRequestsPerSecond != 0.5 || cfg.ServeConfig.GenerationBurst != 2) {
				t.Fatal("custom limits not loaded")
			}
			if tc.name == "defaults" && (cfg.ServeConfig.MaxConcurrentGenerations != 2 || cfg.ServeConfig.GenerationRequestsPerSecond != 1 || cfg.ServeConfig.GenerationBurst != 4) {
				t.Fatal("defaults not loaded")
			}
		})
	}
}

func TestExplicitConfigFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "control.yaml")
	if err := os.WriteFile(path, []byte("serve_config:\n  port: 8123\n"), 0600); err != nil {
		t.Fatal(err)
	}
	cfg, err := LoadConfigFile(path)
	if err != nil || cfg.ServeConfig.Port != 8123 {
		t.Fatalf("cfg=%v err=%v", cfg, err)
	}
}

func TestLifecycleConfiguration(t *testing.T) {
	for _, tc := range []struct {
		name, settings string
		invalid        bool
	}{
		{"defaults", "", false},
		{"disabled idle and session", "  idle_timeout: 0s\n  max_session_duration: 0s\n", false},
		{"zero startup", "  provision_timeout: 0s\n", true},
		{"negative idle", "  idle_timeout: -1s\n", true},
		{"shutdown shorter than cleanup", "  cleanup_timeout: 60s\n  shutdown_timeout: 20s\n", true},
		{"empty state directory", "  state_directory: ''\n", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "config.yaml")
			t.Setenv("STATE_DIRECTORY", "/var/lib/relay")
			if err := os.WriteFile(path, []byte("control_plane_config:\n"+tc.settings), 0600); err != nil {
				t.Fatal(err)
			}
			cfg, err := LoadConfigFile(path)
			if (err != nil) != tc.invalid {
				t.Fatalf("config=%+v err=%v", cfg, err)
			}
			if tc.invalid {
				return
			}
			c := cfg.ControlPlaneConfig
			if c.StateDirectory != "/var/lib/relay" || c.CleanupTimeout != time.Minute || c.ShutdownTimeout != 80*time.Second || c.ProvisionTimeout != 15*time.Minute {
				t.Fatalf("defaults: %+v", c)
			}
			if tc.name == "defaults" && (c.IdleTimeout != 5*time.Minute || c.MaxSessionDuration != 30*time.Minute) {
				t.Fatalf("defaults: %+v", c)
			}
			if tc.name == "disabled idle and session" && (c.IdleTimeout != 0 || c.MaxSessionDuration != 0) {
				t.Fatal("explicit disable ignored")
			}
		})
	}
}
