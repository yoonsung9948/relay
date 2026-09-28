package main

import (
	"context"
	"os"
	"os/signal"
	"syscall"

	"github.com/yoonsung9948/relay/internal/api"
	"github.com/yoonsung9948/relay/internal/config"
	"github.com/yoonsung9948/relay/internal/controlplane"
	"github.com/yoonsung9948/relay/internal/engine"
)

func main() {
	cfg, err := config.LoadConfig()
	if err != nil {
		panic(err)
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	client := engine.NewHttpClient(cfg.EngineConfig.Endpoint)
	control := controlplane.NewControlPlane(ctx, cfg.ControlPlaneConfig, client)
	server := api.NewServer(cfg.ServeConfig, control)
	if err := server.ServeAndListen(ctx); err != nil {
		panic(err)
	}
}
