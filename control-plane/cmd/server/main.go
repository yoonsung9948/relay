package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/yoonsung9948/relay/internal/api"
	"github.com/yoonsung9948/relay/internal/config"
	"github.com/yoonsung9948/relay/internal/controlplane"
	"github.com/yoonsung9948/relay/internal/engine"
	"github.com/yoonsung9948/relay/internal/observability"
)

func configPath(args []string, output io.Writer) (string, error) {
	flags := flag.NewFlagSet("server", flag.ContinueOnError)
	flags.SetOutput(output)
	path := flags.String("config", config.ConfigPath, "Path to the local control-plane YAML")
	if err := flags.Parse(args); err != nil {
		return "", err
	}
	if flags.NArg() != 0 {
		return "", fmt.Errorf("unexpected arguments: %v", flags.Args())
	}
	return *path, nil
}

func main() {
	path, err := configPath(os.Args[1:], os.Stderr)
	if err != nil {
		if err == flag.ErrHelp {
			return
		}
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	cfg, err := config.LoadConfigFile(path)
	if err != nil {
		panic(err)
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	shutdownTracing, err := observability.Init(ctx)
	if err != nil {
		panic(err)
	}
	defer func() {
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = shutdownTracing(shutdownCtx)
	}()
	client := engine.NewHttpClient(cfg.EngineConfig.Endpoint)
	control := controlplane.NewControlPlane(ctx, cfg.ControlPlaneConfig, client)
	defer control.Close()
	recoveryCtx, recoveryCancel := context.WithTimeout(ctx, cfg.ControlPlaneConfig.CleanupTimeout)
	if err := control.Recover(recoveryCtx); err != nil {
		log.Printf("startup recovery blocked; new provisioning disabled: %v", err)
	}
	recoveryCancel()
	server := api.NewServer(cfg.ServeConfig, control)
	if err := server.ServeAndListen(ctx); err != nil {
		panic(err)
	}
}
