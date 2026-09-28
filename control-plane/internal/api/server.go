package api

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/yoonsung9948/relay/internal/config"
	"github.com/yoonsung9948/relay/internal/controlplane"
	"github.com/yoonsung9948/relay/internal/request"
)

func (s *Server) StartEngineHandler(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintf(w, "Starting engine...")
	if err := s.control.StartEngine(r.Context()); err != nil {
		http.Error(w, fmt.Sprintf("Error starting engine: %v", err), http.StatusInternalServerError)
		return
	}
	fmt.Fprintf(w, "Engine started successfully.")
}
func (s *Server) GenerateHandler(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, 8<<10)

	pendingRequest, err := request.BuildPendingRequest(r)
	if err != nil {
		var tooBig *http.MaxBytesError
		if errors.As(err, &tooBig) {
			http.Error(w, "request too large", http.StatusRequestEntityTooLarge)
			return
		}
		http.Error(w, "invalid request", http.StatusBadRequest)
		return
	}

	resp, err := s.control.Generate(r.Context(), pendingRequest)
	if err != nil {
		switch {
		case errors.Is(err, context.DeadlineExceeded):
			http.Error(w, "generation timed out", http.StatusGatewayTimeout)
		default:
			log.Printf("generate failed: %v", err)
			http.Error(w, "engine unavailable", http.StatusServiceUnavailable)
		}
		return
	}

	data, err := json.Marshal(resp)
	if err != nil {
		http.Error(w, "response encoding failed", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	if _, err := w.Write(data); err != nil {
		log.Printf("write response: %v", err)
	}
}

func NewServer(
	cfg config.ServeConfig,
	control *controlplane.ControlPlane,
) *Server {
	if cfg.DemoKey == "" {
		log.Fatal("serve_config.demo_key must be set")
	}

	server := &Server{
		control: control,
	}

	mux := http.NewServeMux()
	generate := withDemoKeyCheck(
		cfg.DemoKey,
		withConcurrencyLimit(
			make(chan struct{}, 2),
			withTimeout(45*time.Second, server.GenerateHandler),
		),
	)
	mux.HandleFunc("POST /generate", generate)
	// mux.HandleFunc("POST /start_engine", server.StartEngineHandler)

	server.httpServer = &http.Server{
		Addr:              fmt.Sprintf("%s:%d", cfg.Host, cfg.Port),
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       10 * time.Second,
		WriteTimeout:      60 * time.Second,
	}

	return server
}

type Server struct {
	httpServer *http.Server
	control    *controlplane.ControlPlane
}

func (s *Server) ServeAndListen(ctx context.Context) error {
	fmt.Printf("Server is running on %s\n", s.httpServer.Addr)
	go func() {
		if err := s.httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			fmt.Printf("HTTP server ListenAndServe: %v\n", err)
		}
	}()

	<-ctx.Done()
	fmt.Println("\nShutting down HTTP server...")
	if err := s.httpServer.Shutdown(context.Background()); err != nil {
		fmt.Printf("HTTP server Shutdown: %v\n", err)
		return err
	}
	if err := s.control.Shutdown(context.Background()); err != nil {
		fmt.Printf("Control plane Shutdown: %v\n", err)
		return err
	}
	fmt.Println("HTTP server shut down successfully.")
	return nil
}
