package api

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net"
	"net/http"
	"time"

	"github.com/yoonsung9948/relay/internal/config"
	"github.com/yoonsung9948/relay/internal/controlplane"
	"github.com/yoonsung9948/relay/internal/controlplane/gpuprovider"
	"github.com/yoonsung9948/relay/internal/request"
)

func (s *Server) StartEngineHandler(w http.ResponseWriter, r *http.Request) {
	err := s.control.StartEngine(
		r.Context(),
		s.control.Cfg.GPUProviderConfig,
	)

	if err != nil {
		if errors.Is(err, gpuprovider.ErrNoCapacity) {
			writeJSON(w, http.StatusServiceUnavailable, map[string]string{
				"error": "no_gpu_available",
			})
			return
		}

		http.Error(
			w,
			"error starting engine",
			http.StatusInternalServerError,
		)
		return
	}

	writeJSON(w, http.StatusAccepted, map[string]string{
		"status": "starting",
	})
}
func (s *Server) HealthHandler(w http.ResponseWriter, r *http.Request) {
	status := s.control.Status()
	data, err := json.Marshal(status)
	if err != nil {
		http.Error(w, "response encoding failed", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	if _, err := w.Write(data); err != nil {
		log.Printf("write response: %v", err)
	}
}

func clientIP(r *http.Request) string {
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		return r.RemoteAddr
	}
	return host
}

func writeJSON(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(data); err != nil {
		log.Printf("write response: %v", err)
	}
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
			withTimeout(cfg.RequestTimeout, server.GenerateHandler),
		),
	)
	health := withTimeout(cfg.RequestTimeout, server.HealthHandler)
	startEngine := withDemoKeyCheck(
		cfg.DemoKey,
		withConcurrencyLimit(
			make(chan struct{}, 2),
			withTimeout(cfg.RequestTimeout, server.StartEngineHandler),
		),
	)

	mux.HandleFunc("POST /generate", generate)
	mux.HandleFunc("POST /start_engine", startEngine)
	mux.HandleFunc("GET /health", health)

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
