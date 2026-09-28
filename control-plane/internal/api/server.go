package api

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"

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
	pendingRequest, err := request.BuildPendingRequest(r)
	if err != nil {
		http.Error(w, fmt.Sprintf("Error building pending request: %v", err), http.StatusBadRequest)
		return
	}
	// if err := s.control.RequestManager.QueueRequest(pendingRequest); err != nil {
	// 	http.Error(w, fmt.Sprintf("Error queuing pending request: %v", err), http.StatusInternalServerError)
	// 	return
	// }

	// directly send the request to the engine for now
	resp, err := s.control.Generate(r.Context(), pendingRequest)
	if err != nil {
		http.Error(w, fmt.Sprintf("Error generating request: %v", err), http.StatusInternalServerError)
		return
	}
	log.Printf("Received pending request: %+v", pendingRequest)

	data, err := json.Marshal(resp)
	if err != nil {
		http.Error(w, "response encoding failed", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	if _, err := w.Write(data); err != nil {
		// Log the write failure; the response is already committed.
		// print the error for now
		fmt.Printf("Error writing response: %v\n", err)
	}
}

func NewServer(
	cfg config.ServeConfig,
	control *controlplane.ControlPlane,
) *Server {
	server := &Server{
		control: control,
	}

	mux := http.NewServeMux()
	mux.HandleFunc("POST /generate", server.GenerateHandler)
	mux.HandleFunc("POST /start_engine", server.StartEngineHandler)

	server.httpServer = &http.Server{
		Addr:    fmt.Sprintf("%s:%d", cfg.Host, cfg.Port),
		Handler: mux,
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
