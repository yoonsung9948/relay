package api

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"math"
	"net"
	"net/http"
	"strconv"
	"time"

	"github.com/yoonsung9948/relay/internal/config"
	"github.com/yoonsung9948/relay/internal/controlplane"
	"github.com/yoonsung9948/relay/internal/controlplane/gpuprovider"
	"github.com/yoonsung9948/relay/internal/engine"
	"github.com/yoonsung9948/relay/internal/observability"
	"github.com/yoonsung9948/relay/internal/ratelimit"
	"github.com/yoonsung9948/relay/internal/request"
	"github.com/yoonsung9948/relay/internal/response"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
)

func (s *Server) StartEngineHandler(w http.ResponseWriter, r *http.Request) {
	err := s.control.StartEngine(
		r.Context(),
		s.control.Cfg.GPUProviderConfig,
	)

	if err != nil {
		var cooldown *ratelimit.CooldownError
		var capacity *controlplane.NoCapacityError
		var conflict *controlplane.EngineStateConflictError
		switch {
		case errors.As(err, &conflict):
			writeRetryError(w, http.StatusConflict, "engine_state_conflict", time.Time{})
		case errors.As(err, &cooldown):
			writeRetryError(w, http.StatusTooManyRequests, "provision_cooldown", cooldown.RetryAt)
		case errors.As(err, &capacity):
			writeRetryError(w, http.StatusServiceUnavailable, "no_gpu_available", capacity.RetryAt)
		case errors.Is(err, gpuprovider.ErrNoCapacity):
			writeRetryError(w, http.StatusServiceUnavailable, "no_gpu_available", time.Time{})
		default:
			http.Error(w, "error starting engine", http.StatusInternalServerError)
		}
		return
	}

	writeJSON(w, http.StatusAccepted, map[string]string{
		"status": "starting",
	})
}

// Retry-After is rounded up so a client never retries before the deadline.
func writeRetryError(w http.ResponseWriter, status int, code string, retryAt time.Time) {
	if !retryAt.IsZero() {
		seconds := max(1, int(math.Ceil(time.Until(retryAt).Seconds())))
		w.Header().Set("Retry-After", strconv.Itoa(seconds))
	}
	w.Header().Set("X-Relay-Rejection", code)
	w.Header().Set("Cache-Control", "no-store")
	writeJSON(w, status, map[string]string{"error": code})
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

func writeJSON(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(data); err != nil {
		log.Printf("write response: %v", err)
	}
}

func writeEngineRejection(w http.ResponseWriter, r *http.Request, err error) bool {
	var rejection *engine.HTTPError
	if errors.As(err, &rejection) && (rejection.StatusCode == 422 || rejection.StatusCode == 503) {
		if rejection.RejectionReason == "engine_capacity" {
			observability.Rejected(r.Context(), "engine_capacity")
			w.Header().Set("X-Relay-Rejection", "engine_capacity")
		} else if rejection.StatusCode == 422 {
			observability.Rejected(r.Context(), "engine_invalid_request")
		}
		if rejection.RetryAfter != "" {
			w.Header().Set("Retry-After", rejection.RetryAfter)
		}
		http.Error(w, "engine rejected request", rejection.StatusCode)
		return true
	}
	return false
}

func (s *Server) GenerateHandler(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, 8<<10)
	defer r.Body.Close()
	serverSpan := trace.SpanFromContext(r.Context())
	outcome := "failed"
	defer func() {
		if r.Context().Err() != nil && outcome != "success" {
			outcome = "cancelled"
		}
		observability.SetOutcome(r.Context(), outcome)
		serverSpan.SetAttributes(attribute.String("request.outcome", outcome))
		if outcome != "success" {
			serverSpan.SetStatus(codes.Error, outcome)
		}
	}()
	_, parseSpan := otel.Tracer("relay.control-plane").Start(r.Context(), "request.parse")
	outputLimit := s.maxOutputTokens
	if outputLimit == 0 {
		outputLimit = config.DefaultMaxOutputTokens
	}
	pendingRequest, err := request.BuildPendingRequestWithLimit(r, outputLimit)
	if err != nil {
		parseSpan.SetStatus(codes.Error, "invalid_request")
	}
	parseSpan.End()
	if err != nil {
		var tooBig *http.MaxBytesError
		if errors.As(err, &tooBig) {
			http.Error(w, "request too large", http.StatusRequestEntityTooLarge)
			return
		}
		http.Error(w, "invalid request", http.StatusBadRequest)
		return
	}
	strategy := "greedy"
	if pendingRequest.Payload.Temperature > 0 {
		strategy = "temperature"
	}
	serverSpan.SetAttributes(
		attribute.Float64("sampling.temperature", pendingRequest.Payload.Temperature),
		attribute.String("sampling.strategy", strategy),
	)
	observability.SetExecution(r.Context(), pendingRequest.Payload.KVCacheMode, pendingRequest.Payload.DecodeBatching)
	if pendingRequest.Payload.Stream {
		stream, err := s.control.GenerateStream(r.Context(), pendingRequest)
		if err != nil {
			if writeEngineRejection(w, r, err) {
				return
			}
			http.Error(w, "generation failed", http.StatusServiceUnavailable)
			return
		}

		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("X-Accel-Buffering", "no")
		controller := http.NewResponseController(w)
		// Clear the global 60-second response deadline for this streaming handler.
		// Individual writes remain bounded; request context bounds generation time.
		_ = controller.SetWriteDeadline(time.Time{})
		_, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "streaming not supported", http.StatusInternalServerError)
			return
		}
		for {
			select {
			case <-r.Context().Done():
				return
			case result, ok := <-stream:
				if !ok {
					return
				}
				if result.Err != nil {
					log.Printf("generation failed: %v", result.Err)

					_ = controller.SetWriteDeadline(time.Now().Add(30 * time.Second))
					data, _ := json.Marshal(response.GenerateChunk{
						Type: "error", RequestID: result.Chunk.RequestID, Error: "Generation failed",
					})
					_, err := fmt.Fprintf(w, "event: error\ndata: %s\n\n", data)
					if err == nil {
						_ = controller.Flush()
					}
					return
				}
				if result.Chunk.RequestID != nil {
					serverSpan.SetAttributes(attribute.String("request.id", *result.Chunk.RequestID))
				}
				data, err := json.Marshal(result.Chunk)
				if err != nil {
					log.Printf("marshal chunk: %v", err)
					return
				}
				_ = controller.SetWriteDeadline(time.Now().Add(30 * time.Second))
				event := ""
				if result.Chunk.Type == "done" {
					event = "event: done\n"
				}
				_, err = fmt.Fprintf(w, "%sdata: %s\n\n", event, data)
				if err != nil {
					log.Printf("write SSE: %v", err)
					return
				}

				if err := controller.Flush(); err != nil {
					return
				}
				if result.Chunk.Type == "done" {
					outcome = "success"
					return
				}
			}
		}
	} else {
		resp, err := s.control.Generate(r.Context(), pendingRequest)
		if err != nil {
			if writeEngineRejection(w, r, err) {
				return
			}
			switch {
			case errors.Is(err, context.DeadlineExceeded):
				http.Error(w, "generation timed out", http.StatusGatewayTimeout)
			default:
				log.Printf("generate failed: %v", err)
				http.Error(w, "engine unavailable", http.StatusServiceUnavailable)
			}
			return
		}

		if resp.RequestID != nil {
			serverSpan.SetAttributes(attribute.String("request.id", *resp.RequestID))
		}
		data, err := json.Marshal(resp)
		if err != nil {
			http.Error(w, "response encoding failed", http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		if _, err := w.Write(data); err != nil {
			log.Printf("write response: %v", err)
		} else {
			outcome = "success"
		}
	}

}

func NewServer(
	cfg config.ServeConfig,
	control *controlplane.ControlPlane,
) *Server {
	if cfg.RequestTimeout <= 0 {
		cfg.RequestTimeout = 180 * time.Second
	}
	if cfg.DemoKey == "" {
		log.Fatal("serve_config.demo_key must be set")
	}

	if cfg.MaxConcurrentGenerations == 0 {
		cfg.MaxConcurrentGenerations = config.DefaultMaxConcurrentGenerations
	}
	if cfg.GenerationRequestsPerSecond == 0 {
		cfg.GenerationRequestsPerSecond = config.DefaultGenerationRequestsPerSecond
	}
	if cfg.GenerationBurst == 0 {
		cfg.GenerationBurst = config.DefaultGenerationBurst
	}
	if cfg.MaxOutputTokens == 0 {
		cfg.MaxOutputTokens = config.DefaultMaxOutputTokens
	}
	capacity := make(chan struct{}, cfg.MaxConcurrentGenerations)
	server := &Server{
		control:                  control,
		metrics:                  observability.NewMetrics(capacity),
		maxConcurrentGenerations: cfg.MaxConcurrentGenerations,
		maxOutputTokens:          cfg.MaxOutputTokens,
		servingInfo:              newServingInfo(cfg, control.Cfg.RateLimitEnabled),
		generationCapacity:       capacity,
	}
	requestLimiter := ratelimit.NewRequestLimiter(cfg.GenerationRequestsPerSecond, cfg.GenerationBurst)

	httpContext, cancelRequests := context.WithCancel(context.Background())
	server.cancelRequests = cancelRequests
	mux := http.NewServeMux()
	generate := withDemoKeyCheck(
		cfg.DemoKey,
		withConcurrencyLimit(
			capacity,
			withRequestRateLimit(requestLimiter, control.Cfg.RateLimitEnabled, withTimeout(cfg.RequestTimeout, server.GenerateHandler)),
		),
	)
	health := withTimeout(cfg.RequestTimeout, server.HealthHandler)
	startEngine := withDemoKeyCheck(
		cfg.DemoKey,
		withTimeout(cfg.RequestTimeout, server.StartEngineHandler),
	)

	mux.Handle("POST /generate", observability.Handler(server.metrics.Handler(generate)))
	mux.HandleFunc("POST /start_engine", startEngine)
	mux.HandleFunc("POST /stop_engine", withDemoKeyCheck(cfg.DemoKey, server.StopEngineHandler))
	mux.HandleFunc("GET /health", health)
	mux.HandleFunc("GET /engine/info", withDemoKeyCheck(cfg.DemoKey, server.EngineInfoHandler))
	mux.HandleFunc("GET /metrics", withDemoKeyCheck(cfg.DemoKey, server.metrics.ServeHTTP))

	server.httpServer = &http.Server{
		BaseContext:       func(net.Listener) context.Context { return httpContext },
		Addr:              fmt.Sprintf("%s:%d", cfg.Host, cfg.Port),
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       10 * time.Second,
		WriteTimeout:      cfg.RequestTimeout + 5*time.Second,
	}

	return server
}

type Server struct {
	cancelRequests           context.CancelFunc
	maxOutputTokens          int
	generationCapacity       chan struct{}
	servingInfo              servingInfo
	maxConcurrentGenerations int
	httpServer               *http.Server
	control                  *controlplane.ControlPlane
	metrics                  *observability.Metrics
}

func (s *Server) StopEngineHandler(w http.ResponseWriter, r *http.Request) {
	if err := s.control.StopEngine(r.Context(), "manual_stop"); err != nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "engine_cleanup_failed"})
		return
	}
	writeJSON(w, http.StatusOK, s.control.Status())
}

func (s *Server) ServeAndListen(ctx context.Context) error {
	listener, listenErr := net.Listen("tcp", s.httpServer.Addr)
	if listenErr != nil {
		cleanupCtx, cancel := context.WithTimeout(context.Background(), s.control.Cfg.ShutdownTimeout)
		defer cancel()
		return errors.Join(listenErr, s.control.Shutdown(cleanupCtx))
	}
	log.Printf("Server is running on %s", listener.Addr())
	serveErrors := make(chan error, 1)
	go func() { serveErrors <- s.httpServer.Serve(listener) }()
	var serveErr error
	select {
	case <-ctx.Done():
	case serveErr = <-serveErrors:
	}
	if s.cancelRequests != nil {
		s.cancelRequests()
	}
	shutdownCtx, cancel := context.WithTimeout(context.Background(), s.control.Cfg.ShutdownTimeout)
	defer cancel()
	controlDone := make(chan error, 1)
	go func() { controlDone <- s.control.Shutdown(shutdownCtx) }()
	drainCtx, drainCancel := context.WithTimeout(shutdownCtx, 5*time.Second)
	httpErr := s.httpServer.Shutdown(drainCtx)
	drainCancel()
	if httpErr != nil {
		httpErr = errors.Join(httpErr, s.httpServer.Close())
	}
	var cleanupErr error
	select {
	case cleanupErr = <-controlDone:
	case <-shutdownCtx.Done():
		cleanupErr = shutdownCtx.Err()
	}
	if errors.Is(serveErr, http.ErrServerClosed) {
		serveErr = nil
	}
	return errors.Join(serveErr, httpErr, cleanupErr)
}
