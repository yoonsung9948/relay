package api

import (
	"context"
	"net/http"
	"time"

	"github.com/yoonsung9948/relay/internal/engine"
)

type admissionInfo struct {
	InFlightGenerations         int     `json:"in_flight_generations"`
	MaxConcurrentGenerations    int     `json:"max_concurrent_generations"`
	GenerationRequestsPerSecond float64 `json:"generation_requests_per_second"`
	GenerationBurst             int     `json:"generation_burst"`
	RateLimitEnabled            bool    `json:"rate_limit_enabled"`
}

type deploymentInfo struct {
	Provider     string  `json:"provider"`
	RequestedGPU string  `json:"requested_gpu"`
	Region       *string `json:"region"`
}

type engineInfoResponse struct {
	Deployment   deploymentInfo        `json:"deployment"`
	Worker       engine.HealthResponse `json:"worker"`
	Admission    admissionInfo         `json:"admission"`
	ControlPlane servingInfo           `json:"control_plane"`
}

func (s *Server) EngineInfoHandler(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()
	w.Header().Set("Cache-Control", "no-store")
	worker, err := s.control.EngineInfo(ctx)
	if err != nil {
		// Upstream errors may contain private addresses. Return a stable public error.
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "engine_info_unavailable"})
		return
	}
	worker.Detail = ""
	// Public information never includes the worker's configuration path.
	if worker.Capabilities != nil {
		worker.Capabilities.ConfigPath = nil
	}
	provider := "unreported"
	configured := s.control.Cfg.GPUProviderConfig
	if configured.VastAI != nil {
		provider = "vastai"
	} else if configured.RunPod != nil {
		provider = "runpod"
	}
	admission := s.servingInfo.Admission
	admission.InFlightGenerations = len(s.generationCapacity)
	writeJSON(w, http.StatusOK, engineInfoResponse{
		Deployment:   deploymentInfo{Provider: provider, RequestedGPU: configured.InstanceConfig.GPU},
		Worker:       worker,
		Admission:    admission,
		ControlPlane: s.servingInfo,
	})
}
