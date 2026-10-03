package gpuprovider

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/yoonsung9948/relay/internal/config"
)

func NewRunPodProvider(cfg *config.RunPodConfig) *RunPodProvider {
	return &RunPodProvider{
		apiKey:  cfg.APIKey,
		image:   cfg.Image,
		baseURL: strings.TrimRight(cfg.BaseURL, "/"),
		client: &http.Client{
			Timeout: cfg.Timeout,
		},
		enginePort: cfg.EnginePort,
		hfToken:    cfg.HFToken,
	}
}

type RunPodProvider struct {
	apiKey     string
	client     *http.Client
	image      string
	baseURL    string
	enginePort int
	hfToken    string
}

type runPodCreateRequest struct {
	Name  string            `json:"name"`
	Image string            `json:"image"`
	GPU   runPodGPU         `json:"gpu"`
	Disk  int               `json:"disk"`
	Ports []string          `json:"ports"`
	Cloud string            `json:"cloud"`
	Env   map[string]string `json:"env,omitempty"`
}

type runPodGPU struct {
	ID    string `json:"id"`
	Count int    `json:"count"`
}

type runPodPod struct {
	ID     string `json:"id"`
	Status string `json:"status"`
}

type runPodErrorResponse struct {
	Detail string `json:"detail"`
	Status int    `json:"status"`
	Title  string `json:"title"`
}

var ErrNoCapacity = errors.New("no gpu capacity")

func runPodEndpoint(id string, port int) string {
	return fmt.Sprintf(
		"https://%s-%d.proxy.runpod.net",
		id,
		port,
	)
}

// runpod api reference
// - `POST /v2/pods` — create/provision your GPU worker.
// - `GET /v2/pods/{id}` — poll lifecycle state, cost, runtime/ports, GPU, datacenter.
// - `POST /v2/pods/{id}/action` with `{"action":"start"|"stop"|"restart"|"terminate"}` — control lifecycle.
// - `DELETE /v2/pods/{id}` — terminate permanently.
// - `GET /v2/pods/{id}/logs` — stream pod logs; very useful while your engine is booting.
// - `GET /v2/catalog/gpus` / catalog GPU-type endpoints — discover GPU types.
// - `GET /v2/catalog/datacenters` — especially useful because it can report GPU availability by datacenter.
// - Later: network-volume endpoints if you decide to persist weights/cache between machines.
func (r *RunPodProvider) Create(
	ctx context.Context,
	spec CreateInstanceSpec,
) (*Instance, error) {
	payload := runPodCreateRequest{
		Name:  spec.Name,
		Image: r.image,
		GPU: runPodGPU{
			ID:    spec.GPU,
			Count: 1,
		},
		Disk: spec.DiskGB,
		Ports: []string{
			fmt.Sprintf("%d/http", r.enginePort),
		},
		Cloud: "SECURE",
		Env: map[string]string{
			"HF_TOKEN": r.hfToken,
		},
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("marshal runpod create request: %w", err)
	}

	req, err := http.NewRequestWithContext(
		ctx,
		http.MethodPost,
		r.baseURL+"/pods",
		bytes.NewReader(body),
	)
	if err != nil {
		return nil, fmt.Errorf("create runpod request: %w", err)
	}

	withAPIKey(req, r.apiKey)

	resp, err := r.client.Do(req)
	if err != nil || resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(resp.Body)

		var apiErr runPodErrorResponse
		if err := json.Unmarshal(body, &apiErr); err == nil {
			detail := strings.ToLower(apiErr.Detail)

			if strings.Contains(
				detail,
				"no longer any instances available",
			) {
				return nil, fmt.Errorf(
					"%w: %s",
					ErrNoCapacity,
					apiErr.Detail,
				)
			}
		}

		return nil, fmt.Errorf(
			"runpod create pod: status=%d body=%s",
			resp.StatusCode,
			body,
		)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusCreated {
		body, _ := io.ReadAll(resp.Body)

		return nil, fmt.Errorf(
			"runpod create pod: status=%d body=%s",
			resp.StatusCode,
			body,
		)
	}

	var pod runPodPod
	if err := json.NewDecoder(resp.Body).Decode(&pod); err != nil {
		return nil, fmt.Errorf(
			"decode runpod create response: %w",
			err,
		)
	}

	endpoint := runPodEndpoint(pod.ID, r.enginePort)

	return &Instance{
		ID:       pod.ID,
		Running:  pod.Status == "RUNNING",
		Status:   pod.Status,
		Endpoint: endpoint,
	}, nil
}

func withAPIKey(req *http.Request, apiKey string) {
	req.Header.Set(
		"Authorization",
		"Bearer "+apiKey,
	)
	req.Header.Set(
		"Content-Type",
		"application/json",
	)
}

func (r *RunPodProvider) Get(ctx context.Context, id string) (*Instance, error) {

	req, err := http.NewRequestWithContext(
		ctx,
		http.MethodGet,
		r.baseURL+"/pods/"+id,
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("get runpod request: %w", err)
	}

	withAPIKey(req, r.apiKey)

	resp, err := r.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("runpod get pod: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)

		return nil, fmt.Errorf(
			"runpod get pod: status=%d body=%s",
			resp.StatusCode,
			body,
		)
	}

	var pod runPodPod
	if err := json.NewDecoder(resp.Body).Decode(&pod); err != nil {
		return nil, fmt.Errorf(
			"decode runpod get response: %w",
			err,
		)
	}

	endpoint := runPodEndpoint(pod.ID, r.enginePort)

	return &Instance{
		ID:       pod.ID,
		Running:  pod.Status == "RUNNING",
		Status:   pod.Status,
		Endpoint: endpoint,
	}, nil
}

type runPodActionRequest struct {
	Action string `json:"action"`
}

func (r *RunPodProvider) Stop(ctx context.Context, id string) error {
	return r.action(ctx, id, "stop")
}

func (r *RunPodProvider) Start(ctx context.Context, id string) error {
	return r.action(ctx, id, "start")
}

func (r *RunPodProvider) Terminate(ctx context.Context, id string) error {
	return r.action(ctx, id, "terminate")
}

func (r *RunPodProvider) action(
	ctx context.Context,
	id string,
	action string,
) error {
	payload := runPodActionRequest{
		Action: action,
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("marshal runpod %s request: %w", action, err)
	}

	req, err := http.NewRequestWithContext(
		ctx,
		http.MethodPost,
		r.baseURL+"/pods/"+id+"/action",
		bytes.NewReader(body),
	)
	if err != nil {
		return fmt.Errorf("create runpod %s request: %w", action, err)
	}

	withAPIKey(req, r.apiKey)

	resp, err := r.client.Do(req)
	if err != nil {
		return fmt.Errorf("runpod %s pod: %w", action, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound {
		return fmt.Errorf("%w: pod %s", ErrNotFound, id)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(resp.Body)

		return fmt.Errorf(
			"runpod %s pod: status=%d body=%s",
			action,
			resp.StatusCode,
			body,
		)
	}

	return nil
}
