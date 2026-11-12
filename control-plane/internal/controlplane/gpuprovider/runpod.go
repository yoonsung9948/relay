package gpuprovider

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"time"

	"github.com/yoonsung9948/relay/internal/config"
)

func NewRunPodProvider(cfg *config.RunPodConfig) *RunPodProvider {
	if cfg.Timeout <= 0 {
		copy := *cfg
		copy.Timeout = 30 * time.Second
		cfg = &copy
	}
	return &RunPodProvider{
		apiKey:  cfg.APIKey,
		image:   cfg.Image,
		baseURL: strings.TrimRight(cfg.BaseURL, "/"),
		client: &http.Client{
			Timeout: cfg.Timeout,
		},
		enginePort: cfg.EnginePort,
		hfToken:    cfg.HFToken,
		engineEnv:  cfg.EngineEnv,
	}
}

type RunPodProvider struct {
	apiKey     string
	client     *http.Client
	image      string
	baseURL    string
	enginePort int
	hfToken    string
	engineEnv  map[string]string
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
	Name   string `json:"name"`
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
	env := make(map[string]string, len(r.engineEnv)+1)
	for key, value := range r.engineEnv {
		env[key] = value
	}
	env["HF_TOKEN"] = r.hfToken
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
		Env:   env,
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
	if err != nil {
		return nil, fmt.Errorf("runpod create pod: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))

		var apiErr runPodErrorResponse
		if err := json.Unmarshal(body, &apiErr); err == nil {
			detail := strings.ToLower(apiErr.Detail)

			if (resp.StatusCode == http.StatusBadRequest || resp.StatusCode == http.StatusServiceUnavailable) && strings.Contains(
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

		return nil, &HTTPError{StatusCode: resp.StatusCode, Operation: "create"}
	}

	if resp.StatusCode != http.StatusCreated {
		return nil, fmt.Errorf("unexpected create response HTTP %d; reconcile attempt", resp.StatusCode)
	}

	var pod runPodPod
	if err := json.NewDecoder(resp.Body).Decode(&pod); err != nil {
		return nil, fmt.Errorf(
			"decode runpod create response: %w",
			err,
		)
	}

	if !validPodID.MatchString(pod.ID) {
		return nil, errors.New("RunPod response missing valid pod id; reconcile attempt")
	}
	endpoint := runPodEndpoint(pod.ID, r.enginePort)

	return &Instance{
		Name:     pod.Name,
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

	if !validPodID.MatchString(id) {
		return nil, errors.New("invalid pod id")
	}
	req, err := http.NewRequestWithContext(
		ctx, http.MethodGet, r.baseURL+"/pods/"+url.PathEscape(id),
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

	if resp.StatusCode == http.StatusNotFound {
		return nil, ErrNotFound
	}
	if resp.StatusCode != http.StatusOK {
		return nil, &HTTPError{StatusCode: resp.StatusCode, Operation: "get"}
	}

	var pod runPodPod
	if err := json.NewDecoder(resp.Body).Decode(&pod); err != nil {
		return nil, fmt.Errorf(
			"decode runpod get response: %w",
			err,
		)
	}

	if !validPodID.MatchString(pod.ID) {
		return nil, errors.New("RunPod response missing valid pod id; reconcile attempt")
	}
	endpoint := runPodEndpoint(pod.ID, r.enginePort)

	return &Instance{
		Name:     pod.Name,
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
	if !validPodID.MatchString(id) {
		return errors.New("invalid pod id")
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodDelete, r.baseURL+"/pods/"+url.PathEscape(id), nil)
	if err != nil {
		return err
	}
	withAPIKey(req, r.apiKey)
	resp, err := r.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusNotFound {
		return ErrNotFound
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return &HTTPError{StatusCode: resp.StatusCode, Operation: "terminate"}
	}
	return nil
}

func (r *RunPodProvider) action(
	ctx context.Context,
	id string,
	action string,
) error {
	if !validPodID.MatchString(id) {
		return errors.New("invalid pod id")
	}
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
		r.baseURL+"/pods/"+url.PathEscape(id)+"/action",
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

var validPodID = regexp.MustCompile(`^[a-zA-Z0-9_-]{1,128}$`)

// A fingerprint binds recovery to the same credentials and API endpoint, without
// storing the API key. Key rotation intentionally requires manual reconciliation.
func (r *RunPodProvider) OwnershipScope() string {
	sum := sha256.Sum256([]byte(r.baseURL + "\x00" + r.apiKey))
	return hex.EncodeToString(sum[:])
}

// FindByName scans every page but returns only the exact durable attempt name.
func (r *RunPodProvider) FindByName(ctx context.Context, name string) ([]*Instance, error) {
	if name == "" {
		return nil, errors.New("empty ownership name")
	}
	var matches []*Instance
	cursor := ""
	seen := map[string]bool{}
	for {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		page, next, err := r.listPage(ctx, cursor)
		if err != nil {
			return nil, err
		}
		for _, pod := range page {
			if pod.Name != name {
				continue
			}
			if !validPodID.MatchString(pod.ID) {
				return nil, errors.New("invalid pod id in ownership lookup")
			}
			matches = append(matches, &Instance{ID: pod.ID, Name: pod.Name, Status: pod.Status, Running: pod.Status == "RUNNING", Endpoint: runPodEndpoint(pod.ID, r.enginePort)})
		}
		if next == "" {
			return matches, nil
		}
		if seen[next] {
			return nil, errors.New("RunPod returned a repeated pagination cursor")
		}
		seen[next] = true
		cursor = next
	}
}
func (r *RunPodProvider) listPage(ctx context.Context, cursor string) ([]runPodPod, string, error) {
	query := url.Values{"limit": {"1000"}}
	if cursor != "" {
		query.Set("cursor", cursor)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, r.baseURL+"/pods?"+query.Encode(), nil)
	if err != nil {
		return nil, "", err
	}
	withAPIKey(req, r.apiKey)
	resp, err := r.client.Do(req)
	if err != nil {
		return nil, "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, "", &HTTPError{StatusCode: resp.StatusCode, Operation: "list"}
	}
	var body struct {
		Pods       []runPodPod `json:"pods"`
		Pagination *struct {
			NextCursor  *string `json:"nextCursor"`
			HasNextPage *bool   `json:"hasNextPage"`
		} `json:"pagination"`
	}
	if err = json.NewDecoder(io.LimitReader(resp.Body, 16<<20)).Decode(&body); err != nil {
		return nil, "", err
	}
	if body.Pagination == nil || body.Pagination.HasNextPage == nil {
		return nil, "", errors.New("RunPod list response missing pagination")
	}
	if !*body.Pagination.HasNextPage {
		return body.Pods, "", nil
	}
	if body.Pagination.NextCursor == nil || *body.Pagination.NextCursor == "" {
		return nil, "", errors.New("RunPod list response missing next cursor")
	}
	return body.Pods, *body.Pagination.NextCursor, nil
}
