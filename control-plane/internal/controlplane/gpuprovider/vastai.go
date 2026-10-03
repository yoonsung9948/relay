package gpuprovider

import (
	"context"
	"net/http"

	"github.com/yoonsung9948/relay/internal/config"
)

func NewVastAIProvider(cfg *config.VastAIConfig) *vastAIProvider {
	return &vastAIProvider{
		apiKey: cfg.APIKey,
		image:  cfg.Image,
		client: &http.Client{
			Timeout: cfg.Timeout,
		},
	}
}

type vastAIProvider struct {
	apiKey string
	image  string
	client *http.Client
}

func (v *vastAIProvider) Create(ctx context.Context, spec CreateInstanceSpec) (*Instance, error) {
	return &Instance{}, nil
}

func (v *vastAIProvider) Get(ctx context.Context, id string) (*Instance, error) {
	return &Instance{}, nil
}

func (v *vastAIProvider) Stop(ctx context.Context, id string) error {
	return nil
}

func (v *vastAIProvider) Start(ctx context.Context, id string) error {
	return nil
}

func (v *vastAIProvider) Terminate(ctx context.Context, id string) error {
	return nil
}
