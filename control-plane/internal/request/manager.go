package request

import "github.com/yoonsung9948/relay/internal/config"

type RequestManager struct {
}

func (r *RequestManager) QueueRequest(request PendingRequest) error {
	// Implement the logic to queue a request here.
	return nil
}

func NewRequestManager(cfg config.RequestManagerConfig) *RequestManager {
	return &RequestManager{}
}
