package controlplane

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/yoonsung9948/relay/internal/request"
	"github.com/yoonsung9948/relay/internal/response"
)

func (c *ControlPlane) beginRequest(parent context.Context) (context.Context, func(), error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.state != StateReady || c.worker == nil || c.closed || c.stop != nil {
		return nil, nil, errors.New("engine not ready")
	}
	if err := parent.Err(); err != nil {
		return nil, nil, err
	}
	ctx, cancel := context.WithCancel(parent)
	c.nextRequest++
	id := c.nextRequest
	attempt := c.worker.AttemptID
	done := make(chan struct{})
	c.requests[id] = trackedRequest{cancel: cancel, done: done}
	c.armLeaseLocked()
	var once sync.Once
	finish := func() {
		once.Do(func() {
			cancel()
			c.mu.Lock()
			defer c.mu.Unlock()
			delete(c.requests, id)
			close(done)
			if c.worker != nil && c.worker.AttemptID == attempt {
				// Activity is only used in this process. Recovery always terminates old workers.
				c.worker.LastActivity = time.Now().UTC()
				c.armLeaseLocked()
			}
		})
	}
	return ctx, finish, nil
}
func (c *ControlPlane) Generate(ctx context.Context, pending request.PendingRequest) (response.GenerateResponse, error) {
	child, finish, err := c.beginRequest(ctx)
	if err != nil {
		return response.GenerateResponse{}, err
	}
	defer finish()
	result, err := c.Client.Generate(child, pending.Payload)
	if err != nil {
		return response.GenerateResponse{}, fmt.Errorf("engine generation failed: %w", err)
	}
	return result, nil
}
func (c *ControlPlane) GenerateStream(ctx context.Context, pending request.PendingRequest) (<-chan response.StreamResponse, error) {
	child, finish, err := c.beginRequest(ctx)
	if err != nil {
		return nil, err
	}
	source, err := c.Client.GenerateStream(child, pending.Payload)
	if err != nil {
		finish()
		return nil, fmt.Errorf("generate stream failed: %w", err)
	}
	output := make(chan response.StreamResponse)
	go func() {
		defer close(output)
		defer finish()
		for {
			select {
			case <-child.Done():
				return
			case result, ok := <-source:
				if !ok {
					return
				}
				select {
				case output <- result:
				case <-child.Done():
					return
				}
				if result.Err != nil || result.Chunk.Type == "done" {
					return
				}
			}
		}
	}()
	return output, nil
}
