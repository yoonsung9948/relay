# Control-plane lifecycle

Relay owns one GPU worker at a time. Startup, generation, timeout expiry, manual stop,
and process shutdown share the same ownership record and cleanup operation.

## Configuration

```yaml
control_plane_config:
  # Optional. Defaults to systemd's STATE_DIRECTORY, otherwise ./state.
  # state_directory: /var/lib/relay
  provision_timeout: 10m
  idle_timeout: 1m
  max_session_duration: 10m
  cleanup_timeout: 60s
  shutdown_timeout: 80s
  provision_cooldown: 1h
  provision_retry_backoff: 15s
```

Keep the existing provider and serving sections alongside these settings. Paths are
relative to the process working directory. `RATE_LIMITED` still defaults to `true`;
setting it to `false` disables admission rate limits, **not** ownership checks,
startup deadlines, idle expiry, session expiry, or cleanup.

- `provision_timeout`: one budget for Create and engine readiness, beginning before
  Create. A browser disconnect does not cancel this workflow.
- `idle_timeout`: starts when the engine becomes ready and resets after the last
  generation finishes. Active requests and streams suspend idle expiry. Zero disables it.
- `max_session_duration`: measured from readiness; cancels active generation at expiry.
  Zero disables it. Startup time is additional to this ready-session budget.
- `cleanup_timeout`: independent budget for joining startup, draining canceled requests,
  verifying ownership, terminating the Pod, and confirming termination. Request draining
  uses at most five seconds of this budget.
- `shutdown_timeout`: overall server shutdown budget; must cover cleanup. HTTP draining
  runs concurrently with GPU cleanup and uses at most five seconds.

## Ownership and restart behavior

Before contacting RunPod, Relay writes a unique owner/attempt name into
`<state_directory>/worker.json`. It records the Pod ID immediately after creation,
before health polling. Writes use a private temporary file, fsync, atomic rename,
and directory fsync. A process lock prevents two Relay processes from using this directory.

On restart, Relay cleans up the recorded worker before allowing a new start. It does
not automatically rent a replacement. When a Create response was lost, it searches all
RunPod listing pages for that **exact attempt name**. It never deletes Pods using a
prefix or sweeps unrelated account resources. Missing/duplicate matches, malformed
records, changed credentials, or failed termination leave provisioning blocked and
ownership retained.

RunPod credentials and API endpoint are fingerprinted, without storing the API key
in the journal. Reconcile outstanding workers before rotating credentials or changing
provider endpoint. A provider 404 is accepted only after checking the recorded ID and
credential scope. A successful DELETE is followed by Get until the same Pod is
terminated or absent. Only then is the record cleared and the engine endpoint removed.
Transient network failures, HTTP 429, and HTTP 5xx are retried within the cleanup budget;
permanent errors stop immediately. Create is never automatically retried.

`GET /health` includes `state`, optional `error`/`instance_id`, and `active_requests`.
Failed cleanup retains the record in `error`; retry the authenticated stop operation:

```sh
curl --fail-with-body -X POST http://127.0.0.1:8080/stop_engine \
  -H "X-Demo-Key: $DEMO_KEY"
```

A successful stop returns 200 with offline status; cleanup failure returns 503.
Concurrent stops share one operation. Disconnecting the caller does not cancel cleanup.

If recovery stays blocked, inspect the logged attempt and RunPod console using the
original account credentials. Confirm the exact Pod's termination before repairing the
journal while Relay is stopped. Do not delete `worker.json` merely to bypass an error.

## Running under systemd

[deploy/relay.service](deploy/relay.service) is an example unit. Install the binary at
`/usr/local/bin/relay`, keep YAML at `/etc/relay/config.yaml`, and put environment variables
in `/etc/relay/relay.env` readable only by root/the service user. The unit assumes a
`relay` system user/group already exists. Adapt these paths to your existing unit.

`StateDirectory=relay` provides persistent `/var/lib/relay` and exports `STATE_DIRECTORY`;
leave `state_directory` out of YAML to use it. Keep that directory across deployments.
The unit's 90-second stop budget covers the default 80-second server budget and tracing
flush. Increase it if you increase the configured shutdown budget.

Stop the old binary cleanly before replacing it. The previous binary did not write this
journal, so manually terminate any old demo Pod before the first rollout. The new
process cannot infer ownership of workers created before journaling was introduced.

For local use from the module directory:

```sh
go run ./cmd/server -config examples/config.yaml
```

## Verification

```sh
go test -race ./...
go vet ./...
CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -o /tmp/relay-linux ./cmd/server
```

Tests use fake providers and local HTTP servers; they never rent a GPU.

## Limits

This remains a single-process demo. Ownership recovery survives a process restart with
its disk intact. VM/disk loss, SIGKILL, machine downtime, provider outages, and an
unresolvable Create response can require manual reconciliation. Timers cannot run while
the VM is down. An external watchdog/provider-side limit would be needed for an
independent billing safeguard. Provision cooldown is still in-memory and resets on restart.
