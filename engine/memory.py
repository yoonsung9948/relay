"""Opt-in allocator measurements, independent of tracing and benchmark clients.

Peaks belong to a period of shared work, never to an individual concurrent
request. Existing engine request IDs let an offline consumer attribute periods.
"""

import json
import logging
import os
import time
from pathlib import Path
from uuid import uuid4

import torch

logger = logging.getLogger("inference.memory")
MAX_WINDOW_REQUESTS = 4096


class CudaMemoryProbe:
    def __init__(self, device):
        self.device = torch.device(device)
        self.supported = self.device.type == "cuda" and torch.cuda.is_available()

    def begin(self):
        if not self.supported:
            return None
        torch.cuda.synchronize(self.device)
        baseline = self.current()
        torch.cuda.reset_peak_memory_stats(self.device)
        return baseline

    def current(self):
        return {
            "allocated_bytes": torch.cuda.memory_allocated(self.device),
            "reserved_bytes": torch.cuda.memory_reserved(self.device),
        }

    def end(self):
        if not self.supported:
            return None, None
        torch.cuda.synchronize(self.device)
        return self.current(), {
            "allocated_bytes": torch.cuda.max_memory_allocated(self.device),
            "reserved_bytes": torch.cuda.max_memory_reserved(self.device),
        }


class MemoryRecorder:
    def __init__(self, path, probe, metadata=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Check writability; each record is flushed when its file handle closes.
        with self.path.open("a"):
            pass
        self.probe = probe
        self.metadata = metadata or {}
        self.instance_id = str(uuid4())
        self.active = set()
        self.window = None
        self.disabled = False

    @classmethod
    def from_env(cls, runner, metadata):
        path = os.environ.get("RELAY_MEMORY_PATH")
        if not path:
            return None
        try:
            pool = runner.pool
            recorder = cls(
                path,
                CudaMemoryProbe(runner.device),
                {
                    **metadata,
                    "device": str(runner.device),
                    "pool_allocated_bytes": pool.allocated_bytes if pool else 0,
                    "pool_slots": len(pool.owners) if pool else 0,
                    "pool_max_seq_len": pool.capacity if pool else None,
                    "dynamic_max_bytes": runner.cache_config.dynamic_max_bytes,
                },
            )
        except Exception:
            logger.exception("memory_recording_unavailable")
            return None
        logger.info("memory_recording_enabled path=%s cuda=%s", path, recorder.probe.supported)
        return recorder

    def request_started(self, request_id):
        if self.disabled:
            return
        request_id = str(request_id)
        if not self.active:
            self.window = {
                "schema_version": 1,
                "kind": "engine_memory_window",
                "window_id": str(uuid4()),
                "instance_id": self.instance_id,
                "pid": os.getpid(),
                "metadata": self.metadata,
                "status": "ok" if self.probe.supported else "unsupported_device",
                "complete": True,
                "request_ids": [],
                "request_ids_truncated": False,
                "request_count": 0,
                "max_in_flight": 0,
                "baseline": None,
                "final": None,
                "peak": None,
            }
            try:
                self.window["baseline"] = self.probe.begin()
            except Exception:
                self.window["status"] = "measurement_error"
                logger.exception("memory_begin_failed")
            self.window["started_at_unix_ns"] = time.time_ns()
        self.active.add(request_id)
        self.window["request_count"] += 1
        if len(self.window["request_ids"]) < MAX_WINDOW_REQUESTS:
            self.window["request_ids"].append(request_id)
        else:
            self.window["request_ids_truncated"] = True
        self.window["max_in_flight"] = max(self.window["max_in_flight"], len(self.active))

    def request_finished(self, request_id):
        request_id = str(request_id)
        if request_id not in self.active:
            return  # Stream cleanup can call cancel_request more than once.
        self.active.remove(request_id)
        if not self.active:
            self._finish(complete=True)

    def close(self):
        if self.window is not None:
            self._finish(complete=False)
        self.active.clear()

    def _finish(self, *, complete):
        window, self.window = self.window, None
        window["complete"] = complete
        if window["status"] == "ok":
            try:
                window["final"], window["peak"] = self.probe.end()
            except Exception:
                window["status"] = "measurement_error"
                logger.exception("memory_end_failed")
        window["ended_at_unix_ns"] = time.time_ns()
        try:
            # One small append per drained period; no per-token work or queues.
            with self.path.open("a") as output:
                output.write(json.dumps(window, allow_nan=False) + "\n")
        except OSError:
            self.disabled = True
            logger.exception("memory_write_failed recording_disabled=true")
