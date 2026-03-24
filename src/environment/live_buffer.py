"""
LiveWorkloadBuffer — rolling window of live cluster CPU measurements.

Streams an Alibaba .npy trace at compressed speed (default 24x: 1 day → 1 hour).
Each 30-second bin is ingested every 1.25 seconds real-time at 24x compression.

In production: replace stream_from_npy() with calls to add() from your
monitoring system (Prometheus, CloudWatch, Datadog) every 30 seconds.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Optional

import numpy as np


class LiveWorkloadBuffer:
    """
    Rolling buffer of (timestamp, cpu_util) measurements.

    - stream_from_npy(path): replay Alibaba .npy in background thread at
      compressed speed (1 day in 1 hour by default)
    - add(cpu_util): ingest a live measurement
    - workload_fn(sim_time): CloudEnv-compatible callable
    - as_numpy(): dump buffer as 1-D numpy array for saving to temp .npy
    """

    def __init__(self, window_seconds: int = 600, bin_size: int = 30):
        self.window_seconds = window_seconds
        self.bin_size       = bin_size
        self._buf: deque    = deque()       # (timestamp, cpu_util)
        self._lock          = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self.streaming      = False

    # ── Ingestion ──────────────────────────────────────────────────────

    def add(self, cpu_util: float, timestamp: Optional[float] = None) -> None:
        """Ingest one CPU utilisation measurement (0–1)."""
        if timestamp is None:
            timestamp = time.time()
        with self._lock:
            self._buf.append((timestamp, float(np.clip(cpu_util, 0.05, 1.0))))
            cutoff = timestamp - self.window_seconds
            while self._buf and self._buf[0][0] < cutoff:
                self._buf.popleft()

    def stream_from_npy(
        self,
        npy_path:    str,
        compression: float = 24.0,   # 1 day → 1 hour real-time
        column:      int   = 0,
        loop:        bool  = True,
        verbose:     bool  = True,
    ) -> threading.Thread:
        """
        Stream an Alibaba .npy file into the buffer in a background thread.

        compression=24 → each 30-s bin arrives every 30/24 = 1.25 s real-time.
        With loop=True the trace repeats; set loop=False to stream once.
        """
        data  = np.load(npy_path)
        rates = np.clip(data[:, column] if data.ndim > 1 else data, 0.05, 1.0)
        interval = self.bin_size / compression   # seconds between ingestions
        n_bins   = len(rates)

        def _stream():
            self.streaming = True
            if verbose:
                total_min = n_bins * self.bin_size / 60
                real_min  = total_min / compression
                print(f"[LiveBuffer] Streaming {npy_path}")
                print(f"[LiveBuffer] {n_bins} bins = {total_min:.0f} min simulated → "
                      f"{real_min:.1f} min real-time  ({compression:.0f}x compression)")
                print(f"[LiveBuffer] 1 sample every {interval:.2f} s")
            idx = 0
            while self.streaming:
                self.add(float(rates[idx % n_bins]))
                idx += 1
                if not loop and idx >= n_bins:
                    break
                time.sleep(interval)
            self.streaming = False
            if verbose:
                print(f"[LiveBuffer] Stream ended. Buffer: {self.n_samples} samples")

        self._thread = threading.Thread(target=_stream, daemon=True)
        self._thread.start()
        return self._thread

    def stop_stream(self) -> None:
        self.streaming = False
        if self._thread:
            self._thread.join(timeout=5)

    # ── Query ──────────────────────────────────────────────────────────

    def has_enough_data(self, min_samples: int = 20) -> bool:
        with self._lock:
            return len(self._buf) >= min_samples

    def workload_fn(self, sim_time: float) -> float:
        """CloudEnv-compatible workload function (wraps around buffer)."""
        with self._lock:
            if not self._buf:
                return 0.5
            rates = [v for _, v in self._buf]
        idx = int(sim_time / self.bin_size) % len(rates)
        return float(rates[idx])

    def as_numpy(self) -> np.ndarray:
        """Return buffer as (N, 4) float32 array (CPU in col 0, zeros elsewhere)."""
        with self._lock:
            rates = np.array([v for _, v in self._buf], dtype=np.float32)
        if len(rates) == 0:
            rates = np.array([0.5], dtype=np.float32)
        out = np.zeros((len(rates), 4), dtype=np.float32)
        out[:, 0] = rates
        return out

    @property
    def n_samples(self) -> int:
        with self._lock:
            return len(self._buf)

    @property
    def mean_util(self) -> float:
        with self._lock:
            if not self._buf:
                return 0.0
            return float(np.mean([v for _, v in self._buf]))

    @property
    def latest_util(self) -> float:
        with self._lock:
            return float(self._buf[-1][1]) if self._buf else 0.0
