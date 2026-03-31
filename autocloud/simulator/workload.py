"""
Workload module — Alibaba Cluster Trace 2018 loader + synthetic fallback.

Primary: AlibabaTraceLoader — loads real per-machine CPU/mem utilization,
         selects 200 highest-variance machines, aggregates to 30s bins,
         produces the 4-feature timeseries used by the Transformer forecaster
         and as the RL simulation workload.

Fallback: SyntheticWorkload — diurnal + spike generator used when the trace
          is not available (e.g. local unit tests).

Alibaba 2018 trace format (machine_usage CSV):
  machine_id, time_stamp, cpu_util_percent, mem_util_percent,
  net_in, net_out, disk_io_percent
  (timestamps are in seconds; granularity varies by file but ~10s)
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List


# 
# Alibaba Trace Loader
# 

class AlibabaTraceLoader:
    """
    Loads and preprocesses the Alibaba Cluster Trace 2018.

    Steps:
      1. Read machine_usage CSV(s) from `data_dir`.
      2. Select `n_machines` machines with highest CPU utilization variance.
      3. Aggregate per-machine data to `bin_size_s`-second bins (default 30s).
      4. Compute cluster-level aggregate features:
             demand_norm   — mean CPU across selected machines (0-1)
             cpu_util      — same as demand_norm
             queue_len_norm— proxy: fraction of machines with CPU > 80%
             hour_of_day   — (timestamp % 86400) / 86400
      5. Split into Day 1 (forecaster training) and Day 2 (RL simulation).
    """

    # Column names in the Alibaba 2018 machine_usage files
    COLUMNS = ["machine_id", "time_stamp", "cpu_util_percent",
               "mem_util_percent", "net_in", "net_out", "disk_io_percent"]

    def __init__(
        self,
        data_dir: str,
        n_machines: int = 200,
        bin_size_s: int = 30,
        day1_start_s: Optional[float] = None,   # auto-detect from trace
        day2_start_s: Optional[float] = None,
        day_duration_s: float = 86400.0,
    ):
        self.data_dir      = data_dir
        self.n_machines    = n_machines
        self.bin_size_s    = bin_size_s
        self.day_duration_s = day_duration_s
        self._day1_start   = day1_start_s
        self._day2_start   = day2_start_s

        self._day1_data: Optional[np.ndarray] = None
        self._day2_data: Optional[np.ndarray] = None
        self._workload_bins: Optional[np.ndarray] = None   # full normalized CPU series
        self._bin_t0: float = 0.0   # absolute sim time of first bin

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    # Threshold above which chunked reading is used (bytes)
    _LARGE_FILE_THRESHOLD = 200 * 1024 * 1024   # 200 MB

    def load(self, verbose: bool = True, chunk_size: int = 500_000) -> None:
        """
        Load and preprocess the trace.  Must be called before get_*().

        For large files (>200 MB) a two-pass chunked strategy is used to
        avoid OOM:
          Pass 1 — stream the file to accumulate per-machine CPU variance.
          Pass 2 — stream again, keeping only the top-N machines.

        Parameters
        ----------
        chunk_size : rows per chunk during streaming (default 500 000)
        """
        csv_files = self._find_csv_files()
        if not csv_files:
            raise FileNotFoundError(
                f"No machine_usage CSV files found in {self.data_dir}."
            )

        if verbose:
            for f in csv_files:
                size_mb = os.path.getsize(f) / 1e6
                print(f"Found: {os.path.basename(f)}  ({size_mb:.0f} MB)")

        # Detect header row: read first row with header=None, check if
        # time_stamp column (col 1) looks numeric.  If it's a string like
        # "time_stamp" the file has a header row.
        raw_sample = pd.read_csv(csv_files[0], header=None, nrows=2)
        try:
            float(str(raw_sample.iloc[0, 1]))
            has_header = False   # first row is already data
        except (ValueError, TypeError):
            has_header = True    # first row is column names

        # Always read with header=None (skip the header row via skiprows when
        # it exists) so column indices stay integer-based throughout.
        use_cols  = [0, 1, 2, 3]   # machine_id, time_stamp, cpu, mem
        dtype_map = {0: str, 1: float, 2: float, 3: float}
        col_names = None   # we rename after read anyway
        skiprows  = [0] if has_header else None

        # Choose loading strategy based on file size
        total_size = sum(os.path.getsize(f) for f in csv_files)
        use_chunks = total_size > self._LARGE_FILE_THRESHOLD

        if use_chunks:
            if verbose:
                print(f"Large file ({total_size/1e6:.0f} MB) — using chunked 2-pass loading ...")
            raw = self._load_chunked(csv_files, chunk_size,
                                     skiprows, use_cols, dtype_map, verbose)
        else:
            dfs = []
            for f in csv_files:
                df = pd.read_csv(
                    f,
                    header=None,
                    skiprows=skiprows,
                    usecols=use_cols,
                    dtype=dtype_map,
                )
                df.columns = ["machine_id", "time_stamp", "cpu_util_percent", "mem_util_percent"]
                dfs.append(df)
            raw = pd.concat(dfs, ignore_index=True)

        # Normalize CPU from percent (0-100) to fraction (0-1)
        raw["cpu"] = pd.to_numeric(raw["cpu_util_percent"], errors="coerce").clip(0, 100) / 100.0
        raw["mem"] = pd.to_numeric(raw["mem_util_percent"], errors="coerce").clip(0, 100) / 100.0
        raw["ts"]  = pd.to_numeric(raw["time_stamp"],       errors="coerce")
        raw = raw.dropna(subset=["cpu", "ts"])

        if verbose:
            t_min, t_max = raw["ts"].min(), raw["ts"].max()
            print(f"Trace spans {(t_max - t_min)/3600:.1f} h, "
                  f"{raw['machine_id'].nunique()} unique machines, "
                  f"{len(raw):,} rows")

        # Select highest-variance machines
        machine_var  = raw.groupby("machine_id")["cpu"].var().sort_values(ascending=False)
        top_machines = machine_var.head(self.n_machines).index.tolist()
        filtered     = raw[raw["machine_id"].isin(top_machines)].copy()

        if verbose:
            print(f"Selected {len(top_machines)} machines with highest CPU variance")

        # Bin to bin_size_s resolution
        t0 = filtered["ts"].min()
        filtered["bin"] = ((filtered["ts"] - t0) / self.bin_size_s).astype(int)

        binned = filtered.groupby("bin").agg(
            cpu_mean=("cpu", "mean"),
            mem_mean=("mem", "mean"),
            cpu_high_frac=("cpu", lambda x: (x > 0.8).mean()),   # queue proxy
        ).reset_index()
        binned["hour_of_day"] = ((binned["bin"] * self.bin_size_s) % 86400) / 86400.0

        # Build feature matrix: [demand_norm, cpu_util, queue_len_norm, hour_of_day]
        feat = np.column_stack([
            binned["cpu_mean"].values,
            binned["cpu_mean"].values,
            binned["cpu_high_frac"].values,
            binned["hour_of_day"].values,
        ]).astype(np.float32)

        # Clip to [0, 1]
        feat = np.clip(feat, 0.0, 1.0)

        n_bins = len(feat)
        bins_per_day = int(self.day_duration_s / self.bin_size_s)   # 2880

        # Determine day boundaries
        if self._day1_start is None:
            day1_bin = 0
        else:
            day1_bin = int((self._day1_start - t0) / self.bin_size_s)

        if self._day2_start is None:
            day2_bin = day1_bin + bins_per_day
        else:
            day2_bin = int((self._day2_start - t0) / self.bin_size_s)

        # Ensure we have enough data
        if day2_bin + bins_per_day > n_bins:
            # Wrap around or truncate
            day2_bin = max(0, n_bins - bins_per_day)
            if verbose:
                print(f"Warning: trace shorter than 2 days; Day 2 starts at bin {day2_bin}")

        self._day1_data = feat[day1_bin : day1_bin + bins_per_day]
        self._day2_data = feat[day2_bin : day2_bin + bins_per_day]
        self._workload_bins = feat
        self._bin_t0 = float(t0)

        if verbose:
            print(f"Day 1: {len(self._day1_data)} bins, "
                  f"mean CPU={self._day1_data[:,0].mean():.3f}")
            print(f"Day 2: {len(self._day2_data)} bins, "
                  f"mean CPU={self._day2_data[:,0].mean():.3f}")

    def get_day1(self) -> np.ndarray:
        """Returns (n_steps, 4) feature array for Day 1 (forecaster training)."""
        self._check_loaded()
        return self._day1_data

    def get_day2(self) -> np.ndarray:
        """Returns (n_steps, 4) feature array for Day 2 (RL simulation)."""
        self._check_loaded()
        return self._day2_data

    def get_full_data(self) -> np.ndarray:
        """Returns the full feature matrix for all available days."""
        self._check_loaded()
        return self._workload_bins

    def get_train_data(self) -> np.ndarray:
        """
        Returns all data except the last day for forecaster training.
        Uses the full trace length rather than just Day 1.
        """
        self._check_loaded()
        bins_per_day = int(self.day_duration_s / self.bin_size_s)
        full = self._workload_bins
        if len(full) > bins_per_day:
            return full[:-bins_per_day]   # all days except the last
        return full   # fallback: trace shorter than 2 days

    def make_workload_fn(self, day: int = 2) -> callable:
        """
        Returns a callable workload_fn(sim_time_seconds) -> arrival_rate_multiplier
        based on the trace data for the given day (1 or 2).

        The multiplier is derived from the CPU utilization time series:
        a value of 1.0 corresponds to the mean CPU utilization.
        """
        self._check_loaded()
        data = self._day1_data if day == 1 else self._day2_data
        cpu_series = data[:, 0]      # demand_norm = mean CPU util
        mean_cpu   = cpu_series.mean() if cpu_series.mean() > 0 else 0.5

        def workload_fn(sim_time_s: float) -> float:
            bin_idx = int(sim_time_s / self.bin_size_s)
            bin_idx = min(bin_idx, len(cpu_series) - 1)
            return float(cpu_series[bin_idx] / max(mean_cpu, 1e-6))

        return workload_fn

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _load_chunked(
        self,
        csv_files: List[str],
        chunk_size: int,
        skiprows,
        use_cols: list,
        dtype_map: dict,
        verbose: bool,
    ) -> pd.DataFrame:
        """
        Two-pass chunked loader for large CSV files.

        Pass 1: accumulate per-machine CPU variance using Welford's online
                algorithm (memory: O(n_machines) — no full file in RAM).
        Pass 2: stream again, keeping only rows for the top-N machines.

        `skiprows=[0]` when the file has a header row, None otherwise.
        """
        from collections import defaultdict

        _read_kwargs = dict(
            header=None,
            skiprows=skiprows,
            usecols=use_cols,
            dtype=dtype_map,
            chunksize=chunk_size,
            on_bad_lines="skip",
        )

        # Pass 1: compute per-machine CPU variance (Welford)
        if verbose:
            print("  Pass 1/2: computing per-machine CPU variance ...")

        wf_count: dict = defaultdict(int)
        wf_mean:  dict = defaultdict(float)
        wf_m2:    dict = defaultdict(float)

        for fpath in csv_files:
            for chunk in pd.read_csv(fpath, **_read_kwargs):
                chunk.columns = ["machine_id", "time_stamp",
                                  "cpu_util_percent", "mem_util_percent"]
                cpu = pd.to_numeric(chunk["cpu_util_percent"], errors="coerce")
                cpu = cpu.clip(0, 100) / 100.0
                chunk = chunk.copy()
                chunk["cpu"] = cpu
                chunk = chunk.dropna(subset=["cpu"])

                for mid, grp in chunk.groupby("machine_id", sort=False):
                    for val in grp["cpu"]:
                        n = wf_count[mid] + 1
                        delta  = val - wf_mean[mid]
                        wf_mean[mid] += delta / n
                        wf_m2[mid]   += delta * (val - wf_mean[mid])
                        wf_count[mid] = n

        machine_var = {
            mid: (wf_m2[mid] / (wf_count[mid] - 1)) if wf_count[mid] > 1 else 0.0
            for mid in wf_count
        }
        top_machines = set(
            sorted(machine_var, key=machine_var.get, reverse=True)[: self.n_machines]
        )
        if verbose:
            print(f"  {len(machine_var)} total machines; keeping top {len(top_machines)}")

        # Pass 2: collect rows for top machines only
        if verbose:
            print("  Pass 2/2: reading selected machines ...")

        selected: List[pd.DataFrame] = []
        for fpath in csv_files:
            for chunk in pd.read_csv(fpath, **_read_kwargs):
                chunk.columns = ["machine_id", "time_stamp",
                                  "cpu_util_percent", "mem_util_percent"]
                mask = chunk["machine_id"].isin(top_machines)
                if mask.any():
                    selected.append(chunk[mask].copy())

        return pd.concat(selected, ignore_index=True)

    def _find_csv_files(self) -> List[str]:
        """
        Search `data_dir` (and one level of subdirectories) for Alibaba
        machine_usage CSV files.  Falls back to any .csv if no machine_usage
        files are found — handles differently-named Kaggle re-uploads.
        """
        if not os.path.isdir(self.data_dir):
            return []

        # Collect all CSVs recursively (max depth 2)
        all_csvs: List[str] = []
        for root, dirs, fnames in os.walk(self.data_dir):
            # Limit depth: skip anything more than 1 level below data_dir
            depth = root.replace(self.data_dir, "").count(os.sep)
            if depth > 1:
                dirs[:] = []
                continue
            for fname in sorted(fnames):
                if fname.endswith(".csv"):
                    all_csvs.append(os.path.join(root, fname))

        # Prefer files with "machine_usage" in the name
        preferred = [p for p in all_csvs if "machine_usage" in os.path.basename(p).lower()]
        if preferred:
            return preferred

        # Fallback: validate any CSV by checking its header/columns
        validated = []
        for path in all_csvs:
            try:
                header = pd.read_csv(path, header=None, nrows=1).iloc[0].tolist()
                # Accept if it looks like an Alibaba trace (7 numeric/id columns)
                if len(header) >= 4:
                    validated.append(path)
            except Exception:
                pass
        return validated

    def _check_loaded(self) -> None:
        if self._day1_data is None:
            raise RuntimeError("Call .load() before accessing data.")


# 
# Synthetic fallback (used for unit tests / when trace unavailable)
# 

class SyntheticWorkload:
    """
    Generates a time-varying arrival rate multiplier.

    Diurnal pattern: 1 + amplitude * sin(2π * t / T_day + phase)
    Spikes: Poisson-distributed onset times, each spike lasts `duration` seconds
            with a 5× rate multiplier during the spike window.
    """

    def __init__(
        self,
        rng: np.random.Generator,
        diurnal_amplitude: float = 0.5,
        diurnal_phase: float = -np.pi / 2,
        noise_std: float = 0.05,
        spike_rate: float = 1 / 3600.0,
        spike_multiplier: float = 5.0,
        spike_duration_range: Tuple[float, float] = (120.0, 300.0),
    ):
        self.rng = rng
        self.diurnal_amplitude = diurnal_amplitude
        self.diurnal_phase = diurnal_phase
        self.noise_std = noise_std
        self.spike_rate = spike_rate
        self.spike_multiplier = spike_multiplier
        self.spike_duration_range = spike_duration_range
        self._spikes: list = []
        self._generate_spikes(duration_s=2 * 86400)

    def _generate_spikes(self, duration_s: float) -> None:
        self._spikes = []
        t = 0.0
        while t < duration_s:
            inter = self.rng.exponential(1.0 / self.spike_rate)
            t += inter
            if t >= duration_s:
                break
            dur = float(self.rng.uniform(*self.spike_duration_range))
            self._spikes.append((t, t + dur))

    def __call__(self, sim_time: float) -> float:
        T_day = 86400.0
        diurnal = 1.0 + self.diurnal_amplitude * np.sin(
            2 * np.pi * sim_time / T_day + self.diurnal_phase
        )
        noise = float(self.rng.normal(0, self.noise_std))
        mult = diurnal + noise
        for (t_start, t_end) in self._spikes:
            if t_start <= sim_time <= t_end:
                mult *= self.spike_multiplier
                break
        return max(0.1, mult)


def generate_forecast_dataset(
    rng: np.random.Generator,
    step_duration: float = 30.0,
    day1_steps: int = 2880,
    day2_steps: int = 2880,
    base_rate: float = 2.0,
    n_nodes_ref: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Synthetic dataset — used when Alibaba trace is unavailable."""
    workload = SyntheticWorkload(rng)

    def _simulate_metrics(n_steps: int, t_offset: float) -> np.ndarray:
        data = np.zeros((n_steps, 4), dtype=np.float32)
        queue_sim = 0.0
        denom = base_rate * (1 + workload.diurnal_amplitude + workload.noise_std * 3)
        for i in range(n_steps):
            t = t_offset + i * step_duration
            rate = workload(t)
            demand_norm = min(rate / denom, 1.0)
            arrivals_per_step = rate * step_duration
            services_per_step = n_nodes_ref * 1.0
            cpu_util = min(arrivals_per_step / max(services_per_step, 1.0), 1.0)
            queue_sim = max(0.0, queue_sim + arrivals_per_step - services_per_step)
            queue_norm = min(queue_sim / 50.0, 1.0)
            hour_of_day = (t % 86400) / 86400.0
            data[i] = [demand_norm, cpu_util, queue_norm, hour_of_day]
        return data

    day1_data = _simulate_metrics(day1_steps, t_offset=0.0)
    day2_data = _simulate_metrics(day2_steps, t_offset=86400.0)
    return day1_data, day2_data


def split_day1(
    day1_data: np.ndarray,
    train_fraction: float = 0.6,
) -> Tuple[np.ndarray, np.ndarray]:
    split = int(len(day1_data) * train_fraction)
    return day1_data[:split], day1_data[split:]


def make_sequences(
    data: np.ndarray,
    seq_len: int = 20,
    horizons: Tuple[int, ...] = (1, 5, 10, 15),
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(data)
    max_horizon = max(horizons)
    xs, ys = [], []
    for i in range(n - seq_len - max_horizon):
        x = data[i : i + seq_len]
        y = np.array([data[i + seq_len + h - 1, 0] for h in horizons])
        xs.append(x)
        ys.append(y)
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)
