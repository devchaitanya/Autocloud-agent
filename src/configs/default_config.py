from dataclasses import dataclass, field
from typing import Dict


@dataclass
class SimConfig:
    n_max: int = 20
    n_min: int = 3
    n_init: int = 5
    step_duration: float = 30.0       # seconds per env step
    grace_period: float = 30.0        # soft-drain window (seconds)
    warmup_period: float = 60.0       # post-boot protection window (seconds)
    episode_steps: int = 120          # 120 steps × 30s = 1-hour episode
    k_jobs: int = 20                  # top-K jobs in obs
    lognormal_mu: float = 2.0
    lognormal_sigma: float = 1.0


@dataclass
class NodeTypeDef:
    cpu: int
    memory: float        # GB
    cost_per_hr: float
    boot_time: float     # seconds


NODE_TYPES: Dict[str, NodeTypeDef] = {
    "small":  NodeTypeDef(cpu=2,  memory=4.0,  cost_per_hr=0.05, boot_time=30.0),
    "medium": NodeTypeDef(cpu=4,  memory=8.0,  cost_per_hr=0.10, boot_time=45.0),
    "large":  NodeTypeDef(cpu=8,  memory=16.0, cost_per_hr=0.20, boot_time=60.0),
    "xlarge": NodeTypeDef(cpu=16, memory=32.0, cost_per_hr=0.40, boot_time=90.0),
}


@dataclass
class ForecastConfig:
    seq_len: int = 20
    d_model: int = 64
    n_heads: int = 4
    d_ff: int = 256
    n_layers: int = 2
    dropout: float = 0.2
    n_horizons: int = 4               # t+1, t+5, t+10, t+15
    mc_samples: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 50
    patience: int = 10
    quantiles: tuple = (0.1, 0.5, 0.9)


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    entropy_coef_min: float = 0.001
    entropy_decay_steps: int = 100_000
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    minibatch_size: int = 64
    update_epochs: int = 4
    buffer_size: int = 2048
    ema_window: int = 1000


@dataclass
class RewardConfig:
    # Scale-Out
    alpha1: float = 2.0    # under-provisioning penalty
    alpha2: float = 1.0    # over-provisioning penalty
    alpha3: float = 0.5    # running cost pressure
    alpha4: float = 0.3    # boot-event penalty
    # Consolidation
    beta1: float = 1.0     # running cost
    beta2: float = 2.0     # SLA bonus
    beta3: float = 0.5     # migration penalty
    beta4: float = 1.0     # memory pressure
    # Scheduling
    gamma1: float = 1.0    # mean wait time
    gamma2: float = 2.0    # SLA violation
    # SLA threshold (seconds)
    sla_latency_ms: float = 500.0
    # Utilization targets
    cpu_high: float = 0.8
    cpu_low: float = 0.3
    mem_high: float = 0.85


@dataclass
class CoordinatorConfig:
    sigma_high: float = 0.3           # uncertainty suppression threshold


@dataclass
class AutoResearchConfig:
    budget: int = 8
    timeout_s: int = 600
    api_model: str = "claude-sonnet-4-6"


@dataclass
class Config:
    sim: SimConfig = field(default_factory=SimConfig)
    forecast: ForecastConfig = field(default_factory=ForecastConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    coordinator: CoordinatorConfig = field(default_factory=CoordinatorConfig)
    autoresearch: AutoResearchConfig = field(default_factory=AutoResearchConfig)
    total_steps: int = 200_000
    eval_seeds: list = field(default_factory=lambda: [0, 1, 2])
    log_interval: int = 1000


DEFAULT_CONFIG = Config()
