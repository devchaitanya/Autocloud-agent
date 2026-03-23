"""
AutoResearch Engine — Karpathy-style LLM-guided experiment loop.

Inspired by github.com/karpathy/autoresearch:
  The LLM reads the current experiment.py code + trial history, then
  rewrites the file with a proposed improvement. If score improves the
  new code is KEPT; otherwise the previous version is RESTORED.

Flow (each iteration):
    1. Read current experiment.py
    2. Build prompt: program.md + current code + history
    3. LLM returns complete new experiment.py
    4. Syntax-check the proposed code
    5. Run trial in subprocess (train.py --experiment_file <tmp>)
    6. Parse score
    7. Keep (update experiment.py) or Discard (restore previous)
    8. Append to results.tsv
"""
from __future__ import annotations

import ast
import copy
import json
import os
import shutil
import sys
import time
from typing import Dict, List, Optional, Tuple, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from autoresearch.subprocess_runner import run_trial, FAILURE_SENTINEL
from configs.default_config import Config, DEFAULT_CONFIG

# Paths (relative to autocloud_agent root)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENT_FILE = os.path.join(_ROOT, "experiment.py")
PROGRAM_FILE    = os.path.join(_ROOT, "program.md")
RESULTS_TSV     = os.path.join(_ROOT, "autoresearch", "results.tsv")


# ------------------------------------------------------------------ #
# Prompt builder
# ------------------------------------------------------------------ #

def _load_program_md() -> str:
    try:
        with open(PROGRAM_FILE) as f:
            return f.read()
    except FileNotFoundError:
        return "Goal: maximize score = mean_SLA_rate - 0.1 × mean_cost."


def _load_experiment_code() -> str:
    try:
        with open(EXPERIMENT_FILE) as f:
            return f.read()
    except FileNotFoundError:
        return ""


def _build_prompt(
    current_code: str,
    history: List[Dict],
    n_iter: int,
    total: int,
    program_md: str,
) -> str:
    if history:
        rows = []
        for h in history:
            status = "KEEP   " if h["kept"] else "DISCARD"
            rows.append(
                f"  [{status}] iter={h['iter']:2d}  score={h['score']:+.4f}  "
                f"sla={h['sla']:.3f}  cost={h['cost']:.4f}  | {h['description']}"
            )
        history_str = "\n".join(rows)
    else:
        history_str = "  (no trials yet — iteration 1 establishes the baseline)"

    return f"""You are an autonomous RL researcher improving the AutoCloud-Agent system.

## Research Program
{program_md}

## Current experiment.py (iteration {n_iter - 1} — {"baseline" if n_iter == 1 else "last kept version"})
```python
{current_code}
```

## Trial History (score = SLA_rate − 0.1×cost, HIGHER is better)
{history_str}

## Your Task — Iteration {n_iter}/{total}

Study the history above. Propose ONE focused change to improve the score.
- Prefer changing a single value so the effect is interpretable.
- If no history yet, run the baseline as-is (copy the file unchanged).
- Think about which reward weight most directly drives the metric you want to improve.

Return ONLY the complete, valid Python content of the new experiment.py.
Rules:
  - File must define `get_config()` returning a valid Config object.
  - Do NOT add new imports beyond `copy` and `configs.default_config`.
  - Do NOT include markdown fences, explanations, or any text outside Python.
  - Start your response with the triple-quoted docstring line.
"""


# ------------------------------------------------------------------ #
# LLM backends
# ------------------------------------------------------------------ #

def _call_llm_anthropic(prompt: str, model: str) -> Optional[str]:
    """Anthropic Claude API."""
    try:
        import anthropic
    except ImportError:
        print("[AutoResearch] pip install anthropic")
        return None
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[AutoResearch] Set ANTHROPIC_API_KEY")
        return None
    client = anthropic.Anthropic(api_key=api_key)
    try:
        msg = client.messages.create(
            model=model, max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text
    except Exception as e:
        print(f"[AutoResearch] Anthropic error: {e}")
        return None


def _call_llm_groq(prompt: str, model: str) -> Optional[str]:
    """Groq API — free tier, fast Llama/Mixtral models.
    Get a free key at https://console.groq.com (no credit card).
    Set: export GROQ_API_KEY=gsk_...
    """
    try:
        from groq import Groq
    except ImportError:
        print("[AutoResearch] pip install groq")
        return None
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("[AutoResearch] Set GROQ_API_KEY (free at console.groq.com)")
        return None
    client = Groq(api_key=api_key)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
        )
        return resp.choices[0].message.content
    except Exception as e:
        print(f"[AutoResearch] Groq error: {e}")
        return None


def _call_llm_ollama(prompt: str, model: str) -> Optional[str]:
    """Ollama — completely local, no API key needed.
    Install: curl -fsSL https://ollama.com/install.sh | sh
    Pull:    ollama pull llama3.2:3b
    Run:     ollama serve  (starts at localhost:11434)
    """
    try:
        import urllib.request, json as _json
        payload = _json.dumps({
            "model": model,
            "prompt": prompt,
            "stream": False,
        }).encode()
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = _json.loads(resp.read())
            return data.get("response", "")
    except Exception as e:
        print(f"[AutoResearch] Ollama error: {e}")
        print("[AutoResearch] Is Ollama running? Try: ollama serve")
        return None


def _call_llm_gemini(prompt: str, model: str) -> Optional[str]:
    """Google Gemini API — free tier available.
    Get a free key at https://aistudio.google.com
    Set: export GEMINI_API_KEY=...
    """
    try:
        import google.generativeai as genai
    except ImportError:
        print("[AutoResearch] pip install google-generativeai")
        return None
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[AutoResearch] Set GEMINI_API_KEY (free at aistudio.google.com)")
        return None
    genai.configure(api_key=api_key)
    try:
        m = genai.GenerativeModel(model)
        resp = m.generate_content(prompt)
        return resp.text
    except Exception as e:
        print(f"[AutoResearch] Gemini error: {e}")
        return None


# Provider dispatch
_LLM_DEFAULTS = {
    "anthropic": "claude-haiku-4-5-20251001",
    "groq":      "llama-3.1-8b-instant",
    "ollama":    "llama3.2:3b",
    "gemini":    "gemini-1.5-flash-8b",
}

_LLM_CALLERS = {
    "anthropic": _call_llm_anthropic,
    "groq":      _call_llm_groq,
    "ollama":    _call_llm_ollama,
    "gemini":    _call_llm_gemini,
}


def _call_llm(prompt: str, provider: str = "groq", model: str = None) -> Optional[str]:
    """Unified LLM call. provider = 'anthropic' | 'groq' | 'ollama' | 'gemini'."""
    if provider not in _LLM_CALLERS:
        print(f"[AutoResearch] Unknown provider '{provider}'. Choose: {list(_LLM_CALLERS)}")
        return None
    if model is None:
        model = _LLM_DEFAULTS[provider]
    return _LLM_CALLERS[provider](prompt, model)


# ------------------------------------------------------------------ #
# Code validation
# ------------------------------------------------------------------ #

def _validate_code(code: str) -> Tuple[bool, str]:
    """Syntax-check and basic sanity check. Returns (ok, reason)."""
    # Strip accidental markdown fences
    lines = code.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    code = "\n".join(lines)

    try:
        ast.parse(code)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"

    if "def get_config" not in code:
        return False, "Missing get_config() function"
    if "return config" not in code:
        return False, "get_config() must return config"

    return True, code   # second element returns cleaned code


def _extract_description(prev_code: str, new_code: str) -> str:
    """Diff the two codes to produce a short description of what changed."""
    prev_lines = set(prev_code.splitlines())
    new_lines  = set(new_code.splitlines())
    added   = [l.strip() for l in new_lines - prev_lines if l.strip() and not l.strip().startswith("#")]
    removed = [l.strip() for l in prev_lines - new_lines if l.strip() and not l.strip().startswith("#")]
    parts = []
    if added:
        parts.append("+" + "; +".join(added[:2]))
    if removed:
        parts.append("-" + "; -".join(removed[:2]))
    return " | ".join(parts) if parts else "no change"


# ------------------------------------------------------------------ #
# Results TSV logging
# ------------------------------------------------------------------ #

def _init_tsv() -> None:
    os.makedirs(os.path.dirname(RESULTS_TSV), exist_ok=True)
    if not os.path.exists(RESULTS_TSV):
        with open(RESULTS_TSV, "w") as f:
            f.write("iter\tscore\tsla_rate\tcost\tstatus\tdescription\n")


def _append_tsv(row: Dict) -> None:
    with open(RESULTS_TSV, "a") as f:
        f.write(
            f"{row['iter']}\t{row['score']:.4f}\t{row['sla']:.4f}\t"
            f"{row['cost']:.4f}\t{row['status']}\t{row['description']}\n"
        )


# ------------------------------------------------------------------ #
# AutoResearch Engine
# ------------------------------------------------------------------ #

class AutoResearchEngine:
    def __init__(
        self,
        n_iterations:   int = 6,
        total_steps:    int = 3000,
        seed:           int = 0,
        trial_timeout:  int = 600,
        verbose:        bool = True,
        llm_provider:   str = "groq",      # "groq" | "ollama" | "anthropic" | "gemini"
        llm_model:      str = None,        # None = use provider default
    ):
        self.n_iterations  = n_iterations
        self.total_steps   = total_steps
        self.seed          = seed
        self.trial_timeout = trial_timeout
        self.verbose       = verbose
        self.llm_provider  = llm_provider
        self.llm_model     = llm_model

        self.history: List[Dict] = []
        self.best_score  = FAILURE_SENTINEL
        self.best_code   = ""

    # ── Main loop ─────────────────────────────────────────────────────

    def run(self) -> Tuple[float, str]:
        """
        Run n_iterations of the Karpathy-style loop.
        Returns (best_score, best_experiment_code).
        """
        _init_tsv()
        program_md = _load_program_md()

        # Save a backup of the original experiment.py
        original_code = _load_experiment_code()
        backup_path   = EXPERIMENT_FILE + ".bak"
        shutil.copy2(EXPERIMENT_FILE, backup_path)

        current_code = original_code

        for b in range(1, self.n_iterations + 1):
            t0 = time.time()
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"[AutoResearch] Iteration {b}/{self.n_iterations}")
                print(f"{'='*60}")

            # ── LLM proposes new experiment.py ─────────────────────
            prompt   = _build_prompt(current_code, self.history, b,
                                     self.n_iterations, program_md)
            raw_text = _call_llm(prompt, provider=self.llm_provider,
                                 model=self.llm_model)

            if raw_text is None:
                print("[AutoResearch] LLM unavailable — skipping iteration")
                continue

            # Validate proposed code (retry once on failure)
            ok, result = _validate_code(raw_text)
            if not ok:
                if self.verbose:
                    print(f"[AutoResearch] Code validation failed ({result}), retrying...")
                raw_text2 = _call_llm(prompt, provider=self.llm_provider,
                                      model=self.llm_model)
                if raw_text2:
                    ok, result = _validate_code(raw_text2)
            if not ok:
                print(f"[AutoResearch] Skipping iteration {b}: {result}")
                continue

            proposed_code = result   # cleaned code string

            desc = _extract_description(current_code, proposed_code)
            if self.verbose:
                print(f"[AutoResearch] Proposed change: {desc}")

            # ── Run trial with proposed code ───────────────────────
            score, metrics = run_trial(
                experiment_code=proposed_code,
                total_steps=self.total_steps,
                seed=self.seed,
                timeout=self.trial_timeout,
            )

            elapsed = time.time() - t0
            sla  = metrics.get("mean_sla",  0.0)
            cost = metrics.get("mean_cost", 0.0)

            if score == FAILURE_SENTINEL:
                err = metrics.get("error", "unknown")
                if self.verbose:
                    print(f"[AutoResearch] Trial FAILED: {err}")
                self.history.append({
                    "iter": b, "score": -1.0, "sla": 0.0, "cost": 0.0,
                    "kept": False, "description": f"CRASH: {err}",
                })
                _append_tsv({"iter": b, "score": -1.0, "sla": 0.0,
                              "cost": 0.0, "status": "crash", "description": desc})
                continue

            kept = score > self.best_score
            status_str = "keep" if kept else "discard"

            if self.verbose:
                arrow = "▲" if kept else "▼"
                print(
                    f"[AutoResearch] {arrow} score={score:+.4f}  "
                    f"sla={sla:.3f}  cost={cost:.4f}  "
                    f"→ {status_str.upper()}  ({elapsed:.0f}s)"
                )

            if kept:
                self.best_score = score
                self.best_code  = proposed_code
                current_code    = proposed_code
                # Write kept version back to experiment.py
                with open(EXPERIMENT_FILE, "w") as f:
                    f.write(proposed_code)
                if self.verbose:
                    print(f"[AutoResearch] *** New best: {score:.4f} — experiment.py updated ***")

            self.history.append({
                "iter": b, "score": score, "sla": sla, "cost": cost,
                "kept": kept, "description": desc,
            })
            _append_tsv({
                "iter": b, "score": score, "sla": sla,
                "cost": cost, "status": status_str, "description": desc,
            })

        # ── Final summary ──────────────────────────────────────────
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[AutoResearch] Finished. Best score: {self.best_score:.4f}")
            self._print_history_table()

        # Restore best code to experiment.py (it's already there if kept)
        if not self.best_code:
            # Nothing improved — restore original
            shutil.copy2(backup_path, EXPERIMENT_FILE)
            if self.verbose:
                print("[AutoResearch] No improvement found — original experiment.py restored")

        return self.best_score, self.best_code or original_code

    # ── Helpers ───────────────────────────────────────────────────────

    def _print_history_table(self) -> None:
        print(f"\n{'iter':>4}  {'score':>8}  {'sla':>6}  {'cost':>8}  {'status':>8}  description")
        print("-" * 70)
        for h in self.history:
            status = "KEEP" if h["kept"] else "discard"
            print(
                f"{h['iter']:>4}  {h['score']:>8.4f}  {h['sla']:>6.3f}  "
                f"{h['cost']:>8.4f}  {status:>8}  {h['description']}"
            )

    def get_results_df(self):
        """Return history as a pandas DataFrame (for notebooks)."""
        try:
            import pandas as pd
            return pd.DataFrame(self.history)
        except ImportError:
            return self.history
