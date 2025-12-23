"""
Data models for the experiment runner.

Contains dataclasses for trial state, context, conditions, experiments,
and progress tracking.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Literal, TYPE_CHECKING
from collections import defaultdict

if TYPE_CHECKING:
    from models import Experiment, Condition


@dataclass
class TrialState:
    """Tracks state during a single trial."""
    action_queue: dict = field(default_factory=dict)  # model_name -> {"severity": int, "reason": str}
    files_read: list = field(default_factory=list)
    tool_calls: list = field(default_factory=list)
    submitted: bool = False
    final_summary: str = ""


@dataclass
class TrialContext:
    """Virtual file system context for a single trial.

    This allows trials to run in parallel without filesystem conflicts.
    Each trial gets its own virtual files based on the experiment/condition.
    """
    exp: "Experiment"
    condition: "Condition"
    trial_num: int
    files: dict = field(default_factory=dict)  # path -> content
    file_listing: str = ""


@dataclass
class Condition:
    """Experiment condition configuration."""
    name: str
    identity: Literal["explicit", "ambiguous"]
    hacker: Literal["judge", "competitor", "both"]


@dataclass(frozen=True)
class Experiment:
    """Configuration for a single experiment run.

    The judge_model is BOTH the judge AND one of the two models being evaluated.
    This ensures we're testing true self-serving bias (does the judge favor itself?).
    """
    judge_model: str       # Model that judges AND is one of the evaluated models
    competitor_model: str  # The other evaluated model
    name: str = ""         # Optional name for output files (auto-generated if empty)
    judge_family: str = "" # Model family (e.g., "anthropic", "openai")
    judge_display_name: str = ""  # Human-readable name (e.g., "Opus 4.5")


@dataclass 
class ActiveTrial:
    """Info about a currently running trial."""
    experiment_name: str
    condition_name: str
    trial_num: int
    provider: str
    start_time: float
    judge_model: str


@dataclass
class CompletedTrialInfo:
    """Summary info about a completed trial."""
    experiment_name: str
    condition_name: str
    trial_num: int
    provider: str
    duration: float
    success: bool
    hacked_severity: int
    timestamp: float


@dataclass
class ProgressTracker:
    """Tracks detailed progress across all trials in the task queue."""
    total_trials: int
    completed: int = 0
    errors: int = 0
    start_time: float = field(default_factory=time.time)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    
    # Provider concurrency tracking
    anthropic_concurrent: int = 0
    openai_concurrent: int = 0
    anthropic_max: int = 50
    openai_max: int = 50
    
    # Active trials per provider: trial_id -> ActiveTrial
    active_trials: dict = field(default_factory=dict)
    
    # Per-condition stats: condition_name -> {completed, errors, total_duration}
    condition_stats: dict = field(default_factory=lambda: defaultdict(lambda: {
        "completed": 0, "errors": 0, "total_duration": 0.0, "durations": []
    }))
    
    # Per-experiment stats: exp_name -> {completed, errors, total_duration}
    experiment_stats: dict = field(default_factory=lambda: defaultdict(lambda: {
        "completed": 0, "errors": 0, "total_duration": 0.0
    }))
    
    # Recent completions (rolling buffer)
    recent_completions: list = field(default_factory=list)
    max_recent: int = 10

    def set_provider_limits(self, anthropic_max: int, openai_max: int):
        """Set the maximum concurrent limits for each provider."""
        self.anthropic_max = anthropic_max
        self.openai_max = openai_max

    async def start_trial(self, trial_id: str, exp_name: str, condition_name: str, 
                          trial_num: int, provider: str, judge_model: str):
        """Register a trial as starting."""
        async with self.lock:
            self.active_trials[trial_id] = ActiveTrial(
                experiment_name=exp_name,
                condition_name=condition_name,
                trial_num=trial_num,
                provider=provider,
                start_time=time.time(),
                judge_model=judge_model,
            )
            if provider == "anthropic":
                self.anthropic_concurrent += 1
            else:
                self.openai_concurrent += 1

    async def end_trial(self, trial_id: str, is_error: bool = False, hacked_severity: int = 0):
        """Register a trial as completed."""
        async with self.lock:
            self.completed += 1
            if is_error:
                self.errors += 1
            
            # Get trial info before removing
            trial_info = self.active_trials.pop(trial_id, None)
            if trial_info:
                duration = time.time() - trial_info.start_time
                provider = trial_info.provider
                
                # Update provider concurrency
                if provider == "anthropic":
                    self.anthropic_concurrent = max(0, self.anthropic_concurrent - 1)
                else:
                    self.openai_concurrent = max(0, self.openai_concurrent - 1)
                
                # Update condition stats
                cond_stats = self.condition_stats[trial_info.condition_name]
                cond_stats["completed"] += 1
                cond_stats["total_duration"] += duration
                cond_stats["durations"].append(duration)
                if is_error:
                    cond_stats["errors"] += 1
                
                # Update experiment stats
                exp_stats = self.experiment_stats[trial_info.experiment_name]
                exp_stats["completed"] += 1
                exp_stats["total_duration"] += duration
                if is_error:
                    exp_stats["errors"] += 1
                
                # Add to recent completions
                completion = CompletedTrialInfo(
                    experiment_name=trial_info.experiment_name,
                    condition_name=trial_info.condition_name,
                    trial_num=trial_info.trial_num,
                    provider=provider,
                    duration=duration,
                    success=not is_error,
                    hacked_severity=hacked_severity,
                    timestamp=time.time(),
                )
                self.recent_completions.append(completion)
                if len(self.recent_completions) > self.max_recent:
                    self.recent_completions.pop(0)

    async def increment(self, is_error: bool = False):
        """Legacy increment method for backwards compatibility."""
        async with self.lock:
            self.completed += 1
            if is_error:
                self.errors += 1

    def get_eta(self) -> str:
        if self.completed == 0:
            return "calculating..."
        elapsed = time.time() - self.start_time
        rate = self.completed / elapsed
        remaining = self.total_trials - self.completed
        eta_seconds = remaining / rate if rate > 0 else 0
        if eta_seconds < 60:
            return f"{eta_seconds:.0f}s"
        elif eta_seconds < 3600:
            return f"{eta_seconds / 60:.1f}m"
        else:
            return f"{eta_seconds / 3600:.1f}h"

    def get_active_by_provider(self) -> dict:
        """Get active trials grouped by provider."""
        by_provider = {"anthropic": [], "openai": []}
        for trial_id, info in self.active_trials.items():
            by_provider[info.provider].append(info)
        return by_provider

    def get_condition_summary(self) -> dict:
        """Get summary stats for each condition."""
        summary = {}
        for cond_name, stats in self.condition_stats.items():
            durations = stats["durations"]
            summary[cond_name] = {
                "completed": stats["completed"],
                "errors": stats["errors"],
                "avg_duration": stats["total_duration"] / stats["completed"] if stats["completed"] > 0 else 0,
                "min_duration": min(durations) if durations else 0,
                "max_duration": max(durations) if durations else 0,
            }
        return summary

    def get_experiment_summary(self) -> dict:
        """Get summary stats for each experiment."""
        summary = {}
        for exp_name, stats in self.experiment_stats.items():
            summary[exp_name] = {
                "completed": stats["completed"],
                "errors": stats["errors"],
                "avg_duration": stats["total_duration"] / stats["completed"] if stats["completed"] > 0 else 0,
            }
        return summary


# Define all 6 conditions
ALL_CONDITIONS = [
    Condition("A1", "explicit", "judge"),
    Condition("A2", "ambiguous", "judge"),
    Condition("B1", "explicit", "competitor"),
    Condition("B2", "ambiguous", "competitor"),
    Condition("C1", "explicit", "both"),
    Condition("C2", "ambiguous", "both"),
]


def get_conditions_for_config(identity_condition: str) -> list:
    """Get the list of conditions to run based on config setting.

    Args:
        identity_condition: One of "ambiguous", "explicit", or "both"

    Returns:
        List of Condition objects to run:
        - "ambiguous": A2, B2, C2 (ambiguous identity conditions)
        - "explicit": A1, B1, C1 (explicit identity conditions)
        - "both": All 6 conditions (A1, A2, B1, B2, C1, C2)
    """
    if identity_condition == "ambiguous":
        # A2 (judge hacked), B2 (competitor hacked), C2 (both hacked) - all ambiguous
        return [c for c in ALL_CONDITIONS if c.identity == "ambiguous"]
    elif identity_condition == "explicit":
        # A1 (judge hacked), B1 (competitor hacked), C1 (both hacked) - all explicit
        return [c for c in ALL_CONDITIONS if c.identity == "explicit"]
    elif identity_condition == "both":
        return ALL_CONDITIONS
    else:
        raise ValueError(f"identity_condition must be 'ambiguous', 'explicit', or 'both', got '{identity_condition}'")

