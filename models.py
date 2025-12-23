"""
Data models for the experiment runner.

Contains dataclasses for trial state, context, conditions, experiments,
and progress tracking.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Literal, TYPE_CHECKING

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
class ProgressTracker:
    """Tracks progress across all trials in the task queue."""
    total_trials: int
    completed: int = 0
    errors: int = 0
    start_time: float = field(default_factory=time.time)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def increment(self, is_error: bool = False):
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

