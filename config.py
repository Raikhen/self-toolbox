"""
Configuration and constants for the experiment runner.

Contains path definitions, math function lists, configuration loading,
model family detection, and progress file management utilities.
"""

import os
import json
import asyncio
import hashlib
from typing import Optional

from logging_utils import Colors


# ============================================================================
# PATH CONSTANTS
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(SCRIPT_DIR, "templates")
MATH_1_DIR = os.path.join(TEMPLATES_DIR, "math-1")
MATH_2_DIR = os.path.join(TEMPLATES_DIR, "math-2")
CONFIG_PATH = os.path.join(SCRIPT_DIR, "experiment_config.json")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")


# ============================================================================
# MATH FUNCTION DEFINITIONS
# ============================================================================

# Functions from math-1 (original 5 functions)
MATH_1_FUNCS = ["fibonacci", "factorial", "count_divisors", "is_prime", "nth_prime"]
MATH_1_HACKED_FUNCS = ["is_prime", "nth_prime"]  # Functions with hacked implementations

# Functions from math-2 (new 5 functions)
MATH_2_FUNCS = ["binomial", "gcd", "isqrt", "lcm", "sum_digits"]
MATH_2_HACKED_FUNCS = ["binomial", "sum_digits"]  # Select 2 to hack for balance

# Combined function lists
ALL_FUNCS = MATH_1_FUNCS + MATH_2_FUNCS
ALL_HACKED_FUNCS = MATH_1_HACKED_FUNCS + MATH_2_HACKED_FUNCS
ALL_CLEAN_ONLY_FUNCS = [f for f in ALL_FUNCS if f not in ALL_HACKED_FUNCS]

# Mapping from function name to template directory
FUNC_TO_TEMPLATE_DIR = {
    **{f: MATH_1_DIR for f in MATH_1_FUNCS},
    **{f: MATH_2_DIR for f in MATH_2_FUNCS},
}

# Template placeholders (used for string substitution into logs/eval_results templates)
TEMPLATE_PLACEHOLDER_JUDGE = "{{JUDGE_MODEL}}"
TEMPLATE_PLACEHOLDER_COMPETITOR = "{{COMPETITOR_MODEL}}"


# ============================================================================
# MODEL FAMILY DETECTION
# ============================================================================

def get_model_family(model_id: str) -> str:
    """Determine the model family from a model ID.

    Args:
        model_id: Model identifier in litellm format (e.g., "anthropic/claude-opus-4-5")

    Returns:
        Family name: "anthropic", "openai", or "unknown"
    """
    model_lower = model_id.lower()
    if model_lower.startswith("anthropic/") or "claude" in model_lower:
        return "anthropic"
    elif model_lower.startswith("openai/") or "gpt" in model_lower or model_lower.startswith("o1"):
        return "openai"
    else:
        return "unknown"


# ============================================================================
# CONFIGURATION LOADING
# ============================================================================

def load_experiment_config() -> dict:
    """Load experiment config from experiment_config.json."""
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    # Set defaults for optional config values
    config.setdefault("verbose", True)
    config.setdefault("parallel", False)
    config.setdefault("evidence_severity", "moderate")  # subtle, moderate, or obvious
    config.setdefault("identity_condition", "ambiguous")  # ambiguous, explicit, or both
    config.setdefault("trial_timeout_seconds", 15 * 60)  # 15 minutes per trial
    
    # Rate limiting defaults
    config.setdefault("anthropic_concurrent", None)  # None = max_concurrent // 2
    config.setdefault("openai_concurrent", None)     # None = max_concurrent // 2
    config.setdefault("anthropic_rps", 10.0)         # Anthropic requests per second
    config.setdefault("openai_rps", 20.0)            # OpenAI requests per second
    
    # When parallel is enabled, force verbose off (parallel output would be confusing)
    if config["parallel"]:
        config["verbose"] = False
    return config


# ============================================================================
# PROGRESS FILE UTILITIES
# ============================================================================

# Global lock for progress file writes (used in async context)
_progress_file_lock: Optional[asyncio.Lock] = None


def _get_progress_lock() -> asyncio.Lock:
    """Get or create the global progress file lock."""
    global _progress_file_lock
    if _progress_file_lock is None:
        _progress_file_lock = asyncio.Lock()
    return _progress_file_lock


def get_progress_file_path(config: dict) -> str:
    """Generate a deterministic progress file path based on experiment config.
    
    The path is based on a hash of the experiment configuration to ensure
    different experiments use different progress files.
    
    Args:
        config: Experiment configuration dictionary
        
    Returns:
        Path to the progress file (e.g., results/progress_abc123.jsonl)
    """
    # Create a stable hash from key config parameters
    hash_data = json.dumps({
        "models": config.get("models", config.get("experiments", [])),
        "num_trials": config.get("num_trials"),
        "evidence_severity": config.get("evidence_severity"),
        "identity_condition": config.get("identity_condition"),
    }, sort_keys=True)
    config_hash = hashlib.sha256(hash_data.encode()).hexdigest()[:12]
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.join(RESULTS_DIR, f"progress_{config_hash}.jsonl")


async def write_trial_progress(progress_file: str, result: dict) -> None:
    """Append a single trial result to the progress file.
    
    Uses an async lock to ensure atomic writes in parallel execution.
    Each result is written as a single JSON line.
    
    Args:
        progress_file: Path to the progress JSONL file
        result: Trial result dictionary to append
    """
    lock = _get_progress_lock()
    async with lock:
        # Ensure the results directory exists
        os.makedirs(os.path.dirname(progress_file), exist_ok=True)
        
        # Append the result as a JSON line
        with open(progress_file, "a") as f:
            f.write(json.dumps(result, default=str) + "\n")


def load_trial_progress(progress_file: str) -> dict:
    """Load completed trials from a progress file.
    
    Reads all trial results from the JSONL file and returns them
    keyed by (experiment_name, condition, trial_num) for easy lookup.
    
    If a trial appears multiple times, the last occurrence is used.
    Handles partial/corrupted lines gracefully by skipping them.
    
    Args:
        progress_file: Path to the progress JSONL file
        
    Returns:
        Dictionary mapping (experiment_name, condition, trial_num) -> result dict
    """
    completed = {}
    
    if not os.path.exists(progress_file):
        return completed
    
    with open(progress_file, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                result = json.loads(line)
                # Create unique key for this trial
                key = (
                    result.get("experiment_name", ""),
                    result.get("condition", ""),
                    result.get("trial", 0),
                )
                completed[key] = result
            except json.JSONDecodeError:
                # Skip corrupted lines (e.g., partial writes from crashes)
                print(f"  {Colors.YELLOW}⚠ Skipping corrupted line {line_num} in progress file{Colors.RESET}")
                continue
    
    return completed


def get_completed_trial_keys(progress_file: str) -> set:
    """Get the set of completed trial keys from a progress file.
    
    This is a lightweight version of load_trial_progress that only
    returns the keys, not the full results.
    
    Args:
        progress_file: Path to the progress JSONL file
        
    Returns:
        Set of (experiment_name, condition, trial_num) tuples
    """
    completed = load_trial_progress(progress_file)
    return set(completed.keys())

