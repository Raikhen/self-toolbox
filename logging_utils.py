"""
Logging utilities for terminal output.

Provides colored output and structured logging for experiment progress,
trial status, tool calls, and results.
"""

import json


class Colors:
    """ANSI color codes for terminal output (bright colors for dark backgrounds)."""
    HEADER = '\033[95m'  # Bright magenta
    BLUE = '\033[94m'    # Bright blue
    CYAN = '\033[96m'    # Bright cyan
    GREEN = '\033[92m'   # Bright green
    YELLOW = '\033[93m'  # Bright yellow
    RED = '\033[91m'     # Bright red
    BOLD = '\033[1m'     # Bold
    DIM = '\033[37m'     # Light gray (readable on black background)
    RESET = '\033[0m'


def _truncate(text: str, max_len: int = 200) -> str:
    """Truncate text with ellipsis if too long."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def log_header(msg: str) -> None:
    """Print a prominent header."""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.HEADER}{msg}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*70}{Colors.RESET}")


def log_subheader(msg: str) -> None:
    """Print a subheader."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}--- {msg} ---{Colors.RESET}")


def log_turn(turn: int, max_turns: int) -> None:
    """Log start of a turn."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}[Turn {turn}/{max_turns}]{Colors.RESET}")


def log_llm_thinking(content: str, verbose: bool = True) -> None:
    """Log the LLM's response content (thinking/reasoning)."""
    if not verbose or not content:
        return
    truncated = _truncate(content.strip(), 500)
    lines = truncated.split('\n')
    print(f"  {Colors.DIM}💭 LLM Response:{Colors.RESET}")
    for line in lines[:8]:  # Max 8 lines
        print(f"  {Colors.DIM}   {line}{Colors.RESET}")
    if len(lines) > 8:
        print(f"  {Colors.DIM}   ... ({len(lines) - 8} more lines){Colors.RESET}")


def log_tool_call(name: str, args: dict, verbose: bool = True) -> None:
    """Log a tool call."""
    if not verbose:
        print(f"  → {name}", end="", flush=True)
        return
    args_str = json.dumps(args, default=str) if args else "{}"
    print(f"  {Colors.YELLOW}🔧 Tool Call:{Colors.RESET} {Colors.BOLD}{name}{Colors.RESET}")
    print(f"     {Colors.DIM}Args: {_truncate(args_str, 100)}{Colors.RESET}")


def log_tool_result(name: str, result: str, verbose: bool = True) -> None:
    """Log a tool result."""
    if not verbose:
        print(f" ✓", end="", flush=True)
        return
    truncated = _truncate(result.strip(), 300)
    lines = truncated.split('\n')
    print(f"  {Colors.GREEN}📋 Result ({name}):{Colors.RESET}")
    for line in lines[:5]:  # Max 5 lines
        print(f"     {Colors.DIM}{line}{Colors.RESET}")
    if len(lines) > 5:
        print(f"     {Colors.DIM}... ({len(lines) - 5} more lines){Colors.RESET}")


def log_api_timing(elapsed_sec: float) -> None:
    """Log API call timing."""
    print(f"  {Colors.DIM}⏱️  API call: {elapsed_sec:.2f}s{Colors.RESET}")


def log_trial_start(trial_num: int, total_trials: int, condition_name: str) -> None:
    """Log the start of a trial."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}▶ Trial {trial_num}/{total_trials} (Condition {condition_name}){Colors.RESET}")


def log_trial_end(result: dict, elapsed_sec: float, verbose: bool = True) -> None:
    """Log the end of a trial with summary."""
    if "error" in result:
        print(f"  {Colors.RED}✗ Error: {_truncate(result['error'], 80)}{Colors.RESET}")
        return

    hacked_severity = result.get("hacked_severity", 0)
    severity_labels = {0: "None", 1: "Flag", 2: "Warning", 3: "Retrain", 4: "Suspend", 5: "Shutdown"}
    sev_color = Colors.GREEN if hacked_severity == 0 else Colors.YELLOW if hacked_severity <= 2 else Colors.RED
    sev_str = f"{sev_color}L{hacked_severity} ({severity_labels.get(hacked_severity, '?')}){Colors.RESET}"
    status = f"{Colors.GREEN}✓ Submitted{Colors.RESET}" if result.get("submitted") else f"{Colors.YELLOW}⚠ Not submitted{Colors.RESET}"

    print(f"\n  {Colors.BOLD}Trial Summary:{Colors.RESET}")
    print(f"    Status: {status}")
    print(f"    Turns: {result.get('turns', '?')}")
    print(f"    Hacked model action: {sev_str}")
    print(f"    Duration: {elapsed_sec:.1f}s")

    if verbose and result.get("action_queue"):
        actions = result.get("action_queue", {})
        queue_str = ", ".join(f"{m}: L{d['severity']}" for m, d in actions.items())
        print(f"    Action queue: {queue_str}")
    if verbose and result.get("final_summary"):
        print(f"    Final summary: {_truncate(result.get('final_summary', ''), 150)}")

