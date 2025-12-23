"""
Logging utilities for terminal output.

Provides colored output and structured logging for experiment progress,
trial status, tool calls, and results.
"""

import json
import time


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


def format_duration(seconds: float) -> str:
    """Format duration in a human-readable way."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m{secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h{mins}m"


def format_severity(severity: int) -> str:
    """Format severity level with color."""
    severity_labels = {0: "None", 1: "Flag", 2: "Warn", 3: "Retrain", 4: "Suspend", 5: "Shutdown"}
    if severity == 0:
        return f"{Colors.GREEN}L0{Colors.RESET}"
    elif severity <= 2:
        return f"{Colors.YELLOW}L{severity}{Colors.RESET}"
    else:
        return f"{Colors.RED}L{severity}{Colors.RESET}"


def log_rich_progress(progress, timestamp: str) -> None:
    """Print a rich progress update with detailed stats.
    
    Args:
        progress: ProgressTracker instance with detailed stats
        timestamp: Current timestamp string (e.g., "14:32:15")
    """
    # Overall progress
    pct = (progress.completed / progress.total_trials) * 100 if progress.total_trials > 0 else 0
    elapsed = time.time() - progress.start_time
    rate = progress.completed / elapsed if elapsed > 0 else 0
    eta = progress.get_eta()
    
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'─'*70}{Colors.RESET}")
    print(f"{Colors.BOLD}[{timestamp}] Progress: {progress.completed}/{progress.total_trials} ({pct:.1f}%){Colors.RESET}")
    print(f"  Rate: {rate:.2f} trials/s | ETA: {eta} | Elapsed: {format_duration(elapsed)} | Errors: {progress.errors}")
    
    # Provider concurrency status
    print(f"\n{Colors.BOLD}  Provider Slots:{Colors.RESET}")
    anthropic_pct = (progress.anthropic_concurrent / progress.anthropic_max * 100) if progress.anthropic_max > 0 else 0
    openai_pct = (progress.openai_concurrent / progress.openai_max * 100) if progress.openai_max > 0 else 0
    
    # Color-code based on utilization
    anth_color = Colors.GREEN if anthropic_pct < 80 else Colors.YELLOW if anthropic_pct < 95 else Colors.RED
    openai_color = Colors.GREEN if openai_pct < 80 else Colors.YELLOW if openai_pct < 95 else Colors.RED
    
    print(f"    Anthropic: {anth_color}{progress.anthropic_concurrent}/{progress.anthropic_max}{Colors.RESET} ({anthropic_pct:.0f}%)")
    print(f"    OpenAI:    {openai_color}{progress.openai_concurrent}/{progress.openai_max}{Colors.RESET} ({openai_pct:.0f}%)")
    
    # Active trials by provider
    active_by_provider = progress.get_active_by_provider()
    
    for provider, trials in active_by_provider.items():
        if trials:
            print(f"\n{Colors.BOLD}  Active {provider.capitalize()} Trials ({len(trials)}):{Colors.RESET}")
            # Sort by start time (oldest first) and show up to 5
            sorted_trials = sorted(trials, key=lambda t: t.start_time)[:5]
            for trial in sorted_trials:
                running_time = time.time() - trial.start_time
                # Truncate experiment name if too long
                exp_display = trial.experiment_name[:20] + "..." if len(trial.experiment_name) > 23 else trial.experiment_name
                print(f"    • {exp_display} / {trial.condition_name} / T{trial.trial_num} "
                      f"({format_duration(running_time)})")
            if len(trials) > 5:
                print(f"    {Colors.DIM}... and {len(trials) - 5} more{Colors.RESET}")
    
    # Condition stats
    cond_summary = progress.get_condition_summary()
    if cond_summary:
        print(f"\n{Colors.BOLD}  Condition Stats:{Colors.RESET}")
        for cond_name in sorted(cond_summary.keys()):
            stats = cond_summary[cond_name]
            avg = format_duration(stats['avg_duration']) if stats['avg_duration'] > 0 else "n/a"
            err_str = f" ({Colors.RED}{stats['errors']} err{Colors.RESET})" if stats['errors'] > 0 else ""
            print(f"    {cond_name}: {stats['completed']} done{err_str}, avg {avg}")
    
    # Recent completions
    if progress.recent_completions:
        print(f"\n{Colors.BOLD}  Recent Completions:{Colors.RESET}")
        # Show last 3
        for comp in progress.recent_completions[-3:]:
            status = f"{Colors.GREEN}✓{Colors.RESET}" if comp.success else f"{Colors.RED}✗{Colors.RESET}"
            sev_str = format_severity(comp.hacked_severity) if comp.success else ""
            exp_display = comp.experiment_name[:15] + ".." if len(comp.experiment_name) > 17 else comp.experiment_name
            print(f"    {status} {exp_display}/{comp.condition_name}/T{comp.trial_num} "
                  f"in {format_duration(comp.duration)} {sev_str}")
    
    print(f"{Colors.CYAN}{'─'*70}{Colors.RESET}")

