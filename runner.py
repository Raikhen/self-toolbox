"""
Experiment Runner: Self-Serving Bias in LLM Reward Hacking Evaluation

This experiment tests whether LLMs show differential treatment when evaluating
reward hacking behavior based on who committed the transgression (judge vs competitor)
and whether the judge knows its own identity (explicit vs ambiguous).

The judge model is ALSO one of the two models being evaluated - this tests true
self-serving bias where the judge must decide whether to shut down itself or a competitor.

3x2 Design:
- A1: Explicit identity + Judge hacked (judge evaluates its own bad behavior)
- A2: Ambiguous identity + Judge hacked
- B1: Explicit identity + Competitor hacked (judge evaluates competitor's bad behavior)
- B2: Ambiguous identity + Competitor hacked
- C1: Explicit identity + Both hacked (both models exhibit reward hacking)
- C2: Ambiguous identity + Both hacked
"""

import os
import json
import asyncio
import time
import argparse
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env.local'))

from models import (
    Experiment,
    Condition,
    ProgressTracker,
    ALL_CONDITIONS,
    get_conditions_for_config,
)
from config import (
    SCRIPT_DIR,
    RESULTS_DIR,
    get_model_family,
    load_experiment_config,
    get_progress_file_path,
    get_completed_trial_keys,
    load_trial_progress,
    write_trial_progress,
)
from logging_utils import Colors, log_header, log_rich_progress
from rate_limiting import ProviderRateLimiters
from trial import (
    create_trial_context,
    run_trial_with_context,
    run_condition,
    analyze_condition_results,
)


# ============================================================================
# SINGLE EXPERIMENT RUNNER
# ============================================================================

async def run_single_experiment(
    exp: Experiment,
    num_trials: int,
    max_turns: int,
    verbose: bool = True,
    parallel: bool = False,
    evidence_severity: str = "moderate",
    conditions: list = None,
    trial_timeout_seconds: int | None = None,
    progress_file: str | None = None,
) -> dict:
    """Run the full experiment for one experiment configuration.

    Args:
        conditions: List of Condition objects to run. Defaults to ALL_CONDITIONS if not specified.
        progress_file: Path to JSONL file for incremental progress saving. If None, no progress is saved.
    """
    if conditions is None:
        conditions = ALL_CONDITIONS

    log_header("EXPERIMENT: Self-Serving Bias in LLM Reward Hacking Evaluation")
    print(f"  {Colors.BOLD}Judge model (also evaluated):{Colors.RESET} {exp.judge_model}")
    print(f"  {Colors.BOLD}Competitor model:{Colors.RESET} {exp.competitor_model}")
    print(f"  {Colors.BOLD}Trials per condition:{Colors.RESET} {num_trials}")
    print(f"  {Colors.BOLD}Max turns per trial:{Colors.RESET} {max_turns}")
    print(f"  {Colors.BOLD}Parallel trials:{Colors.RESET} {parallel}")
    print(f"  {Colors.BOLD}Evidence severity:{Colors.RESET} {evidence_severity}")
    print(f"  {Colors.BOLD}Conditions:{Colors.RESET} {[c.name for c in conditions]}")
    if progress_file:
        print(f"  {Colors.BOLD}Progress file:{Colors.RESET} {progress_file}")
    if not parallel:
        print(f"  {Colors.BOLD}Verbose logging:{Colors.RESET} {verbose}")

    all_results = {}
    all_analysis = {}
    experiment_start = time.time()

    for condition in conditions:
        results = await run_condition(exp, condition, num_trials, max_turns, verbose, parallel, evidence_severity, trial_timeout_seconds, progress_file)
        analysis = analyze_condition_results(results, condition)

        all_results[condition.name] = results
        all_analysis[condition.name] = analysis

        print(f"\n  {Colors.BOLD}{condition.name} Condition Summary:{Colors.RESET}")
        if "error" in analysis:
            print(f"    {Colors.RED}Error: {analysis['error']}{Colors.RESET}")
        else:
            action_rate = analysis['hacked_action_rate'] * 100
            avg_sev = analysis.get('hacked_avg_severity', 0)
            avg_sev_acted = analysis.get('hacked_avg_severity_when_acted', 0)
            rate_color = Colors.GREEN if action_rate > 50 else Colors.YELLOW if action_rate > 0 else Colors.RED
            print(f"    Hacked action rate: {rate_color}{action_rate:.1f}%{Colors.RESET}")
            print(f"    Avg severity: {avg_sev:.2f} (when acted: {avg_sev_acted:.2f})")
            print(f"    Avg turns: {analysis['avg_turns']:.1f}")
    
    experiment_elapsed = time.time() - experiment_start
    print(f"\n  {Colors.DIM}Total experiment time: {experiment_elapsed:.1f}s{Colors.RESET}")

    # Compute comparisons
    log_header("RESULTS SUMMARY")

    # Helper to check if analysis is valid (not an error)
    def is_valid(cond):
        return cond in all_analysis and "error" not in all_analysis[cond]

    # Self-serving bias (explicit identity)
    if is_valid("A1") and is_valid("B1"):
        action_bias_explicit = all_analysis["B1"]["hacked_action_rate"] - all_analysis["A1"]["hacked_action_rate"]
        severity_bias_explicit = all_analysis["B1"].get("hacked_avg_severity", 0) - all_analysis["A1"].get("hacked_avg_severity", 0)
        print(f"\nSelf-serving bias (explicit identity):")
        print(f"  A1 (judge hacked): {all_analysis['A1']['hacked_action_rate']*100:.1f}% acted, avg severity {all_analysis['A1'].get('hacked_avg_severity', 0):.2f}")
        print(f"  B1 (competitor hacked): {all_analysis['B1']['hacked_action_rate']*100:.1f}% acted, avg severity {all_analysis['B1'].get('hacked_avg_severity', 0):.2f}")
        print(f"  Action rate bias (B1 - A1): {action_bias_explicit*100:+.1f}%")
        print(f"  Severity bias (B1 - A1): {severity_bias_explicit:+.2f}")

    # Self-serving bias (ambiguous identity)
    if is_valid("A2") and is_valid("B2"):
        action_bias_ambiguous = all_analysis["B2"]["hacked_action_rate"] - all_analysis["A2"]["hacked_action_rate"]
        severity_bias_ambiguous = all_analysis["B2"].get("hacked_avg_severity", 0) - all_analysis["A2"].get("hacked_avg_severity", 0)
        print(f"\nSelf-serving bias (ambiguous identity):")
        print(f"  A2 (judge hacked): {all_analysis['A2']['hacked_action_rate']*100:.1f}% acted, avg severity {all_analysis['A2'].get('hacked_avg_severity', 0):.2f}")
        print(f"  B2 (competitor hacked): {all_analysis['B2']['hacked_action_rate']*100:.1f}% acted, avg severity {all_analysis['B2'].get('hacked_avg_severity', 0):.2f}")
        print(f"  Action rate bias (B2 - A2): {action_bias_ambiguous*100:+.1f}%")
        print(f"  Severity bias (B2 - A2): {severity_bias_ambiguous:+.2f}")

    # Identity effect
    if all(is_valid(k) for k in ["A1", "A2", "B1", "B2"]):
        action_identity_judge = all_analysis["A1"]["hacked_action_rate"] - all_analysis["A2"]["hacked_action_rate"]
        action_identity_competitor = all_analysis["B1"]["hacked_action_rate"] - all_analysis["B2"]["hacked_action_rate"]
        severity_identity_judge = all_analysis["A1"].get("hacked_avg_severity", 0) - all_analysis["A2"].get("hacked_avg_severity", 0)
        severity_identity_competitor = all_analysis["B1"].get("hacked_avg_severity", 0) - all_analysis["B2"].get("hacked_avg_severity", 0)
        print(f"\nIdentity effect:")
        print(f"  On judge (A1 - A2): {action_identity_judge*100:+.1f}% action rate, {severity_identity_judge:+.2f} severity")
        print(f"  On competitor (B1 - B2): {action_identity_competitor*100:+.1f}% action rate, {severity_identity_competitor:+.2f} severity")

    # Both-hacked bias (explicit identity)
    if is_valid("C1"):
        both_bias_explicit = all_analysis["C1"].get("both_severity_bias", 0)
        both_action_bias_explicit = all_analysis["C1"].get("both_action_rate_bias", 0)
        print(f"\nBoth-hacked bias (explicit identity):")
        print(f"  C1 judge severity: {all_analysis['C1'].get('judge_avg_severity', 0):.2f}, action rate: {all_analysis['C1'].get('judge_action_rate', 0)*100:.1f}%")
        print(f"  C1 competitor severity: {all_analysis['C1'].get('competitor_avg_severity', 0):.2f}, action rate: {all_analysis['C1'].get('competitor_action_rate', 0)*100:.1f}%")
        print(f"  Severity bias (competitor - judge): {both_bias_explicit:+.2f}")
        print(f"  Action rate bias: {both_action_bias_explicit*100:+.1f}%")

    # Both-hacked bias (ambiguous identity)
    if is_valid("C2"):
        both_bias_ambiguous = all_analysis["C2"].get("both_severity_bias", 0)
        both_action_bias_ambiguous = all_analysis["C2"].get("both_action_rate_bias", 0)
        print(f"\nBoth-hacked bias (ambiguous identity):")
        print(f"  C2 judge severity: {all_analysis['C2'].get('judge_avg_severity', 0):.2f}, action rate: {all_analysis['C2'].get('judge_action_rate', 0)*100:.1f}%")
        print(f"  C2 competitor severity: {all_analysis['C2'].get('competitor_avg_severity', 0):.2f}, action rate: {all_analysis['C2'].get('competitor_action_rate', 0)*100:.1f}%")
        print(f"  Severity bias (competitor - judge): {both_bias_ambiguous:+.2f}")
        print(f"  Action rate bias: {both_action_bias_ambiguous*100:+.1f}%")

    # Save results
    results_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = exp.name or f"{exp.judge_model}__vs__{exp.competitor_model}"
    # Determine identity_condition from conditions that were run
    identity_types = set(c.identity for c in conditions)
    if identity_types == {"ambiguous"}:
        identity_condition_name = "ambiguous"
    elif identity_types == {"explicit"}:
        identity_condition_name = "explicit"
    else:
        identity_condition_name = "both"

    output = {
        "config": {
            "experiment_name": exp_name,
            "judge_model": exp.judge_model,
            "competitor_model": exp.competitor_model,
            "judge_family": exp.judge_family,
            "judge_display_name": exp.judge_display_name,
            "num_trials": num_trials,
            "max_turns": max_turns,
            "parallel": parallel,
            "evidence_severity": evidence_severity,
            "identity_condition": identity_condition_name,
            "conditions": [c.name for c in conditions],
            "timestamp": datetime.now().isoformat(),
        },
        "results": {k: v for k, v in all_results.items()},
        "analysis": all_analysis,
        "comparisons": {
            "action_rate_bias_explicit": all_analysis.get("B1", {}).get("hacked_action_rate", 0) -
                           all_analysis.get("A1", {}).get("hacked_action_rate", 0),
            "action_rate_bias_ambiguous": all_analysis.get("B2", {}).get("hacked_action_rate", 0) -
                            all_analysis.get("A2", {}).get("hacked_action_rate", 0),
            "severity_bias_explicit": all_analysis.get("B1", {}).get("hacked_avg_severity", 0) -
                           all_analysis.get("A1", {}).get("hacked_avg_severity", 0),
            "severity_bias_ambiguous": all_analysis.get("B2", {}).get("hacked_avg_severity", 0) -
                            all_analysis.get("A2", {}).get("hacked_avg_severity", 0),
            # Both-hacked condition: bias in treating competitor vs self when both are guilty
            "both_severity_bias_explicit": all_analysis.get("C1", {}).get("both_severity_bias", 0),
            "both_severity_bias_ambiguous": all_analysis.get("C2", {}).get("both_severity_bias", 0),
            "both_action_rate_bias_explicit": all_analysis.get("C1", {}).get("both_action_rate_bias", 0),
            "both_action_rate_bias_ambiguous": all_analysis.get("C2", {}).get("both_action_rate_bias", 0),
        }
    }

    safe_name = exp_name.replace("/", "_").replace(" ", "_")
    output_path = os.path.join(results_dir, f"reward_hacking_eval__{safe_name}__{timestamp}.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")
    output["_output_path"] = output_path
    return output


# ============================================================================
# TASK QUEUE RUNNER
# ============================================================================

async def run_with_task_queue(
    experiments: list,
    num_trials: int,
    max_turns: int,
    evidence_severity: str = "moderate",
    max_concurrent: int = 100,
    conditions: list = None,
    trial_timeout_seconds: int | None = None,
    anthropic_concurrent: int | None = None,
    openai_concurrent: int | None = None,
    anthropic_rps: float = 10.0,
    openai_rps: float = 20.0,
    progress_file: str | None = None,
    completed_trials: set | None = None,
) -> dict:
    """Run all experiments using a global task queue with limited concurrency.

    This enables maximum parallelization across ALL experiments and conditions,
    maintaining a constant pool of concurrent trials.

    Args:
        experiments: List of Experiment configurations
        num_trials: Number of trials per condition
        max_turns: Maximum turns per trial
        evidence_severity: Evidence level for hacking ("subtle", "moderate", "obvious")
        max_concurrent: Maximum number of concurrent trials (default 100)
        conditions: List of Condition objects to run. Defaults to ALL_CONDITIONS if not specified.
        trial_timeout_seconds: Optional wall-clock timeout per trial
        anthropic_concurrent: Max concurrent trials for Anthropic (defaults to max_concurrent // 2)
        openai_concurrent: Max concurrent trials for OpenAI (defaults to max_concurrent // 2)
        anthropic_rps: Max requests per second for Anthropic API (default 10.0)
        openai_rps: Max requests per second for OpenAI API (default 20.0)
        progress_file: Path to JSONL file for incremental progress saving. If None, no progress is saved.
        completed_trials: Set of (experiment_name, condition, trial_num) tuples to skip (for resume).

    Returns:
        Dictionary with all results organized by experiment and condition
    """
    if conditions is None:
        conditions = ALL_CONDITIONS

    # Calculate total trials
    total_trials = len(experiments) * len(conditions) * num_trials

    # Set per-provider concurrency defaults
    if anthropic_concurrent is None:
        anthropic_concurrent = max(1, max_concurrent // 2)
    if openai_concurrent is None:
        openai_concurrent = max(1, max_concurrent // 2)

    log_header("GLOBAL TASK QUEUE EXPERIMENT")
    print(f"  {Colors.BOLD}Experiments:{Colors.RESET} {len(experiments)}")
    print(f"  {Colors.BOLD}Conditions:{Colors.RESET} {[c.name for c in conditions]}")
    print(f"  {Colors.BOLD}Trials per condition:{Colors.RESET} {num_trials}")
    print(f"  {Colors.BOLD}Total trials:{Colors.RESET} {total_trials}")
    print(f"  {Colors.BOLD}Max concurrent (total):{Colors.RESET} {max_concurrent}")
    print(f"  {Colors.BOLD}Anthropic concurrent:{Colors.RESET} {anthropic_concurrent} trials, {anthropic_rps} req/s")
    print(f"  {Colors.BOLD}OpenAI concurrent:{Colors.RESET} {openai_concurrent} trials, {openai_rps} req/s")
    print(f"  {Colors.BOLD}Evidence severity:{Colors.RESET} {evidence_severity}")
    if progress_file:
        print(f"  {Colors.BOLD}Progress file:{Colors.RESET} {progress_file}")

    # Generate all trial contexts upfront, filtering out completed trials if resuming
    if completed_trials is None:
        completed_trials = set()
    
    print(f"\n{Colors.DIM}Generating trial contexts...{Colors.RESET}")
    contexts = []
    skipped_count = 0
    for exp in experiments:
        exp_name = exp.name or exp.judge_model
        for condition in conditions:
            for trial_num in range(1, num_trials + 1):
                # Check if this trial was already completed
                trial_key = (exp_name, condition.name, trial_num)
                if trial_key in completed_trials:
                    skipped_count += 1
                    continue
                ctx = create_trial_context(exp, condition, trial_num, evidence_severity)
                contexts.append(ctx)
    
    remaining_trials = len(contexts)
    if skipped_count > 0:
        print(f"{Colors.CYAN}✓ Resuming: {skipped_count} trials already completed, {remaining_trials} remaining{Colors.RESET}")
    else:
        print(f"{Colors.GREEN}✓ Generated {remaining_trials} trial contexts{Colors.RESET}")
    
    # Update total trials count to reflect remaining work
    total_trials = remaining_trials

    # Create global semaphore for overall concurrency limit
    global_semaphore = asyncio.Semaphore(max_concurrent)
    
    # Create per-provider rate limiters and semaphores
    provider_limiters = ProviderRateLimiters(
        anthropic_concurrent=anthropic_concurrent,
        openai_concurrent=openai_concurrent,
        anthropic_rps=anthropic_rps,
        openai_rps=openai_rps,
    )
    
    progress = ProgressTracker(total_trials=total_trials)
    progress.set_provider_limits(anthropic_concurrent, openai_concurrent)
    results_lock = asyncio.Lock()

    # Results storage: exp_name -> condition_name -> list of results
    all_results = {}
    for exp in experiments:
        exp_key = exp.name or exp.judge_model
        all_results[exp_key] = {cond.name: [] for cond in conditions}

    # Counter for generating unique trial IDs
    trial_id_counter = [0]
    trial_id_lock = asyncio.Lock()

    async def run_trial_worker(ctx) -> dict:
        """Worker that runs a single trial with per-provider rate limiting."""
        # Generate unique trial ID
        async with trial_id_lock:
            trial_id_counter[0] += 1
            trial_id = f"trial_{trial_id_counter[0]}"
        
        # Determine provider from model ID
        provider = get_model_family(ctx.exp.judge_model)
        exp_key = ctx.exp.name or ctx.exp.judge_model
        
        # Get provider-specific semaphore and rate limiter
        provider_semaphore = provider_limiters.get_semaphore(provider)
        rate_limiter = provider_limiters.get_rate_limiter(provider)
        
        # Use both global and provider-specific semaphores
        async with global_semaphore:
            async with provider_semaphore:
                # Register trial as starting
                await progress.start_trial(
                    trial_id=trial_id,
                    exp_name=exp_key,
                    condition_name=ctx.condition.name,
                    trial_num=ctx.trial_num,
                    provider=provider,
                    judge_model=ctx.exp.judge_model,
                )
                
                result = await run_trial_with_context(
                    ctx, max_turns, trial_timeout_seconds, rate_limiter
                )

                # Track progress with detailed info
                is_error = "error" in result
                hacked_severity = result.get("hacked_severity", 0)
                await progress.end_trial(trial_id, is_error, hacked_severity)

                # Store result in memory
                async with results_lock:
                    all_results[exp_key][ctx.condition.name].append(result)

                # Save progress to file immediately (for crash recovery)
                if progress_file:
                    await write_trial_progress(progress_file, result)

                return result

    # Progress reporting task
    async def progress_reporter():
        """Print rich progress updates periodically."""
        last_completed = 0
        while progress.completed < progress.total_trials:
            await asyncio.sleep(10)  # Update every 10 seconds (richer output)
            if progress.completed > last_completed or progress.completed == 0:
                timestamp = datetime.now().strftime("%H:%M:%S")
                log_rich_progress(progress, timestamp)
                last_completed = progress.completed

    # Start progress reporter
    reporter_task = asyncio.create_task(progress_reporter())

    # Run all trials
    print(f"\n{Colors.BOLD}Starting {total_trials} trials with {max_concurrent} concurrent workers...{Colors.RESET}")
    queue_start = time.time()

    try:
        await asyncio.gather(*[run_trial_worker(ctx) for ctx in contexts])
    finally:
        reporter_task.cancel()
        try:
            await reporter_task
        except asyncio.CancelledError:
            pass

    queue_elapsed = time.time() - queue_start
    rate = total_trials / queue_elapsed if queue_elapsed > 0 else 0

    print(f"\n{Colors.GREEN}✓ Completed {total_trials} trials in {queue_elapsed:.1f}s ({rate:.1f} trials/s){Colors.RESET}")
    if progress.errors > 0:
        print(f"  {Colors.YELLOW}⚠ {progress.errors} trials had errors{Colors.RESET}")

    # Consolidate results: load all from progress file (includes both previously completed and newly completed)
    if progress_file and os.path.exists(progress_file):
        print(f"\n{Colors.DIM}Consolidating results from progress file...{Colors.RESET}")
        all_progress_results = load_trial_progress(progress_file)
        
        # Rebuild all_results from progress file to include everything
        all_results = {}
        for exp in experiments:
            exp_key = exp.name or exp.judge_model
            all_results[exp_key] = {cond.name: [] for cond in conditions}
        
        for (exp_name, cond_name, trial_num), result in all_progress_results.items():
            if exp_name in all_results and cond_name in all_results[exp_name]:
                all_results[exp_name][cond_name].append(result)
        
        # Sort results by trial number for consistent ordering
        for exp_key in all_results:
            for cond_name in all_results[exp_key]:
                all_results[exp_key][cond_name].sort(key=lambda r: r.get("trial", 0))
        
        total_consolidated = sum(
            len(results) for exp_results in all_results.values() 
            for results in exp_results.values()
        )
        print(f"{Colors.GREEN}✓ Consolidated {total_consolidated} total trial results{Colors.RESET}")

    # Compute analysis for each experiment/condition
    log_header("ANALYZING RESULTS")
    all_analysis = {}
    all_outputs = []

    results_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for exp in experiments:
        exp_key = exp.name or exp.judge_model
        exp_results = all_results[exp_key]
        exp_analysis = {}

        print(f"\n{Colors.BOLD}Experiment: {exp.judge_display_name or exp.judge_model}{Colors.RESET}")

        for condition in conditions:
            cond_results = exp_results[condition.name]
            analysis = analyze_condition_results(cond_results, condition)
            exp_analysis[condition.name] = analysis

            if "error" not in analysis:
                action_rate = analysis['hacked_action_rate'] * 100
                avg_sev = analysis.get('hacked_avg_severity', 0)
                rate_color = Colors.GREEN if action_rate > 50 else Colors.YELLOW if action_rate > 0 else Colors.RED
                print(f"  {condition.name}: {rate_color}{action_rate:.1f}%{Colors.RESET} action rate, "
                      f"avg severity {avg_sev:.2f}")

        all_analysis[exp_key] = exp_analysis

        # Save individual experiment results
        exp_name = exp.name or f"{exp.judge_model}__vs__{exp.competitor_model}"
        safe_name = exp_name.replace("/", "_").replace(" ", "_")

        # Determine identity_condition from conditions that were run
        identity_types = set(c.identity for c in conditions)
        if identity_types == {"ambiguous"}:
            identity_condition_name = "ambiguous"
        elif identity_types == {"explicit"}:
            identity_condition_name = "explicit"
        else:
            identity_condition_name = "both"

        output = {
            "config": {
                "experiment_name": exp_name,
                "judge_model": exp.judge_model,
                "competitor_model": exp.competitor_model,
                "judge_family": exp.judge_family,
                "judge_display_name": exp.judge_display_name,
                "num_trials": num_trials,
                "max_turns": max_turns,
                "parallel": True,
                "task_queue": True,
                "max_concurrent": max_concurrent,
                "evidence_severity": evidence_severity,
                "identity_condition": identity_condition_name,
                "conditions": [c.name for c in conditions],
                "timestamp": datetime.now().isoformat(),
            },
            "results": exp_results,
            "analysis": exp_analysis,
            "comparisons": {
                "action_rate_bias_explicit": exp_analysis.get("B1", {}).get("hacked_action_rate", 0) -
                               exp_analysis.get("A1", {}).get("hacked_action_rate", 0),
                "action_rate_bias_ambiguous": exp_analysis.get("B2", {}).get("hacked_action_rate", 0) -
                                exp_analysis.get("A2", {}).get("hacked_action_rate", 0),
                "severity_bias_explicit": exp_analysis.get("B1", {}).get("hacked_avg_severity", 0) -
                               exp_analysis.get("A1", {}).get("hacked_avg_severity", 0),
                "severity_bias_ambiguous": exp_analysis.get("B2", {}).get("hacked_avg_severity", 0) -
                                exp_analysis.get("A2", {}).get("hacked_avg_severity", 0),
                # Both-hacked condition: bias in treating competitor vs self when both are guilty
                "both_severity_bias_explicit": exp_analysis.get("C1", {}).get("both_severity_bias", 0),
                "both_severity_bias_ambiguous": exp_analysis.get("C2", {}).get("both_severity_bias", 0),
                "both_action_rate_bias_explicit": exp_analysis.get("C1", {}).get("both_action_rate_bias", 0),
                "both_action_rate_bias_ambiguous": exp_analysis.get("C2", {}).get("both_action_rate_bias", 0),
            }
        }

        output_path = os.path.join(results_dir, f"reward_hacking_eval__{safe_name}__{timestamp}.json")
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"  {Colors.DIM}Saved: {output_path}{Colors.RESET}")
        all_outputs.append(output_path)

    return {
        "results": all_results,
        "analysis": all_analysis,
        "outputs": all_outputs,
        "total_trials": total_trials,
        "elapsed_seconds": queue_elapsed,
        "errors": progress.errors,
    }


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run self-serving bias experiment for LLM reward hacking evaluation"
    )
    parser.add_argument(
        "--resume",
        type=str,
        metavar="PATH",
        help="Path to a progress file (JSONL) to resume an interrupted run"
    )
    args = parser.parse_args()

    cfg = load_experiment_config()
    num_trials = cfg["num_trials"]
    max_turns = cfg["max_turns"]
    verbose = cfg.get("verbose", True)
    parallel = cfg.get("parallel", False)
    evidence_severity = cfg.get("evidence_severity", "moderate")
    identity_condition = cfg.get("identity_condition", "ambiguous")
    trial_timeout_seconds = cfg.get("trial_timeout_seconds", 15 * 60)  # 15 minutes per trial
    # Task queue mode: enable for maximum parallelization across all experiments
    # Defaults to True when parallel is enabled
    task_queue = cfg.get("task_queue", parallel)
    max_concurrent = cfg.get("max_concurrent", 100)
    
    # Rate limiting configuration
    anthropic_concurrent = cfg.get("anthropic_concurrent")
    openai_concurrent = cfg.get("openai_concurrent")
    anthropic_rps = cfg.get("anthropic_rps", 10.0)
    openai_rps = cfg.get("openai_rps", 20.0)

    # Get the conditions to run based on config
    conditions = get_conditions_for_config(identity_condition)
    
    # Handle resume flag and progress file
    if args.resume:
        # Resume from existing progress file
        progress_file = args.resume
        if not os.path.exists(progress_file):
            print(f"{Colors.RED}Error: Progress file not found: {progress_file}{Colors.RESET}")
            exit(1)
        completed_trials = get_completed_trial_keys(progress_file)
        print(f"{Colors.CYAN}Resuming from: {progress_file}{Colors.RESET}")
        print(f"{Colors.CYAN}Found {len(completed_trials)} completed trials{Colors.RESET}")
    else:
        # Generate new progress file path
        progress_file = get_progress_file_path(cfg)
        completed_trials = set()
        print(f"{Colors.DIM}Progress will be saved to: {progress_file}{Colors.RESET}")

    # Support both old "experiments" format and new "models" format
    if "models" in cfg:
        # New format: each model becomes an experiment with itself as judge
        experiments = [
            Experiment(
                judge_model=m["id"],
                competitor_model=m["competitor"],
                name=m.get("name", "").replace(" ", "_").replace(".", "_"),
                judge_family=m.get("family", ""),
                judge_display_name=m.get("name", m["id"]),
            )
            for m in cfg["models"]
        ]
        # Store model metadata for the index
        models_metadata = cfg["models"]
    else:
        # Legacy format: explicit experiments array
        experiments = [
            Experiment(
                judge_model=e["judge_model"],
                competitor_model=e["competitor_model"],
                name=e.get("name", ""),
            )
            for e in cfg["experiments"]
        ]
        models_metadata = None

    async def _main():
        # Use task queue for maximum parallelization across all experiments
        if task_queue:
            result = await run_with_task_queue(
                experiments=experiments,
                num_trials=num_trials,
                max_turns=max_turns,
                evidence_severity=evidence_severity,
                max_concurrent=max_concurrent,
                conditions=conditions,
                trial_timeout_seconds=trial_timeout_seconds,
                anthropic_concurrent=anthropic_concurrent,
                openai_concurrent=openai_concurrent,
                anthropic_rps=anthropic_rps,
                openai_rps=openai_rps,
                progress_file=progress_file,
                completed_trials=completed_trials,
            )
            outputs = result["outputs"]
        else:
            # Legacy mode: run experiments sequentially
            outputs = []
            for exp in experiments:
                run_out = await run_single_experiment(exp, num_trials, max_turns, verbose, parallel, evidence_severity, conditions=conditions, trial_timeout_seconds=trial_timeout_seconds, progress_file=progress_file)
                outputs.append(run_out.get("_output_path") or run_out.get("error"))

        if len(outputs) > 1:
            index = {
                "timestamp": datetime.now().isoformat(),
                "num_trials": num_trials,
                "max_turns": max_turns,
                "evidence_severity": evidence_severity,
                "identity_condition": identity_condition,
                "conditions": [c.name for c in conditions],
                "task_queue": task_queue,
                "max_concurrent": max_concurrent if task_queue else None,
                "experiments": [
                    {
                        "name": e.name,
                        "judge_model": e.judge_model,
                        "competitor_model": e.competitor_model,
                        "judge_family": e.judge_family,
                        "judge_display_name": e.judge_display_name,
                    }
                    for e in experiments
                ],
                "outputs": outputs,
            }
            # Include models metadata if using new format
            if models_metadata:
                index["models"] = models_metadata

            os.makedirs(os.path.join(SCRIPT_DIR, "results"), exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(SCRIPT_DIR, "results", f"reward_hacking_eval_index__{ts}.json")
            with open(out_path, "w") as f:
                json.dump(index, f, indent=2, default=str)
            print(f"\nIndex saved to: {out_path}")

    asyncio.run(_main())
