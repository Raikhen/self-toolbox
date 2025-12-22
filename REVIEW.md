Deep Codebase Review: Self-Serving Bias in LLM Reward Hacking Evaluation

Executive Summary

This is a well-structured AI safety research project testing whether LLMs show self-serving bias when evaluating reward hacking. However, I've identified several methodological issues, potential bugs, and areas for improvement.

---

⚠️ Potential Bugs

8. No Multiple Comparison Correction

Location: analyze.py:298-458

When running chi-square tests, permutation tests, etc. across multiple conditions, no Bonferroni or FDR correction is applied. With 4 conditions being compared, the family-wise error rate inflates.

---

📊 Analysis Script Issues

10. Hardcoded Condition Names Throughout

Location: analyze.py:309, analyze.py:670-671, etc.

## Condition names "A1", "A2", "B1", "B2" are string literals scattered throughout. Adding new conditions would require changes in many places.

💡 Recommendations

High Priority

1. Fix identity leakage by using neutral labels in ambiguous conditions
2. Increase num_trials to at least 20-30 per condition for statistical power
3. Fix race condition in ProgressTracker.get_eta()

Medium Priority

4. Add unit tests for core functions 5. Consolidate execute_tool / execute_tool_ctx into one function
5. Add multiple comparison correction to statistical tests
6. Validate that eval.js test cases match eval_results templates

Low Priority

8. Use aiofiles for async file I/O in trial context creation
9. Consider making hacked code comments less obviously suspicious
10. Add configuration validation (check all model IDs are valid, etc.)

---

✅ What's Done Well

- Clean 2×2 factorial experimental design
- Virtual file system for parallel execution (elegant engineering)
- File order randomization to control for primacy effects
- Style variants to control for coding preferences
- Comprehensive logging with color-coded output
- Bootstrap and Wilson score confidence intervals - Cohen's d effect size calculations with interpretations
- Support for both legacy and multi-model config formats
