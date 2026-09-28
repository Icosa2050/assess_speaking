# Baseline Metric Semantics Implementation Plan

> Archive status, 2026-05-20: completed and retained as historical evidence. Do not execute this file as an active plan.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make CEFR baseline metrics avoid false “Open” statuses when a learner exceeds the requested pace or uses linguistic cohesion not captured by crude exact counters.

**Architecture:** Keep the change inside baseline evaluation. WPM remains an assessed gate, but only the minimum pace is enforced; phrase-count metrics become observed evidence because semantic language quality is handled by the rubric and coaching.

**Tech Stack:** Python `unittest`/`pytest`, existing React baseline display contract.

---

### Task 1: Regression Tests

**Files:**
- Modify: `tests/test_assess_speaking.py`

- [x] **Step 1: Add a regression for fast C1 speech**

Add a test near `test_evaluate_baseline_returns_expected_flags`:

```python
def test_evaluate_baseline_passes_fast_speech_and_observes_phrase_counters(self):
    metrics = {"wpm": 172.6, "fillers": 0, "cohesion_markers": 1, "complexity_index": 14}
    result = assess_speaking.evaluate_baseline("C1", metrics)

    self.assertTrue(result["passed"])
    self.assertEqual(result["targets"]["wpm"]["expected"], "≥110")
    self.assertTrue(result["targets"]["wpm"]["ok"])
    self.assertEqual(result["targets"]["wpm"]["status"], "pass")
    self.assertEqual(result["targets"]["cohesion_markers"]["status"], "observed")
    self.assertIsNone(result["targets"]["cohesion_markers"]["ok"])
    self.assertEqual(result["targets"]["complexity_index"]["status"], "observed")
    self.assertIsNone(result["targets"]["complexity_index"]["ok"])
```

- [x] **Step 2: Add a regression for genuinely slow C1 speech**

```python
def test_evaluate_baseline_still_fails_slow_speech(self):
    metrics = {"wpm": 90, "fillers": 0, "cohesion_markers": 8, "complexity_index": 8}
    result = assess_speaking.evaluate_baseline("C1", metrics)

    self.assertFalse(result["passed"])
    self.assertFalse(result["targets"]["wpm"]["ok"])
    self.assertEqual(result["targets"]["wpm"]["status"], "fail")
```

- [x] **Step 3: Add regressions for observed counters and missing hard metrics**

```python
def test_evaluate_baseline_observed_phrase_counters_do_not_fail_c1(self):
    metrics = {"wpm": 120, "fillers": 0, "cohesion_markers": 0, "complexity_index": 0}
    result = assess_speaking.evaluate_baseline("C1", metrics)

    self.assertTrue(result["passed"])
    self.assertEqual(result["targets"]["cohesion_markers"]["status"], "observed")
    self.assertIsNone(result["targets"]["cohesion_markers"]["expected"])
    self.assertEqual(result["targets"]["complexity_index"]["status"], "observed")
    self.assertIsNone(result["targets"]["complexity_index"]["expected"])


def test_evaluate_baseline_missing_hard_metric_is_not_assessed(self):
    metrics = {"wpm": 120, "cohesion_markers": 0, "complexity_index": 0}
    result = assess_speaking.evaluate_baseline("C1", metrics)

    self.assertFalse(result["valid"])
    self.assertFalse(result["passed"])
    self.assertEqual(result["missing_required_metrics"], ["fillers"])
    self.assertEqual(result["targets"]["fillers"]["status"], "not_assessed")
    self.assertIsNone(result["targets"]["fillers"]["ok"])
    self.assertTrue(result["targets"]["wpm"]["ok"])
```

- [x] **Step 4: Verify the new tests fail before implementation**

Run:

```bash
.venv/bin/python -m pytest tests/test_assess_speaking.py::ParsingAndBaselineTests -q
```

Expected before implementation: failures showing WPM above the old maximum, observed counters still displaying threshold-shaped expectations, or missing fillers being treated as a pass.

### Task 2: Baseline Evaluation

**Files:**
- Modify: `assess_speaking.py`

- [x] **Step 1: Remove closed-interval WPM gating**

Delete the local `within_range` helper from `evaluate_baseline`.

- [x] **Step 2: Make WPM minimum-only**

Change the WPM target to:

```python
"wpm": _baseline_target(
    expected=f"≥{cfg['wpm_min']}",
    actual=metrics.get("wpm"),
    ok=metrics.get("wpm", 0) >= cfg["wpm_min"],
    not_assessed=not_assessed,
),
```

- [x] **Step 3: Make exact phrase counters observed-only**

Set both `cohesion_markers` and `complexity_index` to `observed_only=True` and `expected=None` so they remain visible as evidence but cannot fail the baseline or display threshold-shaped gate text.

- [x] **Step 4: Treat missing hard-gate metrics as not assessed**

Add required metric tracking for `wpm` and `fillers`; if either is missing, the target should be `status="not_assessed"`, `ok=None`, the baseline should be `valid=False`, and `missing_required_metrics` should name the missing metric.

- [x] **Step 5: Verify focused tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_assess_speaking.py::ParsingAndBaselineTests -q
.venv/bin/python -m pytest tests/test_assess_speaking.py -q
```
