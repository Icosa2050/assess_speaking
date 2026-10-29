# CI run recovery

GitHub Actions has occasionally left a PR run `in_progress` after every job
finished, with rerun and cancel rejected. Runs for PR #23 also disappeared from
the Actions listing. These are provider run-state incidents; a successful job
list is not evidence that the PR's required checks completed.

1. Record the PR head SHA and inspect both `gh run list --branch <branch>` and
   `gh api repos/Icosa2050/assess_speaking/commits/<sha>/check-runs`. Keep the
   run URL or ID and job conclusions in the PR record.
2. For a normal failed run, rerun only the failed job with
   `gh run rerun <run-id> --job <job-id>`. This is appropriate for an isolated
   Ubuntu mirror timeout before tests; the install steps retain explicit limits.
3. If the run cannot be rerun or cancelled, or no run can be found for the head
   SHA, check [GitHub Status](https://www.githubstatus.com/) and trigger a fresh
   PR run from a new head commit. Verify the new SHA has all seven completed
   checks before treating the PR as green. A manual `workflow_dispatch` run is
   useful for diagnosis but does not replace the PR's required checks.

Keep the old run ID and error in the PR discussion so an Actions platform
incident is distinguishable from a test or install failure.
