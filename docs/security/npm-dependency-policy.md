# npm Dependency Policy

Goal: keep the npm surface small, reproducible, and reviewable.

This policy applies to changes touching `frontend/package.json`, npm lockfiles, install scripts, JavaScript/TypeScript build tooling, or CI steps that install frontend dependencies.

## Default Position

Do not add a dependency unless it clearly beats a native API or small internal implementation.

Do not update dependencies for freshness alone. Update only for:

- an applicable security advisory,
- a functional bug affecting this project,
- required platform or toolchain compatibility,
- a narrowly scoped maintenance task approved for this repository.

## Required Review

Before adding or updating a package, check:

- direct need: why this package is necessary,
- maintenance: recent useful activity and issue response,
- ownership: identifiable project and maintainers,
- package shape: no unexpected files, binaries, obfuscation, or network fetches,
- lifecycle scripts: reject install-time scripts by default,
- dependency tree: avoid large or surprising transitive additions,
- provenance and signatures where available,
- release timing: avoid newly published versions unless fixing an active issue.

## Hard Reject By Default

Reject packages that:

- are unnecessary for the task,
- use `preinstall`, `install`, `postinstall`, or `prepare` without a documented exception,
- are installed from git URLs, tarballs, or arbitrary URLs,
- add suspicious optional dependencies,
- add unexplained binaries or generated blobs,
- are newly published by unknown or changed maintainers,
- replace a simple internal implementation.

## Existing Dependencies

Do not broad-update. Do not run `npm update` as a cleanup step.

For approved updates:

- make the smallest version movement that solves the reason,
- prefer exact versions for changed direct dependencies,
- regenerate and inspect the lockfile,
- call out new packages, removed packages, lifecycle scripts, and maintainer or provenance anomalies,
- run the relevant frontend checks.

## CI Rules

Use `npm ci` for reproducible installs.

Dependency installation should run without secrets or write-scoped tokens. If install scripts must be enabled, document the reason, the affected packages, and why the install step cannot safely run with scripts disabled.

## Agent Output

For any dependency change, report:

- risk assessment,
- approve or reject recommendation,
- reason for the change,
- lockfile and package diff summary,
- lifecycle script findings,
- verification run.

## References

- npm install and lockfile behavior: https://docs.npmjs.com/cli/v11/commands/npm-install/
- npm lifecycle scripts: https://docs.npmjs.com/cli/v7/using-npm/scripts/
- npm package provenance: https://docs.npmjs.com/viewing-package-provenance/
- GitHub npm supply-chain hardening plan: https://github.blog/security/supply-chain-security/our-plan-for-a-more-secure-npm-supply-chain/
- TanStack npm compromise advisory: https://github.com/TanStack/router/security/advisories/GHSA-g7cv-rxg3-hmpx
