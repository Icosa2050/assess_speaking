# Frontend npm advisory triage — 2026-10-30

`npm --prefix frontend audit --json` reported six vulnerable package nodes before
this update: three moderate, three high. All six belong to the frontend's
development or build dependency tree; none is imported by the shipped app
runtime. The audit after the locked updates reports zero findings.

| Package | Severity | Exposure | Smallest safe fix and decision |
| --- | --- | --- | --- |
| `@vitest/mocker` | Moderate | Vitest test tooling only | `4.1.11` → `4.1.12` through the Vitest patch; applied. |
| `jsdom` | Moderate | DOM test environment only | Direct `jsdom` stays at `29.0.2`; its vulnerable `undici` child is fixed by the existing override moving to `7.30.1`; applied. |
| `source-map-js` | High | Build tooling only | `1.2.1` → `1.2.2` patch through an exact override; applied. |
| `undici` | High | `jsdom` test tooling only | `7.29.0` → `7.30.1` minor update to the existing override; applied. |
| `vite` | High | Development server and build tooling only | `8.0.16` → `8.1.6`, the earliest fixed 8.1 patch; applied. Its advisories concern development middleware and editor launching, including Windows paths. |
| `vitest` | Moderate | Test runner only | `4.1.11` → `4.1.12` patch; applied. |

The npm policy review found no new or removed packages, git or tarball sources,
or new install-script flags in the lockfile. No new transitive package maintainer
is introduced through a package addition. The updated direct packages retain
their established Vite and Vitest project ownership. The lockfile changes 36 existing package versions,
including Vitest companion packages and Rolldown's existing platform bindings.
The registry publishes a `prepare` script for undici's source repository, but
the locked registry tarball has no install-script flag and `npm ci --ignore-scripts`
does not run it. All four fix versions predate this triage by at least two weeks.

No residual advisory is accepted for this snapshot. Re-run the audit for new
advisories before the next dependency change.
