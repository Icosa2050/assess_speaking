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
does not run it. The Vite, Vitest, undici, and source-map-js fix versions all
predate this triage by at least two weeks.

## Package safety checks

| Package | Maintenance and issue response | Published package shape | Dependency tree | Registry integrity evidence |
| --- | --- | --- | --- | --- |
| `vite@8.1.6` | Active [Vite project](https://github.com/vitejs/vite); this fixed release followed the reported Vite advisories and was published October 6. | 35 published files: expected CLI JS, client declarations, and build output; no native binary or install entry point. | Existing Vite/Rolldown tree; no new package names. | Registry signatures and SLSA provenance advertised. |
| `vitest@4.1.12` | Active [Vitest project](https://github.com/vitest-dev/vitest); this patch addresses the mocker advisories and was published October 15. | 114 published files: test runner JS, declarations, and build output; no native binary or install entry point. | Existing `@vitest/*` companions move to the same patch; no new package names. | Registry signature and SLSA provenance advertised. |
| `undici@7.30.1` | Active [Node.js undici project](https://github.com/nodejs/undici); this release addresses the cited undici advisories and was published October 15. | 210 published files: HTTP library JS and docs; no native binary or install entry point in the registry tarball. | Existing jsdom child through the existing override; no new package names. | Registry signatures and SLSA provenance advertised. |
| `source-map-js@1.2.2` | [source-map-js project](https://github.com/7rulnik/source-map-js); this patch addresses its indexed source-map advisory and was published September 30. | 18 published files: source-map JS, declarations, docs, and license; no native binary or install entry point. | Existing build-tooling child, exact override; no new package names. | Registry signatures advertised; no provenance attestation shown in package metadata. |

`npm pack --dry-run --json` supplied the published file lists. The lockfile has
no new git or tarball sources and no new `hasInstallScript` flags. The installed
tree passed `npm --prefix frontend audit signatures`: all 246 packages have
verified registry signatures and 57 have verified attestations. No unexpected
install-time network fetches or generated binaries were found in these four
package tarballs; expected Rolldown platform bindings were already present in
the prior lockfile.

No residual advisory is accepted for this snapshot. Re-run the audit for new
advisories before the next dependency change.
