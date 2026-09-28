# Frontend dependency lock review — 2026-09-28

Recommendation: accept the locked dependency changes with lifecycle scripts disabled.
Risk is moderate: the patch updates the router and development toolchain, while
retaining the existing libraries and their responsibilities. No new product dependency
is introduced. Registry signatures provide integrity evidence, not a guarantee that
package code is safe.

## Scope and rationale

- Pin existing direct dependencies to exact versions and add `frontend/package-lock.json`.
- Replace the existing unpinned CI `npx jscpd` download with development dependency
  `jscpd@4.0.8`, preserving the duplication gate with a reproducible dependency tree.
  Its registry metadata identifies maintainer `apk` and no install lifecycle scripts.
- The first lock preserved existing versions. Its audit reported eight findings
  (six high, two moderate). Make targeted security updates instead of `npm update`
  or `npm audit fix --force`:

| Package | Previous | Selected |
| --- | --- | --- |
| react-router-dom / react-router | 7.14.2 | 7.18.2 |
| vite | 8.0.10 | 8.0.16 |
| vitest and its internal packages | 4.1.5 | 4.1.11 |
| nanoid (override) | 3.3.11 | 3.3.18 |
| postcss (override) | 8.5.10 | 8.5.23 |
| undici (override) | 7.25.0 | 7.29.0 |

The overrides pin existing transitive dependencies to patched versions. Remove them
when the parent constraints and lock naturally retain patched versions. The Vite
patch requires Rolldown 1.0.3 and matching optional platform bindings; associated
type/helper packages also change. Comparing the initial lock with the security
update adds/removes no package paths. The final lock has 276 package entries plus
the root, including optional platform packages; 246 packages install on this Mac.

The audit findings include framework/server modes that this client-side application
does not use. A zero audit result is not evidence that every original advisory was
exploitable here. The updates remove the flagged versions from the reproducible tree.

Primary advisory references:

- [React Router advisory](https://github.com/remix-run/react-router/security/advisories/GHSA-chx6-hx7r-mcp5)
- [Vite advisory](https://github.com/vitejs/vite/security/advisories/GHSA-fx2h-pf6j-xcff)
- [Vitest advisory](https://github.com/vitest-dev/vitest/security/advisories/GHSA-82fw-gwwq-j7x9)

## Installation and provenance

All resolved package URLs use the npm registry and have lockfile integrity metadata;
there are no git or arbitrary tarball sources. Native optional bindings belong to
the existing Rolldown/Lightning CSS build tools. Registry metadata and installed
package manifests were inspected for the changed direct/security packages.

The lock marks two optional `fsevents` entries as having install scripts. Undici's
manifest also declares a contributor `prepare` script. These are documented
exceptions to retaining packages with lifecycle declarations, **not permission to
execute those scripts**: installs use `npm ci --ignore-scripts`. Tests and builds
pass with them disabled. CI also uses `contents: read` and checkout
`persist-credentials: false`; no write token or application secrets are provided to
the install steps.

## Verification

- `npm ci --ignore-scripts`: successful, using a writable temporary npm cache.
- `npm audit`: zero vulnerabilities after the targeted patches.
- `npm audit signatures`: all 246 installed packages have verified registry
  signatures; 57 have verified attestations.
- Frontend: 106 unit/component tests pass; application and E2E TypeScript check
  passes; production build passes (existing large-chunk warning remains).
- Duplication gate: zero clones in the backend/core scope.

Browser execution remains an environment verification gap. On 2026-09-28 the
Chromium cache was missing its framework binary; the installed Chrome fallback
aborted at launch (`browserType.launch: Target page, context or browser has been
closed`, `SIGABRT`, with `kill EPERM` during cleanup). No browser test body ran.
The temporary fallback configuration lives outside the repository; CI retains
the normal Playwright-managed Chromium configuration.

Smallest repeated probe (2026-09-28):

```sh
npm --prefix frontend run test:e2e -- --config=/private/tmp/assess-browser-review.config.mts tests/e2e/smokeLocalGuest.spec.ts --reporter=line
```

The temporary config imports the repository config and overrides only the browser
channel to `chrome` and absolute test/output paths. The managed browser could
display the frontend home screen, but its connection became unavailable before
an interactive flow could be verified. Neither attempt substitutes for an E2E pass.
