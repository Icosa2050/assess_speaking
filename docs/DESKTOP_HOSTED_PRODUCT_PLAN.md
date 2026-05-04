# Desktop And Hosted Product Plan

Last updated: 2026-05-01
Status: Approved migration direction; phase-2 local guest implementation recovered and ready for final parity cleanup

## Summary

This plan defines the approved path from the current Streamlit desktop shell to
a product shape that supports both:
1. a packaged desktop app
2. a hosted web product

Locked decisions:
1. shared frontend: `React + TypeScript + Vite`
2. desktop shell: `Tauri v2`
3. backend runtime: keep `FastAPI` plus the existing Python assessment runtime
4. hosted auth foundation: `Supabase Auth`
5. hosted data foundation: `Supabase Postgres` for relational data and
   `Supabase Storage` for uploaded audio and persisted report artifacts
6. hosted background jobs: `Redis + RQ`
7. desktop login policy: optional sign-in, not required for local-only use

This keeps the current local-first desktop story intact while creating a
deliberate second deployment mode for hosted users instead of pretending the
existing local backend can simply be deployed as-is.

For the current migration sequence:
1. macOS and Windows remain the first-class desktop packaging targets
2. Linux remains a no-regression target until the shared frontend is stable

## Product Shape

### Desktop mode

Desktop remains the primary local product:
1. Tauri launches the shared React app
2. Tauri starts or reuses the local FastAPI backend on `127.0.0.1`
3. the backend continues to use the existing app-data layout and local files
4. guest mode remains supported and is the default first-run path
5. sign-in is optional and uses the system browser plus PKCE
6. desktop tokens are stored in OS-backed secure storage, not plain files

Desktop sign-in is for cloud-linked capabilities only:
1. hosted account access
2. future cloud history or sync features
3. future entitlement or subscription checks

Desktop sign-in must not block:
1. local runtime setup
2. local recording and assessment
3. local history browsing for local reports
4. offline use of the local product

### Hosted mode

Hosted is a separate deployment mode, not a direct copy of the local backend:
1. the shared React app runs as a hosted web frontend
2. FastAPI runs as a hosted API
3. Supabase Auth owns Google and Facebook login
4. FastAPI verifies Supabase-issued JWTs on protected endpoints
5. hosted relational state lives in Supabase Postgres
6. hosted uploads and large report artifacts live in Supabase Storage
7. long-running assessments are executed by hosted Python workers through
   `Redis + RQ`

Hosted mode must not depend on:
1. local app-data paths
2. `history.csv`
3. local backend state files
4. the current multiprocessing-only job manager

## Architecture

### Shared frontend

The frontend should be one app with deployment-specific adapters:
1. routes and UI behavior are shared across desktop and hosted
2. environment adapters decide whether API calls go to local FastAPI or hosted
   FastAPI
3. auth UI is shared, but desktop and web use different session plumbing
4. localization must continue to come from the repo locale files rather than
   new hardcoded strings

Recommended frontend building blocks:
1. `React + TypeScript + Vite`
2. `TanStack Query` for API state and job polling
3. `Zustand` for local session and screen state that currently lives in
   `st.session_state`
4. `shadcn/ui` plus `Radix` primitives for the component layer

### Local backend mode

Local backend mode continues to use the current behavior:
1. `app_backend/app.py` remains the local API contract
2. `app_backend/jobs.py` continues to own local background execution for desktop
3. reports, uploads, jobs, and logs continue to use the local app-data root
4. local-only endpoints remain auth-free

### Hosted backend mode

Hosted backend mode needs explicit new seams:
1. storage interfaces for reports, uploads, history, and job state
2. user identity resolution from verified Supabase JWTs
3. per-user or per-organization data ownership instead of machine-local state
4. worker-safe job submission instead of local multiprocessing state files
5. environment-based runtime mode so local and hosted stay in one repo without
   conflating their storage and auth behavior

Hosted protected endpoints should require JWT auth for:
1. uploads
2. assessments
3. history list and detail
4. future account-linked settings

Health and basic readiness endpoints may stay public if they do not reveal user
data or secrets.

## Auth And Identity

Supabase Auth is the approved auth foundation.

Why:
1. social login is required for the hosted product
2. hosted mode will also need relational user data and storage
3. the repo already needs a real hosted persistence layer, not just identity
4. this keeps auth and hosted data moving in one architectural direction

Auth rules:
1. hosted web always requires auth before user-scoped actions
2. desktop local-only mode does not require auth
3. desktop sign-in uses the system browser with PKCE and returns to the app via
   a desktop-safe callback flow
4. FastAPI must verify Supabase JWTs and derive the authenticated subject from
   those claims
5. app-owned roles, entitlements, and data access rules are enforced in FastAPI
   after JWT verification, not in the frontend

Identity model defaults:
1. one app user row per Supabase `auth.users` identity
2. hosted assessment records belong to an authenticated user id
3. local guest mode keeps using local app-data and does not require a cloud
   account
4. if a signed-in desktop user also performs local assessments, local artifacts
   remain local unless a later sync feature is explicitly implemented

## Implementation Phases

### Phase 1: Freeze local contract and add deployment seams

Goal:
1. keep the existing local backend stable
2. prepare the codebase for a second deployment mode without changing user
   behavior yet

Required changes:
1. treat the current `/v1/*` endpoints as the stable local API contract
2. add shared runtime metadata with `deployment_mode`, `launch_mode`,
   `packaging_safe`, and `auth_mode` so code can distinguish local desktop mode
   from hosted mode without inventing parallel schemas
3. extract interfaces around history, report, upload, and job persistence
4. keep local implementations backed by the current app-data layout
5. keep local mode auth-free

Done when:
1. local desktop behavior is unchanged
2. backend storage and job behavior can be swapped by runtime mode
3. no hosted code path reads `history.csv` or local app-data paths directly

### Phase 2: Replace Streamlit with the shared frontend for local desktop

Goal:
1. replace Streamlit without taking on hosted multi-user scope yet

Required changes:
1. build the shared React app against the existing local FastAPI contract
2. package the desktop app with Tauri
3. port the current learner flow first:
   - Home
   - Runtime Setup
   - Session Setup
   - Speak
   - Review
   - History
4. preserve the current local app-data layout and backend bootstrap behavior
5. keep localization parity with the existing locale files

Implementation status as of 2026-05-01:
1. the shared React shell is live against the phase-2 local FastAPI contract
2. Tauri can launch the shared frontend and start or reuse the local backend
3. local guest browser lanes now cover the shared frontend smoke path plus
   History, Settings, Settings -> Runtime Setup return-flow, and support-bundle
   runtime-health regression flows
4. Runtime Setup and Settings share the local runtime-management API for saved
   connections, secret-state preservation, UI locale, Whisper model persistence,
   connection testing, default/delete actions, and model download state
5. hosted auth, hosted persistence, and signed-in desktop behavior remain out of
   scope for this phase and are still tracked in later phases

Done when:
1. the shared frontend can fully replace the current Streamlit desktop shell
2. local guest mode supports the full existing assessment flow
3. no current local-only user is forced to create an account

### Phase 3: Add hosted persistence and hosted worker execution

Goal:
1. create a true hosted deployment mode instead of reusing local filesystem
   assumptions

Required changes:
1. add hosted storage implementations backed by Supabase Postgres and Supabase
   Storage
2. add hosted job submission and polling backed by Redis and RQ
3. keep local desktop using the existing local job manager
4. model hosted history and review payload ownership by authenticated user
5. move hosted uploads and persisted report files out of local disk assumptions

Done when:
1. hosted assessments no longer depend on local file paths or local process
   state
2. hosted users can create, poll, and review assessments independently
3. local and hosted deployments use the same public API concepts with different
   persistence implementations

### Phase 4: Add hosted auth and optional desktop sign-in

Goal:
1. enable hosted login without breaking the local-first desktop experience

Required changes:
1. integrate Supabase Auth in the shared frontend
2. add protected hosted routes and JWT verification in FastAPI
3. support Google and Facebook login through Supabase
4. add optional desktop sign-in using the system browser and PKCE
5. store desktop auth material in OS-backed secure storage
6. keep guest desktop mode as a supported default path

Done when:
1. hosted users can sign in and reach only their own hosted data
2. desktop users can ignore sign-in and still use the local app
3. desktop users can sign in without losing offline local capability

### Phase 5: Rollout, parity, and cleanup

Goal:
1. finish parity and retire Streamlit as the shipped shell

Required changes:
1. port secondary screens after the main learner flow is stable
2. keep current E2E and backend regression coverage green
3. add separate local and hosted test lanes
4. update packaging and deployment docs
5. document clear support boundaries for local guest mode, signed-in desktop
   mode, and hosted mode

Done when:
1. Streamlit is no longer required for normal product use
2. desktop and hosted users have clearly documented supported paths
3. the repo has one canonical frontend direction

## Public Interfaces And Contracts

The plan assumes these stable product contracts:
1. the existing `/v1/health`, `/v1/diagnostics`, `/v1/runtime`,
   `/v1/uploads`, `/v1/assessments`, `/v1/history`, and `/v1/samples`
   concepts remain canonical
2. hosted mode adds auth requirements and user ownership, but should avoid
   inventing a different product API shape
3. desktop sign-in must not be required to call local-only flows
4. job lifecycle states remain `queued`, `running`, `completed`, `failed`, and
   `cancelled`

## Test And Verification Plan

### Local desktop guest mode

1. app launches through Tauri and starts or reuses the local backend
2. no sign-in is required for local setup or local assessments
3. local uploads, assessments, review, and history use local app-data only
4. app works offline after dependencies and local providers are available

### Hosted web mode

1. Google and Facebook sign-in succeed through Supabase Auth
2. protected endpoints reject unauthenticated access
3. authenticated users only see their own hosted history and reports
4. long-running assessments move through the expected hosted queue states
5. hosted uploads and reports land in hosted storage, not local filesystem

### Desktop optional sign-in mode

1. system-browser sign-in succeeds through PKCE
2. tokens are stored securely and can be revoked cleanly
3. sign-out does not remove local guest-mode capability
4. local artifacts stay local unless a later sync feature explicitly moves them

### Regression coverage

1. current backend API tests stay green in local mode
2. current local learner flow stays green in desktop mode
3. hosted mode adds coverage for JWT verification, per-user isolation, and
   hosted storage adapters

## Assumptions And Defaults

1. `Supabase Auth` is the locked auth foundation
2. desktop login is optional, not required
3. local guest mode remains a first-class supported product path
4. hosted mode is a second deployment mode that requires explicit storage and
   job abstractions
5. mobile remains a later follow-on and should consume stable backend contracts,
   not UI exports
6. provider credentials stored through app-shell `secret_ref` remain separate
   from any future optional desktop sign-in tokens or hosted identity material
7. this plan does not replace `docs/PUBLIC_INFERENCE_ARCHITECTURE_PLAN.md`;
   that document remains a separate hosted inference beta option rather than the
   default end-user product architecture
