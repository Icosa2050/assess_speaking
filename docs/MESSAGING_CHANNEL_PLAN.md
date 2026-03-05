# Messaging Channel Expansion Plan

Last updated: 2026-03-04  
Owner: project maintainers  
Status: proposed

## 1. Purpose

Define whether and how `assess_speaking` should be exposed through chat channels (Telegram, WhatsApp, Signal or similar), and provide an alternative roadmap if chat-channel expansion is not pursued.

## 2. Decision Summary

1. Recommended path: run a Telegram-first pilot.
2. WhatsApp: optional second phase if there is a validated business need.
3. Signal: do not target for bot/service integration in the current plan.
4. If chat channels are not pursued, focus on web/API + LMS depth for better ROI with lower platform risk.

## 3. Why This Direction

1. Current code is CLI-centric and processes one file per run, but core steps are modular enough to wrap in a service.
2. Integration shape already exists (LMS adapters), so adding channel adapters is consistent with the architecture direction.
3. Telegram has the lowest implementation and operational friction for a pilot.
4. WhatsApp introduces business onboarding, policy constraints, and additional operational burden.
5. Signal currently has weak fit for formal bot/business integration.

## 4. Current State Snapshot

1. Entry point: `assess_speaking.py` orchestrates transcription, metrics, rubric generation, and output.
2. Output artifact: structured JSON report with optional training suggestions.
3. Existing external integration pattern: Canvas/Moodle helper module in `lms.py`.
4. Test baseline: unit tests for core functions and CLI pathways.

## 5. Target Capability

Allow a user to send a voice message to a bot/channel, receive acknowledgment, and get an assessment result when processing is complete.

## 6. Constraints and Assumptions

1. Audio processing can be CPU-intensive; request/response should not block webhook handlers.
2. Channel payload and file-size limits vary; adapters must validate and reject unsupported media cleanly.
3. LLM and ASR failures are expected edge cases; service must return actionable failure messages.
4. PII/voice data must have explicit retention policy.

## 7. Architecture Plan

## 7.1 New Components

1. `service/api.py`
2. Minimal HTTP service for channel webhooks and health checks.
3. `service/adapters/telegram.py`, `service/adapters/whatsapp.py`
4. Channel-specific message parsing, media retrieval, and reply formatting.
5. `service/jobs.py`
6. Async job queue layer (Redis-based queue recommended) and job status persistence.
7. `service/runner.py`
8. Wraps existing assessment pipeline functions for worker execution.
9. `service/store.py`
10. Persistent session tracking (`received`, `processing`, `done`, `failed`).

## 7.2 Processing Flow

1. Receive webhook event and validate signature/token.
2. Extract voice/audio metadata and channel user ID.
3. Persist job record and enqueue background task.
4. Worker downloads media, converts audio as needed, runs assessment.
5. Store report and send formatted result message back via adapter.

## 8. Delivery Plan (Channel Track)

## Phase 0: Foundation (1 week)

1. Refactor pipeline orchestration into importable function(s) with stable input/output contract.
2. Define service config model (`env`-driven).
3. Add structured logging and run IDs.

Exit criteria:

1. CLI still works.
2. Service wrapper can execute one local end-to-end job without channel integration.

## Phase 1: Telegram Pilot (1-2 weeks)

1. Implement Telegram webhook endpoint and update parser.
2. Implement media download and audio normalization.
3. Implement async execution and response messaging.
4. Add failure responses for unsupported media, timeout, or model errors.

Exit criteria:

1. Real Telegram bot can process voice messages end-to-end.
2. Median job runtime and failure rate are measured.

## Phase 2: Hardening (1 week)

1. Add rate limits, request auth checks, idempotency handling.
2. Add retention policy and purge jobs.
3. Add basic monitoring and alerting for failed jobs.

Exit criteria:

1. Service can operate continuously with controlled error behavior.

## Phase 3: WhatsApp Decision Gate (0.5 week)

1. Validate demand and operational readiness.
2. If go: implement WhatsApp adapter with template/policy-compliant responses.
3. If no-go: keep Telegram + API as supported channels.

Go criteria:

1. Clear user demand and owner for compliance operations.
2. Acceptable projected cost and maintenance overhead.

## 9. Should We Build It

Proceed only if at least one is true:

1. Primary user interaction is already chat-first.
2. Channel delivery will materially improve completion rate or acquisition.
3. Team can support operational ownership (monitoring, policy, incident response).

Do not proceed now if:

1. Team wants fastest value with lowest complexity.
2. There is no clear channel adoption hypothesis.
3. Operational ownership is undefined.

## 10. Alternative Plan (If Not Chat Channels)

## 10.1 API + Lightweight Web Upload (Recommended fallback)

1. Build a small REST endpoint for file upload and report retrieval.
2. Add minimal web page for uploading audio and viewing results.
3. Reuse same async worker architecture without channel-specific overhead.

Benefits:

1. Lower platform policy risk.
2. Better debuggability and observability.
3. Easier later channel expansion via adapters.

## 10.2 LMS-First Expansion

1. Finish production-grade Canvas/Moodle flows (IDs, auth scopes, better error mapping).
2. Add submission status reconciliation and retries.
3. Provide instructor-grade reporting export.

Benefits:

1. Aligned with existing code direction.
2. Direct value for education workflows.

## 11. Risks and Mitigations

1. Risk: long-running jobs block incoming requests.
2. Mitigation: strict async boundary and worker timeout policies.
3. Risk: high media processing cost.
4. Mitigation: input limits, model-size policy, and per-channel quotas.
5. Risk: privacy and retention concerns.
6. Mitigation: configurable retention windows, explicit deletion workflow, no secret logging.

## 12. Recommended Immediate Next Steps

1. Approve or reject Phase 1 Telegram pilot.
2. If approved, implement Phase 0 foundation refactor first.
3. Define success KPI before any channel launch (for example: completion rate, median turnaround time, weekly active users).
