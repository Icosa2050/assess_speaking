---
version: alpha
name: Vostavo Learner Coach
description: A calm, motivating speaking-practice interface for adult language learners.
colors:
  ink: "#243033"
  muted: "#5B676B"
  surface: "#F8FAF8"
  surface-soft: "#EDF3EF"
  surface-raised: "#FFFFFF"
  primary: "#276A73"
  primary-hover: "#1F5961"
  progress: "#3F7A52"
  focus: "#6B5CA5"
  caution: "#94472B"
  border-soft: "#CED8D2"
typography:
  display:
    fontFamily: Inter
    fontSize: 40px
    fontWeight: 700
    lineHeight: 1.1
    letterSpacing: 0
  headline:
    fontFamily: Inter
    fontSize: 28px
    fontWeight: 650
    lineHeight: 1.2
    letterSpacing: 0
  title:
    fontFamily: Inter
    fontSize: 18px
    fontWeight: 650
    lineHeight: 1.35
    letterSpacing: 0
  body:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: 400
    lineHeight: 1.55
    letterSpacing: 0
  label:
    fontFamily: Inter
    fontSize: 13px
    fontWeight: 650
    lineHeight: 1.2
    letterSpacing: 0
rounded:
  sm: 4px
  md: 8px
  full: 999px
spacing:
  xs: 4px
  sm: 8px
  md: 16px
  lg: 24px
  xl: 32px
  xxl: 48px
components:
  button-primary:
    backgroundColor: "{colors.primary}"
    textColor: "{colors.surface-raised}"
    typography: "{typography.label}"
    rounded: "{rounded.md}"
    padding: 12px
  button-primary-hover:
    backgroundColor: "{colors.primary-hover}"
    textColor: "{colors.surface-raised}"
    typography: "{typography.label}"
    rounded: "{rounded.md}"
    padding: 12px
  card:
    backgroundColor: "{colors.surface-raised}"
    textColor: "{colors.ink}"
    rounded: "{rounded.md}"
    padding: 24px
  system-status:
    backgroundColor: "{colors.surface-soft}"
    textColor: "{colors.muted}"
    typography: "{typography.label}"
    rounded: "{rounded.md}"
    padding: 16px
  field-frame:
    backgroundColor: "{colors.border-soft}"
    textColor: "{colors.ink}"
    typography: "{typography.body}"
    rounded: "{rounded.md}"
    padding: 16px
  progress-pill:
    backgroundColor: "{colors.progress}"
    textColor: "{colors.surface-raised}"
    typography: "{typography.label}"
    rounded: "{rounded.full}"
    padding: 8px
  focus-chip:
    backgroundColor: "{colors.focus}"
    textColor: "{colors.surface-raised}"
    typography: "{typography.label}"
    rounded: "{rounded.full}"
    padding: 8px
  warning-inline:
    backgroundColor: "{colors.surface-soft}"
    textColor: "{colors.caution}"
    typography: "{typography.body}"
    rounded: "{rounded.md}"
    padding: 16px
---

## Overview

Vostavo should feel like a calm speaking coach: focused, steady, and encouraging. The product is a practice companion first and a local-AI control panel second. The interface should reduce performance anxiety by making the next practice action obvious and keeping technical setup out of the main learning loop.

## Colors

Use a balanced light palette, not a beige editorial theme and not a single-hue blue dashboard. `primary` is for the main learner action. `progress` is for growth and completed states. `focus` is for selected practice focus. `caution` is for setup or review warnings.

## Typography

Use Inter for all UI text. Avoid viewport-scaled type. Prompts and scores may use `display` or `headline`, but dense tool panels, settings, and setup forms should use compact `title`, `body`, and `label` levels.

## Layout

Lead every screen with the learner's next action. Home should prioritize starting practice. Speak should prioritize prompt and recording. Review and History should prioritize feedback, next focus, and retry/new-session actions. Runtime setup is reachable but secondary.

## Elevation & Depth

Use soft surfaces and thin borders. Avoid nested cards, decorative orbs, bokeh, and broad gradients. Use stable dimensions for toolbars, status rows, recording controls, and route navigation so text and state changes do not shift layout.

## Shapes

Use 8px radius or less for cards and controls, except pills/chips where `full` is appropriate. Keep icon buttons square enough to read as controls.

## Components

Primary buttons start or continue practice. Secondary buttons navigate or reveal setup details. Empty states always provide a next step. System status should be a compact row, not a primary screen destination when everything is healthy.

## Do's and Don'ts

Do preserve localization, semantic IDs, route guards, and existing app architecture. Do make setup progressive and learner-friendly. Do keep technical labels available in advanced sections. Do not hide runtime errors. Do not make Runtime Setup a peer of Speak/Review once configured. Do not use landing-page hero composition inside the application shell.
