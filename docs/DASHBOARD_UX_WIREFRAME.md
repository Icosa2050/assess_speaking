# Dashboard UX Wireframe

## Direction

The dashboard should stop behaving like a long tools page and become a guided practice flow.

- Top-level navigation: `Ueben`, `Verlauf`, `Einstellungen`
- `Ueben`: one linear session flow from setup to speaking to review
- `Verlauf`: saved attempts, trends, and detailed reports
- `Einstellungen`: UI locale, learning-language library, and advanced model controls

This keeps `ui_locale` and `learning_language` separate:

- `ui_locale`: labels, instructions, coaching shell
- `learning_language`: prompt content, transcript checks, speaking task

## Practice Flow

```text
+--------------------------------------------------+
| [Ueben] [Verlauf] [Einstellungen]                |
+--------------------------------------------------+
| Phase 1: Setup                                   |
|                                                  |
|  Italiano · B1  [Bearbeiten]                     |
|  Thema: [ Zufall / eigenes Thema / Bibliothek ]  |
|  Dauer: [----- 60s -----]                        |
|                                                  |
|  [ Aufgabe generieren ]                          |
+--------------------------------------------------+
| Phase 2: Speak                                   |
|                                                  |
|  Aufgabe:                                        |
|  "Parla di un viaggio che hai fatto..."          |
|                                                  |
|  - punto 1                                       |
|  - punto 2                                       |
|  - punto 3                                       |
|                                                  |
|  [ Tipps anzeigen ]                              |
|                                                  |
|  [ Aufnahme starten ]                            |
|  Datei hochladen                                 |
|  Neue Aufgabe                                    |
+--------------------------------------------------+
| Phase 3: Review                                  |
|                                                  |
|  Transcript:                                     |
|  "Sono andato..."                                |
|                                                  |
|  Feedback (UI locale):                           |
|  "Du hast ..."                                   |
|                                                  |
|  [ Nochmal ueben ] [ Im Verlauf speichern ]      |
+--------------------------------------------------+
```

## Notes

- Avoid two-column practice layouts on mobile.
- Keep upload as a fallback action, not a competing primary mode.
- Move Prompt Trainer out of the first-run practice surface.
- Show only one primary CTA per phase.
- Theme/library management should not live in the sidebar on mobile-first flows.
