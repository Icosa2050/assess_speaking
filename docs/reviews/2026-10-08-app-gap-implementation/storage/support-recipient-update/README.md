# Owner-selected support recipient

The default recipient is now `info@frommherz-it.ch`, remains editable and is passed to the existing explicit attached-draft action. No email was sent. The existing failure/retry component case now verifies the default first, then a changed recipient on retry. All 249 frontend tests, typecheck and the isolated localhost Settings support smoke passed.

The rebuilt internal DMG SHA-256 is `c4c2c2c9938e630586a9e8ca5369511801b4702eb25f3a3ab24561fb171f1f14`. This frontend-only iteration reused the unchanged backend helper verified on the prior `c527…` artifact. Native Settings in a fresh disposable copied app showed the selected default, and deep signature verification passed. Full backend/Ubuntu/storage/Mail-handoff acceptance in the parent record belongs to its previously recorded hashes; it was not rerun or reassigned to this new hash.
