# Ubuntu browser capability probe

Date: October 8, 2026

The app owner authorized using the Ubuntu server through `ssh p50`, with GitHub available as an alternative. SSH succeeds outside the local network sandbox. The server runs Ubuntu 24.04.5 LTS x86_64, Python 3.12.3 and Docker 29.8.2. Node is not on the host PATH.

An isolated container using `mcr.microsoft.com/playwright:v1.59.1-noble` provided Node 24.14.1 and the matching browser binaries. Chromium 147.0.7727.15 and WebKit 26.4 each launched headlessly, rendered a synthetic page and accepted a synthetic text file through a file input. The process exited successfully. Full results and the image digest are in [the JSON record](2026-10-08-ubuntu-browser-probe.json).

The image/version pairing follows the [official Playwright Docker guidance](https://playwright.dev/docs/docker). This probe used only official tool packages and synthetic generic browser content. No repository source, app audio, learner report or credential was transferred. Installation used disabled npm lifecycle scripts, followed by a locked `npm ci`. The container was removed after execution; its temporary package/results workspace and the image cache remain for further verification. Host packages and existing services were not changed.

## Commands and limits

Initial restricted SSH command:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=10 p50 'uname -srm'
```

The restrictive local sandbox returned `Operation not permitted` for TCP port 22. The same authorized SSH connection succeeded when network access was permitted. The image pull was bounded by 240 seconds:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=10 p50 'timeout 240 docker pull mcr.microsoft.com/playwright:v1.59.1-noble'
```

The generic browser probe was bounded by 180 seconds and used a temporary workspace, a removable container, two CPUs, 2 GB of memory and 512 MB of shared memory. The [exact invocation](2026-10-08-ubuntu-browser-probe-command.txt) includes the synthetic probe script and package commands. No host ports, SSH agent, Docker socket or user configuration directories were mounted into the container.

## Application acceptance still pending

This demonstrates browser availability on Ubuntu, not the app's two WebKit upload journeys or full CI acceptance. The planned app probe needs selected current source and six tracked synthetic CEFR WAVs, approximately 3.8 MB, in a fresh test workspace.

Automatic approval review rejected that upload because using the server was not considered specific authorization to export private repository source and audio fixtures. A question requesting approval for that exact payload is pending. The upload did not execute. Application testing will use that approval or another explicitly authorized source location; no indirect transfer was attempted.

Use p50 for Ubuntu browser/backend/frontend verification. Continue macOS DMG build/signing and native WKWebView/microphone/Keychain acceptance on the Mac. Actual GitHub workflow evidence remains tied to the commit the workflow tested.
