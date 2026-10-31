# What leaves the device

Vostavo stores recordings, transcripts, reports and drafts on the device. The
selected speech recognition and assessment providers determine which practice
content is sent elsewhere. This describes the current application paths, not
the provider's own retention policy.

| Choice | Practice content sent to a provider |
| --- | --- |
| Local Whisper speech recognition | None. Audio is decoded and transcribed locally. Downloading model weights contacts the model host, but does not upload a recording. |
| Local Ollama or LM Studio assessment | None. The assessment prompt and transcript go to the configured localhost service. |
| OpenRouter, Groq, xAI or ChatGPT assessment | The transcript and assessment prompt are sent to that service. OpenRouter can forward them to the upstream provider serving the selected model. The prompt can include the selected task and scoring context. |
| OpenAI-compatible assessment | The same transcript and prompt go to the configured URL. Only a loopback address such as `localhost` or `127.0.0.1` stays on this device; private-LAN and remote endpoints send them off-device. |
| Ollama Cloud assessment | The transcript and prompt go to Ollama's cloud endpoint. |
| Groq cloud speech recognition | Audio chunks are sent to Groq for transcription. The resulting transcript can then be sent to the selected assessment provider. |

Local Whisper weights are downloaded on demand into the user's cache; the DMG
does not ship them. During provider processing, recordings are not sent to cloud
assessment providers when local transcription is selected. You can separately
choose to include recordings or uploaded audio in a support bundle and share it
yourself. Check the chosen provider's terms and privacy policy before using
personal or sensitive recordings.
