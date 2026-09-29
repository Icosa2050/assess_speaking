import { useState } from "react";
import { buildApiUrl } from "@/lib/runtime/environment";

export function RecordingReplay({ sessionId, label, unavailable }: { sessionId: string; label: string; unavailable: string }) {
  const [failed, setFailed] = useState(false);
  return <div style={{ display: "grid", gap: ".5rem", minWidth: 0 }}>
    <strong>{label}</strong>
    {failed ? <p role="status">{unavailable}</p> : <audio
      controls preload="none" aria-label={label}
      src={buildApiUrl(`/v1/history/${encodeURIComponent(sessionId)}/audio`)}
      onError={() => setFailed(true)} style={{ width: "100%", minWidth: 0 }}
    />}
  </div>;
}
