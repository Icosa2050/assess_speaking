import { useEffect, useState } from "react";
import { RecordingReplay } from "@/components/history/RecordingReplay";
import type { Rehearsal } from "@/lib/rehearsal/session";
import { loadPartRecording } from "@/lib/rehearsal/storage";

export function RehearsalAudio({ session, index, label, unavailable, downloadLabel }: {
  session: Rehearsal; index: number; label: string; unavailable: string; downloadLabel: string;
}) {
  const [audio, setAudio] = useState({ url: "", extension: "webm", loaded: false });
  const reportId = session.parts[index].reportId;
  useEffect(() => {
    let active = true;
    let url = "";
    setAudio({ url: "", extension: "webm", loaded: false });
    loadPartRecording(session, index).then(blob => {
      if (!active) return;
      if (blob) url = URL.createObjectURL(blob);
      setAudio({ url, extension: blob?.type.includes("mp4") ? "m4a" : blob?.type.includes("ogg") ? "ogg" : "webm", loaded: true });
    }).catch(() => { if (active) setAudio({ url: "", extension: "webm", loaded: true }); });
    return () => { active = false; if (url) URL.revokeObjectURL(url); };
  }, [session.id, index]);
  if (audio.url) return <div>
    <strong>{label}</strong>
    <audio controls preload="none" src={audio.url} aria-label={label} style={{ width: "100%" }} />
    <a href={audio.url} download={`rehearsal-${session.id}-part-${index + 1}.${audio.extension}`}>{downloadLabel}</a>
  </div>;
  if (reportId) return <RecordingReplay sessionId={reportId} label={label} unavailable={unavailable} />;
  return audio.loaded ? <p role="status">{unavailable}</p> : null;
}
