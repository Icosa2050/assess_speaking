import { useId, useState } from "react";
import styles from "./TranscriptFeedback.module.css";

type Suggestion = { original: string; suggestion: string; explanation: string };
type Translate = (key: string, vars?: Record<string, string | number>) => string;

type Segment = { text: string; suggestion?: Suggestion };
const isWordChar = (value: string | undefined): boolean => Boolean(value && /[\p{L}\p{N}_]/u.test(value));
export function transcriptSegments(transcript: string, suggestions: Suggestion[]): Segment[] {
  const matches: { start: number; end: number; suggestion: Suggestion }[] = [];
  for (const suggestion of suggestions) {
    if (!suggestion.original.trim() || !suggestion.explanation.trim()) continue;
    let start = transcript.indexOf(suggestion.original);
    while (start >= 0) {
      const end = start + suggestion.original.length;
      const wholePhrase = !(isWordChar(transcript[start - 1]) && isWordChar(suggestion.original[0])) &&
        !(isWordChar(transcript[end]) && isWordChar(suggestion.original.at(-1)));
      if (wholePhrase && !matches.some(match => start < match.end && end > match.start)) {
        matches.push({ start, end, suggestion });
        break;
      }
      start = transcript.indexOf(suggestion.original, start + 1);
    }
  }
  matches.sort((a, b) => a.start - b.start);
  const segments: Segment[] = [];
  let cursor = 0;
  for (const match of matches) {
    if (match.start > cursor) segments.push({ text: transcript.slice(cursor, match.start) });
    segments.push({ text: transcript.slice(match.start, match.end), suggestion: match.suggestion });
    cursor = match.end;
  }
  if (cursor < transcript.length) segments.push({ text: transcript.slice(cursor) });
  return segments;
}

export function TranscriptFeedback({ transcript, suggestions, translate }: {
  transcript: string; suggestions: Suggestion[]; translate: Translate;
}) {
  const id = useId();
  const [open, setOpen] = useState<number | null>(null);
  const segments = transcriptSegments(transcript, suggestions);
  if (!segments.some(segment => segment.suggestion)) return null;
  let annotationIndex = -1;
  return <div className={styles.panel} data-testid="transcript-feedback">
    <p className={styles.label}>{translate("history_slice10.annotation_intro")}</p>
    <p className={styles.transcript}>{segments.map((segment, index) => {
      if (!segment.suggestion) return <span key={index}>{segment.text}</span>;
      annotationIndex += 1;
      const number = annotationIndex;
      const detailId = `${id}-annotation-${number}`;
      return <span key={index} className={styles.annotation}>
        <button type="button" className={styles.phrase} aria-expanded={open === number} aria-controls={detailId}
          aria-label={translate("history_slice10.annotation_label", { phrase: segment.text })}
          aria-describedby={open === number ? detailId : undefined} onClick={() => setOpen(open === number ? null : number)}>
          {segment.text}<span aria-hidden="true"> ✦</span>
        </button>
        <span id={detailId} className={styles.detail} hidden={open !== number}>
          {translate("history_slice10.annotation_suggestion", { suggestion: segment.suggestion.suggestion })} {segment.suggestion.explanation}
        </span>
      </span>;
    })}</p>
  </div>;
}
