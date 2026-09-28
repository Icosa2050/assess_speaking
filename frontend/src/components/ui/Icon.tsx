import { useId, type CSSProperties, type ReactNode } from "react";

import styles from "./visualPrimitives.module.css";

export type IconName =
  | "arrow-right"
  | "check"
  | "guide"
  | "headphones"
  | "history"
  | "language"
  | "microphone"
  | "play"
  | "settings"
  | "sparkle"
  | "stop"
  | "target"
  | "upload"
  | "warning";

type IconProps = {
  className?: string;
  name: IconName;
  size?: number;
  title?: string;
};

const iconPaths: Record<IconName, ReactNode> = {
  "arrow-right": (
    <>
      <path d="M5 12h14" />
      <path d="m13 6 6 6-6 6" />
    </>
  ),
  check: <path d="m5 12 4 4 10-10" />,
  guide: (
    <>
      <path d="M5 5.5A3.5 3.5 0 0 1 8.5 2H20v17H8.5A3.5 3.5 0 0 0 5 22z" />
      <path d="M5 5.5V22" />
      <path d="M9 6h7" />
      <path d="M9 10h6" />
    </>
  ),
  headphones: (
    <>
      <path d="M4 13a8 8 0 0 1 16 0" />
      <path d="M4 13v4a3 3 0 0 0 3 3h1v-7H7a3 3 0 0 0-3 3" />
      <path d="M20 13v4a3 3 0 0 1-3 3h-1v-7h1a3 3 0 0 1 3 3" />
    </>
  ),
  history: (
    <>
      <path d="M3 12a9 9 0 1 0 3-6.7" />
      <path d="M3 4v5h5" />
      <path d="M12 7v5l3 2" />
    </>
  ),
  language: (
    <>
      <path d="M4 5h9" />
      <path d="M9 3v2" />
      <path d="M6 9c1.2 2.5 3.3 4.3 6 5" />
      <path d="M12 5c-.5 3.5-2.7 6.2-7 8" />
      <path d="M14 21l4-9 4 9" />
      <path d="M16 17h4" />
    </>
  ),
  microphone: (
    <>
      <rect x="9" y="3" width="6" height="11" rx="3" />
      <path d="M5 11a7 7 0 0 0 14 0" />
      <path d="M12 18v3" />
      <path d="M8 21h8" />
    </>
  ),
  play: <path d="M8 5v14l11-7z" />,
  settings: (
    <>
      <path d="M12 15.5a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7z" />
      <path d="M19.4 15a1.8 1.8 0 0 0 .35 2l.05.05a2.1 2.1 0 1 1-3 3l-.05-.05a1.8 1.8 0 0 0-2-.35 1.8 1.8 0 0 0-1 1.65V21a2.1 2.1 0 1 1-4.2 0v-.08a1.8 1.8 0 0 0-1.2-1.7 1.8 1.8 0 0 0-1.95.4l-.05.05a2.1 2.1 0 1 1-3-3l.05-.05a1.8 1.8 0 0 0 .35-2 1.8 1.8 0 0 0-1.65-1H2a2.1 2.1 0 1 1 0-4.2h.08a1.8 1.8 0 0 0 1.7-1.2 1.8 1.8 0 0 0-.4-1.95l-.05-.05a2.1 2.1 0 1 1 3-3l.05.05a1.8 1.8 0 0 0 2 .35A1.8 1.8 0 0 0 9.4 2.1V2a2.1 2.1 0 1 1 4.2 0v.08a1.8 1.8 0 0 0 1.2 1.7 1.8 1.8 0 0 0 1.95-.4l.05-.05a2.1 2.1 0 1 1 3 3l-.05.05a1.8 1.8 0 0 0-.35 2c.28.68.94 1.13 1.67 1.14H22a2.1 2.1 0 1 1 0 4.2h-.08a1.8 1.8 0 0 0-1.7 1.2z" />
    </>
  ),
  sparkle: (
    <>
      <path d="m12 3 1.7 5.1L19 10l-5.3 1.9L12 17l-1.7-5.1L5 10l5.3-1.9z" />
      <path d="m19 16 .7 2.1L22 19l-2.3.9L19 22l-.7-2.1L16 19l2.3-.9z" />
      <path d="m5 3 .7 2.1L8 6l-2.3.9L5 9l-.7-2.1L2 6l2.3-.9z" />
    </>
  ),
  stop: <rect x="7" y="7" width="10" height="10" rx="1.5" />,
  target: (
    <>
      <circle cx="12" cy="12" r="8" />
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v3" />
      <path d="M12 19v3" />
      <path d="M2 12h3" />
      <path d="M19 12h3" />
    </>
  ),
  upload: (
    <>
      <path d="M12 16V4" />
      <path d="m7 9 5-5 5 5" />
      <path d="M5 20h14" />
      <path d="M5 16v4" />
      <path d="M19 16v4" />
    </>
  ),
  warning: (
    <>
      <path d="M12 3 2.8 20h18.4z" />
      <path d="M12 9v5" />
      <path d="M12 17h.01" />
    </>
  ),
};

export const Icon = ({
  className,
  name,
  size = 20,
  title,
}: IconProps) => {
  const titleId = useId();
  const style = { "--icon-size": `${size}px` } as CSSProperties;
  const accessibilityProps = title
    ? { "aria-labelledby": titleId, role: "img" }
    : { "aria-hidden": "true" as const };

  return (
    <svg
      className={[styles.icon, className].filter(Boolean).join(" ")}
      fill="none"
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="2"
      style={style}
      viewBox="0 0 24 24"
      {...accessibilityProps}
    >
      {title ? <title id={titleId}>{title}</title> : null}
      {iconPaths[name]}
    </svg>
  );
};
