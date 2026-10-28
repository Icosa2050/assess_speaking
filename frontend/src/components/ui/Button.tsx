import type { ButtonHTMLAttributes, ReactNode } from "react";

import styles from "./controls.module.css";

type ButtonProps = Omit<ButtonHTMLAttributes<HTMLButtonElement>, "size"> & {
  children: ReactNode;
  variant?: "primary" | "secondary" | "quiet" | "danger";
  size?: "small" | "default" | "large";
} & ({ loading?: false; loadingLabel?: never } | { loading: boolean; loadingLabel: string });

export function Button({
  children,
  variant = "secondary",
  size = "default",
  loading = false,
  loadingLabel,
  className = "",
  disabled,
  type = "button",
  ...props
}: ButtonProps) {
  return <><button
    {...props}
    type={type}
    disabled={disabled || loading}
    aria-busy={loading || undefined}
    className={[styles.button, styles[variant], styles[size], className].filter(Boolean).join(" ")}
  >
    <span className={styles.buttonContent}>{children}</span>
  </button>
  {loadingLabel !== undefined && <span className={styles.loadingStatus} role="status">{loading ? loadingLabel : ""}</span>}</>;
}
