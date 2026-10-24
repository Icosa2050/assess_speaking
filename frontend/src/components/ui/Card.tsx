import type { DetailsHTMLAttributes, HTMLAttributes, ReactNode } from "react";

import styles from "./controls.module.css";

type CardProps = HTMLAttributes<HTMLElement> & {
  as?: "div" | "section" | "aside";
  children: ReactNode;
  subtle?: boolean;
};

export function Card({ as: Element = "div", children, subtle = false, className = "", ...props }: CardProps) {
  return <Element {...props} className={`${styles.card} ${subtle ? styles.cardSubtle : ""} ${className}`.trim()}>{children}</Element>;
}

export function Disclosure({ title, children, className = "", ...props }: DetailsHTMLAttributes<HTMLDetailsElement> & { title: ReactNode; children: ReactNode }) {
  return <details {...props} className={`${styles.disclosure} ${className}`.trim()}>
    <summary>{title}</summary>
    <div className={styles.disclosureBody}>{children}</div>
  </details>;
}
