import { useId, type InputHTMLAttributes, type ReactNode, type SelectHTMLAttributes, type TextareaHTMLAttributes } from "react";

import styles from "./controls.module.css";

type SharedFieldProps = {
  label: ReactNode;
  hint?: ReactNode;
  error?: ReactNode;
  optional?: ReactNode;
  unit?: ReactNode;
  className?: string;
};

type FieldProps = Omit<InputHTMLAttributes<HTMLInputElement>, "size"> & SharedFieldProps;
type SelectProps = SelectHTMLAttributes<HTMLSelectElement> & SharedFieldProps & { children: ReactNode };
type TextAreaProps = TextareaHTMLAttributes<HTMLTextAreaElement> & SharedFieldProps;

function describedBy(existing: string | undefined, hintId: string | undefined, errorId: string | undefined) {
  return [existing, hintId, errorId].filter(Boolean).join(" ") || undefined;
}

export function Field({ label, hint, error, optional, unit, className = "", id, "aria-describedby": description, ...props }: FieldProps) {
  const generatedId = useId();
  const inputId = id || generatedId;
  const hintId = hint ? `${inputId}-hint` : undefined;
  const errorId = error ? `${inputId}-error` : undefined;
  return <div className={`${styles.field} ${className}`.trim()}>
    <label htmlFor={inputId}>{label}{optional && <span className={styles.optional}> {optional}</span>}</label>
    <div className={styles.inputWrap}>
      <input {...props} id={inputId} aria-describedby={describedBy(description, hintId, errorId)} aria-invalid={error ? true : props["aria-invalid"]} className={styles.input} />
      {unit && <span className={styles.unit}>{unit}</span>}
    </div>
    {hint && <p id={hintId} className={styles.hint}>{hint}</p>}
    {error && <p id={errorId} className={styles.fieldError}>{error}</p>}
  </div>;
}

export function Select({ label, hint, error, optional, unit, className = "", id, "aria-describedby": description, children, ...props }: SelectProps) {
  const generatedId = useId();
  const selectId = id || generatedId;
  const hintId = hint ? `${selectId}-hint` : undefined;
  const errorId = error ? `${selectId}-error` : undefined;
  return <div className={`${styles.field} ${className}`.trim()}>
    <label htmlFor={selectId}>{label}{optional && <span className={styles.optional}> {optional}</span>}</label>
    <div className={styles.inputWrap}>
      <select {...props} id={selectId} aria-describedby={describedBy(description, hintId, errorId)} aria-invalid={error ? true : props["aria-invalid"]} className={styles.input}>{children}</select>
      {unit && <span className={styles.unit}>{unit}</span>}
    </div>
    {hint && <p id={hintId} className={styles.hint}>{hint}</p>}
    {error && <p id={errorId} className={styles.fieldError}>{error}</p>}
  </div>;
}

export function TextArea({ label, hint, error, optional, className = "", id, "aria-describedby": description, ...props }: TextAreaProps) {
  const generatedId = useId();
  const inputId = id || generatedId;
  const hintId = hint ? `${inputId}-hint` : undefined;
  const errorId = error ? `${inputId}-error` : undefined;
  return <div className={`${styles.field} ${className}`.trim()}>
    <label htmlFor={inputId}>{label}{optional && <span className={styles.optional}> {optional}</span>}</label>
    <textarea {...props} id={inputId} aria-describedby={describedBy(description, hintId, errorId)} aria-invalid={error ? true : props["aria-invalid"]} className={`${styles.input} ${styles.textarea}`} />
    {hint && <p id={hintId} className={styles.hint}>{hint}</p>}
    {error && <p id={errorId} className={styles.fieldError}>{error}</p>}
  </div>;
}
