import "@testing-library/jest-dom/vitest";

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { Button } from "./Button";
import { Card, Disclosure } from "./Card";
import { Field, Select, TextArea } from "./Field";
import { StatusBanner } from "./StatusBanner";

describe("shared controls", () => {
  it("keeps a loading action's visible label and native disabled semantics", () => {
    render(<Button variant="primary" loading loadingLabel="Saving backup">Create backup</Button>);

    const button = screen.getByRole("button", { name: /Create backup/ });
    expect(button).toBeDisabled();
    expect(button).toHaveAttribute("aria-busy", "true");
    expect(button).toHaveAccessibleName("Create backup");
    expect(button).toHaveTextContent("Create backup");
    expect(screen.getByRole("status")).toHaveTextContent("Saving backup");
  });

  it("associates field hints and errors with a labeled native input", () => {
    render(<Field label="Learner name" hint="Used in saved history" error="Enter a name" required />);

    const input = screen.getByRole("textbox", { name: "Learner name" });
    expect(input).toBeRequired();
    expect(input).toHaveAttribute("aria-invalid", "true");
    expect(input).toHaveAccessibleDescription("Used in saved history Enter a name");
  });

  it("preserves native select and multiline field controls", () => {
    render(<>
      <Select label="Microphone" defaultValue="built-in"><option value="built-in">Built-in</option></Select>
      <TextArea label="Notes" hint="Optional context" />
    </>);

    expect(screen.getByRole("combobox", { name: "Microphone" })).toHaveValue("built-in");
    expect(screen.getByRole("textbox", { name: "Notes" })).toHaveAccessibleDescription("Optional context");
  });

  it("uses an urgent role only for an urgent banner and keeps disclosures native", () => {
    render(<>
      <StatusBanner title="Backup available" tone="success">Your data is ready.</StatusBanner>
      <StatusBanner title="Recovery required" tone="error" urgent>Choose a safe action.</StatusBanner>
      <Card><Disclosure title="Archived attempts"><p>One attempt</p></Disclosure></Card>
    </>);

    expect(screen.getByRole("status")).toHaveTextContent("Backup available");
    expect(screen.getByRole("alert")).toHaveTextContent("Recovery required");
    expect(screen.getByText("Archived attempts").closest("summary")).toBeInTheDocument();
  });
});
