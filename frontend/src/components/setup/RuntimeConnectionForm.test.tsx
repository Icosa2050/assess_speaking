import "@testing-library/jest-dom/vitest";
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { RuntimeConnectionForm } from "./RuntimeConnectionForm";

const models = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"];
afterEach(cleanup);

describe("Groq feedback setup", () => {
  for (const locale of ["en", "it"] as const) {
    it(`offers Groq without advanced settings and submits each model (${locale})`, () => {
      const onTest = vi.fn();
      const onSave = vi.fn();
      render(<RuntimeConnectionForm locale={locale} initialDraft={{label:"",base_url:"",provider_choice:"ollama_local",model:"local-model",api_key:"unrelated-key"}}
        initialSecretState="absent" onTest={onTest} onSave={onSave}
        detectedModels={["whisper-large-v3", "canopylabs/orpheus-v1-english"]} detectedModelMessage="Old discovery message" />);
      const provider = screen.getByTestId("runtime_connection.provider");
      expect(within(provider).getByRole("option", {name:/Groq/})).toBeInTheDocument();
      fireEvent.change(provider, {target:{value:"groq"}});
      expect(screen.getByTestId("runtime_connection.api_key")).toHaveValue("");
      const picker = screen.getByTestId("runtime_connection.model");
      expect(picker.tagName).toBe("SELECT");
      expect(picker).toHaveValue(models[0]);
      expect(within(picker).getAllByRole("option").map(option => option.getAttribute("value"))).toEqual(models);
      expect(screen.queryByRole("option", {name:"whisper-large-v3"})).not.toBeInTheDocument();
      expect(screen.queryByText("Old discovery message")).not.toBeInTheDocument();
      fireEvent.change(screen.getByTestId("runtime_connection.api_key"), {target:{value:"test-only-key"}});
      for (const model of models) {
        fireEvent.change(picker, {target:{value:model}});
        fireEvent.click(screen.getByTestId("runtime_connection.test_connection"));
        fireEvent.click(screen.getByTestId("runtime_connection.save_connection"));
        expect(onTest).toHaveBeenLastCalledWith(expect.objectContaining({provider_choice:"groq",model,api_key:"test-only-key",base_url:"https://api.groq.com/openai/v1"}));
        expect(onSave).toHaveBeenLastCalledWith({clearSavedSecret:false,draft:expect.objectContaining({provider_choice:"groq",model})});
      }
    });
  }

  it("requires an explicit replacement for an unsupported saved model", () => {
    const onTest = vi.fn(); const onSave = vi.fn();
    render(<RuntimeConnectionForm locale="en" variant="settings"
      initialDraft={{label:"",base_url:"",provider_choice:"groq",model:"whisper-large-v3"}} initialSecretState="present"
      onTest={onTest} onSave={onSave} />);
    fireEvent.click(screen.getByTestId("runtime_connection.test_connection"));
    fireEvent.click(screen.getByTestId("runtime_connection.save_connection"));
    expect(onTest).not.toHaveBeenCalled(); expect(onSave).not.toHaveBeenCalled();
    expect(screen.getByTestId("runtime_connection.form_status")).toHaveTextContent("Choose a supported feedback model");
    fireEvent.change(screen.getByTestId("runtime_connection.model"), {target:{value:models[1]}});
    fireEvent.click(screen.getByTestId("runtime_connection.save_connection"));
    expect(onSave).toHaveBeenCalledWith({clearSavedSecret:false,draft:expect.objectContaining({model:models[1],api_key:""})});
  });

  it("preserves the saved Groq selection on reopening Settings", () => {
    render(<RuntimeConnectionForm locale="it" variant="settings"
      initialDraft={{label:"",base_url:"",provider_choice:"groq",model:models[2]}} initialSecretState="present"
      onTest={vi.fn()} onSave={vi.fn()} />);
    expect(screen.getByTestId("runtime_connection.model")).toHaveValue(models[2]);
    expect(screen.getByTestId("runtime_connection.api_key")).toHaveValue("");
  });
});
