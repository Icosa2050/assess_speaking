#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::collections::HashMap;
use std::env;
use std::error::Error;
use std::path::{Path, PathBuf};
use std::process::Command;

use tauri::Manager;

const DESKTOP_API_BASE_URL_ENV_VAR: &str = "VOSTAVO_DESKTOP_API_BASE_URL";
const DEPLOYMENT_MODE_ENV_VAR: &str = "VOSTAVO_DEPLOYMENT_MODE";
const LAUNCH_MODE_ENV_VAR: &str = "VOSTAVO_LAUNCH_MODE";
const DESKTOP_PACKAGING_SAFE_ENV_VAR: &str = "VOSTAVO_DESKTOP_PACKAGING_SAFE";
const AUTH_MODE_ENV_VAR: &str = "VOSTAVO_AUTH_MODE";
const PYTHON_BIN_ENV_VAR: &str = "PYTHON_BIN";

struct DesktopRuntimeBridge {
    api_base_url: String,
    deployment_mode: String,
    launch_mode: String,
    packaging_safe: bool,
    auth_mode: String,
}

fn escape_js(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('\'', "\\'")
        .replace('"', "\\\"")
        .replace('`', "\\`")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\u{2028}', "\\u2028")
        .replace('\u{2029}', "\\u2029")
        .replace("</", "<\\/")
}

fn packaging_safe_from_env(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes"
    )
}

fn desktop_bridge_from_env() -> Option<DesktopRuntimeBridge> {
    let api_base_url = env::var(DESKTOP_API_BASE_URL_ENV_VAR).ok()?;
    let trimmed = api_base_url.trim().to_string();
    if trimmed.is_empty() {
        return None;
    }
    Some(DesktopRuntimeBridge {
        api_base_url: trimmed,
        deployment_mode: env::var(DEPLOYMENT_MODE_ENV_VAR).unwrap_or_else(|_| "local".to_string()),
        launch_mode: env::var(LAUNCH_MODE_ENV_VAR).unwrap_or_else(|_| "repo".to_string()),
        packaging_safe: env::var(DESKTOP_PACKAGING_SAFE_ENV_VAR)
            .map(|value| packaging_safe_from_env(&value))
            .unwrap_or(false),
        auth_mode: env::var(AUTH_MODE_ENV_VAR).unwrap_or_else(|_| "guest".to_string()),
    })
}

fn repo_root() -> PathBuf {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .unwrap_or_else(|| manifest_dir.to_path_buf())
}

fn parse_bootstrap_lines(stdout: &str) -> HashMap<String, String> {
    let mut values = HashMap::new();
    for line in stdout.lines() {
        if let Some((key, value)) = line.split_once('=') {
            values.insert(key.trim().to_string(), value.trim().to_string());
        }
    }
    values
}

fn command_is_available(executable: &str) -> bool {
    Command::new(executable).arg("--version").output().is_ok()
}

fn resolve_python_executable(root: &Path) -> PathBuf {
    if let Ok(value) = env::var(PYTHON_BIN_ENV_VAR) {
        let trimmed = value.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }

    for candidate in [
        root.join(".venv").join("bin").join("python"),
        root.join(".venv").join("Scripts").join("python.exe"),
    ] {
        if candidate.exists() {
            return candidate;
        }
    }

    let candidate_names: &[&str] = if cfg!(target_os = "windows") {
        &["python", "py"]
    } else {
        &["python3", "python"]
    };
    for candidate in candidate_names {
        if command_is_available(candidate) {
            return PathBuf::from(candidate);
        }
    }

    PathBuf::from(candidate_names[0])
}

fn bootstrap_from_repo_launcher() -> Result<DesktopRuntimeBridge, Box<dyn Error>> {
    let root = repo_root();
    let python_executable = resolve_python_executable(&root);
    let output = Command::new(&python_executable)
        .arg(root.join("scripts").join("bootstrap_backend.py"))
        .current_dir(&root)
        .output()
        .map_err(|error| {
            std::io::Error::other(format!(
                "failed to run desktop bootstrap with {}: {}",
                python_executable.display(),
                error
            ))
        })?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let detail = format!("desktop bootstrap failed: {}", stderr.trim());
        return Err(std::io::Error::other(detail).into());
    }

    let values = parse_bootstrap_lines(&String::from_utf8_lossy(&output.stdout));
    let api_base_url = values
        .get(DESKTOP_API_BASE_URL_ENV_VAR)
        .cloned()
        .unwrap_or_default();
    if api_base_url.trim().is_empty() {
        return Err(std::io::Error::other("desktop bootstrap did not return a backend URL").into());
    }

    Ok(DesktopRuntimeBridge {
        api_base_url,
        deployment_mode: values
            .get(DEPLOYMENT_MODE_ENV_VAR)
            .cloned()
            .unwrap_or_else(|| "local".to_string()),
        launch_mode: values
            .get(LAUNCH_MODE_ENV_VAR)
            .cloned()
            .unwrap_or_else(|| "repo".to_string()),
        packaging_safe: values
            .get(DESKTOP_PACKAGING_SAFE_ENV_VAR)
            .map(|value| packaging_safe_from_env(value))
            .unwrap_or(false),
        auth_mode: values
            .get(AUTH_MODE_ENV_VAR)
            .cloned()
            .unwrap_or_else(|| "guest".to_string()),
    })
}

fn desktop_runtime_bridge() -> Result<DesktopRuntimeBridge, Box<dyn Error>> {
    if let Some(runtime) = desktop_bridge_from_env() {
        return Ok(runtime);
    }

    bootstrap_from_repo_launcher().map_err(|error| {
        std::io::Error::other(format!(
            "could not initialize Vostavo desktop bridge from {} or launcher bootstrap: {}",
            DESKTOP_API_BASE_URL_ENV_VAR, error
        ))
        .into()
    })
}

fn desktop_bridge_script(runtime: &DesktopRuntimeBridge) -> String {
    format!(
        "window.__VOSTAVO_DESKTOP__ = Object.freeze({{ apiBaseUrl: '{}', deploymentMode: '{}', launchMode: '{}', packagingSafe: {}, authMode: '{}' }});",
        escape_js(&runtime.api_base_url),
        escape_js(&runtime.deployment_mode),
        escape_js(&runtime.launch_mode),
        if runtime.packaging_safe { "true" } else { "false" },
        escape_js(&runtime.auth_mode),
    )
}

fn main() {
    let runtime = desktop_runtime_bridge().expect("could not initialize Vostavo desktop bridge");
    tauri::Builder::default()
        .setup(move |app| {
            let window = app
                .get_webview_window("main")
                .ok_or_else(|| std::io::Error::other("main webview window is missing"))?;
            window.eval(desktop_bridge_script(&runtime))?;
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running Vostavo desktop shell");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escape_js_escapes_script_and_literal_boundaries() {
        let escaped = escape_js("a'b\"c`d\\e\nf\rg\u{2028}h\u{2029}</script>");

        assert_eq!(
            escaped,
            "a\\'b\\\"c\\`d\\\\e\\nf\\rg\\u2028h\\u2029<\\/script>"
        );
    }
}
