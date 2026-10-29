#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
mod desktop;
mod support;
use std::sync::{mpsc, Mutex};
use std::time::Duration;
use tauri::{webview::PageLoadEvent, Manager};

fn log_startup_issue(message: &str) {
    use std::io::Write;
    let logs = desktop::data_root().join("logs");
    let _ = std::fs::create_dir_all(&logs);
    if let Ok(mut log) = std::fs::OpenOptions::new().create(true).append(true).open(logs.join("desktop-startup.log")) {
        let _ = writeln!(log, "{message}");
    }
}

fn page_loaded(receiver: &mpsc::Receiver<()>, timeout: Duration) -> bool {
    receiver.recv_timeout(timeout).is_ok()
}

fn bridge(runtime: &desktop::Runtime) -> String {
    format!("window.__VOSTAVO_DESKTOP__ = Object.freeze(Object.assign({{}}, {}, {{ saveLearnerBackup: (backupId) => window.__TAURI_INTERNALS__.invoke(\"save_learner_backup\", {{ backupId }}), draftSupportEmail: (bundleId, recipient) => window.__TAURI_INTERNALS__.invoke(\"draft_support_email\", {{ bundleId, recipient }}), saveSupportBundle: (bundleId) => window.__TAURI_INTERNALS__.invoke(\"save_support_bundle\", {{ bundleId }}) }}));", serde_json::json!({
        "apiBaseUrl": runtime.api_base_url, "sessionToken": runtime.session_token,
        "mediaToken": runtime.media_token,
        "deploymentMode": "local", "launchMode": runtime.launch_mode,
        "packagingSafe": true, "authMode": "guest"
    }))
}

fn main() {
    let previous_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |information| {
        log_startup_issue(&format!("Desktop startup failure: {information}"));
        previous_hook(information);
    }));
    let mut context = tauri::generate_context!();
    // macOS reads the Dock/application icon from Contents/Resources/Vostavo.icns.
    // The source PNG is 16-bit, which Tauri's window-icon decoder misinterprets
    // as twice as many RGBA pixels. Do not install that Windows-style icon.
    #[cfg(target_os = "macos")]
    context.set_default_window_icon(None);
    let app = tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![support::save_support_bundle, support::save_learner_backup, support::draft_support_email])
        .manage(Mutex::new(None::<desktop::BackendOwner>))
        .setup(|app| {
            let (loading_loaded, loading_signal) = mpsc::channel();
            tauri::WebviewWindowBuilder::new(app, "loading", tauri::WebviewUrl::App("loading.html".into()))
                .title("Vostavo").inner_size(560.0, 300.0).resizable(false).visible(false)
                .on_page_load(move |window, payload| {
                    if payload.event() == PageLoadEvent::Finished && window.show().is_ok() {
                        let _ = loading_loaded.send(());
                    }
                }).build()?;
            let loading_handle = app.handle().clone();
            std::thread::spawn(move || {
                if !page_loaded(&loading_signal, Duration::from_secs(3)) {
                    log_startup_issue("Loading page did not finish in 3 seconds; revealing the loading window");
                    if let Some(window) = loading_handle.get_webview_window("loading") { let _ = window.show(); }
                }
            });
            let handle = app.handle().clone();
            let config = app.config().app.windows.iter().find(|window| window.label == "main").cloned().ok_or("Main window configuration is missing")?;
            std::thread::spawn(move || {
                if let Err(error) = support::cleanup_draft_attachments(&desktop::data_root()) {
                    eprintln!("Support attachment cleanup: {error}");
                }
                match desktop::BackendOwner::launch() {
                    Ok((owner, runtime)) => {
                        *handle.state::<Mutex<Option<desktop::BackendOwner>>>().lock().unwrap() = Some(owner);
                        let (main_loaded, main_signal) = mpsc::channel();
                        let result = tauri::WebviewWindowBuilder::from_config(&handle, &config)
                            .and_then(|builder| builder.on_navigation(|url| {
                                matches!((url.scheme(), url.host_str()), ("tauri", Some("localhost")) | ("http", Some("tauri.localhost")) | ("https", Some("tauri.localhost")))
                                    || cfg!(debug_assertions) && url.host_str() == Some("127.0.0.1") && url.port() == Some(4173)
                            }).visible(false).on_page_load(move |window, payload| {
                                if payload.event() == PageLoadEvent::Finished && window.show().is_ok() {
                                    let _ = main_loaded.send(());
                                    if let Some(loading) = window.app_handle().get_webview_window("loading") { let _ = loading.close(); }
                                }
                            }).initialization_script(&bridge(&runtime)).build());
                        match result {
                            Ok(_) => {
                                let fallback_handle = handle.clone();
                                std::thread::spawn(move || {
                                    if !page_loaded(&main_signal, Duration::from_secs(8)) {
                                        log_startup_issue("Main page did not finish in 8 seconds; revealing the main window");
                                        if let Some(main) = fallback_handle.get_webview_window("main") {
                                            if main.show().is_ok() {
                                                if let Some(loading) = fallback_handle.get_webview_window("loading") { let _ = loading.close(); }
                                            } else {
                                                log_startup_issue("Main window could not be revealed after the page-load timeout");
                                            }
                                        }
                                    }
                                });
                            }
                            Err(error) => startup_failure(&handle, &error.to_string()),
                        }
                    }
                    Err(error) => startup_failure(&handle, &error),
                }
            });
            Ok(())
        })
        .build(context)
        .expect("Could not create the Vostavo application");
    app.run(|handle, event| {
        if let tauri::RunEvent::Exit = event {
            handle.state::<Mutex<Option<desktop::BackendOwner>>>().lock().unwrap().take();
        }
    });
}

fn startup_failure(handle: &tauri::AppHandle, error: &str) {
    let message = format!("{error}\n\nStartup log: {}", desktop::data_root().join("logs/desktop-startup.log").display());
    rfd::MessageDialog::new().set_title("Vostavo").set_description(&message).set_level(rfd::MessageLevel::Error).show();
    handle.exit(1);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_load_signal_prevents_the_fallback() {
        let (sender, receiver) = mpsc::channel();
        sender.send(()).unwrap();
        assert!(page_loaded(&receiver, Duration::ZERO));
    }

    #[test]
    fn missing_or_disconnected_signal_uses_the_bounded_fallback() {
        let (sender, receiver) = mpsc::channel();
        assert!(!page_loaded(&receiver, Duration::ZERO));
        drop(sender);
        assert!(!page_loaded(&receiver, Duration::ZERO));
    }
}
