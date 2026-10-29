#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
mod desktop;
mod draft_store;
mod support;
use std::sync::Mutex;
use tauri::Manager;

const DRAFT_BRIDGE_JS: &str = r#"({
    list: () => window.__TAURI_INTERNALS__.invoke("draft_list"),
    recoveryIds: () => window.__TAURI_INTERNALS__.invoke("draft_recovery_ids"),
    saveSetup: (setup, draftId, expectedRevision) => window.__TAURI_INTERNALS__.invoke("draft_save_setup", { setup, draftId, expectedRevision }),
    beginTake: (draftId, expectedRevision, take) => window.__TAURI_INTERNALS__.invoke("draft_begin_take", { draftId, expectedRevision, take }),
    writeTakeChunk: (transferId, bytes) => {
      if (!Array.isArray(bytes) || bytes.length === 0 || bytes.length > 262144) return Promise.reject(new Error("Take chunk exceeds limit"));
      return window.__TAURI_INTERNALS__.invoke("draft_write_take_chunk", Uint8Array.from(bytes), { headers: { "X-Vostavo-Transfer-Id": transferId } });
    },
    abortTake: (transferId) => window.__TAURI_INTERNALS__.invoke("draft_abort_take", { transferId }),
    abortTakeForDraft: (draftId) => window.__TAURI_INTERNALS__.invoke("draft_abort_take_for_draft", { draftId }),
    finishTake: (transferId) => window.__TAURI_INTERNALS__.invoke("draft_finish_take", { transferId }),
    loadTakeChunk: (draftId, mediaId, index) => window.__TAURI_INTERNALS__.invoke("draft_load_take_chunk", { draftId, mediaId, index }),
    markSubmission: (draftId, expectedRevision, status, submission) => window.__TAURI_INTERNALS__.invoke("draft_mark_submission", { draftId, expectedRevision, status, submission }),
    discard: (draftId) => window.__TAURI_INTERNALS__.invoke("draft_discard", { draftId }),
    expire: () => window.__TAURI_INTERNALS__.invoke("draft_expire")
})"#;

fn bridge(runtime: &desktop::Runtime) -> String {
    format!("window.__VOSTAVO_DESKTOP__ = Object.freeze(Object.assign({{}}, {}, {{ saveLearnerBackup: (backupId) => window.__TAURI_INTERNALS__.invoke(\"save_learner_backup\", {{ backupId }}), draftSupportEmail: (bundleId, recipient) => window.__TAURI_INTERNALS__.invoke(\"draft_support_email\", {{ bundleId, recipient }}), saveSupportBundle: (bundleId) => window.__TAURI_INTERNALS__.invoke(\"save_support_bundle\", {{ bundleId }}), draftStore: Object.freeze({}) }}));", serde_json::json!({
        "apiBaseUrl": runtime.api_base_url, "sessionToken": runtime.session_token,
        "mediaToken": runtime.media_token,
        "deploymentMode": "local", "launchMode": runtime.launch_mode,
        "packagingSafe": true, "authMode": "guest"
    }), DRAFT_BRIDGE_JS)
}

fn main() {
    let previous_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |information| {
        use std::io::Write;
        let logs = desktop::data_root().join("logs");
        let _ = std::fs::create_dir_all(&logs);
        if let Ok(mut log) = std::fs::OpenOptions::new().create(true).append(true).open(logs.join("desktop-startup.log")) {
            let _ = writeln!(log, "Desktop startup failure: {information}");
        }
        previous_hook(information);
    }));
    let mut context = tauri::generate_context!();
    // macOS reads the Dock/application icon from Contents/Resources/Vostavo.icns.
    // The source PNG is 16-bit, which Tauri's window-icon decoder misinterprets
    // as twice as many RGBA pixels. Do not install that Windows-style icon.
    #[cfg(target_os = "macos")]
    context.set_default_window_icon(None);
    let app = tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![support::save_support_bundle, support::save_learner_backup, support::draft_support_email,
            draft_store::draft_list, draft_store::draft_recovery_ids, draft_store::draft_save_setup, draft_store::draft_begin_take,
            draft_store::draft_write_take_chunk, draft_store::draft_abort_take, draft_store::draft_abort_take_for_draft, draft_store::draft_finish_take, draft_store::draft_load_take_chunk,
            draft_store::draft_mark_submission, draft_store::draft_discard, draft_store::draft_expire])
        .manage(Mutex::new(None::<desktop::BackendOwner>))
        .manage(draft_store::unopened_store())
        .setup(|app| {
            tauri::WebviewWindowBuilder::new(app, "loading", tauri::WebviewUrl::App("loading.html".into()))
                .title("Vostavo").inner_size(560.0, 300.0).resizable(false).build()?;
            let handle = app.handle().clone();
            let config = app.config().app.windows.iter().find(|window| window.label == "main").cloned().ok_or("Main window configuration is missing")?;
            std::thread::spawn(move || {
                match desktop::BackendOwner::launch() {
                    Ok((owner, runtime)) => {
                        *handle.state::<Mutex<Option<desktop::BackendOwner>>>().lock().unwrap() = Some(owner);
                        draft_store::initialize_store(&handle.state::<draft_store::DraftState>());
                        if let Err(error) = support::cleanup_draft_attachments(&desktop::data_root()) {
                            eprintln!("Support attachment cleanup: {error}");
                        }
                        let result = tauri::WebviewWindowBuilder::from_config(&handle, &config)
                            .and_then(|builder| builder.on_navigation(|url| {
                                matches!((url.scheme(), url.host_str()), ("tauri", Some("localhost")) | ("http", Some("tauri.localhost")) | ("https", Some("tauri.localhost")))
                                    || cfg!(debug_assertions) && url.host_str() == Some("127.0.0.1") && url.port() == Some(4173)
                            }).initialization_script(&bridge(&runtime)).build());
                        if let Err(error) = result {
                            startup_failure(&handle, &error.to_string());
                        } else if let Some(window) = handle.get_webview_window("loading") { let _ = window.close(); }
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
