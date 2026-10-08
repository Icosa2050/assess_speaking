#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
mod desktop;
mod support;
use std::sync::Mutex;
use tauri::Manager;

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
        .invoke_handler(tauri::generate_handler![support::save_support_bundle, support::save_learner_backup, support::draft_support_email])
        .manage(Mutex::new(None::<desktop::BackendOwner>))
        .setup(|app| {
            tauri::WebviewWindowBuilder::new(app, "loading", tauri::WebviewUrl::App("loading.html".into()))
                .title("Vostavo").inner_size(560.0, 300.0).resizable(false).build()?;
            let handle = app.handle().clone();
            let config = app.config().app.windows.iter().find(|window| window.label == "main").cloned().ok_or("Main window configuration is missing")?;
            std::thread::spawn(move || {
                match desktop::BackendOwner::launch() {
                    Ok((owner, runtime)) => {
                        *handle.state::<Mutex<Option<desktop::BackendOwner>>>().lock().unwrap() = Some(owner);
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
