use base64::Engine;
use std::fs;
use std::path::Path;

const DEFAULT_ICON_PNG_BASE64: &str =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR4XmNgAAIAAAUAAQYUdaMAAAAASUVORK5CYII=";
const ICON_EXTENSIONS: &[&str] = &["png", "ico", "icns", "svg", "webp"];

fn has_existing_icon(icon_dir: &Path) -> bool {
    let Ok(entries) = fs::read_dir(icon_dir) else {
        return false;
    };
    entries.flatten().any(|entry| {
        let path = entry.path();
        path.is_file()
            && path
                .extension()
                .and_then(|extension| extension.to_str())
                .map(|extension| {
                    ICON_EXTENSIONS
                        .iter()
                        .any(|known| extension.eq_ignore_ascii_case(known))
                })
                .unwrap_or(false)
    })
}

fn write_default_icon_if_needed(icon_dir: &Path) {
    fs::create_dir_all(icon_dir).expect("failed to create src-tauri/icons");
    if has_existing_icon(icon_dir) {
        return;
    }

    let icon_path = icon_dir.join("icon.png");
    let icon_bytes = base64::engine::general_purpose::STANDARD
        .decode(DEFAULT_ICON_PNG_BASE64)
        .expect("failed to decode default Tauri icon");
    fs::write(&icon_path, icon_bytes).expect("failed to write default Tauri icon");
}

fn main() {
    write_default_icon_if_needed(Path::new("icons"));

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=tauri.conf.json");
    println!("cargo:rerun-if-changed=../dist");
    tauri_build::build()
}
