use base64::Engine;
use std::fs;
use std::path::Path;

const DEFAULT_ICON_PNG_BASE64: &str =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR4XmNgAAIAAAUAAQYUdaMAAAAASUVORK5CYII=";

fn main() {
    let icon_dir = Path::new("icons");
    let icon_path = icon_dir.join("icon.png");
    fs::create_dir_all(icon_dir).expect("failed to create src-tauri/icons");
    let icon_bytes = base64::engine::general_purpose::STANDARD
        .decode(DEFAULT_ICON_PNG_BASE64)
        .expect("failed to decode default Tauri icon");
    fs::write(&icon_path, icon_bytes).expect("failed to write default Tauri icon");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=tauri.conf.json");
    println!("cargo:rerun-if-changed=../dist");
    tauri_build::build()
}
