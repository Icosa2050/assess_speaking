use std::fs::{self, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};

pub fn bundle_source(root: &Path, id: &str) -> Result<PathBuf, String> {
    if !id.starts_with("bundle_") || id.len() != 19 || !id[7..].bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
        return Err("Invalid support package identifier".into());
    }
    let directory = root.join("tmp/support-bundles");
    let source = directory.join(format!("{id}.zip"));
    let metadata = fs::symlink_metadata(&source).map_err(|_| "Support package expired or is unavailable. Create it again.".to_string())?;
    if !metadata.is_file() || metadata.file_type().is_symlink() {
        return Err("Invalid support package file".into());
    }
    let age = metadata.modified().map_err(|_| "Support package timestamp is unavailable".to_string())?
        .elapsed().map_err(|_| "Invalid support package timestamp".to_string())?;
    if age > std::time::Duration::from_secs(24 * 60 * 60) {
        return Err("Support package expired. Create it again.".into());
    }
    let canonical = source.canonicalize().map_err(|_| "Support package is unavailable".to_string())?;
    let root = root.canonicalize().map_err(|_| "Support package root is unavailable".to_string())?;
    let directory = directory.canonicalize().map_err(|_| "Support package directory is unavailable".to_string())?;
    if !directory.starts_with(root) || canonical.parent() != Some(directory.as_path()) {
        return Err("Invalid support package location".into());
    }
    Ok(canonical)
}

pub fn atomic_save(source: &Path, destination: &Path) -> io::Result<()> {
    if destination.exists() && source.canonicalize()? == destination.canonicalize()? { return Ok(()); }
    let parent = destination.parent().ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "Missing destination directory"))?;
    // A same-directory temporary file ensures an interrupted copy never damages
    // an existing destination. create_new also refuses collisions and symlinks.
    let nonce = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_err(io::Error::other)?.as_nanos();
    let temporary = parent.join(format!(".vostavo-support-{}-{nonce}.tmp", std::process::id()));
    let result = (|| {
        let mut input = fs::File::open(source)?;
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)] {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut output = options.open(&temporary)?;
        io::copy(&mut input, &mut output)?;
        output.sync_all()?;
        drop(output);
        fs::rename(&temporary, destination)
    })();
    if result.is_err() { let _ = fs::remove_file(&temporary); }
    result
}

#[tauri::command]
pub async fn save_support_bundle(window: tauri::WebviewWindow, bundle_id: String) -> Result<serde_json::Value, String> {
    if window.label() != "main" { return Err("Support packages can only be saved from the main window".into()); }
    bundle_source(&crate::desktop::data_root(), &bundle_id)?;
    let (sender, receiver) = std::sync::mpsc::channel();
    let suggested = format!("Vostavo-support-{bundle_id}.zip");
    window.run_on_main_thread(move || {
        let chosen = rfd::FileDialog::new().set_title("Save Vostavo support package")
            .set_file_name(suggested).add_filter("ZIP archive", &["zip"]).save_file();
        let _ = sender.send(chosen);
    }).map_err(|_| "Could not open the save dialog".to_string())?;
    tauri::async_runtime::spawn_blocking(move || {
        let destination = receiver.recv().map_err(|_| "Save dialog closed unexpectedly".to_string())?;
        let Some(destination) = destination else { return Ok(serde_json::json!({"status":"cancelled"})); };
        // Validate again after the dialog: cleanup may have expired the package.
        let source = bundle_source(&crate::desktop::data_root(), &bundle_id)?;
        atomic_save(&source, &destination).map_err(|_| "Could not save the support package. Check free space and folder permissions, then retry.".to_string())?;
        Ok(serde_json::json!({"status":"saved"}))
    }).await.map_err(|_| "Support package save failed".to_string())?
}

#[cfg(test)]
mod tests {
    use super::*;
    fn directory() -> PathBuf {
        let path = std::env::temp_dir().join(format!("vostavo-support-test-{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        fs::create_dir_all(path.join("tmp/support-bundles")).unwrap(); path
    }
    #[test]
    fn refuses_arbitrary_paths_missing_files_and_symlinks() {
        let root=directory();
        for id in ["../private", "bundle_ABCDEF123456", "bundle_0123456789ab"] { assert!(bundle_source(&root,id).is_err()); }
        let file=root.join("tmp/support-bundles/bundle_0123456789ab.zip"); fs::write(&file,b"fixture").unwrap();
        assert!(bundle_source(&root,"bundle_0123456789ab").is_ok());
        #[cfg(unix)] {
            fs::remove_file(&file).unwrap(); std::os::unix::fs::symlink(root.join("private"),&file).unwrap();
            assert!(bundle_source(&root,"bundle_0123456789ab").is_err());
        }
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn backup_sources_are_owned_and_recipients_cannot_inject_scripts() {
        let root = directory(); fs::create_dir(root.join("maintenance")).unwrap();
        let id = "0123456789abcdef0123456789abcdef";
        fs::write(root.join(format!("maintenance/backup_{id}.zip")), b"fixture").unwrap();
        assert!(backup_source(&root, id).is_ok());
        assert!(backup_source(&root, "../source").is_err());
        for recipient in ["support@example.com", "help+desktop@example.org"] { assert!(valid_recipient(recipient)); }
        for recipient in ["", "a@", "a@@example.com", "a@example.com\nrun", "a@example.com;", "\"a\"@example.com"] { assert!(!valid_recipient(recipient)); }
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn replaces_only_after_complete_copy_and_preserves_existing_file_on_failure() {
        let root=directory();let source=root.join("source.zip");let destination=root.join("saved.zip");
        fs::write(&destination,b"old").unwrap();
        assert!(atomic_save(&source,&destination).is_err());assert_eq!(fs::read(&destination).unwrap(),b"old");
        fs::write(&source,b"complete-fixture").unwrap();atomic_save(&source,&destination).unwrap();
        assert_eq!(fs::read(&destination).unwrap(),b"complete-fixture");
        #[cfg(unix)] {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(fs::metadata(&destination).unwrap().permissions().mode() & 0o777, 0o600);
        }
        assert!(!fs::read_dir(&root).unwrap().any(|e|e.unwrap().file_name().to_string_lossy().ends_with(".tmp")));
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn mail_attachment_survives_download_cleanup() {
        let root = directory();
        let source = root.join("tmp/support-bundles/bundle_0123456789ab.zip");
        fs::write(&source, b"private draft fixture").unwrap();
        let attachment = draft_attachment(&root, &source).unwrap();
        fs::remove_file(source).unwrap();
        assert_eq!(fs::read(attachment).unwrap(), b"private draft fixture");
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn expires_only_old_owned_draft_attachments() {
        let root = directory();
        let drafts = root.join("support-email-drafts"); fs::create_dir(&drafts).unwrap();
        let old = drafts.join("bundle_0123456789ab.zip");
        let fresh = drafts.join("bundle_0123456789ac.zip");
        let unrelated = drafts.join("learner.zip");
        let now = std::time::SystemTime::now();
        let expired = now - std::time::Duration::from_secs(8 * 24 * 60 * 60);
        for file in [&old, &fresh, &unrelated] { fs::write(file, b"private fixture").unwrap(); }
        for file in [&old, &unrelated] {
            fs::File::options().write(true).open(file).unwrap()
                .set_times(fs::FileTimes::new().set_modified(expired)).unwrap();
        }
        #[cfg(unix)] {
            std::os::unix::fs::symlink(&unrelated, drafts.join("bundle_0123456789ad.zip")).unwrap();
        }
        cleanup_draft_attachments(&root).unwrap();
        assert!(!old.exists()); assert!(fresh.exists()); assert!(unrelated.exists());
        #[cfg(unix)] { assert!(fs::symlink_metadata(drafts.join("bundle_0123456789ad.zip")).is_ok()); }
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn startup_cleanup_does_not_follow_a_linked_draft_directory() {
        let root = directory();
        assert!(cleanup_draft_attachments(&root).is_ok());
        #[cfg(unix)] {
            let external = directory();
            let retained = external.join("bundle_0123456789ab.zip");
            fs::write(&retained, b"retained fixture").unwrap();
            std::os::unix::fs::symlink(&external, root.join("support-email-drafts")).unwrap();
            assert!(cleanup_draft_attachments(&root).is_err());
            assert_eq!(fs::read(retained).unwrap(), b"retained fixture");
            fs::remove_dir_all(external).unwrap();
        }
        fs::remove_dir_all(root).unwrap();
    }
}

pub fn backup_source(root: &Path, id: &str) -> Result<PathBuf, String> {
    if id.len() != 32 || !id.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
        return Err("Invalid learner backup identifier".into());
    }
    let directory = root.join("maintenance");
    let source = directory.join(format!("backup_{id}.zip"));
    let metadata = fs::symlink_metadata(&source).map_err(|_| "Backup is unavailable. Create it again.".to_string())?;
    if !metadata.is_file() || metadata.file_type().is_symlink() { return Err("Invalid backup file".into()); }
    let root = root.canonicalize().map_err(|_| "Backup root is unavailable".to_string())?;
    let directory = directory.canonicalize().map_err(|_| "Backup directory is unavailable".to_string())?;
    let canonical = source.canonicalize().map_err(|_| "Backup is unavailable".to_string())?;
    if !directory.starts_with(root) || canonical.parent() != Some(directory.as_path()) { return Err("Invalid backup location".into()); }
    Ok(canonical)
}

#[tauri::command]
pub async fn save_learner_backup(window: tauri::WebviewWindow, backup_id: String) -> Result<serde_json::Value, String> {
    if window.label() != "main" { return Err("Backups can only be saved from the main window".into()); }
    backup_source(&crate::desktop::data_root(), &backup_id)?;
    let (sender, receiver) = std::sync::mpsc::channel();
    let suggested = format!("Vostavo-backup-{backup_id}.zip");
    window.run_on_main_thread(move || {
        let chosen = rfd::FileDialog::new().set_title("Save Vostavo learner backup")
            .set_file_name(suggested).add_filter("ZIP archive", &["zip"]).save_file();
        let _ = sender.send(chosen);
    }).map_err(|_| "Could not open the save dialog".to_string())?;
    tauri::async_runtime::spawn_blocking(move || {
        let destination = receiver.recv().map_err(|_| "Save dialog closed unexpectedly".to_string())?;
        let Some(destination) = destination else { return Ok(serde_json::json!({"status":"cancelled"})); };
        let source = backup_source(&crate::desktop::data_root(), &backup_id)?;
        atomic_save(&source, &destination).map_err(|_| "Could not save the backup. Check free space and permissions, then retry.".to_string())?;
        Ok(serde_json::json!({"status":"saved"}))
    }).await.map_err(|_| "Backup save failed".to_string())?
}

pub fn valid_recipient(recipient: &str) -> bool {
    let mut parts = recipient.split('@');
    let local = parts.next().unwrap_or("");
    let domain = parts.next().unwrap_or("");
    recipient.len() <= 254 && !local.is_empty() && local.chars().any(|c| c.is_ascii_alphanumeric()) && !domain.contains("..") && domain.contains('.') && !domain.starts_with('.') && !domain.ends_with('.')
        && parts.next().is_none() && recipient.bytes().all(|b| b.is_ascii_alphanumeric() || b"._+-@".contains(&b))
}

fn prune_draft_attachments(directory: &Path, now: std::time::SystemTime) -> Result<(), String> {
    for entry in fs::read_dir(directory).map_err(|_| "Could not check retained email attachments".to_string())? {
        let entry = entry.map_err(|_| "Could not check retained email attachments".to_string())?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let owned = name.strip_prefix("bundle_").and_then(|n| n.strip_suffix(".zip"))
            .is_some_and(|id| id.len() == 12 && id.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)));
        let metadata = fs::symlink_metadata(entry.path()).map_err(|_| "Could not check retained email attachments".to_string())?;
        let expired = metadata.modified().ok().and_then(|modified| now.duration_since(modified).ok())
            .is_some_and(|age| age > std::time::Duration::from_secs(7 * 24 * 60 * 60));
        if owned && metadata.is_file() && !metadata.file_type().is_symlink() && expired {
            fs::remove_file(entry.path()).map_err(|_| "Could not remove an expired email attachment".to_string())?;
        }
    }
    Ok(())
}

fn validate_draft_directory(root: &Path, directory: &Path) -> Result<(), String> {
    let metadata = fs::symlink_metadata(directory).map_err(|_| "Email attachment directory is unavailable".to_string())?;
    if !metadata.is_dir() || metadata.file_type().is_symlink()
        || !directory.canonicalize().map_err(|_| "Email attachment directory is unavailable".to_string())?.starts_with(root.canonicalize().map_err(|_| "App data is unavailable".to_string())?) {
        return Err("Invalid email attachment directory".into());
    }
    Ok(())
}

pub fn cleanup_draft_attachments(root: &Path) -> Result<(), String> {
    let directory = root.join("support-email-drafts");
    match fs::symlink_metadata(&directory) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(_) => return Err("Email attachment directory is unavailable".into()),
        Ok(_) => {}
    }
    validate_draft_directory(root, &directory)?;
    prune_draft_attachments(&directory, std::time::SystemTime::now())
}

pub fn draft_attachment(root: &Path, source: &Path) -> Result<PathBuf, String> {
    // Mail may read an attachment after its automation reply. Keep a private
    // draft-owned copy independent of the expiring download package.
    let directory = root.join("support-email-drafts");
    fs::create_dir_all(&directory).map_err(|_| "Could not retain the email attachment".to_string())?;
    validate_draft_directory(root, &directory)?;
    #[cfg(unix)] {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&directory, fs::Permissions::from_mode(0o700)).map_err(|_| "Could not protect email attachments".to_string())?;
    }
    // Expire app-owned copies on the next draft handoff; keep newer copies so
    // Mail has time to import attachments independently of download cleanup.
    prune_draft_attachments(&directory, std::time::SystemTime::now())?;
    let destination = directory.join(source.file_name().ok_or("Missing attachment filename")?);
    atomic_save(source, &destination).map_err(|_| "Could not retain the email attachment. Save the ZIP and attach it manually.".to_string())?;
    Ok(destination)
}

#[tauri::command]
pub async fn draft_support_email(window: tauri::WebviewWindow, bundle_id: String, recipient: String) -> Result<serde_json::Value, String> {
    if window.label() != "main" || !valid_recipient(&recipient) { return Err("Enter a valid support email address in the main window".into()); }
    let source = bundle_source(&crate::desktop::data_root(), &bundle_id)?;
    #[cfg(target_os = "macos")]
    {
        tauri::async_runtime::spawn_blocking(move || {
            let source = draft_attachment(&crate::desktop::data_root(), &source)?;
            // Parameters are argv, never executable AppleScript text. This makes
            // an editable Mail draft; the learner reviews and sends it in Mail.
            let script = r#"on run arguments
                set recipientAddress to item 1 of arguments
                set attachmentPath to item 2 of arguments
                tell application "Mail"
                    set messageDraft to make new outgoing message with properties {subject:"Vostavo support package", content:"Please describe the problem and the steps to reproduce it here.\n\n", visible:true}
                    tell messageDraft
                        make new to recipient at end of to recipients with properties {address:recipientAddress}
                        make new attachment with properties {file name:POSIX file attachmentPath} at after last paragraph of content
                    end tell
                    delay 1
                    if (count of attachments of content of messageDraft) is 0 then error "Mail did not attach the ZIP"
                    activate
                end tell
                return "drafted"
            end run"#;
            let mut child = std::process::Command::new("/usr/bin/osascript").args(["-e", script, "--", &recipient]).arg(source)
                .stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null()).spawn()
                .map_err(|_| "Could not open Mail. Save the ZIP and attach it manually.".to_string())?;
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
            loop {
                match child.try_wait().map_err(|_| "Mail draft status is unavailable".to_string())? {
                    Some(status) if status.success() => return Ok(serde_json::json!({"status":"drafted"})),
                    Some(_) => return Err("Mail could not verify the attachment. Discard any incomplete draft, allow Vostavo to control Mail, then retry or attach the saved ZIP manually.".into()),
                    None if std::time::Instant::now() >= deadline => { let _ = child.kill(); let _ = child.wait(); return Err("Mail did not respond. Check the permission prompt or save and attach the ZIP manually.".into()); },
                    None => std::thread::sleep(std::time::Duration::from_millis(100)),
                }
            }
        }).await.map_err(|_| "Support email draft failed".to_string())?
    }
    #[cfg(not(target_os = "macos"))]
    { let _ = source; Err("Email attachment drafts currently require macOS Mail. Save the ZIP and attach it manually.".into()) }
}
