//! Private, local-only draft storage. No renderer value is used as a path.
use aes_gcm::{
    aead::{Aead, KeyInit, Payload},
    Aes256Gcm, Nonce,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet},
    fs::{self, File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    os::unix::fs::{DirBuilderExt, OpenOptionsExt, PermissionsExt},
    path::{Path, PathBuf},
    sync::Mutex,
    time::{SystemTime, UNIX_EPOCH},
};

const CHUNK: usize = 256 * 1024;
const MAX_TAKE: u64 = 100 * 1024 * 1024;
const MAX_TOTAL: u64 = 250 * 1024 * 1024;
const RESERVE: u64 = 512 * 1024 * 1024;
const TTL: u64 = 7 * 24 * 60 * 60 * 1000;
const MAGIC: &[u8; 4] = b"VD01";
const KEY_SERVICE: &str = "com.vostavo.desktop.drafts.v1";
const KEY_ACCOUNT: &str = "content-key";

type StoreResult<T> = Result<T, String>;
fn err(message: &str) -> String {
    message.to_owned()
}
fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
fn random<const N: usize>() -> StoreResult<[u8; N]> {
    let mut bytes = [0; N];
    File::open("/dev/urandom")
        .and_then(|mut file| file.read_exact(&mut bytes))
        .map_err(|_| err("Secure randomness is unavailable"))?;
    Ok(bytes)
}
fn new_id() -> StoreResult<String> {
    Ok(random::<16>()?.iter().map(|b| format!("{b:02x}")).collect())
}
fn encrypted_media_size(size: u64) -> u64 {
    4 + size + size.div_ceil(CHUNK as u64) * 36
}
fn valid_id(id: &str) -> bool {
    id.len() == 32
        && id
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}
fn checked_id(id: &str) -> StoreResult<&str> {
    if valid_id(id) {
        Ok(id)
    } else {
        Err(err("Invalid draft identifier"))
    }
}
fn normalized_speaker(text: &str) -> String {
    text.trim().to_lowercase()
}
fn text_ok(text: &str, max: usize) -> bool {
    text.len() <= max && !text.contains('\0')
}
fn sync_dir(path: &Path) -> StoreResult<()> {
    File::open(path)
        .and_then(|f| f.sync_all())
        .map_err(|_| err("Could not sync draft directory"))
}

fn secure_dir(path: &Path) -> StoreResult<()> {
    match fs::symlink_metadata(path) {
        Ok(meta) if meta.is_dir() && !meta.file_type().is_symlink() => {}
        Ok(_) => return Err(err("Draft directory is unsafe")),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => fs::DirBuilder::new()
            .mode(0o700)
            .create(path)
            .map_err(|_| err("Could not create draft directory"))?,
        Err(_) => return Err(err("Draft directory is unavailable")),
    }
    if fs::symlink_metadata(path)
        .map_err(|_| err("Draft directory is unavailable"))?
        .uid()
        != unsafe { libc::geteuid() }
    {
        return Err(err("Draft directory has another owner"));
    }
    fs::set_permissions(path, fs::Permissions::from_mode(0o700))
        .map_err(|_| err("Could not protect draft directory"))
}
fn regular_file(path: &Path) -> StoreResult<bool> {
    match fs::symlink_metadata(path) {
        Ok(meta)
            if meta.is_file()
                && !meta.file_type().is_symlink()
                && meta.nlink() == 1
                && meta.uid() == unsafe { libc::geteuid() }
                && meta.permissions().mode() & 0o777 == 0o600 =>
        {
            Ok(true)
        }
        Ok(_) => Err(err("Draft file is unsafe")),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(_) => Err(err("Draft file is unavailable")),
    }
}
use std::os::unix::fs::MetadataExt;
fn create_private(path: &Path) -> StoreResult<File> {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)
        .map_err(|_| err("Could not create private draft file"))
}
fn open_private(path: &Path) -> StoreResult<File> {
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)
        .map_err(|_| err("Saved draft file is unavailable"))?;
    let meta = file
        .metadata()
        .map_err(|_| err("Saved draft file is unavailable"))?;
    if !meta.is_file()
        || meta.nlink() != 1
        || meta.uid() != unsafe { libc::geteuid() }
        || meta.permissions().mode() & 0o777 != 0o600
    {
        return Err(err("Draft file is unsafe"));
    }
    Ok(file)
}

trait KeySource: Send + Sync {
    fn load(&self) -> StoreResult<Option<[u8; 32]>>;
    fn create(&self, key: &[u8; 32]) -> StoreResult<()>;
}
struct MacKeychain {
    service: String,
}
impl MacKeychain {
    fn production() -> Self {
        Self {
            service: KEY_SERVICE.into(),
        }
    }
}
#[cfg(target_os = "macos")]
impl KeySource for MacKeychain {
    fn load(&self) -> StoreResult<Option<[u8; 32]>> {
        use security_framework::passwords::{generic_password, PasswordOptions};
        let mut options = PasswordOptions::new_generic_password(&self.service, KEY_ACCOUNT);
        options.set_access_synchronized(Some(false));
        match generic_password(options) {
            Ok(value) => value
                .try_into()
                .map(Some)
                .map_err(|_| err("Saved take cannot be opened on this Mac")),
            Err(error) if error.code() == -25300 => Ok(None),
            Err(error) => Err(format!(
                "Mac Keychain is unavailable; local save is disabled (code {})",
                error.code()
            )),
        }
    }
    fn create(&self, key: &[u8; 32]) -> StoreResult<()> {
        use security_framework::passwords::{set_generic_password_options, PasswordOptions};
        let mut options = PasswordOptions::new_generic_password(&self.service, KEY_ACCOUNT);
        options.set_access_synchronized(Some(false));
        set_generic_password_options(key, options).map_err(|error| {
            format!(
                "Mac Keychain is unavailable; local save is disabled (code {})",
                error.code()
            )
        })
    }
}
#[cfg(not(target_os = "macos"))]
impl KeySource for MacKeychain {
    fn load(&self) -> StoreResult<Option<[u8; 32]>> {
        Err(err("Local drafts require macOS Keychain"))
    }
    fn create(&self, _: &[u8; 32]) -> StoreResult<()> {
        Err(err("Local drafts require macOS Keychain"))
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Setup {
    session_id: String,
    speaker_id: String,
    learning_language: String,
    learning_language_label: String,
    cefr_level: String,
    theme_id: String,
    theme_label: String,
    task_family: String,
    duration_sec: u32,
    prompt_id: String,
    prompt_text: String,
    retry_of_session_id: String,
    coaching_goal: String,
    retry_goal_eligible: bool,
}
impl Setup {
    fn validate(&self) -> StoreResult<String> {
        for (text, max) in [
            (&self.session_id, 128),
            (&self.speaker_id, 256),
            (&self.learning_language, 2),
            (&self.learning_language_label, 100),
            (&self.theme_id, 256),
            (&self.theme_label, 1000),
            (&self.task_family, 64),
            (&self.prompt_id, 256),
            (&self.prompt_text, 12000),
            (&self.retry_of_session_id, 128),
            (&self.coaching_goal, 12000),
        ] {
            if !text_ok(text, max) {
                return Err(err("Draft setup exceeds text limits"));
            }
        }
        let speaker = normalized_speaker(&self.speaker_id);
        if speaker.is_empty()
            || !["en", "it"].contains(&self.learning_language.as_str())
            || !["B1", "B2", "C1"].contains(&self.cefr_level.as_str())
            || ![
                "travel_narrative",
                "personal_experience",
                "opinion_monologue",
                "picture_description",
                "free_monologue",
            ]
            .contains(&self.task_family.as_str())
            || ![60, 90, 120, 180].contains(&self.duration_sec)
        {
            return Err(err("Invalid draft setup"));
        }
        Ok(speaker)
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Take {
    #[serde(default)]
    media_id: String,
    original_filename: String,
    mime: String,
    size: u64,
    sha256: String,
    duration_sec: f64,
    capture_ended_at: u64,
    input_method: String,
}
impl Take {
    fn validate(&self) -> StoreResult<()> {
        checked_id(&self.media_id)?;
        if !text_ok(&self.original_filename, 256)
            || self.original_filename.is_empty()
            || self.original_filename.chars().any(char::is_control)
            || self.original_filename.contains('/')
            || self.original_filename.contains('\\')
            || ![
                "audio/webm",
                "audio/webm;codecs=opus",
                "audio/mp4",
                "audio/mp4;codecs=mp4a.40.2",
                "audio/wav",
                "audio/mpeg",
                "audio/ogg",
                "audio/ogg;codecs=opus",
                "audio/x-wav",
            ]
            .contains(&self.mime.as_str())
            || self.size == 0
            || self.size > MAX_TAKE
            || self.sha256.len() != 64
            || !self
                .sha256
                .bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
            || !self.duration_sec.is_finite()
            || self.duration_sec <= 0.0
            || self.duration_sec > 1200.0
            || !["record", "upload"].contains(&self.input_method.as_str())
        {
            return Err(err("Invalid take metadata"));
        }
        Ok(())
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Submission {
    upload_request_id: String,
    audio_id: String,
    assessment_request_id: String,
    assessment_id: String,
    request_fingerprint: String,
    frozen_request_body: String,
    last_confirmed_phase: String,
}
impl Submission {
    fn validate(&self) -> StoreResult<()> {
        if [
            &self.upload_request_id,
            &self.audio_id,
            &self.assessment_request_id,
            &self.assessment_id,
            &self.request_fingerprint,
        ]
        .iter()
        .any(|s| !text_ok(s, 128))
            || !text_ok(&self.frozen_request_body, 16_384)
            || !["none", "upload", "assessment", "completed"]
                .contains(&self.last_confirmed_phase.as_str())
        {
            return Err(err("Invalid submission receipt"));
        }
        if !self.frozen_request_body.is_empty() {
            let body: serde_json::Value = serde_json::from_str(&self.frozen_request_body)
                .map_err(|_| err("Invalid submission request"))?;
            if !safe_request_body(&body) {
                return Err(err("Submission request contains prohibited fields"));
            }
        }
        Ok(())
    }
    fn validate_for(&self, status: &str) -> StoreResult<()> {
        self.validate()?;
        if self.upload_request_id.is_empty()
            || matches!(status, "uploaded" | "assessment_uncertain" | "assessing")
                && self.audio_id.is_empty()
            || matches!(status, "assessment_uncertain" | "assessing")
                && (self.assessment_request_id.is_empty()
                    || self.request_fingerprint.is_empty()
                    || self.frozen_request_body.is_empty())
            || status == "assessing" && self.assessment_id.is_empty()
        {
            return Err(err("Incomplete submission receipt"));
        }
        if matches!(status, "assessment_uncertain" | "assessing") {
            let body: serde_json::Value = serde_json::from_str(&self.frozen_request_body)
                .map_err(|_| err("Invalid submission request"))?;
            if body["request_id"].as_str() != Some(self.assessment_request_id.as_str())
                || body["audio_id"].as_str() != Some(self.audio_id.as_str())
                || body["sharing_fingerprint"].as_str() != Some(self.request_fingerprint.as_str())
            {
                return Err(err("Submission request identity mismatch"));
            }
        }
        Ok(())
    }
}
fn safe_request_body(value: &serde_json::Value) -> bool {
    let Some(fields) = value.as_object() else {
        return false;
    };
    const ALLOWED: &[&str] = &[
        "sharing_fingerprint",
        "request_id",
        "resume_assessment_id",
        "audio_id",
        "whisper",
        "provider",
        "llm_model",
        "expected_language",
        "feedback_language",
        "speaker_id",
        "task_family",
        "theme",
        "target_duration_sec",
        "target_cefr",
        "prompt_id",
        "prompt_text",
        "retry_of_session_id",
        "language_profile_key",
        "label",
        "notes",
        "dry_run",
    ];
    fields.iter().all(|(name, value)| {
        ALLOWED.contains(&name.as_str()) && !value.is_object() && !value.is_array()
    })
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Manifest {
    version: u8,
    draft_id: String,
    speaker_id: String,
    setup: Setup,
    created_at: u64,
    updated_at: u64,
    expires_at: u64,
    revision: u64,
    status: String,
    take: Option<Take>,
    submission: Option<Submission>,
}
impl std::fmt::Debug for Manifest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Manifest { private }")
    }
}
impl Manifest {
    fn validate(&self, id: &str, now: u64) -> StoreResult<()> {
        if self.version != 1
            || self.draft_id != id
            || self.speaker_id != self.setup.validate()?
            || self.created_at > self.updated_at
            || self.updated_at > now.saturating_add(60_000)
            || self.expires_at < self.created_at
            || self.expires_at > self.updated_at.saturating_add(TTL)
            || self.revision == 0
            || ![
                "setup",
                "take_ready",
                "upload_uncertain",
                "uploaded",
                "assessment_uncertain",
                "assessing",
            ]
            .contains(&self.status.as_str())
        {
            return Err(err("Invalid saved draft"));
        }
        if let Some(take) = &self.take {
            take.validate()?;
            if self.expires_at != take.capture_ended_at.saturating_add(TTL) {
                return Err(err("Invalid saved take expiry"));
            }
            if self.status == "setup" {
                return Err(err("Invalid saved draft state"));
            }
        } else if self.status != "setup" {
            return Err(err("Invalid saved draft state"));
        }
        if self.status == "setup" || self.status == "take_ready" {
            if self.submission.is_some() {
                return Err(err("Invalid saved draft state"));
            }
        } else {
            self.submission
                .as_ref()
                .ok_or_else(|| err("Missing submission receipt"))?
                .validate_for(&self.status)?;
        }
        Ok(())
    }
}

fn seal(key: &[u8; 32], id: &str, kind: &str, plain: &[u8]) -> StoreResult<Vec<u8>> {
    let cipher = Aes256Gcm::new_from_slice(key).map_err(|_| err("Encryption is unavailable"))?;
    let nonce = random::<12>()?;
    let aad = format!("vostavo-draft-v1:{kind}:{id}");
    let nonce_ref =
        Nonce::try_from(nonce.as_slice()).map_err(|_| err("Encryption is unavailable"))?;
    let encrypted = cipher
        .encrypt(
            &nonce_ref,
            Payload {
                msg: plain,
                aad: aad.as_bytes(),
            },
        )
        .map_err(|_| err("Could not encrypt draft"))?;
    let mut out = Vec::with_capacity(4 + 12 + encrypted.len());
    out.extend(MAGIC);
    out.extend(nonce);
    out.extend(encrypted);
    Ok(out)
}
fn open(key: &[u8; 32], id: &str, kind: &str, bytes: &[u8]) -> StoreResult<Vec<u8>> {
    if bytes.len() < 32 || &bytes[..4] != MAGIC {
        return Err(err("Saved draft is damaged"));
    }
    let cipher = Aes256Gcm::new_from_slice(key).map_err(|_| err("Encryption is unavailable"))?;
    let aad = format!("vostavo-draft-v1:{kind}:{id}");
    let nonce_ref = Nonce::try_from(&bytes[4..16]).map_err(|_| err("Saved draft is damaged"))?;
    cipher
        .decrypt(
            &nonce_ref,
            Payload {
                msg: &bytes[16..],
                aad: aad.as_bytes(),
            },
        )
        .map_err(|_| err("Saved draft failed authentication"))
}

struct Incoming {
    draft_id: String,
    revision: u64,
    take: Take,
    file: File,
    path: PathBuf,
    bytes: u64,
    hash: Sha256,
    chunk_index: u32,
    replaced_size: u64,
}
#[derive(Debug)]
struct CommitFailure {
    message: String,
    published: bool,
}
impl From<String> for CommitFailure {
    fn from(message: String) -> Self {
        Self {
            message,
            published: false,
        }
    }
}
pub struct DraftStore {
    dir: PathBuf,
    key: Box<dyn KeySource>,
    incoming: HashMap<String, Incoming>,
    verified: HashSet<String>,
    #[cfg(test)]
    fail_manifest_sync: bool,
}
impl DraftStore {
    pub fn new(root: &Path) -> StoreResult<Self> {
        secure_dir(root)?;
        let drafts = root.join("drafts");
        secure_dir(&drafts)?;
        let dir = drafts.join("v1");
        secure_dir(&dir)?;
        Ok(Self {
            dir,
            key: Box::new(MacKeychain::production()),
            incoming: HashMap::new(),
            verified: HashSet::new(),
            #[cfg(test)]
            fail_manifest_sync: false,
        })
    }
    #[cfg(test)]
    fn test(root: &Path, key: Box<dyn KeySource>) -> Self {
        secure_dir(root).unwrap();
        let drafts = root.join("drafts");
        secure_dir(&drafts).unwrap();
        let dir = drafts.join("v1");
        secure_dir(&dir).unwrap();
        Self {
            dir,
            key,
            incoming: HashMap::new(),
            verified: HashSet::new(),
            fail_manifest_sync: false,
        }
    }
    fn manifest_path(&self, id: &str) -> StoreResult<PathBuf> {
        Ok(self.dir.join(format!("{}.manifest", checked_id(id)?)))
    }
    fn media_path(&self, draft_id: &str, media_id: &str) -> StoreResult<PathBuf> {
        Ok(self.dir.join(format!(
            "{}-{}.media",
            checked_id(draft_id)?,
            checked_id(media_id)?
        )))
    }
    fn key(&self) -> StoreResult<[u8; 32]> {
        if let Some(key) = self.key.load()? {
            return Ok(key);
        }
        if fs::read_dir(&self.dir)
            .map_err(|_| err("Draft directory is unavailable"))?
            .any(|row| {
                row.ok()
                    .is_some_and(|entry| entry.file_name().to_string_lossy().ends_with(".manifest"))
            })
        {
            return Err(err("Saved take cannot be opened on this Mac"));
        }
        let key = random::<32>()?;
        self.key.create(&key)?;
        self.key
            .load()?
            .ok_or_else(|| err("Mac Keychain is unavailable; local save is disabled"))
    }
    fn read(&self, id: &str, key: &[u8; 32], now: u64) -> StoreResult<Manifest> {
        let path = self.manifest_path(id)?;
        if !regular_file(&path)? {
            return Err(err("Draft was not found"));
        }
        let mut file = open_private(&path)?;
        if file
            .metadata()
            .map_err(|_| err("Could not read saved draft"))?
            .len()
            > 64 * 1024
        {
            return Err(err("Saved draft exceeds limit"));
        }
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)
            .map_err(|_| err("Could not read saved draft"))?;
        let plain = open(key, id, "manifest", &bytes)?;
        let manifest: Manifest =
            serde_json::from_slice(&plain).map_err(|_| err("Invalid saved draft"))?;
        manifest.validate(id, now)?;
        Ok(manifest)
    }
    fn manifests(&self, key: &[u8; 32], now: u64) -> StoreResult<Vec<Manifest>> {
        let mut found = Vec::new();
        for entry in fs::read_dir(&self.dir).map_err(|_| err("Draft directory is unavailable"))? {
            let entry = entry.map_err(|_| err("Draft directory is unavailable"))?;
            let name = entry.file_name().to_string_lossy().into_owned();
            if let Some(id) = name.strip_suffix(".manifest") {
                checked_id(id)?;
                found.push(self.read(id, key, now)?);
            }
        }
        Ok(found)
    }
    fn verify_media(&self, manifest: &Manifest, key: &[u8; 32]) -> StoreResult<()> {
        let Some(take) = &manifest.take else {
            return Ok(());
        };
        let path = self.media_path(&manifest.draft_id, &take.media_id)?;
        if !regular_file(&path)? {
            return Err(err("Saved take is unavailable"));
        }
        let mut file = open_private(&path)?;
        let mut magic = [0; 4];
        file.read_exact(&mut magic)
            .map_err(|_| err("Saved take is damaged"))?;
        let count = take.size.div_ceil(CHUNK as u64);
        if &magic != MAGIC
            || file
                .metadata()
                .map_err(|_| err("Saved take is damaged"))?
                .len()
                != encrypted_media_size(take.size)
        {
            return Err(err("Saved take is damaged"));
        }
        let mut digest = Sha256::new();
        for index in 0..count {
            digest.update(read_chunk(
                &mut file,
                key,
                &manifest.draft_id,
                &take.media_id,
                take.size,
                index,
            )?);
        }
        if format!("{:x}", digest.finalize()) != take.sha256 {
            return Err(err("Saved take failed integrity check"));
        }
        Ok(())
    }
    fn commit(&self, manifest: &Manifest, key: &[u8; 32]) -> Result<(), CommitFailure> {
        let final_path = self.manifest_path(&manifest.draft_id)?;
        regular_file(&final_path)?;
        let plain = serde_json::to_vec(manifest).map_err(|_| err("Could not encode draft"))?;
        if plain.len().saturating_add(32) > 64 * 1024 {
            return Err(err("Saved draft exceeds limit").into());
        }
        let temp = self.dir.join(format!("{}.tmp", new_id()?));
        let bytes = seal(key, &manifest.draft_id, "manifest", &plain)?;
        let mut file = create_private(&temp)?;
        let result = (|| -> Result<(), CommitFailure> {
            file.write_all(&bytes)
                .and_then(|_| file.sync_all())
                .map_err(|_| err("Could not save draft"))?;
            fs::rename(&temp, &final_path).map_err(|_| err("Could not commit draft"))?;
            #[cfg(test)]
            if self.fail_manifest_sync {
                return Err(CommitFailure {
                    message: err("Could not sync draft directory"),
                    published: true,
                });
            }
            sync_dir(&self.dir).map_err(|message| CommitFailure {
                message,
                published: true,
            })
        })();
        if result.is_err() {
            let _ = fs::remove_file(temp);
        }
        result
    }
    pub fn list(&mut self) -> StoreResult<Vec<Manifest>> {
        let now = now_ms();
        self.expire(now)?;
        let key = self.key()?;
        let manifests = self.manifests(&key, now)?;
        for manifest in &manifests {
            self.verify_media(manifest, &key)?;
        }
        Ok(manifests)
    }
    pub fn recovery_ids(&self) -> StoreResult<Vec<String>> {
        // This intentionally needs no Keychain access. A lost item leaves only
        // opaque file names available so the user can still discard local data.
        let mut ids = Vec::new();
        for entry in fs::read_dir(&self.dir).map_err(|_| err("Draft directory is unavailable"))? {
            let entry = entry.map_err(|_| err("Draft directory is unavailable"))?;
            let name = entry.file_name().to_string_lossy().into_owned();
            if let Some(id) = name.strip_suffix(".manifest") {
                checked_id(id)?;
                if !regular_file(&entry.path())? {
                    return Err(err("Draft file is unsafe"));
                }
                ids.push(id.to_owned());
                if ids.len() > 64 {
                    return Err(err("Too many saved drafts"));
                }
            }
        }
        ids.sort();
        Ok(ids)
    }
    pub fn save_setup(
        &mut self,
        setup: Setup,
        draft_id: Option<String>,
        expected_revision: Option<u64>,
    ) -> StoreResult<Manifest> {
        let speaker = setup.validate()?;
        let now = now_ms();
        self.expire(now)?;
        let key = self.key()?;
        let all = self.manifests(&key, now)?;
        let previous = match draft_id {
            Some(id) => Some(self.read(&id, &key, now)?),
            None => None,
        };
        if let Some(old) = &previous {
            if old.revision != expected_revision.ok_or_else(|| err("Draft revision is required"))? {
                return Err(err("Draft changed in another window"));
            }
            if !["setup", "take_ready"].contains(&old.status.as_str()) {
                return Err(err("Submission receipt is frozen"));
            }
            if old.speaker_id != speaker {
                return Err(err("Learner cannot be changed on a saved draft"));
            }
        } else if expected_revision.is_some() {
            return Err(err("Unexpected draft revision"));
        }
        if all.iter().any(|m| {
            m.speaker_id == speaker && previous.as_ref().is_none_or(|p| p.draft_id != m.draft_id)
        }) {
            return Err(err("A draft already exists for this learner"));
        }
        if previous.is_none() && all.len() >= 3 {
            return Err(err("Three learner drafts are already saved"));
        }
        let mut manifest = previous.unwrap_or(Manifest {
            version: 1,
            draft_id: new_id()?,
            speaker_id: speaker,
            setup: setup.clone(),
            created_at: now,
            updated_at: now,
            expires_at: now + TTL,
            revision: 0,
            status: "setup".into(),
            take: None,
            submission: None,
        });
        manifest.setup = setup;
        manifest.updated_at = now;
        manifest.revision += 1;
        if manifest.take.is_none() {
            manifest.expires_at = now + TTL;
        }
        self.commit(&manifest, &key)
            .map_err(|error| error.message)?;
        Ok(manifest)
    }
    pub fn begin_take(
        &mut self,
        draft_id: String,
        expected_revision: u64,
        mut take: Take,
    ) -> StoreResult<String> {
        let now = now_ms();
        let key = self.key()?;
        let old = self.read(&draft_id, &key, now)?;
        if old.expires_at <= now || old.revision != expected_revision {
            return Err(err("Draft changed or expired"));
        }
        if !["setup", "take_ready"].contains(&old.status.as_str()) {
            return Err(err("Submission receipt is frozen"));
        }
        take.media_id = new_id()?;
        take.validate()?;
        if take.capture_ended_at < old.created_at || take.capture_ended_at > now {
            return Err(err("Invalid take capture time"));
        }
        let existing: u64 = self
            .manifests(&key, now)?
            .iter()
            .filter_map(|m| m.take.as_ref().map(|t| encrypted_media_size(t.size)))
            .sum();
        let old_size = old
            .take
            .as_ref()
            .map_or(0, |t| encrypted_media_size(t.size));
        let incoming_size = encrypted_media_size(take.size);
        let pending: u64 = self
            .incoming
            .values()
            .map(|item| encrypted_media_size(item.take.size).saturating_sub(item.replaced_size))
            .sum();
        if existing
            .saturating_add(pending)
            .saturating_add(incoming_size.saturating_sub(old_size))
            > MAX_TOTAL
        {
            return Err(err("Local draft storage limit reached"));
        }
        let free = free_bytes(&self.dir)?;
        if free < incoming_size.saturating_mul(2).saturating_add(RESERVE) {
            return Err(err("Not enough free disk space to save take"));
        }
        if self.incoming.values().any(|item| item.draft_id == draft_id) {
            return Err(err("A take save is already in progress"));
        }
        let transfer = new_id()?;
        let path = self.dir.join(format!("{transfer}.tmp"));
        let mut file = create_private(&path)?;
        file.write_all(MAGIC)
            .map_err(|_| err("Could not start encrypted take"))?;
        self.incoming.insert(
            transfer.clone(),
            Incoming {
                draft_id,
                revision: expected_revision,
                take,
                file,
                path,
                bytes: 0,
                hash: Sha256::new(),
                chunk_index: 0,
                replaced_size: old_size,
            },
        );
        Ok(transfer)
    }
    pub fn write_take_chunk(&mut self, transfer_id: &str, bytes: Vec<u8>) -> StoreResult<u64> {
        checked_id(transfer_id)?;
        let key = self.key()?;
        let item = self
            .incoming
            .get_mut(transfer_id)
            .ok_or_else(|| err("Take save was not found"))?;
        if bytes.is_empty()
            || bytes.len() > CHUNK
            || item.bytes.saturating_add(bytes.len() as u64) > item.take.size
        {
            return Err(err("Take chunk exceeds declared size"));
        }
        // All nonfinal chunks have a fixed size, keeping read offsets bounded.
        if bytes.len() != CHUNK && item.bytes + bytes.len() as u64 != item.take.size {
            return Err(err("Nonfinal take chunk is short"));
        }
        let aad = format!(
            "{}:{}:{}",
            item.draft_id, item.take.media_id, item.chunk_index
        );
        let sealed = seal(&key, &aad, "media", &bytes)?;
        item.file
            .write_all(&(sealed.len() as u32).to_be_bytes())
            .and_then(|_| item.file.write_all(&sealed))
            .map_err(|_| err("Could not write encrypted take"))?;
        item.hash.update(&bytes);
        item.bytes += bytes.len() as u64;
        item.chunk_index += 1;
        Ok(item.bytes)
    }
    pub fn abort_take(&mut self, transfer_id: &str) -> StoreResult<()> {
        checked_id(transfer_id)?;
        if let Some(item) = self.incoming.remove(transfer_id) {
            drop(item.file);
            fs::remove_file(item.path).map_err(|_| err("Could not cancel take save"))?;
            sync_dir(&self.dir)?;
        }
        Ok(())
    }
    pub fn finish_take(&mut self, transfer_id: &str) -> StoreResult<Manifest> {
        checked_id(transfer_id)?;
        let item = self
            .incoming
            .remove(transfer_id)
            .ok_or_else(|| err("Take save was not found"))?;
        let result = (|| -> StoreResult<Manifest> {
            if item.bytes != item.take.size
                || format!("{:x}", item.hash.clone().finalize()) != item.take.sha256
            {
                return Err(err("Take size or digest mismatch"));
            }
            item.file
                .sync_all()
                .map_err(|_| err("Could not sync encrypted take"))?;
            let key = self.key()?;
            let now = now_ms();
            let mut manifest = self.read(&item.draft_id, &key, now)?;
            if manifest.revision != item.revision || manifest.expires_at <= now {
                return Err(err("Draft changed or expired"));
            }
            let final_path = self.media_path(&item.draft_id, &item.take.media_id)?;
            if regular_file(&final_path)? {
                return Err(err("Take identifier collision"));
            }
            fs::rename(&item.path, &final_path)
                .map_err(|_| err("Could not commit encrypted take"))?;
            sync_dir(&self.dir)?;
            let old = manifest.take.replace(item.take.clone());
            manifest.status = "take_ready".into();
            manifest.submission = None;
            manifest.updated_at = now;
            manifest.expires_at = item.take.capture_ended_at.saturating_add(TTL);
            manifest.revision += 1;
            if let Err(error) = self.commit(&manifest, &key) {
                if !error.published {
                    let _ = fs::remove_file(&final_path);
                }
                return Err(error.message);
            }
            self.verified
                .insert(format!("{}:{}", manifest.draft_id, item.take.media_id));
            if let Some(old) = old {
                let _ = fs::remove_file(self.media_path(&manifest.draft_id, &old.media_id)?);
            }
            Ok(manifest)
        })();
        if result.is_err() {
            let _ = fs::remove_file(&item.path);
        }
        result
    }
    pub fn load_take_chunk(
        &mut self,
        draft_id: &str,
        media_id: &str,
        index: u32,
    ) -> StoreResult<Vec<u8>> {
        checked_id(media_id)?;
        let key = self.key()?;
        let now = now_ms();
        let manifest = self.read(draft_id, &key, now)?;
        if manifest.expires_at <= now {
            return Err(err("Draft expired"));
        }
        let take = manifest.take.ok_or_else(|| err("Draft has no take"))?;
        if take.media_id != media_id || index as u64 >= take.size.div_ceil(CHUNK as u64) {
            return Err(err("Take chunk was not found"));
        }
        let path = self.media_path(draft_id, media_id)?;
        if !regular_file(&path)? {
            return Err(err("Saved take is unavailable"));
        }
        let mut file = open_private(&path)?;
        let mut magic = [0; 4];
        file.read_exact(&mut magic)
            .map_err(|_| err("Saved take is damaged"))?;
        if &magic != MAGIC {
            return Err(err("Saved take is damaged"));
        }
        let count = take.size.div_ceil(CHUNK as u64);
        let expected_file_len = 4 + take.size + count * 36;
        if file
            .metadata()
            .map_err(|_| err("Saved take is damaged"))?
            .len()
            != expected_file_len
        {
            return Err(err("Saved take is damaged"));
        }
        let identity = format!("{draft_id}:{media_id}");
        if !self.verified.contains(&identity) {
            let mut digest = Sha256::new();
            for chunk in 0..count {
                digest.update(read_chunk(
                    &mut file, &key, draft_id, media_id, take.size, chunk,
                )?);
            }
            if format!("{:x}", digest.finalize()) != take.sha256 {
                return Err(err("Saved take failed integrity check"));
            }
            self.verified.insert(identity);
        }
        read_chunk(&mut file, &key, draft_id, media_id, take.size, index as u64)
    }
    pub fn mark_submission(
        &mut self,
        draft_id: &str,
        expected_revision: u64,
        status: String,
        submission: Submission,
    ) -> StoreResult<Manifest> {
        submission.validate_for(&status)?;
        if ![
            "upload_uncertain",
            "uploaded",
            "assessment_uncertain",
            "assessing",
        ]
        .contains(&status.as_str())
        {
            return Err(err("Invalid submission state"));
        }
        let key = self.key()?;
        let now = now_ms();
        let mut manifest = self.read(draft_id, &key, now)?;
        if manifest.revision != expected_revision
            || manifest.take.is_none()
            || manifest.expires_at <= now
        {
            return Err(err("Draft changed or expired"));
        }
        manifest.status = status;
        manifest.submission = Some(submission);
        manifest.revision += 1;
        manifest.updated_at = now;
        self.commit(&manifest, &key)
            .map_err(|error| error.message)?;
        Ok(manifest)
    }
    pub fn discard(&mut self, draft_id: &str) -> StoreResult<()> {
        let path = self.manifest_path(draft_id)?;
        if regular_file(&path)? {
            fs::remove_file(path).map_err(|_| err("Could not discard draft"))?;
            sync_dir(&self.dir)?;
        }
        // A lost Keychain item or damaged manifest must still be discardable.
        // Names are generated by this store and only the exact opaque-ID prefix
        // is removed; no caller path or manifest-provided path is followed.
        let prefix = format!("{draft_id}-");
        for entry in fs::read_dir(&self.dir).map_err(|_| err("Draft directory is unavailable"))? {
            let entry = entry.map_err(|_| err("Draft directory is unavailable"))?;
            let name = entry.file_name().to_string_lossy().into_owned();
            if let Some(media_id) = name
                .strip_prefix(&prefix)
                .and_then(|name| name.strip_suffix(".media"))
            {
                if valid_id(media_id) {
                    if !regular_file(&entry.path())? {
                        return Err(err("Draft file is unsafe"));
                    }
                    fs::remove_file(entry.path()).map_err(|_| err("Could not discard take"))?;
                }
            }
        }
        let active: Vec<_> = self
            .incoming
            .iter()
            .filter(|(_, value)| value.draft_id == draft_id)
            .map(|(id, _)| id.clone())
            .collect();
        for id in active {
            if let Some(value) = self.incoming.remove(&id) {
                let _ = fs::remove_file(value.path);
            }
        }
        sync_dir(&self.dir)?;
        Ok(())
    }
    pub fn expire(&mut self, now: u64) -> StoreResult<u32> {
        let key = self.key()?;
        let mut expired = 0;
        for manifest in self.manifests(&key, now)? {
            if manifest.expires_at <= now {
                self.discard(&manifest.draft_id)?;
                expired += 1;
            }
        }
        // A temp or media file is never recovered as an independent take. Only
        // objects referenced by a committed manifest survive a launch scan.
        let referenced: HashSet<_> = self
            .manifests(&key, now)?
            .iter()
            .filter_map(|m| {
                m.take
                    .as_ref()
                    .map(|t| format!("{}-{}.media", m.draft_id, t.media_id))
            })
            .collect();
        let active: HashSet<_> = self
            .incoming
            .values()
            .filter_map(|v| v.path.file_name().map(|n| n.to_string_lossy().into_owned()))
            .collect();
        for entry in fs::read_dir(&self.dir).map_err(|_| err("Draft directory is unavailable"))? {
            let entry = entry.map_err(|_| err("Draft directory is unavailable"))?;
            let name = entry.file_name().to_string_lossy().into_owned();
            if (name.ends_with(".tmp") && !active.contains(&name))
                || (name.ends_with(".media") && !referenced.contains(&name))
            {
                if !regular_file(&entry.path())? {
                    return Err(err("Draft file is unsafe"));
                }
                fs::remove_file(entry.path()).map_err(|_| err("Could not clean draft storage"))?;
            }
        }
        sync_dir(&self.dir)?;
        Ok(expired)
    }
}
fn read_chunk(
    file: &mut File,
    key: &[u8; 32],
    draft_id: &str,
    media_id: &str,
    total: u64,
    index: u64,
) -> StoreResult<Vec<u8>> {
    let count = total.div_ceil(CHUNK as u64);
    let expected = if index + 1 == count {
        (total - index * CHUNK as u64) as usize
    } else {
        CHUNK
    };
    file.seek(SeekFrom::Start(4 + index * (CHUNK as u64 + 36)))
        .map_err(|_| err("Saved take is damaged"))?;
    let mut size = [0; 4];
    file.read_exact(&mut size)
        .map_err(|_| err("Saved take is damaged"))?;
    if u32::from_be_bytes(size) as usize != expected + 32 {
        return Err(err("Saved take is damaged"));
    }
    let mut encrypted = vec![0; expected + 32];
    file.read_exact(&mut encrypted)
        .map_err(|_| err("Saved take is damaged"))?;
    let aad = format!("{draft_id}:{media_id}:{index}");
    let plain = open(key, &aad, "media", &encrypted)?;
    if plain.len() != expected {
        return Err(err("Saved take is damaged"));
    }
    Ok(plain)
}
fn free_bytes(path: &Path) -> StoreResult<u64> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    let path =
        CString::new(path.as_os_str().as_bytes()).map_err(|_| err("Invalid draft directory"))?;
    let mut stats = std::mem::MaybeUninit::<libc::statvfs>::uninit();
    if unsafe { libc::statvfs(path.as_ptr(), stats.as_mut_ptr()) } != 0 {
        return Err(err("Could not check free disk space"));
    }
    let stats = unsafe { stats.assume_init() };
    Ok((stats.f_bavail as u64).saturating_mul(stats.f_frsize as u64))
}

pub type DraftState = Mutex<Option<StoreResult<DraftStore>>>;
pub fn unopened_store() -> DraftState {
    Mutex::new(None)
}
pub fn initialize_store(state: &DraftState) {
    let result = DraftStore::new(&crate::desktop::data_root()).map(|mut store| {
        // Keychain failure leaves the shell usable, while all draft commands
        // continue to return a clear error and never fall back to plaintext.
        let _ = store.expire(now_ms());
        store
    });
    *state.lock().unwrap() = Some(result);
}
fn with_store<T>(
    state: tauri::State<'_, DraftState>,
    f: impl FnOnce(&mut DraftStore) -> StoreResult<T>,
) -> StoreResult<T> {
    let mut store = state
        .lock()
        .map_err(|_| err("Draft store is unavailable"))?;
    f(store
        .as_mut()
        .ok_or_else(|| err("Draft store is unavailable"))?
        .as_mut()
        .map_err(|error| error.clone())?)
}
#[tauri::command]
pub fn draft_list(state: tauri::State<'_, DraftState>) -> StoreResult<Vec<Manifest>> {
    with_store(state, |s| s.list())
}
#[tauri::command]
pub fn draft_recovery_ids(state: tauri::State<'_, DraftState>) -> StoreResult<Vec<String>> {
    with_store(state, |s| s.recovery_ids())
}
#[tauri::command]
pub fn draft_save_setup(
    state: tauri::State<'_, DraftState>,
    setup: Setup,
    draft_id: Option<String>,
    expected_revision: Option<u64>,
) -> StoreResult<Manifest> {
    with_store(state, |s| s.save_setup(setup, draft_id, expected_revision))
}
#[tauri::command]
pub fn draft_begin_take(
    state: tauri::State<'_, DraftState>,
    draft_id: String,
    expected_revision: u64,
    take: Take,
) -> StoreResult<String> {
    with_store(state, |s| s.begin_take(draft_id, expected_revision, take))
}
#[tauri::command]
pub fn draft_write_take_chunk(
    state: tauri::State<'_, DraftState>,
    transfer_id: String,
    bytes: Vec<u8>,
) -> StoreResult<u64> {
    with_store(state, |s| s.write_take_chunk(&transfer_id, bytes))
}
#[tauri::command]
pub fn draft_abort_take(
    state: tauri::State<'_, DraftState>,
    transfer_id: String,
) -> StoreResult<()> {
    with_store(state, |s| s.abort_take(&transfer_id))
}
#[tauri::command]
pub fn draft_finish_take(
    state: tauri::State<'_, DraftState>,
    transfer_id: String,
) -> StoreResult<Manifest> {
    with_store(state, |s| s.finish_take(&transfer_id))
}
#[tauri::command]
pub fn draft_load_take_chunk(
    state: tauri::State<'_, DraftState>,
    draft_id: String,
    media_id: String,
    index: u32,
) -> StoreResult<Vec<u8>> {
    with_store(state, |s| s.load_take_chunk(&draft_id, &media_id, index))
}
#[tauri::command]
pub fn draft_mark_submission(
    state: tauri::State<'_, DraftState>,
    draft_id: String,
    expected_revision: u64,
    status: String,
    submission: Submission,
) -> StoreResult<Manifest> {
    with_store(state, |s| {
        s.mark_submission(&draft_id, expected_revision, status, submission)
    })
}
#[tauri::command]
pub fn draft_discard(state: tauri::State<'_, DraftState>, draft_id: String) -> StoreResult<()> {
    with_store(state, |s| s.discard(&draft_id))
}
#[tauri::command]
pub fn draft_expire(state: tauri::State<'_, DraftState>) -> StoreResult<u32> {
    with_store(state, |s| s.expire(now_ms()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    struct TestRoot(PathBuf);
    impl TestRoot {
        fn new() -> Self {
            let path =
                std::env::temp_dir().join(format!("vostavo-draft-test-{}", new_id().unwrap()));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for TestRoot {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    struct TestKey {
        available: AtomicBool,
        present: AtomicBool,
    }
    impl TestKey {
        fn ready() -> Self {
            Self {
                available: AtomicBool::new(true),
                present: AtomicBool::new(true),
            }
        }
    }
    impl KeySource for TestKey {
        fn load(&self) -> StoreResult<Option<[u8; 32]>> {
            if !self.available.load(Ordering::SeqCst) {
                return Err(err("Mac Keychain is unavailable; local save is disabled"));
            }
            Ok(self.present.load(Ordering::SeqCst).then_some([7; 32]))
        }
        fn create(&self, _: &[u8; 32]) -> StoreResult<()> {
            Err(err("Mac Keychain is unavailable; local save is disabled"))
        }
    }
    fn setup(speaker: &str) -> Setup {
        Setup {
            session_id: "draft-session".into(),
            speaker_id: speaker.into(),
            learning_language: "it".into(),
            learning_language_label: "Italiano".into(),
            cefr_level: "B1".into(),
            theme_id: "travel".into(),
            theme_label: "Travel".into(),
            task_family: "travel_narrative".into(),
            duration_sec: 90,
            prompt_id: "prompt".into(),
            prompt_text: "Describe a trip".into(),
            retry_of_session_id: "".into(),
            coaching_goal: "".into(),
            retry_goal_eligible: false,
        }
    }
    fn take(bytes: &[u8]) -> Take {
        Take {
            media_id: "".into(),
            original_filename: "take.webm".into(),
            mime: "audio/webm".into(),
            size: bytes.len() as u64,
            sha256: format!("{:x}", Sha256::digest(bytes)),
            duration_sec: 5.0,
            capture_ended_at: now_ms(),
            input_method: "record".into(),
        }
    }
    fn store(root: &TestRoot) -> DraftStore {
        DraftStore::test(&root.0, Box::new(TestKey::ready()))
    }
    #[test]
    fn authenticated_round_trip_and_tamper_rejection() {
        let root = TestRoot::new();
        let mut store = store(&root);
        let draft = store.save_setup(setup("  Alex  "), None, None).unwrap();
        let bytes = b"a short playable take";
        let transfer = store
            .begin_take(draft.draft_id.clone(), draft.revision, take(bytes))
            .unwrap();
        assert_eq!(
            store.write_take_chunk(&transfer, bytes.to_vec()).unwrap(),
            bytes.len() as u64
        );
        let ready = store.finish_take(&transfer).unwrap();
        let media_id = &ready.take.as_ref().unwrap().media_id;
        assert_eq!(
            store.load_take_chunk(&ready.draft_id, media_id, 0).unwrap(),
            bytes
        );
        let path = store.media_path(&ready.draft_id, media_id).unwrap();
        let mut ciphertext = fs::read(&path).unwrap();
        *ciphertext.last_mut().unwrap() ^= 1;
        fs::write(&path, ciphertext).unwrap();
        assert!(store
            .load_take_chunk(&ready.draft_id, media_id, 0)
            .unwrap_err()
            .contains("authentication"));
        assert!(store.list().unwrap_err().contains("authentication"));
        let path = store.manifest_path(&ready.draft_id).unwrap();
        let mut ciphertext = fs::read(&path).unwrap();
        *ciphertext.last_mut().unwrap() ^= 1;
        fs::write(&path, ciphertext).unwrap();
        assert!(store.list().unwrap_err().contains("authentication"));
    }
    #[test]
    fn incomplete_atomic_write_recovers_previous_pair_and_prunes_orphans() {
        let root = TestRoot::new();
        let mut store = store(&root);
        let draft = store.save_setup(setup("Alex"), None, None).unwrap();
        let first = b"first take";
        let transfer = store
            .begin_take(draft.draft_id.clone(), draft.revision, take(first))
            .unwrap();
        store.write_take_chunk(&transfer, first.to_vec()).unwrap();
        let ready = store.finish_take(&transfer).unwrap();
        let second = b"replacement";
        let transfer = store
            .begin_take(ready.draft_id.clone(), ready.revision, take(second))
            .unwrap();
        store.write_take_chunk(&transfer, second.to_vec()).unwrap();
        let manifest_temp = store.dir.join(format!("{}.tmp", new_id().unwrap()));
        let mut file = create_private(&manifest_temp).unwrap();
        file.write_all(
            &seal(
                &[7; 32],
                &ready.draft_id,
                "manifest",
                b"unfinished replacement",
            )
            .unwrap(),
        )
        .unwrap();
        file.sync_all().unwrap();
        let orphan_media = store
            .media_path(&ready.draft_id, &new_id().unwrap())
            .unwrap();
        create_private(&orphan_media)
            .unwrap()
            .write_all(b"orphan")
            .unwrap();
        // Process death drops the writer while the encrypted private temp remains.
        drop(store);
        let mut recovered = self::store(&root);
        recovered.expire(now_ms()).unwrap();
        assert_eq!(recovered.list().unwrap()[0].revision, ready.revision);
        assert_eq!(
            recovered
                .load_take_chunk(&ready.draft_id, &ready.take.unwrap().media_id, 0)
                .unwrap(),
            first
        );
        assert!(!recovered.dir.join(format!("{transfer}.tmp")).exists());
        assert!(!manifest_temp.exists());
        assert!(!orphan_media.exists());
    }
    #[test]
    fn unopened_secondary_state_does_not_prune_active_transfer() {
        let root = TestRoot::new();
        let mut active = store(&root);
        let draft = active.save_setup(setup("Alex"), None, None).unwrap();
        let transfer = active
            .begin_take(draft.draft_id, draft.revision, take(b"active"))
            .unwrap();
        let temp = active.dir.join(format!("{transfer}.tmp"));
        assert!(temp.exists());
        let secondary = unopened_store();
        assert!(secondary.lock().unwrap().is_none());
        assert!(temp.exists());
        drop(active);
        let mut recovered = self::store(&root);
        recovered.expire(now_ms()).unwrap();
        assert!(!temp.exists());
    }
    #[test]
    fn manifest_sync_failure_after_rename_preserves_recoverable_media() {
        let root = TestRoot::new();
        let mut store = store(&root);
        let draft = store.save_setup(setup("Alex"), None, None).unwrap();
        let first = b"first take";
        let transfer = store
            .begin_take(draft.draft_id.clone(), draft.revision, take(first))
            .unwrap();
        store.write_take_chunk(&transfer, first.to_vec()).unwrap();
        let ready = store.finish_take(&transfer).unwrap();
        let old_media = store
            .media_path(&ready.draft_id, &ready.take.as_ref().unwrap().media_id)
            .unwrap();
        let second = b"replacement";
        let transfer = store
            .begin_take(ready.draft_id.clone(), ready.revision, take(second))
            .unwrap();
        store.write_take_chunk(&transfer, second.to_vec()).unwrap();
        store.fail_manifest_sync = true;
        assert!(store.finish_take(&transfer).unwrap_err().contains("sync"));
        assert!(old_media.exists());
        drop(store);
        let mut recovered = self::store(&root);
        recovered.expire(now_ms()).unwrap();
        let published = recovered.list().unwrap().remove(0);
        assert_eq!(published.revision, ready.revision + 1);
        assert_eq!(
            recovered
                .load_take_chunk(&published.draft_id, &published.take.unwrap().media_id, 0)
                .unwrap(),
            second
        );
    }
    #[test]
    fn oversized_manifest_does_not_replace_readable_revision() {
        let root = TestRoot::new();
        let mut store = store(&root);
        let first = store.save_setup(setup("Alex"), None, None).unwrap();
        let mut changed = setup("Alex");
        changed.prompt_text = "\u{0001}".repeat(12_000);
        assert!(store
            .save_setup(changed, Some(first.draft_id.clone()), Some(first.revision))
            .unwrap_err()
            .contains("limit"));
        let recovered = store.list().unwrap().remove(0);
        assert_eq!(recovered.revision, first.revision);
        assert_eq!(recovered.setup.prompt_text, first.setup.prompt_text);
    }
    #[test]
    fn expiry_limits_revisions_and_learner_isolation() {
        let root = TestRoot::new();
        let mut store = store(&root);
        let first = store.save_setup(setup(" Alex "), None, None).unwrap();
        assert!(store
            .save_setup(setup("alex"), None, None)
            .unwrap_err()
            .contains("already exists"));
        assert!(store
            .save_setup(setup("Alex"), Some(first.draft_id.clone()), Some(0))
            .unwrap_err()
            .contains("another window"));
        store.save_setup(setup("Bea"), None, None).unwrap();
        store.save_setup(setup("Cia"), None, None).unwrap();
        assert!(store
            .save_setup(setup("Dee"), None, None)
            .unwrap_err()
            .contains("Three"));
        assert!(store
            .begin_take(
                first.draft_id.clone(),
                first.revision,
                take(&vec![0; CHUNK + 1])
            )
            .is_ok());
        assert_eq!(store.expire(first.expires_at - 1).unwrap(), 0);
        assert_eq!(store.expire(first.expires_at + TTL).unwrap(), 3);
        assert!(store.list().unwrap().is_empty());
    }
    #[test]
    fn chunk_size_digest_and_path_limits_fail() {
        let root = TestRoot::new();
        let mut store = store(&root);
        let draft = store.save_setup(setup("Alex"), None, None).unwrap();
        let mut oversized = take(b"x");
        oversized.size = MAX_TAKE + 1;
        assert!(store
            .begin_take(draft.draft_id.clone(), draft.revision, oversized)
            .is_err());
        let bytes = vec![1; CHUNK + 1];
        let transfer = store
            .begin_take(draft.draft_id.clone(), draft.revision, take(&bytes))
            .unwrap();
        assert!(store
            .write_take_chunk(&transfer, bytes)
            .unwrap_err()
            .contains("chunk"));
        assert!(store.read("../etc/passwd", &[7; 32], now_ms()).is_err());
        assert!(store
            .load_take_chunk("../etc/passwd", "../passwd", 0)
            .is_err());
        store.abort_take(&transfer).unwrap();
        store.abort_take(&transfer).unwrap();
        assert!(!store.dir.join(format!("{transfer}.tmp")).exists());
        let transfer = store
            .begin_take(draft.draft_id.clone(), draft.revision, take(b"other"))
            .unwrap();
        assert!(store.finish_take(&transfer).unwrap_err().contains("digest"));
    }
    #[test]
    fn keychain_unavailable_or_lost_fails_closed() {
        let root = TestRoot::new();
        let key = TestKey {
            available: AtomicBool::new(false),
            present: AtomicBool::new(false),
        };
        let mut store = DraftStore::test(&root.0, Box::new(key));
        assert!(store
            .save_setup(setup("Alex"), None, None)
            .unwrap_err()
            .contains("Keychain is unavailable"));
        assert!(fs::read_dir(&store.dir).unwrap().next().is_none());
        let mut good = super::tests::store(&root);
        good.save_setup(setup("Alex"), None, None).unwrap();
        let lost = TestKey {
            available: AtomicBool::new(true),
            present: AtomicBool::new(false),
        };
        let mut store = DraftStore::test(&root.0, Box::new(lost));
        assert!(store.list().unwrap_err().contains("cannot be opened"));
        let id = store.recovery_ids().unwrap().remove(0);
        store.discard(&id).unwrap();
        store.discard(&id).unwrap();
        assert!(fs::read_dir(&store.dir).unwrap().next().is_none());
        assert!(store.recovery_ids().unwrap().is_empty());
        let mut restored = super::tests::store(&root);
        assert!(restored.save_setup(setup("Alex"), None, None).is_ok());
    }
    #[test]
    fn symlinked_storage_is_rejected() {
        use std::os::unix::fs::symlink;
        let root = TestRoot::new();
        let outside = TestRoot::new();
        symlink(&outside.0, root.0.join("drafts")).unwrap();
        assert!(DraftStore::new(&root.0).err().unwrap().contains("unsafe"));
        let clean = TestRoot::new();
        let store = store(&clean);
        let outside_file = clean.0.join("outside");
        create_private(&outside_file).unwrap();
        let linked = store
            .dir
            .join(format!("{}-{}.media", new_id().unwrap(), new_id().unwrap()));
        symlink(&outside_file, &linked).unwrap();
        assert!(regular_file(&linked).is_err());
    }
    #[test]
    fn total_limit_and_unsafe_receipt_are_rejected() {
        let root = TestRoot::new();
        let mut store = store(&root);
        let first = store.save_setup(setup("Alex"), None, None).unwrap();
        let second = store.save_setup(setup("Bea"), None, None).unwrap();
        let third = store.save_setup(setup("Cia"), None, None).unwrap();
        let key = [7; 32];
        for draft in [first.clone(), second, third] {
            let mut fake = draft;
            let mut metadata = take(b"x");
            metadata.media_id = new_id().unwrap();
            metadata.size = 90 * 1024 * 1024;
            fake.expires_at = metadata.capture_ended_at + TTL;
            fake.updated_at = metadata.capture_ended_at;
            fake.take = Some(metadata);
            fake.status = "take_ready".into();
            store.commit(&fake, &key).unwrap();
        }
        let mut incoming = take(b"x");
        incoming.size = 80 * 1024 * 1024;
        assert!(store
            .begin_take(first.draft_id.clone(), first.revision, incoming)
            .err()
            .unwrap()
            .contains("storage limit"));
        let receipt = Submission {
            upload_request_id: "id".into(),
            audio_id: "".into(),
            assessment_request_id: "".into(),
            assessment_id: "".into(),
            request_fingerprint: "".into(),
            frozen_request_body: r#"{"nested":{"apiKey":"do-not-store"}}"#.into(),
            last_confirmed_phase: "none".into(),
        };
        assert!(receipt.validate().is_err());
        let safe = Submission { frozen_request_body: r#"{"audio_id":"audio-1","request_id":"request-2","sharing_fingerprint":"fingerprint"}"#.into(),
            audio_id: "audio-1".into(), assessment_request_id: "request-2".into(), request_fingerprint: "fingerprint".into(), ..receipt.clone() };
        assert!(safe.validate_for("assessment_uncertain").is_ok());
        let unsafe_connection = Submission {
            frozen_request_body: r#"{"llm_base_url":"http://example.com?key=secret"}"#.into(),
            ..safe.clone()
        };
        assert!(unsafe_connection.validate().is_err());
    }
    #[test]
    fn concurrent_transfers_reserve_aggregate_quota() {
        let root = TestRoot::new();
        let mut store = store(&root);
        let drafts = ["Alex", "Bea", "Cia"]
            .map(|speaker| store.save_setup(setup(speaker), None, None).unwrap());
        let mut metadata = take(b"x");
        metadata.size = MAX_TAKE;
        let first = store
            .begin_take(
                drafts[0].draft_id.clone(),
                drafts[0].revision,
                metadata.clone(),
            )
            .unwrap();
        let second = store
            .begin_take(
                drafts[1].draft_id.clone(),
                drafts[1].revision,
                metadata.clone(),
            )
            .unwrap();
        assert!(store
            .begin_take(
                drafts[2].draft_id.clone(),
                drafts[2].revision,
                metadata.clone()
            )
            .unwrap_err()
            .contains("storage limit"));
        store.abort_take(&first).unwrap();
        let third = store
            .begin_take(drafts[2].draft_id.clone(), drafts[2].revision, metadata)
            .unwrap();
        store.abort_take(&second).unwrap();
        store.abort_take(&third).unwrap();
    }
    #[test]
    fn future_capture_time_cannot_publish_unreadable_manifest() {
        let root = TestRoot::new();
        let mut store = store(&root);
        let draft = store.save_setup(setup("Alex"), None, None).unwrap();
        let mut future = take(b"take");
        future.capture_ended_at = now_ms() + 60_000;
        assert!(store
            .begin_take(draft.draft_id.clone(), draft.revision, future)
            .unwrap_err()
            .contains("capture time"));
        let transfer = store
            .begin_take(draft.draft_id.clone(), draft.revision, take(b"take"))
            .unwrap();
        store.write_take_chunk(&transfer, b"take".to_vec()).unwrap();
        let ready = store.finish_take(&transfer).unwrap();
        assert_eq!(store.list().unwrap()[0].revision, ready.revision);
    }
    #[test]
    fn old_transfer_cannot_overwrite_new_revision_and_permissions_are_private() {
        let root = TestRoot::new();
        let mut store = store(&root);
        let first = store.save_setup(setup("Alex"), None, None).unwrap();
        let bytes = b"take";
        let transfer = store
            .begin_take(first.draft_id.clone(), first.revision, take(bytes))
            .unwrap();
        store.write_take_chunk(&transfer, bytes.to_vec()).unwrap();
        let newer = store
            .save_setup(
                setup("Alex"),
                Some(first.draft_id.clone()),
                Some(first.revision),
            )
            .unwrap();
        assert!(store
            .finish_take(&transfer)
            .err()
            .unwrap()
            .contains("changed"));
        assert_eq!(store.list().unwrap()[0].revision, newer.revision);
        assert_eq!(
            fs::metadata(&store.dir).unwrap().permissions().mode() & 0o777,
            0o700
        );
        assert_eq!(
            fs::metadata(store.manifest_path(&first.draft_id).unwrap())
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o600
        );
    }
    #[test]
    fn submission_receipt_is_revisioned_and_does_not_extend_take_expiry() {
        let root = TestRoot::new();
        let mut store = store(&root);
        let draft = store.save_setup(setup("Alex"), None, None).unwrap();
        let bytes = b"take";
        let transfer = store
            .begin_take(draft.draft_id.clone(), draft.revision, take(bytes))
            .unwrap();
        store.write_take_chunk(&transfer, bytes.to_vec()).unwrap();
        let ready = store.finish_take(&transfer).unwrap();
        let receipt = Submission {
            upload_request_id: "request-1".into(),
            audio_id: "".into(),
            assessment_request_id: "".into(),
            assessment_id: "".into(),
            request_fingerprint: "".into(),
            frozen_request_body: "".into(),
            last_confirmed_phase: "none".into(),
        };
        assert!(store
            .mark_submission(
                &ready.draft_id,
                ready.revision,
                "uploaded".into(),
                receipt.clone()
            )
            .is_err());
        let pending = store
            .mark_submission(
                &ready.draft_id,
                ready.revision,
                "upload_uncertain".into(),
                receipt,
            )
            .unwrap();
        assert_eq!(pending.expires_at, ready.expires_at);
        assert!(store
            .mark_submission(
                &ready.draft_id,
                ready.revision,
                "upload_uncertain".into(),
                pending.submission.clone().unwrap()
            )
            .is_err());
        assert!(store
            .save_setup(
                setup("Alex"),
                Some(ready.draft_id.clone()),
                Some(pending.revision)
            )
            .is_err());
        assert!(store
            .begin_take(ready.draft_id.clone(), pending.revision, take(bytes))
            .is_err());
    }
    #[cfg(target_os = "macos")]
    #[test]
    fn real_keychain_round_trip_uses_isolated_non_synchronizing_item() {
        use security_framework::passwords::{delete_generic_password_options, PasswordOptions};
        let service = format!("com.vostavo.desktop.drafts.test.{}", new_id().unwrap());
        let keychain = MacKeychain {
            service: service.clone(),
        };
        let result = (|| -> StoreResult<()> {
            assert!(keychain.load()?.is_none());
            keychain.create(&[9; 32])?;
            assert_eq!(keychain.load()?, Some([9; 32]));
            let root = TestRoot::new();
            let mut store = DraftStore::test(
                &root.0,
                Box::new(MacKeychain {
                    service: service.clone(),
                }),
            );
            let draft = store.save_setup(setup("Mac keychain learner"), None, None)?;
            let bytes = b"keychain-backed encrypted media";
            let transfer = store.begin_take(draft.draft_id.clone(), draft.revision, take(bytes))?;
            store.write_take_chunk(&transfer, bytes.to_vec())?;
            let ready = store.finish_take(&transfer)?;
            assert_eq!(store.list()?.len(), 1);
            assert_eq!(
                store.load_take_chunk(&ready.draft_id, &ready.take.unwrap().media_id, 0)?,
                bytes
            );
            Ok(())
        })();
        let mut options = PasswordOptions::new_generic_password(&service, KEY_ACCOUNT);
        options.set_access_synchronized(Some(false));
        let _ = delete_generic_password_options(options);
        result.unwrap();
    }
    #[test]
    #[ignore = "manual 100 MiB storage throughput probe"]
    fn hundred_mib_chunked_storage_probe() {
        let root = TestRoot::new();
        let mut store = store(&root);
        let draft = store.save_setup(setup("Throughput"), None, None).unwrap();
        let chunk = vec![0x5a; CHUNK];
        let mut hash = Sha256::new();
        for _ in 0..400 {
            hash.update(&chunk);
        }
        let mut metadata = take(&chunk);
        metadata.size = MAX_TAKE;
        metadata.sha256 = format!("{:x}", hash.finalize());
        let started = std::time::Instant::now();
        let transfer = store
            .begin_take(draft.draft_id.clone(), draft.revision, metadata)
            .unwrap();
        for _ in 0..400 {
            store.write_take_chunk(&transfer, chunk.clone()).unwrap();
        }
        let ready = store.finish_take(&transfer).unwrap();
        let write_seconds = started.elapsed().as_secs_f64();
        let media_id = ready.take.unwrap().media_id;
        drop(store);
        let mut store = self::store(&root);
        let started = std::time::Instant::now();
        assert_eq!(
            store
                .load_take_chunk(&ready.draft_id, &media_id, 399)
                .unwrap(),
            chunk
        );
        eprintln!(
            "100 MiB encrypted storage: write {:.2}s, cold verification/read {:.2}s",
            write_seconds,
            started.elapsed().as_secs_f64()
        );
    }
}
