use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::os::fd::AsRawFd;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

pub struct Runtime {
    pub api_base_url: String,
    pub session_token: String,
    pub media_token: String,
    pub launch_mode: String,
}

pub struct BackendOwner {
    child: Child,
    _input: ChildStdin,
    _lock: File,
    launch_dir: PathBuf,
}

pub fn helper_path(executable: &Path) -> Option<PathBuf> {
    let macos = executable.parent()?;
    if macos.file_name()? != "MacOS" || macos.parent()?.file_name()? != "Contents" {
        return None;
    }
    Some(macos.parent()?.join("Helpers/VostavoBackend.app/Contents/MacOS/vostavo-backend"))
}

fn default_root(home: &Path, area: &str) -> PathBuf {
    let current = home.join("Library").join(area).join("Vostavo");
    let legacy = home.join("Library").join(area).join("Speaking Studio");
    if current.join(".vostavo-root").exists() { return current; }
    if legacy.join(".vostavo-root").exists() || legacy.read_dir().map(|mut rows| rows.next().is_some()).unwrap_or(false) { legacy } else { current }
}

pub fn data_root() -> PathBuf {
    env::var_os("VOSTAVO_HOME").map(PathBuf::from).unwrap_or_else(|| default_root(&PathBuf::from(env::var_os("HOME").unwrap_or_default()), "Application Support"))
}

fn health(port: u16, token: &str) -> bool {
    let address: SocketAddr = format!("127.0.0.1:{port}").parse().unwrap();
    let Ok(mut stream) = TcpStream::connect_timeout(&address, Duration::from_millis(500)) else { return false; };
    let _ = stream.set_read_timeout(Some(Duration::from_secs(1)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(1)));
    if write!(stream, "GET /v1/health HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nX-Vostavo-Session: {token}\r\nConnection: close\r\n\r\n").is_err() { return false; }
    let mut bytes = [0; 64];
    let Ok(count) = stream.read(&mut bytes) else { return false; };
    String::from_utf8_lossy(&bytes[..count]).starts_with("HTTP/1.1 200 ")
}

impl BackendOwner {
    pub fn launch() -> Result<(Self, Runtime), String> {
        let root = data_root();
        let cache = env::var_os("VOSTAVO_CACHE_HOME").map(PathBuf::from).unwrap_or_else(|| default_root(&PathBuf::from(env::var_os("HOME").unwrap_or_default()), "Caches"));
        fs::create_dir_all(root.join("logs")).map_err(|e| e.to_string())?;
        fs::create_dir_all(&cache).map_err(|e| e.to_string())?;
        let lock = OpenOptions::new().create(true).truncate(false).write(true).open(root.join("desktop.lock")).map_err(|e| e.to_string())?;
        if unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
            return Err("Vostavo is already running. Open its existing window.".into());
        }
        let mut entropy = [0_u8; 80];
        File::open("/dev/urandom").and_then(|mut f| f.read_exact(&mut entropy)).map_err(|e| e.to_string())?;
        let token: String = entropy[..32].iter().map(|value| format!("{value:02x}")).collect();
        let media_token: String = entropy[32..64].iter().map(|value| format!("{value:02x}")).collect();
        let nonce: String = entropy[64..].iter().map(|value| format!("{value:02x}")).collect();
        let launch_dir = root.join("tmp").join(format!("desktop-{}-{nonce}", std::process::id()));
        fs::create_dir_all(&launch_dir).map_err(|e| e.to_string())?;
        let ready = launch_dir.join("ready.json");
        let executable = env::current_exe().map_err(|e| e.to_string())?;
        let mut command;
        let launch_mode;
        if let Some(helper) = helper_path(&executable) {
            if !helper.is_file() { return Err("The packaged backend is missing. Reinstall Vostavo.".into()); }
            command = Command::new(helper);
            launch_mode = "packaged";
        } else {
            #[cfg(debug_assertions)]
            {
                let repo = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
                command = Command::new(repo.join(".venv/bin/python"));
                command.arg(repo.join("scripts/run_backend.py"));
                launch_mode = "repo";
            }
            #[cfg(not(debug_assertions))]
            { return Err("Launch Vostavo from its installed application bundle.".into()); }
        }
        command.env_clear();
        for key in ["HOME", "USER", "LOGNAME", "TMPDIR", "LANG", "LC_ALL"] {
            if let Some(value) = env::var_os(key) { command.env(key, value); }
        }
        command.env("PATH", "/usr/bin:/bin:/usr/sbin:/sbin");
        command.args(["--host", "127.0.0.1", "--port", "0", "--desktop-owned", "--ready-file"])
            .arg(&ready).arg("--app-data-dir").arg(&root).arg("--cache-dir").arg(&cache)
            .current_dir(&root).stdin(Stdio::piped());
        let log = OpenOptions::new().create(true).append(true).open(root.join("logs/desktop-startup.log")).map_err(|e| e.to_string())?;
        command.stdout(log.try_clone().map_err(|e| e.to_string())?).stderr(log);
        unsafe {
            command.pre_exec(|| {
                if libc::setpgid(0, 0) == -1 { return Err(std::io::Error::last_os_error()); }
                Ok(())
            });
        }
        let mut child = command.spawn().map_err(|e| e.to_string())?;
        let input = child.stdin.take().ok_or("Backend stdin is unavailable")?;
        let mut owner = Self { child, _input: input, _lock: lock, launch_dir };
        writeln!(owner._input, "{token}\n{media_token}").map_err(|e| e.to_string())?;
        let deadline = Instant::now() + Duration::from_secs(90);
        while Instant::now() < deadline {
            if owner.child.try_wait().map_err(|e| e.to_string())?.is_some() { return Err("The backend exited during startup.".into()); }
            if let Ok(text) = fs::read_to_string(&ready) {
                if let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) {
                    if let Some(port) = value["port"].as_u64().and_then(|value| u16::try_from(value).ok()) {
                        if health(port, &token) {
                            return Ok((owner, Runtime { api_base_url: format!("http://127.0.0.1:{port}"), session_token: token, media_token, launch_mode: launch_mode.into() }));
                        }
                    }
                }
            }
            thread::sleep(Duration::from_millis(150));
        }
        Err("The backend did not become ready within 90 seconds.".into())
    }
}

impl Drop for BackendOwner {
    fn drop(&mut self) {
        let group = -(self.child.id() as i32);
        unsafe { libc::kill(group, libc::SIGTERM); }
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline {
            if self.child.try_wait().ok().flatten().is_some() { break; }
            thread::sleep(Duration::from_millis(100));
        }
        unsafe { libc::kill(group, libc::SIGKILL); }
        let _ = self.child.wait();
        let _ = fs::remove_dir_all(&self.launch_dir);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn helper_is_resolved_relative_to_relocated_app() {
        assert_eq!(helper_path(Path::new("/tmp/Applications/Vostavo.app/Contents/MacOS/vostavo-desktop")).unwrap(), PathBuf::from("/tmp/Applications/Vostavo.app/Contents/Helpers/VostavoBackend.app/Contents/MacOS/vostavo-backend"));
        assert!(helper_path(Path::new("/tmp/vostavo-desktop")).is_none());
    }
}
