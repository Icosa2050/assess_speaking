# Built only from the clean production environment, never the development venv.
import json
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules, copy_metadata

root = Path(SPECPATH).parent
version = json.loads((root / 'frontend/src-tauri/tauri.conf.json').read_text())['version']
datas = [(str(root / 'locales'), 'locales'), (str(root / 'samples/cefr'), 'samples/cefr'),
         (str(root / 'assessment_runtime/data'), 'assessment_runtime/data')]
binaries = []
hidden = []
for package in ('app_backend', 'app_core', 'assess_core', 'assessment_runtime'):
    hidden += collect_submodules(package)
for package in ('faster_whisper', 'onnxruntime', 'ctranslate2', 'certifi', 'keyring'):
    datas += collect_data_files(package)
    datas += copy_metadata(package.replace('_', '-'))
    binaries += collect_dynamic_libs(package)
hidden += ['keyring.backends.macOS', 'keyring.backends.fail', 'uvicorn.logging', 'uvicorn.loops.auto',
           'uvicorn.loops.asyncio', 'uvicorn.protocols.http.auto', 'uvicorn.protocols.http.h11_impl',
           'uvicorn.protocols.websockets.auto', 'uvicorn.lifespan.on', 'uvicorn.lifespan.off',
           'assessment_runtime.responses_client', 'assess_speaking', 'scripts.progress_dashboard']
a = Analysis([str(root / 'scripts/run_backend.py')], pathex=[str(root)], binaries=binaries,
             datas=datas, hiddenimports=hidden,
             excludes=['pytest', 'playwright', 'coverage', 'tkinter', 'IPython', 'streamlit', 'pandas'],
             noarchive=False)
for module, *_ in a.pure:
    if module.split('.')[0] in {'pytest', 'playwright', 'coverage', 'IPython', 'streamlit', 'pandas'}:
        raise RuntimeError(f'Development-only module in packaged runtime: {module}')
for destination, *_ in a.datas:
    if Path(destination).name == '.env':
        raise RuntimeError('Environment secrets must never be bundled')
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='vostavo-backend', console=False,
          strip=False, upx=False, target_arch='arm64')
collection = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='vostavo-backend')
app = BUNDLE(collection, name='VostavoBackend.app', bundle_identifier='com.vostavo.desktop.backend',
             info_plist={'CFBundleShortVersionString': version, 'CFBundleVersion': version,
                         'LSBackgroundOnly': True, 'LSMinimumSystemVersion': '13.4'})
