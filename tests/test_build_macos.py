from pathlib import Path
from unittest.mock import patch

from scripts import build_macos


def test_native_audit_preserves_mac_os_14_floor(tmp_path: Path) -> None:
    binary = tmp_path / "Vostavo.app" / "Contents" / "MacOS" / "vostavo-desktop"
    with patch.object(build_macos, "macho_files", return_value=[binary]), patch.object(
        build_macos,
        "output",
        side_effect=["arm64", "binary:\n\t/usr/lib/libSystem.B.dylib", "minos 13.4", ""],
    ):
        result = build_macos.audit(tmp_path / "Vostavo.app")

    assert build_macos.MINIMUM_MACOS == "14.0"
    assert result["minimum_macos"] == "14.0"
