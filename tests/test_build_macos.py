import ast
from pathlib import Path
from unittest.mock import patch
import pytest

from scripts import build_macos


def test_internal_acceptance_command_is_required_only_in_internal_binary(tmp_path: Path) -> None:
    binary = tmp_path / 'vostavo-desktop'
    binary.write_bytes(b'ordinary release shell')
    build_macos.assert_delivery_feature(binary, release=True)
    with pytest.raises(RuntimeError):
        build_macos.assert_delivery_feature(binary, release=False)
    binary.write_bytes(b'ordinary shell VOSTAVO_INTERNAL_DMG_ACCEPTANCE VOSTAVO_INTERNAL_DMG_KEY_SERVICE')
    build_macos.assert_delivery_feature(binary, release=False)
    with pytest.raises(RuntimeError):
        build_macos.assert_delivery_feature(binary, release=True)


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


def test_frozen_helper_declares_the_same_mac_os_floor() -> None:
    spec = ast.parse((build_macos.ROOT / "packaging/vostavo-backend.spec").read_text())
    bundle = next(
        node for node in ast.walk(spec)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "BUNDLE"
    )
    info_plist = next(keyword.value for keyword in bundle.keywords if keyword.arg == "info_plist")
    assert isinstance(info_plist, ast.Dict)
    minimum = next(
        value for key, value in zip(info_plist.keys, info_plist.values)
        if ast.literal_eval(key) == "LSMinimumSystemVersion"
    )
    assert ast.literal_eval(minimum) == build_macos.MINIMUM_MACOS
