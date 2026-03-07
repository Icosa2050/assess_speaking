from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import requests


class TelegramClient:
    def __init__(self, token: str, *, timeout_sec: float = 30.0):
        self.token = token
        self.timeout_sec = timeout_sec

    @property
    def _api_base(self) -> str:
        return f"https://api.telegram.org/bot{self.token}"

    @property
    def _file_base(self) -> str:
        return f"https://api.telegram.org/file/bot{self.token}"

    def _api_call(self, method: str, *, data: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        resp = requests.post(
            f"{self._api_base}/{method}",
            data=data or {},
            timeout=self.timeout_sec,
        )
        resp.raise_for_status()
        payload = resp.json()
        if not payload.get("ok"):
            raise RuntimeError(f"Telegram API error on {method}: {payload}")
        return payload.get("result", {})

    def get_file_path(self, file_id: str) -> str:
        result = self._api_call("getFile", data={"file_id": file_id})
        file_path = result.get("file_path")
        if not file_path:
            raise RuntimeError(f"Telegram getFile returned no file_path for file_id={file_id}")
        return str(file_path)

    def download_file(self, file_path: str, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(f"{self._file_base}/{file_path}", timeout=self.timeout_sec) as resp:
            resp.raise_for_status()
            destination.write_bytes(resp.content)
        return destination

    def send_message(self, chat_id: int, text: str, *, reply_to_message_id: Optional[int] = None) -> None:
        data: dict[str, Any] = {"chat_id": chat_id, "text": text}
        if reply_to_message_id is not None:
            data["reply_to_message_id"] = reply_to_message_id
        self._api_call("sendMessage", data=data)

    def send_document(
        self,
        chat_id: int,
        document_path: Path,
        *,
        caption: str = "",
        reply_to_message_id: Optional[int] = None,
    ) -> None:
        data: dict[str, Any] = {"chat_id": chat_id, "caption": caption}
        if reply_to_message_id is not None:
            data["reply_to_message_id"] = reply_to_message_id
        with document_path.open("rb") as fh:
            resp = requests.post(
                f"{self._api_base}/sendDocument",
                data=data,
                files={"document": fh},
                timeout=self.timeout_sec,
            )
        resp.raise_for_status()
        payload = resp.json()
        if not payload.get("ok"):
            raise RuntimeError(f"Telegram API error on sendDocument: {payload}")

