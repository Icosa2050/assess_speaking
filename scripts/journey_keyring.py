"""Process-only credential storage for isolated live tests; never uses the OS keychain.

Select explicitly with PYTHON_KEYRING_BACKEND=scripts.journey_keyring.MemoryKeyring.
This substitutes only secret storage, not ASR, provider requests or assessments.
"""
from keyring.backend import KeyringBackend
from keyring.errors import PasswordDeleteError


class MemoryKeyring(KeyringBackend):
    priority = 1
    _secrets: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, username: str) -> str | None:
        return self._secrets.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self._secrets[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        if (service, username) not in self._secrets:
            raise PasswordDeleteError("No test credential exists for this account")
        del self._secrets[(service, username)]
