"""Spawn targets for cross-process spending and identical-prompt locking."""
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace


def run(root, barrier, output, kind):
    from app_core.cloud_policy import CloudPolicyError, SpendingLedger
    root = Path(root)
    barrier.wait(timeout=10)
    if kind == 'ledger':
        try:
            output.put(('reserved', SpendingLedger(root/'app').reserve(Decimal(3), 5, 'paid')))
        except CloudPolicyError:
            output.put(('blocked', None))
        return
    from tests.helpers.cloud_backend import install_transport
    from app_backend.cloud_runtime import CloudRuntime
    from app_core.state import AppPreferences, ProviderConnection
    install_transport(root)
    conn = ProviderConnection(connection_id='paid', provider_kind='openrouter', default_model='vendor/paid', secret_ref='fixture')
    state = SimpleNamespace(prefs=AppPreferences(connections=[conn], active_connection_id='paid'))
    runtime = CloudRuntime(root/'app', state, lambda _: 'fixture', cache_root=root/'stages')
    result = runtime.handle({'operation':'completion', 'provider':'openrouter', 'model':'vendor/paid', 'prompt':'fixture identical prompt', 'timeout':5, 'schema':{'name':'rubric','schema':{'type':'object'}}})
    output.put(('result', result['cost_usd']))
