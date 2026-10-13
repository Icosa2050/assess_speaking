"""Inherited fixture-only network guard, including multiprocessing spawn."""
import ipaddress
import json
import os
from pathlib import Path
import socket

if os.environ.get('VOSTAVO_FIXTURE_GUARD_LOG'):
    def local(host):
        if host in ('localhost', '', None):
            return True
        try:
            return ipaddress.ip_address(host).is_loopback
        except ValueError:
            return False
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_resolve = socket.getaddrinfo
    def connect(self, address):
        if self.family != socket.AF_UNIX and not local(address[0]):
            raise PermissionError('Fixture blocked non-loopback connection')
        return original_connect(self, address)
    def connect_ex(self, address):
        if self.family != socket.AF_UNIX and not local(address[0]):
            raise PermissionError('Fixture blocked non-loopback connection')
        return original_connect_ex(self, address)
    def resolve(host, *args, **kwargs):
        if not local(host):
            raise PermissionError('Fixture blocked non-loopback DNS')
        return original_resolve(host, *args, **kwargs)
    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    socket.getaddrinfo = resolve
    with socket.socket() as probe:
        try:
            probe.connect(('192.0.2.1', 443))
        except PermissionError:
            with Path(os.environ['VOSTAVO_FIXTURE_GUARD_LOG']).open('a') as log:
                log.write(json.dumps({'pid': os.getpid(), 'guard': True}) + '\n')
        else:
            raise RuntimeError('Fixture guard inactive')

# Optional local coverage instrumentation; never required by production or CI fixtures.
if os.environ.get('COVERAGE_PROCESS_START'):
    import coverage
    coverage.process_startup()
