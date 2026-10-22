"""SSRF guard: only fetch http(s) URLs whose host resolves exclusively to public addresses."""

import asyncio
import ipaddress
import socket
from collections.abc import Awaitable, Callable
from urllib.parse import urlsplit

Resolver = Callable[[str], Awaitable[list[str]]]
UrlGuard = Callable[[str], Awaitable[bool]]


async def resolve_host(host: str) -> list[str]:
    infos = await asyncio.get_running_loop().getaddrinfo(host, None, type=socket.SOCK_STREAM)
    return [str(info[4][0]) for info in infos]


def _is_public_ip(address: str) -> bool:
    ip = ipaddress.ip_address(address)
    return ip.is_global and not ip.is_multicast


async def is_public_url(url: str, resolve: Resolver = resolve_host) -> bool:
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return False

    host = parts.hostname
    try:
        return _is_public_ip(host)
    except ValueError:
        pass  # not an IP literal; resolve it

    try:
        addresses = await resolve(host)
    except OSError:
        return False
    return bool(addresses) and all(_is_public_ip(a) for a in addresses)
