import pytest

from cwas.tools.url_safety import is_public_url


def resolver(mapping: dict[str, list[str]]):
    async def resolve(host: str) -> list[str]:
        if host not in mapping:
            raise OSError(f"cannot resolve {host}")
        return mapping[host]

    return resolve


PUBLIC = resolver({"example.com": ["93.184.215.14"], "internal.corp": ["10.0.0.5"]})


async def test_accepts_https_url_resolving_to_public_ip() -> None:
    assert await is_public_url("https://example.com/post", resolve=PUBLIC)


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com/x",
        "javascript:alert(1)",
        "https://",
        "not a url",
    ],
)
async def test_rejects_non_http_or_malformed_urls(url: str) -> None:
    assert not await is_public_url(url, resolve=PUBLIC)


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://localhost/",
        "http://10.1.2.3/",
        "http://192.168.0.10/",
        "http://172.16.5.5/",
        "http://169.254.169.254/latest/meta-data",
        "http://[::1]/",
        "http://0.0.0.0/",
    ],
)
async def test_rejects_private_loopback_and_metadata_addresses(url: str) -> None:
    assert not await is_public_url(url, resolve=resolver({"localhost": ["127.0.0.1"]}))


async def test_rejects_hostname_that_resolves_to_private_ip() -> None:
    assert not await is_public_url("https://internal.corp/page", resolve=PUBLIC)


async def test_rejects_hostname_that_cannot_be_resolved() -> None:
    assert not await is_public_url("https://nowhere.invalid/", resolve=PUBLIC)
