"""ASGI entrypoint: `uvicorn cwas.main:app`."""

from cwas.api.app import create_app

app = create_app()
