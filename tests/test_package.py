from __future__ import annotations

import importlib.metadata

import oscillon as m


def test_version() -> None:
    assert importlib.metadata.version("oscillon") == m.__version__
