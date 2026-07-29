"""Opt-in live integration probes (skipped unless env vars are set).

Enable with env vars, e.g.::

    AIRE_LIVE_REDIS=redis://localhost:6379/0 pytest -m live
    AIRE_LIVE_PGVECTOR=postgresql://... pytest -m live
    AIRE_LIVE_NEO4J=bolt://neo4j:password@localhost:7687 pytest -m live
    AIRE_LIVE_QDRANT=http://localhost:6333 pytest -m live
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.live


def _env(name: str) -> str | None:
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else None


@pytest.mark.skipif(not _env("AIRE_LIVE_REDIS"), reason="set AIRE_LIVE_REDIS to enable")
def test_live_redis_ping() -> None:
    import redis

    url = _env("AIRE_LIVE_REDIS")
    assert url is not None
    client = redis.Redis.from_url(url)
    assert client.ping() is True


@pytest.mark.skipif(not _env("AIRE_LIVE_PGVECTOR"), reason="set AIRE_LIVE_PGVECTOR to enable")
def test_live_pgvector_connect() -> None:
    import psycopg

    dsn = _env("AIRE_LIVE_PGVECTOR")
    assert dsn is not None
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT 1")
        assert cur.fetchone() == (1,)


@pytest.mark.skipif(not _env("AIRE_LIVE_NEO4J"), reason="set AIRE_LIVE_NEO4J to enable")
def test_live_neo4j_connect() -> None:
    from neo4j import GraphDatabase

    uri = _env("AIRE_LIVE_NEO4J")
    assert uri is not None
    # Accept bolt://user:pass@host:port or bolt://host with NEO4J_AUTH
    driver = GraphDatabase.driver(uri)
    with driver.session() as session:
        result = session.run("RETURN 1 AS n")
        assert result.single()["n"] == 1
    driver.close()


@pytest.mark.skipif(not _env("AIRE_LIVE_QDRANT"), reason="set AIRE_LIVE_QDRANT to enable")
def test_live_qdrant_ready() -> None:
    import httpx

    base = _env("AIRE_LIVE_QDRANT")
    assert base is not None
    response = httpx.get(f"{base.rstrip('/')}/readyz", timeout=10.0)
    assert response.status_code == 200
