"""Verify real adapters and a generated container using disposable owned services.

Requires Docker, uv and the redis/pgvector/neo4j test profiles. Ports are random
loopback bindings. Only containers, images and networks created here are removed.
"""

# ruff: noqa: S603, S607
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

from aire.deployment import generate_artifacts, lock_requirements

SERVICES = {
    "qdrant": ("qdrant/qdrant:v1.13.6", 6333, [], []),
    "chroma": ("chromadb/chroma:1.0.20", 8000, [], []),
    "weaviate": (
        "cr.weaviate.io/semitechnologies/weaviate:1.28.8",
        8080,
        [
            "AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true",
            "DEFAULT_VECTORIZER_MODULE=none",
            "CLUSTER_HOSTNAME=qa",
        ],
        [],
    ),
    "milvus": (
        "milvusdb/milvus:v2.6.2",
        19530,
        ["DEPLOY_MODE=STANDALONE", "ETCD_USE_EMBED=true", "COMMON_STORAGETYPE=local"],
        ["milvus", "run", "standalone"],
    ),
    "pgvector": (
        "pgvector/pgvector:pg17",
        5432,
        ["POSTGRES_PASSWORD=aireqa", "POSTGRES_DB=aireqa"],
        [],
    ),
    "redis": ("redis:8.2-alpine", 6379, [], []),
    "neo4j": (
        "neo4j:5.26-community",
        7687,
        ["NEO4J_AUTH=none", "NEO4J_server_memory_heap_max__size=512m"],
        [],
    ),
}


def run(*args: str, **kwargs: Any) -> str:
    return subprocess.check_output(list(args), text=True, **kwargs).strip()


def wait_for(check: Any, *, seconds: float = 180) -> None:
    deadline = time.monotonic() + seconds
    error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            if check():
                return
        except Exception as exc:
            error = exc
        time.sleep(0.5)
    raise RuntimeError(f"readiness deadline exceeded: {error}")


def probe(kind: str, url: str) -> bool:
    if kind == "pgvector":
        import psycopg

        with psycopg.connect(url, connect_timeout=2) as conn:
            return conn.execute("SELECT 1").fetchone()[0] == 1
    if kind == "redis":
        import redis

        with redis.Redis.from_url(url, socket_connect_timeout=2) as client:
            return bool(client.ping())
    if kind == "neo4j":
        from neo4j import GraphDatabase

        with GraphDatabase.driver(url, auth=None, connection_timeout=2) as driver:
            driver.verify_connectivity()
        return True
    if kind == "milvus":
        response = httpx.post(url + "/v2/vectordb/collections/list", json={}, timeout=2)
        return response.status_code == 200 and response.json().get("code") == 0
    path = {
        "qdrant": "/readyz",
        "chroma": "/api/v2/heartbeat",
        "weaviate": "/v1/.well-known/ready",
    }[kind]
    return httpx.get(url + path, timeout=2).status_code == 200


def verify_app(
    wheel: Path, network: str, containers: dict[str, str], start: Any, prefix: str
) -> str:
    image = prefix + ":test"
    with tempfile.TemporaryDirectory(prefix="aire-deploy-qa-") as directory:
        root = Path(directory)
        generate_artifacts(
            root,
            wheel=wheel,
            extra_requirements=["aire-ai[pgvector,redis]==" + __import__("aire").__version__],
        )
        target = (
            "aarch64-unknown-linux-gnu"
            if platform.machine() in {"arm64", "aarch64"}
            else "x86_64-unknown-linux-gnu"
        )
        lock_requirements(root, python_platform=target, find_links=root / "wheels")
        subprocess.run(["docker", "build", "-t", image, str(root)], check=True)
    database = f"postgresql://postgres:aireqa@{containers['pgvector']}:5432/aireqa"
    redis_url = f"redis://{containers['redis']}:6379/0"
    app, port = start(
        "app", image, 8000, [f"AIRE_DATABASE_URL={database}", f"AIRE_REDIS_URL={redis_url}"], []
    )
    url = f"http://127.0.0.1:{port}"
    wait_for(lambda: httpx.get(url + "/ready", timeout=5).status_code == 200)
    code = """
import asyncio, os
from aire import AI
from aire.rag.pgvector import PgVectorStore
from aire.rag.types import Document
async def main():
    knowledge = AI.rag.create(store=PgVectorStore(os.environ['AIRE_DATABASE_URL']))
    doc = Document(id='container-qa', text='Aire verifies short lived access tokens.')
    await knowledge.ingest([doc])
asyncio.run(main())
"""
    run("docker", "exec", app, "python", "-c", code)
    for _ in range(2):
        response = httpx.post(
            url + "/v1/ask", json={"question": "What does Aire verify?"}, timeout=15
        )
        assert response.status_code == 200, response.text
        assert response.json()["citations"], response.text
    assert response.json()["usage"]["input_tokens"] == 0, (
        "second generation must replay the Redis cache"
    )
    for backend in ["pgvector", "redis"]:
        run("docker", "stop", containers[backend])
        assert httpx.get(url + "/health").status_code == 200
        assert httpx.get(url + "/ready", timeout=10).status_code == 503
        run("docker", "start", containers[backend])
        wait_for(lambda: httpx.get(url + "/ready", timeout=10).status_code == 200)
    bad, _ = start(
        "bad-app",
        image,
        8000,
        ["AIRE_DATABASE_URL=postgresql://postgres:aireqa@127.0.0.1:1/aireqa"],
        [],
    )
    wait_for(
        lambda: run("docker", "inspect", "--format", "{{.State.Running}}", bad) == "false",
        seconds=30,
    )
    assert run("docker", "inspect", "--format", "{{.State.ExitCode}}", bad) != "0"
    run("docker", "stop", app)
    assert run("docker", "inspect", "--format", "{{.State.ExitCode}}", app) == "0"
    print(
        "Generated container passed: hash install, request/citations, cache replay, "
        "two outages/recovery, fail-closed startup, graceful shutdown.",
        flush=True,
    )
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path)
    args = parser.parse_args()
    prefix = "aire-qa-" + uuid.uuid4().hex[:10]
    network = prefix + "-network"
    containers: dict[str, str] = {}
    image: str | None = prefix + ":test" if args.wheel else None
    run("docker", "network", "create", "--label", "aire.qa=completion040", network)

    def start(
        kind: str, image_name: str, port: int, environment: list[str], command: list[str]
    ) -> tuple[str, str]:
        name = prefix + "-" + kind
        containers[kind] = name
        options = [
            "docker",
            "run",
            "-d",
            "--name",
            name,
            "--label",
            "aire.qa=completion040",
            "--network",
            network,
            "-p",
            f"127.0.0.1::{port}",
        ]
        for value in environment:
            options.extend(["-e", value])
        run(*options, image_name, *command)
        bindings = json.loads(
            run("docker", "inspect", "--format", "{{json .NetworkSettings.Ports}}", name)
        )
        return name, str(bindings[f"{port}/tcp"][0]["HostPort"])

    try:
        environment = dict(os.environ)
        for kind, (image_name, port, variables, command) in SERVICES.items():
            _, mapped = start(kind, image_name, port, variables, command)
            patterns = {
                "pgvector": "postgresql://postgres:aireqa@127.0.0.1:{port}/aireqa",
                "redis": "redis://127.0.0.1:{port}/0",
                "neo4j": "bolt://127.0.0.1:{port}",
            }
            url = patterns.get(kind, "http://127.0.0.1:{port}").format(port=mapped)
            wait_for(lambda kind=kind, url=url: probe(kind, url))
            environment["AIRE_LIVE_" + kind.upper()] = url
            print("Ready:", kind, flush=True)
        subprocess.run(
            [sys.executable, "-m", "pytest", "tests/live/test_adapter_contracts.py"],
            check=True,
            env=environment,
        )
        if args.wheel:
            verify_app(args.wheel.resolve(), network, containers, start, prefix)
    except BaseException:
        for name in containers.values():
            subprocess.run(["docker", "logs", "--tail", "12", name], check=False)
        raise
    finally:
        for name in reversed(list(containers.values())):
            subprocess.run(["docker", "rm", "-fv", name], check=False, capture_output=True)
        subprocess.run(["docker", "network", "rm", network], check=False, capture_output=True)
        if image:
            subprocess.run(["docker", "image", "rm", image], check=False, capture_output=True)


if __name__ == "__main__":
    main()
