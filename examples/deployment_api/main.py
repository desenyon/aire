"""Deployment — serve a knowledge assistant as a production FastAPI app.

Run:  python examples/deployment_api/main.py

Builds the assistant offline, wraps it with AI.deploy.api() (auth + rate
limit + observability), exercises it with the in-process test client, and
generates deployment artifacts (Dockerfile, entrypoint, .env.template).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from aire import AI


def main() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="aire-deploy-"))
    (workdir / "policy.md").write_text(
        "Refunds are allowed within 30 days of purchase with a receipt."
    )

    assistant = (
        AI.project("support_assistant")
        .documents(str(workdir))
        .model("mock:echo")
        .vector_store("local:default")
    )
    assistant.index()

    app = AI.deploy.api(
        assistant.knowledge(),
        title="support API",
        auth_token="dev-token",
        rate_limit_per_minute=60,
        metrics=AI.observe.metrics,
    )

    from fastapi.testclient import TestClient

    client = TestClient(app)
    print("health:", client.get("/health").json())
    print("ready: ", client.get("/ready").json())
    print("unauthorized:", client.post("/v1/ask", json={"question": "hi"}).status_code)
    response = client.post(
        "/v1/ask",
        json={"question": "what is the refund policy?"},
        headers={"Authorization": "Bearer dev-token"},
    )
    print("answer:", response.json()["answer"][:80])

    out = AI.deploy.artifacts(workdir / "artifacts")
    print("\nartifacts:", [Path(f).name for f in out.files])


if __name__ == "__main__":
    main()
