"""Model gateway — an OpenAI-compatible server in front of any providers.

Run:  python examples/gateway/main.py

Creates a gateway offline (mock + callable models stand in for real providers),
then exercises it through the in-process test client: aliases with fallback
chains, round-robin routing, streaming SSE, embeddings and the manifest. To run
it for real against local/hosted providers:

    aire gateway -m ollama:llama3.2 \\
                 -a smart=anthropic:claude-sonnet-4-5,openai:gpt-4o-mini \\
                 --auth-token sk-internal --port 4000

and point any OpenAI client at http://127.0.0.1:4000/v1.
"""

from __future__ import annotations

from aire import AI


def main() -> None:
    AI.models.register_callable("shout", lambda prompt: prompt.upper())

    app = AI.gateway.create(
        models=["mock:echo"],
        aliases={
            # First candidate wins; on failure the request falls through the chain.
            "robust": ["unknown_provider:offline", "mock:echo"],
            # Round-robin between two backends.
            "balanced": ["mock:echo", "callable:shout"],
        },
        embeddings={"emb": "local:hashing"},
        routing="round_robin",
        auth_token="dev-token",
    )

    from fastapi.testclient import TestClient

    client = TestClient(app)
    headers = {"Authorization": "Bearer dev-token"}

    print("manifest:", client.get("/v1/gateway/manifest", headers=headers).json()["chat_models"])

    response = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={"model": "robust", "messages": [{"role": "user", "content": "hello gateway"}]},
    )
    body = response.json()
    print("\nfallback chain resolved to:", body["aire"]["resolved_model"])
    print("answer:", body["choices"][0]["message"]["content"])

    for _ in range(2):
        r = client.post(
            "/v1/chat/completions",
            headers=headers,
            json={"model": "balanced", "messages": [{"role": "user", "content": "round robin"}]},
        )
        print("round robin →", r.json()["aire"]["resolved_model"])

    streamed = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "mock:echo",
            "messages": [{"role": "user", "content": "stream this"}],
            "stream": True,
        },
    )
    chunks = [line for line in streamed.text.splitlines() if line.startswith("data:")]
    print(f"\nstreaming: {len(chunks)} SSE events, last is {chunks[-1]!r}")

    vectors = client.post(
        "/v1/embeddings", headers=headers, json={"model": "emb", "input": "embed me"}
    ).json()
    print("embedding dim:", len(vectors["data"][0]["embedding"]))


if __name__ == "__main__":
    main()
