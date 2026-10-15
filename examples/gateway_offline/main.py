"""Send an actual in-process HTTP request through the offline gateway."""

from fastapi.testclient import TestClient

from aire.core.config import Settings
from aire.core.runtime import Runtime
from aire.deployment.gateway import create_gateway


def main() -> None:
    runtime = Runtime(Settings())
    app = create_gateway(runtime, aliases={"echo": "mock:echo"}, close_runtime=True)
    with TestClient(app) as client:
        assert client.get("/ready").status_code == 200
        response = client.post(
            "/v1/chat/completions",
            json={"model": "echo", "messages": [{"role": "user", "content": "gateway request"}]},
        )
        assert response.status_code == 200
        text = response.json()["choices"][0]["message"]["content"]
        assert text == "gateway request"
        print("Gateway HTTP request passed:", text)
    assert runtime._closed


if __name__ == "__main__":
    main()
