"""Build an OpenAI-compat gateway app offline (no long bind)."""

from aire import AI


def main() -> None:
    app = AI.gateway.create(models=["mock:echo"], aliases={"echo": "mock:echo"})
    # Construct only — do not uvicorn.run for a long-lived server in this example.
    title = getattr(app, "title", type(app).__name__)
    routes = [getattr(r, "path", str(r)) for r in getattr(app, "routes", [])]
    print("gateway app:", title)
    print("route count:", len(routes))
    print("sample routes:", routes[:8])
    print("describe:", AI.gateway.describe()["kind"])


if __name__ == "__main__":
    main()
