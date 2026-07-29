"""Load OpenAPI tools from an in-memory dict (no network)."""

from aire.tools.openapi import load_openapi, openapi_to_tools

SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "Pets", "version": "1.0.0"},
    "servers": [{"url": "https://example.com"}],
    "paths": {
        "/pets": {
            "get": {
                "operationId": "listPets",
                "summary": "List pets",
                "responses": {"200": {"description": "ok"}},
            }
        }
    },
}


def main() -> None:
    loaded = load_openapi(SPEC)
    tools = openapi_to_tools(loaded, base_url="https://example.com")
    print("operations:", [t.name for t in tools])
    if tools:
        print("side_effect:", tools[0].spec.side_effect)


if __name__ == "__main__":
    main()
