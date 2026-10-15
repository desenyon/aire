import sys

from aire.mcp.client import MCPClient
from aire.models.base import run_sync


def test_mcp_real_stdio_roundtrip():
    async def check():
        async with MCPClient([sys.executable, "-m", "aire.mcp"]) as client:
            tools = await client.list_tools()
            assert any(t["name"] == "calculator" for t in tools)
            assert float(await client.call_tool("calculator", {"expression": "2+3"})) == 5
            resources = await client.list_resources()
            assert resources
            assert await client.read_resource(resources[0]["uri"])
            prompts = await client.list_prompts()
            assert prompts
            assert await client.get_prompt(prompts[0]["name"])
        assert client._process is None

    run_sync(check())


def test_http_session_auth_progress_and_roots_callback():
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from aire.mcp.http_client import MCPHttpClient

    roots_received = threading.Event()
    sampling_received = threading.Event()
    unsupported_received = threading.Event()
    progress = []
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            assert self.headers.get("Authorization") == "Bearer qa"
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(body)
            if body.get("id") == "roots-request" and "result" in body:
                roots_received.set()
                self.send_response(202)
                self.end_headers()
                return
            if body.get("id") == "sampling-request" and "result" in body:
                assert body["result"]["content"]["text"] == "sampled"
                sampling_received.set()
                self.send_response(202)
                self.end_headers()
                return
            if body.get("id") == "unsupported-request" and "error" in body:
                assert body["error"]["code"] == -32601
                unsupported_received.set()
                self.send_response(202)
                self.end_headers()
                return
            if "id" not in body:
                self.send_response(202)
                self.end_headers()
                return
            if body["method"] == "initialize":
                result = {"serverInfo": {"name": "qa"}, "capabilities": {"tools": {}}}
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("mcp-session-id", "qa-session")
                self.end_headers()
                self.wfile.write(
                    json.dumps({"jsonrpc": "2.0", "id": body["id"], "result": result}).encode()
                )
                return
            assert self.headers.get("mcp-session-id") == "qa-session"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()

            def emit(value):
                self.wfile.write(("data: " + json.dumps(value) + "\n\n").encode())
                self.wfile.flush()

            emit({"jsonrpc": "2.0", "id": "roots-request", "method": "roots/list"})
            assert roots_received.wait(3), "client must reply before final response"
            emit(
                {
                    "jsonrpc": "2.0",
                    "id": "sampling-request",
                    "method": "sampling/createMessage",
                    "params": {"messages": []},
                }
            )
            assert sampling_received.wait(3)
            emit({"jsonrpc": "2.0", "id": "unsupported-request", "method": "client/unsupported"})
            assert unsupported_received.wait(3)
            emit({"jsonrpc": "2.0", "method": "notifications/progress", "params": {"progress": 1}})
            emit(
                {
                    "jsonrpc": "2.0",
                    "id": body["id"],
                    "result": {"tools": [{"name": "qa", "inputSchema": {"type": "object"}}]},
                }
            )

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    async def check():
        async with MCPHttpClient(
            f"http://127.0.0.1:{server.server_port}/mcp",
            headers={"Authorization": "Bearer qa"},
            on_progress=progress.append,
            sampling_handler=lambda params: {
                "role": "assistant",
                "model": "qa",
                "content": {"type": "text", "text": "sampled"},
            },
        ) as client:
            assert (await client.list_tools())[0]["name"] == "qa"
        assert client._http is None

    try:
        run_sync(check())
        assert progress == [{"progress": 1}]
        assert roots_received.is_set()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_stdio_request_deadline_and_pending_cleanup():
    import asyncio

    import pytest

    from aire.mcp.client import MCPError

    async def check():
        client = MCPClient(
            [sys.executable, "-c", "import time; time.sleep(30)"], request_timeout=0.02
        )
        client._process = await asyncio.create_subprocess_exec(
            *client.command, stdin=asyncio.subprocess.PIPE
        )
        try:
            with pytest.raises(MCPError, match="timed out"):
                await client.list_tools()
            assert not client._pending
        finally:
            await client.close()

    run_sync(check())


def test_http_overall_deadline_and_cancelled_startup_cleanup(monkeypatch):
    import asyncio

    import pytest

    from aire.mcp.http_client import MCPHttpClient
    from aire.mcp.protocol import MCPError

    async def check():
        client = MCPHttpClient("http://unused", timeout=0.01)

        async def never(*args, **kwargs):
            await asyncio.Event().wait()

        monkeypatch.setattr(client, "_post", never)
        with pytest.raises(MCPError, match="timed out"):
            await asyncio.wait_for(client.list_tools(), timeout=0.1)
        monkeypatch.setattr(client, "_request", never)
        connecting = asyncio.create_task(client.connect())
        await asyncio.sleep(0.01)
        connecting.cancel()
        try:
            with pytest.raises(asyncio.CancelledError):
                await connecting
            assert client._http is None
        finally:
            await client.close()

    run_sync(check())


def test_stdio_startup_cancellation_reaps_subprocess():
    import asyncio

    import pytest

    async def check():
        client = MCPClient([sys.executable, "-c", "import time; time.sleep(30)"])
        connecting = asyncio.create_task(client.connect())
        for _ in range(100):
            if client._process is not None:
                break
            await asyncio.sleep(0.001)
        connecting.cancel()
        try:
            with pytest.raises(asyncio.CancelledError):
                await connecting
            assert client._process is None
        finally:
            await client.close()

    run_sync(check())


def test_http_malformed_or_unauthorized_initialization_closes_client(monkeypatch):
    import httpx
    import pytest

    from aire.mcp.http_client import MCPHttpClient
    from aire.mcp.protocol import MCPError

    original = httpx.AsyncClient

    async def check():
        for response in [
            httpx.Response(401),
            httpx.Response(200, content=b"bad-json"),
            httpx.Response(200, json={"jsonrpc": "2.0", "id": "wrong", "result": {}}),
        ]:
            monkeypatch.setattr(
                httpx,
                "AsyncClient",
                lambda response=response, **kw: original(
                    transport=httpx.MockTransport(lambda req: response), **kw
                ),
            )
            client = MCPHttpClient("http://qa")
            with pytest.raises(MCPError):
                await client.connect()
            assert client._http is None

    run_sync(check())
