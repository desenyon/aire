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
