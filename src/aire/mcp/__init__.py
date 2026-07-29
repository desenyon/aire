"""Model Context Protocol: subset — tools, resources, prompts over stdio + HTTP."""

from aire.mcp.client import MCPClient
from aire.mcp.http_client import MCPHttpClient
from aire.mcp.protocol import MCPError
from aire.mcp.server import MCPServer

__all__ = ["MCPClient", "MCPError", "MCPHttpClient", "MCPServer"]
