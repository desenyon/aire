"""Model Context Protocol: expose and consume tools, zero-dependency stdio."""

from aire.mcp.client import MCPClient
from aire.mcp.protocol import MCPError
from aire.mcp.server import MCPServer

__all__ = ["MCPClient", "MCPError", "MCPServer"]
