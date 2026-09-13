"""Shared error types for tool-level refusals.

`ToolError` subclasses fastmcp's ToolError so MCP clients receive a structured
error result while service code (and tests) have a stable, importable type
independent of the transport layer.
"""

from __future__ import annotations

from fastmcp.exceptions import ToolError as _FastMCPToolError


class ToolError(_FastMCPToolError):
    """A tool refused to proceed (validation failure, gate refusal, budget
    enforcement, tombstone block). Carries a human-readable message; callers
    that need structured detail pass it in the message payload."""
