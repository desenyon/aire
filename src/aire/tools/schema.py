"""Compile the supported JSON-schema subset into executable Pydantic contracts."""

from __future__ import annotations

from types import GenericAlias
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from aire.core.errors import ConfigurationError


def resolve_schema(schema: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(schema, dict):
        raise ConfigurationError(
            "unsupported non-object tool schema", code="tool.schema_unsupported"
        )
    unsupported = {
        "allOf",
        "oneOf",
        "not",
        "if",
        "then",
        "else",
        "dependentSchemas",
        "patternProperties",
    } & schema.keys()
    if unsupported:
        raise ConfigurationError(
            f"unsupported tool schema keywords: {sorted(unsupported)}",
            code="tool.schema_unsupported",
        )
    if "$ref" not in schema:
        return schema
    reference = schema["$ref"]
    if not isinstance(reference, str) or not reference.startswith("#/"):
        raise ConfigurationError(
            "only local schema references are supported", code="tool.schema_ref"
        )
    value: Any = root
    try:
        for key in reference[2:].split("/"):
            value = value[key.replace("~1", "/").replace("~0", "~")]
    except (KeyError, TypeError) as exc:
        raise ConfigurationError("unresolved schema reference", code="tool.schema_ref") from exc
    if not isinstance(value, dict):
        raise ConfigurationError(
            "schema reference must resolve to an object", code="tool.schema_ref"
        )
    merged = {**value, **{key: val for key, val in schema.items() if key != "$ref"}}
    if "$ref" in merged:
        raise ConfigurationError(
            "chained/recursive schema reference is unsupported", code="tool.schema_ref"
        )
    return resolve_schema(merged, root)


def schema_model(
    name: str, schema: dict[str, Any], *, root: dict[str, Any] | None = None, depth: int = 0
) -> type[BaseModel]:
    root = root if root is not None else schema
    schema = resolve_schema(schema, root)
    if depth > 12:
        raise ConfigurationError(
            "recursive/deep tool schema is unsupported", code="tool.schema_depth"
        )
    fields: dict[str, Any] = {}
    required = set(schema.get("required", []))
    if required - schema.get("properties", {}).keys():
        raise ConfigurationError(
            "required keys outside properties are unsupported", code="tool.schema_unsupported"
        )
    if isinstance(schema.get("additionalProperties"), dict):
        raise ConfigurationError(
            "typed additional properties are unsupported", code="tool.schema_unsupported"
        )
    for key, raw in schema.get("properties", {}).items():
        prop = resolve_schema(raw, root)
        annotation = _schema_type(f"{name}_{key}", prop, root, depth + 1)
        options = {
            target: prop[source]
            for source, target in (
                ("minimum", "ge"),
                ("maximum", "le"),
                ("exclusiveMinimum", "gt"),
                ("exclusiveMaximum", "lt"),
                ("minLength", "min_length"),
                ("maxLength", "max_length"),
                ("minItems", "min_length"),
                ("maxItems", "max_length"),
                ("pattern", "pattern"),
            )
            if source in prop
        }
        fields[key] = (
            annotation,
            Field(
                default=... if key in required else prop.get("default"),
                description=prop.get("description"),
                **options,
            ),
        )
    return create_model(
        name,
        __config__=ConfigDict(
            extra="allow" if schema.get("additionalProperties", True) else "forbid"
        ),
        **fields,
    )


def _schema_type(name: str, schema: dict[str, Any], root: dict[str, Any], depth: int) -> Any:
    schema = resolve_schema(schema, root)
    if depth > 12:
        raise ConfigurationError(
            "recursive/deep tool schema is unsupported", code="tool.schema_depth"
        )
    if "enum" in schema:
        if not schema["enum"]:
            raise ConfigurationError(
                "empty schema enum is unsupported", code="tool.schema_unsupported"
            )
        return Literal[tuple(schema["enum"])]
    if "const" in schema:
        return Literal[schema["const"]]
    if schema.get("nullable"):
        return _union_type(name, [{**schema, "nullable": False}, {"type": "null"}], root, depth)
    if "anyOf" in schema:
        return _union_type(name, schema["anyOf"], root, depth)
    kind = schema.get("type", "object" if "properties" in schema else None)
    if isinstance(kind, list):
        return _union_type(name, [{**schema, "type": variant} for variant in kind], root, depth)
    if kind == "object":
        return schema_model(name, schema, root=root, depth=depth)
    if kind == "array":
        return GenericAlias(
            list, _schema_type(name + "_item", schema.get("items", {}), root, depth + 1)
        )
    result = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "null": type(None),
    }.get(str(kind), Any)
    return result | None if schema.get("nullable") else result


def _union_type(name: str, variants: list[dict[str, Any]], root: dict[str, Any], depth: int) -> Any:
    if not variants:
        raise ConfigurationError(
            "empty schema union is unsupported", code="tool.schema_unsupported"
        )
    result = _schema_type(name, variants[0], root, depth + 1)
    for variant in variants[1:]:
        result = result | _schema_type(name, variant, root, depth + 1)
    return result
