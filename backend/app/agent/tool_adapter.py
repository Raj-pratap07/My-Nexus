"""
Adapter between MCP tools and model-visible function definitions.

Only tools passed in (the manager's ``permitted_tools``) are ever converted.
The catalog keeps an explicit alias -> qualified-name map, so the name a
model emits is never used directly to reach an MCP tool.
"""
from __future__ import annotations

import copy
import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Any, Mapping

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

logger = logging.getLogger(__name__)

# Fully qualified names ("research.search_openalex_works") already satisfy
# this and are exposed unchanged; anything else gets a sanitized alias.
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_MAX_DESCRIPTION = 1024


class ToolAdapterError(Exception):
    """Base class for tool-call validation failures."""


class UnknownToolError(ToolAdapterError):
    pass


class ToolArgumentError(ToolAdapterError):
    pass


@dataclass(frozen=True)
class ToolBinding:
    alias: str  # name shown to the model
    qualified_name: str  # actual MCP tool name, e.g. research.search_*
    description: str
    schema: dict[str, Any]


class ToolCatalog:
    def __init__(self, bindings: list[ToolBinding]) -> None:
        self._by_alias = {b.alias: b for b in bindings}

    @property
    def bindings(self) -> dict[str, ToolBinding]:
        return dict(self._by_alias)

    def __len__(self) -> int:
        return len(self._by_alias)

    def definitions(self) -> list[dict[str, Any]]:
        """Ollama/OpenAI-style function definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": b.alias,
                    "description": b.description,
                    "parameters": copy.deepcopy(b.schema),
                },
            }
            for b in self._by_alias.values()
        ]

    def resolve(self, alias: Any) -> ToolBinding:
        if not isinstance(alias, str) or alias not in self._by_alias:
            raise UnknownToolError(
                f"Unknown function '{str(alias)[:80]}'. "
                f"Available: {sorted(self._by_alias)}"
            )
        return self._by_alias[alias]

    @staticmethod
    def validate_arguments(binding: ToolBinding, arguments: Any) -> None:
        if not isinstance(arguments, dict):
            raise ToolArgumentError("Arguments must be a JSON object.")

        # Default-deny: MCP schemas usually omit additionalProperties, which
        # JSON Schema treats as "allowed". Unless the schema explicitly opts
        # in, reject any argument the schema does not declare.
        properties = binding.schema.get("properties", {})
        if binding.schema.get("additionalProperties", False) is False:
            unexpected = sorted(set(arguments) - set(properties))
            if unexpected:
                raise ToolArgumentError(
                    f"Unexpected arguments for '{binding.alias}': "
                    f"{[str(k)[:40] for k in unexpected[:5]]}. "
                    f"Allowed: {sorted(properties)}"
                )

        errors = sorted(
            Draft202012Validator(binding.schema).iter_errors(arguments),
            key=lambda e: list(e.absolute_path),
        )
        if errors:
            details = "; ".join(
                f"{'.'.join(str(p) for p in e.absolute_path) or '<root>'}: "
                f"{e.message[:120]}"
                for e in errors[:3]
            )
            raise ToolArgumentError(
                f"Invalid arguments for '{binding.alias}': {details}"
            )


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")[:50] or "tool"


def build_catalog(tools: Mapping[str, Any]) -> ToolCatalog:
    """
    Convert discovered MCP tools into a ToolCatalog.

    Tools with a malformed name or an invalid input schema are skipped
    (never exposed). Names are processed in sorted order so aliases are
    deterministic.
    """
    bindings: list[ToolBinding] = []
    used: set[str] = set()
    deferred: list[tuple[str, dict, str]] = []

    for qualified in sorted(tools):
        tool = tools[qualified]
        server, sep, short = qualified.partition(".")
        if not sep or not server or not short:
            logger.warning("Skipping tool without a server namespace.")
            continue

        schema = getattr(tool, "input_schema", None)
        if schema is None:
            schema = {"type": "object", "properties": {}}
        if not isinstance(schema, dict) or schema.get("type") != "object":
            logger.warning("Skipping %s: input schema is not an object.", qualified)
            continue
        try:
            Draft202012Validator.check_schema(schema)
        except SchemaError:
            logger.warning("Skipping %s: invalid input schema.", qualified)
            continue

        description = (getattr(tool, "description", None) or "")[:_MAX_DESCRIPTION]
        entry = (qualified, copy.deepcopy(schema), description)

        if _SAFE_NAME.match(qualified):
            used.add(qualified)
            bindings.append(ToolBinding(qualified, qualified, description, entry[1]))
        else:
            deferred.append(entry)

    # Unsafe names get a hash suffix; loop until the alias is unique.
    for qualified, schema, description in deferred:
        digest = hashlib.sha256(qualified.encode("utf-8")).hexdigest()
        length = 8
        alias = f"{_sanitize(qualified)}_{digest[:length]}"
        while alias in used and length < len(digest):
            length += 4
            alias = f"{_sanitize(qualified)}_{digest[:length]}"
        used.add(alias)
        bindings.append(ToolBinding(alias, qualified, description, schema))

    return ToolCatalog(bindings)
