"""
User identity boundary.

Routes depend on ``AuthProvider`` and receive a ``Principal``; nothing
else in the application decides who the caller is. The only provider in
this milestone is ``LocalTokenAuth``: a single static bearer token mapped
to one fixed local user, meant for loopback development ONLY. A real
multi-user provider can replace it without touching routes or services.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Protocol

from app.core.security import tokens_equal

LOCAL_USER_ID = uuid.uuid5(uuid.NAMESPACE_URL, "urn:my-nexus:local-dev-user")


class AuthenticationError(Exception):
    pass


@dataclass(frozen=True)
class Principal:
    user_id: uuid.UUID


class AuthProvider(Protocol):
    def authenticate(self, authorization_header: str | None) -> Principal:
        """Return the caller's Principal or raise AuthenticationError."""


class LocalTokenAuth:
    def __init__(self, token: str, user_id: uuid.UUID = LOCAL_USER_ID) -> None:
        if not token:
            raise ValueError("token must not be empty")
        self._token = token
        self._user_id = user_id

    def authenticate(self, authorization_header: str | None) -> Principal:
        scheme, _, supplied = (authorization_header or "").partition(" ")
        if scheme.lower() != "bearer" or not supplied.strip():
            raise AuthenticationError("Missing bearer token.")
        if not tokens_equal(supplied.strip(), self._token):
            raise AuthenticationError("Invalid bearer token.")
        return Principal(user_id=self._user_id)
