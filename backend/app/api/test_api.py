"""
Offline API tests (FastAPI TestClient, SQLite, fake agent runtime).

Run from backend/:
    .venv/Scripts/python.exe -m unittest app.api.test_api -v
"""
from __future__ import annotations

import ast
import unittest
import uuid
from pathlib import Path

from fastapi.testclient import TestClient

from app.agent.orchestrator import AgentResult, ToolCallRecord
from app.core.auth import AuthenticationError, Principal
from app.core.errors import StartupError
from app.main import create_app
from app.testutil import TEST_TOKEN, TempDb, create_schema, make_settings

USER_A, USER_B = uuid.uuid4(), uuid.uuid4()
TOKEN_A, TOKEN_B = "a" * 40, "b" * 40
HDR_A = {"Authorization": f"Bearer {TOKEN_A}"}
HDR_B = {"Authorization": f"Bearer {TOKEN_B}"}


class TwoUserAuth:
    def authenticate(self, header):
        scheme, _, token = (header or "").partition(" ")
        users = {TOKEN_A: USER_A, TOKEN_B: USER_B}
        if scheme.lower() != "bearer" or token not in users:
            raise AuthenticationError("nope")
        return Principal(users[token])


class FakeRuntime:
    instances: list["FakeRuntime"] = []

    def __init__(self, settings, *, fail_start=False, answer="Hello from the model.", boom=None):
        self.started = self.stopped = False
        self.fail_start, self.answer, self.boom = fail_start, answer, boom
        self.calls = []
        FakeRuntime.instances.append(self)

    async def start(self):
        if self.fail_start:
            raise StartupError("Could not start the Research MCP server (OSError).")
        self.started = True

    async def stop(self):
        self.stopped = True

    async def run(self, message, history):
        self.calls.append((message, list(history)))
        if self.boom:
            raise self.boom
        rec = ToolCallRecord("research.search_crossref_works",
                             "research.search_crossref_works", {"query": "x"}, True, None, "succeeded")
        return AgentResult(self.answer, "complete", [rec])


class ApiCase(unittest.TestCase):
    def setUp(self):
        FakeRuntime.instances.clear()
        self.db = TempDb()
        import asyncio
        from app.testutil import make_engine

        async def prep():
            engine = await make_engine(self.db.url)
            await engine.dispose()

        asyncio.run(prep())
        self.settings = make_settings(self.db.url)
        self.addCleanup(self.db.cleanup)

    def client(self, *, auth=None, runtime_factory=FakeRuntime, **kw):
        app = create_app(self.settings, auth=auth or TwoUserAuth(), runtime_factory=runtime_factory)
        c = TestClient(app, raise_server_exceptions=False, **kw)
        c.__enter__()
        self.addCleanup(lambda: c.__exit__(None, None, None))
        return c

    def new_conv(self, c, headers=HDR_A, title=None):
        r = c.post("/v1/conversations", json={"title": title} if title else {}, headers=headers)
        self.assertEqual(r.status_code, 201, r.text)
        return r.json()["id"]


class LifecycleTests(ApiCase):
    def test_startup_starts_runtime_once_and_shutdown_stops_it(self):
        app = create_app(self.settings, auth=TwoUserAuth(), runtime_factory=FakeRuntime)
        with TestClient(app) as c:
            self.assertEqual(c.get("/health").json(), {"status": "ok"})
            for _ in range(3):  # many requests, still one runtime
                self.new_conv(c)
            self.assertEqual(len(FakeRuntime.instances), 1)
            self.assertTrue(FakeRuntime.instances[0].started)
            self.assertFalse(FakeRuntime.instances[0].stopped)
        self.assertTrue(FakeRuntime.instances[0].stopped)

    def test_runtime_start_failure_is_clear_and_cleans_up(self):
        app = create_app(self.settings, auth=TwoUserAuth(),
                         runtime_factory=lambda s: FakeRuntime(s, fail_start=True))
        with self.assertLogs("app.main", level="ERROR") as logs:
            with self.assertRaises(StartupError) as ctx:
                with TestClient(app):
                    pass
        self.assertIn("Research MCP server", str(ctx.exception))
        self.assertIn("Startup failed", logs.output[0])
        self.assertTrue(FakeRuntime.instances[0].stopped)

    def test_unreachable_database_fails_startup_without_leaking_path(self):
        settings = make_settings("sqlite+aiosqlite:///Z:/missing-dir/very-secret.db")
        app = create_app(settings, auth=TwoUserAuth(), runtime_factory=FakeRuntime)
        with self.assertLogs("app.main", level="ERROR") as logs:
            with self.assertRaises(StartupError) as ctx:
                with TestClient(app):
                    pass
        self.assertNotIn("very-secret", str(ctx.exception) + "".join(logs.output))
        self.assertEqual(FakeRuntime.instances, [])  # agent never started without a DB

    def test_unmigrated_database_fails_startup(self):
        settings = make_settings(self.db.url, skip_schema_check=False)
        app = create_app(settings, auth=TwoUserAuth(), runtime_factory=FakeRuntime)
        with self.assertLogs("app.main", level="ERROR"):
            with self.assertRaisesRegex(StartupError, "alembic upgrade head"):
                with TestClient(app):
                    pass

    def test_non_loopback_host_is_refused(self):
        for host in ("0.0.0.0", "192.168.1.5", "example.com", "::"):
            with self.subTest(host=host), self.assertRaisesRegex(StartupError, "non-loopback"):
                create_app(make_settings(self.db.url, host=host))

    def test_short_dev_token_is_refused_without_echoing_it(self):
        with self.assertRaises(StartupError) as ctx:
            create_app(make_settings(self.db.url, dev_token="short-secret"))
        self.assertNotIn("short-secret", str(ctx.exception))


class AuthTests(ApiCase):
    def test_every_v1_route_requires_authentication(self):
        c = self.client()
        cid = uuid.uuid4()
        for method, path, body in (
            ("post", "/v1/conversations", {}),
            ("get", "/v1/conversations", None),
            ("get", f"/v1/conversations/{cid}", None),
            ("delete", f"/v1/conversations/{cid}", None),
            ("get", f"/v1/conversations/{cid}/messages", None),
            ("post", f"/v1/conversations/{cid}/messages", {"content": "x"}),
        ):
            for headers in ({}, {"Authorization": "Bearer wrong"}, {"Authorization": "Basic abc"}):
                with self.subTest(path=path, headers=headers):
                    r = getattr(c, method)(path, headers=headers, **({"json": body} if body is not None else {}))
                    self.assertEqual(r.status_code, 401)
                    self.assertEqual(r.headers["www-authenticate"], "Bearer")

    def test_user_identity_cannot_be_supplied_by_the_client(self):
        c = self.client()
        r = c.post("/v1/conversations", json={"user_id": str(USER_B)}, headers=HDR_A)
        self.assertEqual(r.status_code, 422)
        cid = self.new_conv(c, HDR_A)
        r = c.get(f"/v1/conversations/{cid}", headers={**HDR_A, "X-User-Id": str(USER_B)})
        self.assertEqual(r.status_code, 200)

    def test_default_local_token_auth_works(self):
        app = create_app(self.settings, runtime_factory=FakeRuntime)
        with TestClient(app) as c:
            self.assertEqual(c.get("/v1/conversations").status_code, 401)
            ok = c.get("/v1/conversations", headers={"Authorization": f"Bearer {TEST_TOKEN}"})
            self.assertEqual(ok.status_code, 200)

    def test_docs_and_openapi_are_not_exposed(self):
        c = self.client()
        for path in ("/docs", "/redoc", "/openapi.json"):
            self.assertEqual(c.get(path).status_code, 404)


class OwnershipApiTests(ApiCase):
    def test_users_cannot_access_each_others_conversations(self):
        c = self.client()
        cid = self.new_conv(c, HDR_A, "private")
        c.post(f"/v1/conversations/{cid}/messages", json={"content": "secret plans"}, headers=HDR_A)

        for method, path, kw in (
            ("get", f"/v1/conversations/{cid}", {}),
            ("get", f"/v1/conversations/{cid}/messages", {}),
            ("post", f"/v1/conversations/{cid}/messages", {"json": {"content": "hi"}}),
            ("delete", f"/v1/conversations/{cid}", {}),
        ):
            r = getattr(c, method)(path, headers=HDR_B, **kw)
            self.assertEqual(r.status_code, 404, (method, path))
            self.assertNotIn("secret plans", r.text)
        self.assertEqual(c.get("/v1/conversations", headers=HDR_B).json(), [])
        self.assertEqual(len(c.get(f"/v1/conversations/{cid}/messages", headers=HDR_A).json()), 2)
        # The agent was never invoked on behalf of B.
        self.assertEqual(len(FakeRuntime.instances[0].calls), 1)

    def test_random_conversation_id_is_404_same_as_foreign(self):
        c = self.client()
        r = c.get(f"/v1/conversations/{uuid.uuid4()}", headers=HDR_A)
        self.assertEqual(r.status_code, 404)
        self.assertEqual({k: v for k, v in r.json()["error"].items() if k != "request_id"},
                         {"code": "not_found", "message": "Resource not found.",
                          "details": None})


class ConversationFlowTests(ApiCase):
    def test_full_turn(self):
        c = self.client()
        cid = self.new_conv(c)
        r = c.post(f"/v1/conversations/{cid}/messages", json={"content": " Find RAG papers "}, headers=HDR_A)
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["user_message"]["content"], "Find RAG papers")
        self.assertEqual(body["assistant_message"]["content"], "Hello from the model.")
        self.assertEqual([(t["tool_name"], t["status"]) for t in body["tool_executions"]],
                         [("research.search_crossref_works", "succeeded")])
        self.assertEqual(set(body["tool_executions"][0]), {"tool_name", "status", "created_at"})
        self.assertEqual(c.get(f"/v1/conversations/{cid}", headers=HDR_A).json()["title"], "Find RAG papers")
        self.assertEqual(c.delete(f"/v1/conversations/{cid}", headers=HDR_A).status_code, 204)
        self.assertEqual(c.get(f"/v1/conversations/{cid}", headers=HDR_A).status_code, 404)

    def test_pagination_bounds(self):
        c = self.client()
        for bad in ("limit=0", "limit=101", "offset=-1"):
            self.assertEqual(c.get(f"/v1/conversations?{bad}", headers=HDR_A).status_code, 422)


class InputValidationTests(ApiCase):
    def test_message_validation(self):
        c = self.client()
        cid = self.new_conv(c)
        url = f"/v1/conversations/{cid}/messages"
        bad_bodies = [
            {}, {"content": ""}, {"content": "   "}, {"content": "x" * 8001},
            {"content": "a\x00b"}, {"content": 5}, {"content": None},
            {"content": "hi", "role": "system"},   # no client-supplied roles
            {"content": "hi", "tool_calls": []},
            {"role": "tool", "content": "hi"},
            {"messages": [{"role": "system", "content": "x"}]},
        ]
        for body in bad_bodies:
            with self.subTest(body=str(body)[:40]):
                self.assertEqual(c.post(url, json=body, headers=HDR_A).status_code, 422)
        self.assertEqual(c.get(url, headers=HDR_A).json(), [])
        self.assertEqual(FakeRuntime.instances[0].calls, [])

    def test_title_validation_and_malformed_ids(self):
        c = self.client()
        for body in ({"title": ""}, {"title": "x" * 201}, {"title": 5}, {"extra": 1}):
            self.assertEqual(c.post("/v1/conversations", json=body, headers=HDR_A).status_code, 422)
        self.assertEqual(c.get("/v1/conversations/not-a-uuid", headers=HDR_A).status_code, 422)

    def test_validation_errors_do_not_echo_submitted_values(self):
        c = self.client()
        cid = self.new_conv(c)
        r = c.post(f"/v1/conversations/{cid}/messages",
                   json={"content": "x" * 8001, "role": "TOPSECRETVALUE"}, headers=HDR_A)
        self.assertEqual(r.status_code, 422)
        self.assertNotIn("TOPSECRETVALUE", r.text)
        self.assertNotIn('"input"', r.text)  # no echoed-value field

    def test_oversized_body_is_rejected_early(self):
        c = self.client()
        cid = self.new_conv(c)
        r = c.post(f"/v1/conversations/{cid}/messages", content=b'{"content":"' + b"x" * 70_000 + b'"}',
                   headers={**HDR_A, "Content-Type": "application/json"})
        self.assertEqual(r.status_code, 413)

    def test_unexpected_host_header_is_rejected(self):
        c = self.client(base_url="http://evil.example.com")
        self.assertEqual(c.get("/health").status_code, 400)


class CredentialAndErrorTests(ApiCase):
    def test_credentials_in_user_message_rejected_and_not_stored(self):
        c = self.client()
        cid = self.new_conv(c)
        r = c.post(f"/v1/conversations/{cid}/messages",
                   json={"content": "use github_pat_" + "A1b2" * 10}, headers=HDR_A)
        self.assertEqual(r.status_code, 422)
        self.assertNotIn("github_pat_", r.text)
        self.assertEqual(c.get(f"/v1/conversations/{cid}/messages", headers=HDR_A).json(), [])

    def test_model_unavailable_is_503_without_internals(self):
        from app.agent.providers import ProviderConnectionError

        c = self.client(runtime_factory=lambda s: FakeRuntime(
            s, boom=ProviderConnectionError("Cannot reach Ollama at http://localhost:11434")))
        cid = self.new_conv(c)
        r = c.post(f"/v1/conversations/{cid}/messages", json={"content": "hi"}, headers=HDR_A)
        self.assertEqual(r.status_code, 503)
        self.assertNotIn("11434", r.text)
        self.assertNotIn("Ollama", r.text)

    def test_unexpected_exception_gives_generic_500_without_trace(self):
        c = self.client(runtime_factory=lambda s: FakeRuntime(
            s, boom=RuntimeError("password=hunter2hunter2 /srv/app/internal.py")))
        cid = self.new_conv(c)
        with self.assertLogs("app.main", level="ERROR") as logs:
            r = c.post(f"/v1/conversations/{cid}/messages", json={"content": "hi"}, headers=HDR_A)
        self.assertEqual(r.status_code, 500)
        self.assertEqual(set(r.json()), {"error"})
        self.assertEqual(r.json()["error"]["code"], "internal_error")
        for leaked in ("hunter2", "internal.py", "Traceback", "RuntimeError"):
            self.assertNotIn(leaked, r.text)
        log_text = "".join(logs.output)
        for leaked in ("hunter2", "internal.py", "Traceback"):
            self.assertNotIn(leaked, log_text)
        self.assertIn("RuntimeError", log_text)  # type is logged, message is not

    def test_assistant_output_containing_a_secret_is_redacted(self):
        c = self.client(runtime_factory=lambda s: FakeRuntime(s, answer=f"here: {TEST_TOKEN} ghp_{'c' * 36}"))
        cid = self.new_conv(c)
        r = c.post(f"/v1/conversations/{cid}/messages", json={"content": "hi"}, headers=HDR_A)
        text = r.json()["assistant_message"]["content"]
        self.assertNotIn(TEST_TOKEN, text)
        self.assertNotIn("ghp_", text)


# ----- Phase 4 M1: error envelope, failed turns, tool history ------------------
from app.agent.providers import ProviderConnectionError, ProviderTimeoutError  # noqa: E402
from app.schemas.conversation import ErrorResponse  # noqa: E402

ENVELOPE_KEYS = {"code", "message", "request_id", "details"}


class ScriptedRuntime(FakeRuntime):
    """Plays scripted outcomes in order: an AgentResult, or an exception to raise."""

    script: list = []

    async def run(self, message, history):
        self.calls.append((message, list(history)))
        outcome = self.script.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def scripted(*outcomes):
    def factory(settings):
        runtime = ScriptedRuntime(settings)
        runtime.script = list(outcomes)
        return runtime
    return factory


def tool(name, outcome, args=None):
    return ToolCallRecord(name, name if not name.startswith("?") else None,
                          args or {}, outcome != "rejected", None, outcome)


def result(answer="ok", *tools):
    return AgentResult(answer, "complete", list(tools))


class ErrorEnvelopeTests(ApiCase):
    def assert_error(self, r, status, code, *, details=None):
        self.assertEqual(r.status_code, status, r.text)
        body = r.json()
        ErrorResponse.model_validate(body)                 # exactly {"error": {...}}
        self.assertEqual(set(body), {"error"})
        self.assertEqual(set(body["error"]), ENVELOPE_KEYS)
        self.assertEqual(body["error"]["code"], code)
        self.assertEqual(body["error"]["request_id"], r.headers["x-request-id"])
        self.assertRegex(body["error"]["request_id"], r"^[0-9a-f]{12}$")
        if details is not None:
            self.assertEqual(body["error"]["details"], details)
        return body["error"]

    def test_401_missing_and_invalid_token(self):
        c = self.client()
        for headers in ({}, {"Authorization": "Bearer wrong"}):
            with self.subTest(headers=headers):
                r = c.get("/v1/conversations", headers=headers)
                err = self.assert_error(r, 401, "unauthenticated", details=None)
                self.assertEqual(err["message"], "Authentication required.")
                self.assertEqual(r.headers["www-authenticate"], "Bearer")

    def test_404_missing_resource_and_unknown_route(self):
        c = self.client()
        self.assert_error(c.get(f"/v1/conversations/{uuid.uuid4()}", headers=HDR_A),
                          404, "not_found", details=None)
        self.assert_error(c.get("/no/such/route", headers=HDR_A), 404, "not_found")

    def test_405_keeps_the_allow_header(self):
        r = self.client().put("/v1/conversations", headers=HDR_A, json={})
        self.assert_error(r, 405, "method_not_allowed", details=None)
        # Passed through from Starlette unchanged (it names the first matching
        # route's methods, so only its presence is asserted).
        self.assertTrue(r.headers["allow"])

    def test_409_conversation_busy(self):
        c = self.client()
        cid = self.new_conv(c)
        c.app.state.conversation_service._in_flight[uuid.UUID(cid)] = True
        err = self.assert_error(c.post(f"/v1/conversations/{cid}/messages",
                                       json={"content": "hi"}, headers=HDR_A),
                                409, "conversation_busy", details=None)
        self.assertEqual(err["message"], "This conversation is already processing a message.")

    def test_413_request_too_large(self):
        c = self.client()
        r = c.post("/v1/conversations", content=b"{" + b" " * 70_000 + b"}",
                   headers={**HDR_A, "Content-Type": "application/json"})
        self.assert_error(r, 413, "request_too_large", details=None)

    def test_422_validation_details_are_useful_and_never_echo_input(self):
        c = self.client()
        cid = self.new_conv(c)
        r = c.post(f"/v1/conversations/{cid}/messages",
                   json={"content": "x" * 8001, "role": "TOPSECRETVALUE"}, headers=HDR_A)
        err = self.assert_error(r, 422, "validation_error")
        self.assertEqual(err["message"], "Request validation failed.")
        self.assertTrue(err["details"])
        for item in err["details"]:
            self.assertEqual(set(item), {"loc", "msg", "type"})
        self.assertIn(["body", "content"], [d["loc"] for d in err["details"]])
        self.assertIn("extra_forbidden", [d["type"] for d in err["details"]])
        self.assertNotIn("TOPSECRETVALUE", r.text)

    def test_422_credential_rejected(self):
        c = self.client()
        cid = self.new_conv(c)
        r = c.post(f"/v1/conversations/{cid}/messages",
                   json={"content": "use github_pat_" + "A1b2" * 10}, headers=HDR_A)
        self.assert_error(r, 422, "credential_rejected", details=None)
        self.assertNotIn("github_pat_", r.text)

    def test_503_points_at_the_recorded_failed_turn(self):
        c = self.client(runtime_factory=scripted(ProviderTimeoutError(
            "Ollama did not respond within 600.0s at http://localhost:11434")))
        cid = self.new_conv(c)
        r = c.post(f"/v1/conversations/{cid}/messages", json={"content": "hi"}, headers=HDR_A)
        err = self.assert_error(r, 503, "agent_unavailable")
        self.assertEqual(err["message"], "The language model is currently unavailable.")
        msgs = c.get(f"/v1/conversations/{cid}/messages", headers=HDR_A).json()
        self.assertEqual(err["details"], {"reason": "model_timeout",
                                          "user_message_id": msgs[0]["id"],
                                          "assistant_message_id": msgs[1]["id"]})
        for leaked in ("11434", "Ollama", "600.0"):
            self.assertNotIn(leaked, r.text)

    def test_500_is_generic_and_its_id_matches_the_log(self):
        c = self.client(runtime_factory=scripted(RuntimeError("password=hunter2")))
        cid = self.new_conv(c)
        with self.assertLogs("app.main", level="ERROR") as logs:
            r = c.post(f"/v1/conversations/{cid}/messages", json={"content": "hi"},
                       headers=HDR_A)
        err = self.assert_error(r, 500, "internal_error", details=None)
        self.assertEqual(err["message"], "Internal server error.")
        self.assertIn(f"request_id={err['request_id']}", "".join(logs.output))
        self.assertNotIn("hunter2", r.text)

    def test_every_response_carries_a_distinct_request_id(self):
        c = self.client()
        ids = {c.get("/health").headers["x-request-id"],
               c.get("/v1/conversations", headers=HDR_A).headers["x-request-id"]}
        self.assertEqual(len(ids), 2)

    def test_trusted_host_400_is_the_documented_plain_text_exception(self):
        r = self.client(base_url="http://evil.example.com").get("/health")
        self.assertEqual(r.status_code, 400)
        self.assertTrue(r.headers["content-type"].startswith("text/plain"))
        self.assertRegex(r.headers["x-request-id"], r"^[0-9a-f]{12}$")


class FailedTurnApiTests(ApiCase):
    def test_failed_turn_is_listed_and_the_next_message_works(self):
        c = self.client(runtime_factory=scripted(
            result("first"), ProviderConnectionError("http://localhost:11434 down"),
            result("third")))
        cid = self.new_conv(c)
        url = f"/v1/conversations/{cid}/messages"
        self.assertEqual(c.post(url, json={"content": "one"}, headers=HDR_A).status_code, 200)
        self.assertEqual(c.post(url, json={"content": "two"}, headers=HDR_A).status_code, 503)
        r = c.post(url, json={"content": "two again"}, headers=HDR_A)
        self.assertEqual(r.status_code, 200)
        self.assertEqual((r.json()["assistant_message"]["status"],
                          r.json()["assistant_message"]["error_code"]), ("completed", None))

        msgs = c.get(url, headers=HDR_A).json()
        self.assertEqual([(m["role"], m["status"], m["error_code"]) for m in msgs], [
            ("user", "completed", None), ("assistant", "completed", None),
            ("user", "completed", None), ("assistant", "failed", "model_unreachable"),
            ("user", "completed", None), ("assistant", "completed", None)])
        self.assertEqual(msgs[3]["content"], "The language model could not be reached.")
        self.assertNotIn("11434", c.get(url, headers=HDR_A).text)
        # The model's history for "two again" omits the failed pair.
        history = FakeRuntime.instances[0].calls[2][1]
        self.assertEqual([h["content"] for h in history], ["one", "first"])

    def test_unexpected_failure_is_recorded_as_internal_error(self):
        c = self.client(runtime_factory=scripted(RuntimeError("boom")))
        cid = self.new_conv(c)
        with self.assertLogs("app.main", level="ERROR"):
            r = c.post(f"/v1/conversations/{cid}/messages", json={"content": "hi"},
                       headers=HDR_A)
        self.assertEqual(r.status_code, 500)
        last = c.get(f"/v1/conversations/{cid}/messages", headers=HDR_A).json()[-1]
        self.assertEqual((last["role"], last["status"], last["error_code"]),
                         ("assistant", "failed", "internal_error"))


class ToolHistoryApiTests(ApiCase):
    def messages(self, c, cid, query=""):
        r = c.get(f"/v1/conversations/{cid}/messages{query}", headers=HDR_A)
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    def test_messages_without_tools_have_an_empty_list(self):
        c = self.client(runtime_factory=scripted(result("plain")))
        cid = self.new_conv(c)
        c.post(f"/v1/conversations/{cid}/messages", json={"content": "hi"}, headers=HDR_A)
        msgs = self.messages(c, cid)
        self.assertEqual([m["tool_executions"] for m in msgs], [[], []])

    def test_succeeded_failed_and_rejected_tools_are_attached_in_order(self):
        c = self.client(runtime_factory=scripted(result(
            "done",
            tool("github.get_file_contents", "succeeded",
                 {"path": "SECRET_ARGUMENT_VALUE.md"}),
            tool("research.search_openalex_works", "failed"),
            tool("?hallucinated", "rejected"))))
        cid = self.new_conv(c)
        sent = c.post(f"/v1/conversations/{cid}/messages", json={"content": "go"},
                      headers=HDR_A).json()
        user, assistant = self.messages(c, cid)
        self.assertEqual(user["tool_executions"], [])
        got = assistant["tool_executions"]
        self.assertEqual([(t["tool_name"], t["status"]) for t in got[:2]],
                         [("github.get_file_contents", "succeeded"),
                          ("research.search_openalex_works", "failed")])
        self.assertEqual(got[2]["status"], "rejected")
        self.assertTrue(got[2]["tool_name"].startswith("unknown:"))
        for t in got:
            self.assertEqual(set(t), {"tool_name", "status", "created_at"})
        # Arguments are never stored, so they can never be returned.
        self.assertNotIn("SECRET_ARGUMENT_VALUE",
                         c.get(f"/v1/conversations/{cid}/messages", headers=HDR_A).text)
        # Send response: same executions, top-level list kept for compatibility.
        # (Compared by name/status: SQLite reads timestamps back without a
        # zone, so created_at differs in format between the two on SQLite.)
        self.assertEqual(sent["tool_executions"], sent["assistant_message"]["tool_executions"])
        self.assertEqual([(t["tool_name"], t["status"])
                          for t in sent["assistant_message"]["tool_executions"]],
                         [(t["tool_name"], t["status"]) for t in got])

    def test_each_assistant_message_has_only_its_own_tools_and_pages_work(self):
        c = self.client(runtime_factory=scripted(
            result("a", tool("research.one", "succeeded")),
            result("b", tool("research.two", "succeeded"), tool("research.three", "failed"))))
        cid = self.new_conv(c)
        for text in ("q1", "q2"):
            c.post(f"/v1/conversations/{cid}/messages", json={"content": text}, headers=HDR_A)
        msgs = self.messages(c, cid)
        self.assertEqual([[t["tool_name"] for t in m["tool_executions"]] for m in msgs],
                         [[], ["research.one"], [], ["research.two", "research.three"]])
        page = self.messages(c, cid, "?limit=1&offset=3")
        self.assertEqual([t["tool_name"] for t in page[0]["tool_executions"]],
                         ["research.two", "research.three"])

    def test_completed_messages_report_status(self):
        c = self.client(runtime_factory=scripted(result()))
        cid = self.new_conv(c)
        c.post(f"/v1/conversations/{cid}/messages", json={"content": "hi"}, headers=HDR_A)
        for m in self.messages(c, cid):
            self.assertEqual((m["status"], m["error_code"]), ("completed", None))


class LayeringTests(unittest.TestCase):
    def test_api_layer_never_imports_mcp_or_the_agent_stack(self):
        forbidden = ("app.mcp", "mcp", "app.agent", "custom_mcp_servers")
        api_dir = Path(__file__).parent
        for path in api_dir.glob("*.py"):
            if path.name.startswith("test_"):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                for name in names:
                    with self.subTest(file=path.name, name=name):
                        self.assertFalse(
                            any(name == f or name.startswith(f + ".") for f in forbidden), name
                        )

    def test_services_and_database_do_not_import_mcp(self):
        root = Path(__file__).resolve().parents[1]
        for pkg in ("services", "database", "schemas"):
            for path in (root / pkg).glob("*.py"):
                if path.name.startswith("test_"):
                    continue
                src = path.read_text(encoding="utf-8")
                with self.subTest(file=f"{pkg}/{path.name}"):
                    self.assertNotIn("app.mcp", src)
                    self.assertNotIn("MCPConnectionManager", src)


if __name__ == "__main__":
    unittest.main()
