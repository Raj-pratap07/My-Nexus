"""
MANUAL live end-to-end test: README summary, then a Known-limitations
follow-up, through the running FastAPI app. Two cases, each in its OWN fresh
conversation (so one case's answer can never become the other's history):

  implicit  turn 2 is the exact question of the originally failed
            conversation, verbatim (app/agent/eval_fixtures/readme_followup.json);
  explicit  turn 2 is EXPLICIT_FOLLOWUP_QUESTION (app/agent/github_prompt_eval.py),
            which tells the model to fetch README.md again.
Turn 1 is the fixture's exact turn-1 request in both cases.

Prerequisites (this script starts nothing):
  * Ollama running locally with the configured model (default qwen3:4b-instruct);
  * the API running:  cd backend; .\\.venv\\Scripts\\python.exe -m app.serve
  * NEXUS_DEV_TOKEN (and the GitHub MCP settings the server uses) in the
    project .env. The token is read here and never printed.

Network: only http://<NEXUS_HOST>:<NEXUS_PORT> (loopback). The SERVER then
calls local Ollama and the GitHub MCP endpoint. No paid LLM API is used.
Research tools cannot be disabled through the API; this script does not ask
for them and FAILS the run if the model calls one anyway.

Three groups of checks per case, reported independently:
  RETRIEVAL (gated)  turn 1 and turn 2 each ran github.get_file_contents
                     successfully. The API exposes tool NAME and STATUS only,
                     not the path, so this proves A file was read, not WHICH.
  INTEGRITY (gated)  stop reasons, no research tools, no rejected/failed
                     calls, exactly 4 stored messages, stored answer matches.
  ACCURACY (ADVISORY, never gated)  app.agent.grounding_judge compares the
                     turn-2 answer with the LOCAL checkout's README.md, which
                     may differ from GitHub's default branch. Lexical proxy,
                     not semantic proof. A successful GitHub call does not
                     prove the answer is accurate.

Side effect: creates one conversation per case (kept, for inspection) with
four messages each.

Run from backend/:
    .\\.venv\\Scripts\\python.exe readme_followup_e2e.py [--case implicit|explicit|both]

Exit codes: 0 = every RETRIEVAL and INTEGRITY check passed in every case run,
            1 = one of them failed (accuracy never changes the exit code),
            2 = setup/HTTP error.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv

from app.agent.github_prompt_eval import EXPLICIT_FOLLOWUP_QUESTION
from app.agent.grounding_judge import grade_limitations

BACKEND = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND.parent
FIXTURE = BACKEND / "app" / "agent" / "eval_fixtures" / "readme_followup.json"
LOCAL_README = PROJECT_ROOT / "README.md"

GITHUB_FILE_TOOL = "github.get_file_contents"
RESEARCH_PREFIX = "research."
# Per HTTP request. The server makes several Ollama calls per turn (each with
# its own server-side timeout, 180 s by default), so the client must wait
# longer than one of them. The previous failure was a 120 s client timeout.
TURN_TIMEOUT_S = 900
SHORT_TIMEOUT_S = 15

CASES = ("implicit", "explicit")
ACCURACY_WARNING = (
    "ACCURACY is ADVISORY: a lexical comparison with the LOCAL README.md, which may "
    "differ from GitHub's default branch. It is not semantic proof, and a successful "
    "GitHub call does not prove the answer is accurate.")


class SetupFailure(Exception):
    pass


@dataclass
class CaseReport:
    case: str
    conversation_id: str
    elapsed: tuple[float, float]
    retrieval: list[tuple[str, bool]]
    integrity: list[tuple[str, bool]]
    accuracy: str                          # pass | fail | ungradeable
    accuracy_detail: str
    accuracy_report: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def gated_ok(self) -> bool:
        return all(ok for _, ok in self.retrieval + self.integrity)


def load_questions(fixture: dict) -> dict[str, tuple[str, str]]:
    """case -> (turn-1 request, turn-2 question). Implicit is verbatim."""
    turn1 = fixture["history"][0]["content"]
    return {"implicit": (turn1, fixture["question"]),
            "explicit": (turn1, EXPLICIT_FOLLOWUP_QUESTION)}


def github_read_succeeded(turn: dict) -> bool:
    return any(t["tool_name"] == GITHUB_FILE_TOOL and t["status"] == "succeeded"
               for t in turn["tool_executions"])


def retrieval_checks(t1: dict, t2: dict) -> list[tuple[str, bool]]:
    return [
        ("turn 1: github.get_file_contents succeeded", github_read_succeeded(t1)),
        ("turn 2: FRESH github.get_file_contents succeeded (this turn)",
         github_read_succeeded(t2)),
    ]


def integrity_checks(t1: dict, t2: dict, stored: list[dict]) -> list[tuple[str, bool]]:
    executions = t1["tool_executions"] + t2["tool_executions"]
    return [
        ("turn 1: stop_reason == complete", t1["stop_reason"] == "complete"),
        ("turn 2: stop_reason == complete", t2["stop_reason"] == "complete"),
        ("no research tools called",
         not any(t["tool_name"].startswith(RESEARCH_PREFIX) for t in executions)),
        ("no rejected/failed tool calls", all(t["status"] == "succeeded" for t in executions)),
        ("conversation holds exactly 4 messages: user, assistant, user, assistant",
         [m["role"] for m in stored] == ["user", "assistant", "user", "assistant"]),
        ("stored turn-2 answer matches the response",
         len(stored) == 4 and stored[3]["content"] == t2["assistant_message"]["content"]),
    ]


def accuracy_diagnostic(answer: str, question: str, readme: str | None,
                        retrieved_this_turn: bool) -> tuple[str, str, str]:
    """(status, detail, full report). Advisory only; never gated."""
    if readme is None:
        return "ungradeable", "local README.md could not be read", ""
    report = grade_limitations(answer, readme, extra_quote_sources=[question])
    if not report.section_found:
        return ("ungradeable", "local README.md has no '## Known limitations' section",
                report.explain())
    failed = report.failed_checks
    detail = "lexical checks passed" if not failed else f"lexical checks failed: {failed}"
    if not retrieved_this_turn:
        detail += ("; NOT GROUNDED: no file was retrieved in turn 2, so any match comes "
                   "from earlier context, not from a source read in this request")
    return ("pass" if not failed else "fail"), detail, report.explain()


def build_report(case: str, conv_id: str, t1: dict, t2: dict, stored: list[dict],
                 question: str, readme: str | None) -> CaseReport:
    retrieval = retrieval_checks(t1, t2)
    status, detail, explanation = accuracy_diagnostic(
        t2["assistant_message"]["content"], question, readme, github_read_succeeded(t2))
    return CaseReport(case, conv_id, (t1["_elapsed_s"], t2["_elapsed_s"]), retrieval,
                      integrity_checks(t1, t2, stored), status, detail, explanation)


def print_report(r: CaseReport) -> None:
    print(f"\n=== SUMMARY: {r.case} case ===")
    print(f"conversation: {r.conversation_id}")
    print(f"elapsed: turn 1 {r.elapsed[0]:.1f} s, turn 2 {r.elapsed[1]:.1f} s")
    for title, checks in (("RETRIEVAL (gated)", r.retrieval),
                          ("INTEGRITY (gated)", r.integrity)):
        print(f"{title}:")
        for name, ok in checks:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("  note: the API returns tool name and status, not the path; a succeeded "
          "GitHub read proves a file was read, not which one.")
    print("ACCURACY (ADVISORY, not gated):")
    print(f"  {r.accuracy}: {r.accuracy_detail}")
    print(f"  {ACCURACY_WARNING}")
    if r.accuracy_report:
        print("  per-claim report (for human review):")
        for line in r.accuracy_report.splitlines():
            print(f"    {line}")


# ----- HTTP -------------------------------------------------------------------
def api(method: str, url: str, token: str, body: dict | None = None,
        timeout: float = SHORT_TIMEOUT_S):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except HTTPError as exc:
        # The API's own error body (no headers, so no token) is safe to show.
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise SetupFailure(f"{method} {url} -> HTTP {exc.code}: {detail}") from None
    except (URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        raise SetupFailure(f"{method} {url} failed: {type(exc).__name__}: {reason}") from None


def send(base: str, token: str, conv_id: str, content: str, label: str) -> dict:
    print(f"\n=== {label} ===")
    print(f"USER: {content}")
    print(f"(waiting up to {TURN_TIMEOUT_S} s ...)", flush=True)
    start = time.perf_counter()
    status, data = api("POST", f"{base}/v1/conversations/{conv_id}/messages", token,
                       {"content": content}, timeout=TURN_TIMEOUT_S)
    elapsed = time.perf_counter() - start
    data["_elapsed_s"] = elapsed
    print(f"HTTP {status} | elapsed {elapsed:.1f} s | stop_reason: {data['stop_reason']}")
    execs = data["tool_executions"]
    if execs:
        for i, t in enumerate(execs, 1):
            print(f"  tool {i}: {t['tool_name']:40} status={t['status']}")
    else:
        print("  tool executions: none")
    print(f"\nASSISTANT:\n{data['assistant_message']['content']}")
    return data


def run_case(base: str, token: str, case: str, turn1: str, turn2: str,
             readme: str | None) -> CaseReport:
    title = f"README follow-up E2E ({case}) {datetime.now():%Y-%m-%d %H:%M:%S}"
    _, conv = api("POST", f"{base}/v1/conversations", token, {"title": title})
    conv_id = conv["id"]
    print(f"\nCreated conversation {conv_id} ({title!r})")
    t1 = send(base, token, conv_id, turn1, f"[{case}] TURN 1: README summary")
    t2 = send(base, token, conv_id, turn2, f"[{case}] TURN 2: Known limitations follow-up")
    _, stored = api("GET", f"{base}/v1/conversations/{conv_id}/messages", token)
    return build_report(case, conv_id, t1, t2, stored, turn2, readme)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual live README follow-up E2E "
                                                 "(see module docstring).")
    parser.add_argument("--case", choices=[*CASES, "both"], default="both")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cases = list(CASES) if args.case == "both" else [args.case]
    load_dotenv(PROJECT_ROOT / ".env")
    token = os.getenv("NEXUS_DEV_TOKEN", "")
    host = os.getenv("NEXUS_HOST", "127.0.0.1")
    port = os.getenv("NEXUS_PORT", "8000")
    base = f"http://{host}:{port}"
    try:
        readme = LOCAL_README.read_text(encoding="utf-8")
    except OSError:
        readme = None
    reports: list[CaseReport] = []
    try:
        if not token:
            raise SetupFailure("NEXUS_DEV_TOKEN is not set in the project .env.")
        questions = load_questions(json.loads(FIXTURE.read_text(encoding="utf-8")))
        status, _ = api("GET", f"{base}/health", token)
        print(f"API reachable at {base} (health HTTP {status}). Token loaded (not shown).")
        print(f"Cases: {', '.join(cases)} (one fresh conversation each).")
        print(ACCURACY_WARNING)
        for case in cases:
            turn1, turn2 = questions[case]
            reports.append(run_case(base, token, case, turn1, turn2, readme))
    except SetupFailure as exc:
        for r in reports:
            print_report(r)
        print(f"\nSETUP/HTTP ERROR: {exc}")
        return 2

    for r in reports:
        print_report(r)
    print("\n=== OVERALL ===")
    for r in reports:
        print(f"  {r.case:9} retrieval {'PASS' if all(ok for _, ok in r.retrieval) else 'FAIL'}"
              f" | integrity {'PASS' if all(ok for _, ok in r.integrity) else 'FAIL'}"
              f" | accuracy (advisory) {r.accuracy}")
    print("\nNOT checked automatically (verify by hand):")
    print("  * WHICH file turn 2 read (the API does not expose the path).")
    print("  * Whether turn 2 lists exactly the README's '## Known limitations' items and")
    print("    nothing invented: read the per-claim report above against README.md on")
    print("    GitHub's default branch.")
    passed = all(r.gated_ok for r in reports)
    print(f"\nRESULT: {'PASS' if passed else 'FAIL'} (retrieval and integrity checks only)")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
