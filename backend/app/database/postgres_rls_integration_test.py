"""
PostgreSQL RLS integration test for the initial migration.

Runs ONLY against a disposable, empty, LOCAL PostgreSQL database -- never
Supabase or any other remote server. NEXUS_TEST_PG_URL is read from the
process environment only; the project .env (which holds the real
DATABASE_URL) is never loaded.

    $env:NEXUS_TEST_PG_URL = "postgresql://postgres@localhost:5432/nexus_rls_test"
    .venv/Scripts/python.exe -m app.database.postgres_rls_integration_test

What it checks (upgrade -> checks -> downgrade):
  * RLS enabled, not forced, on the five application tables and alembic_version;
  * no RLS policies exist;
  * anon/authenticated hold no privileges and are denied on all six tables;
  * with a privilege re-granted, RLS alone still yields 0 rows and blocks writes;
  * a NON-superuser table owner (like Supabase's postgres role) bypasses RLS;
  * ON DELETE CASCADE still works for the owner;
  * downgrade drops all five tables; alembic_version stays, still secured.

Refusals -- every one happens before anything is changed.
  Before connecting (nothing has been sent to any server):
  * PGHOSTADDR, PGSERVICE and PGSERVICEFILE must be unset, and the URL may
    carry no query parameters except sslmode and connect_timeout (others,
    such as host, hostaddr or service, can redirect libpq to another server);
  * the URL host must be loopback;
  * the database name must start with "test_" or end with "_test".
  After connecting, before any change:
  * the connection's server address (inet_server_addr()) must be loopback,
    or NULL (Unix-domain socket);
  * the connecting role must be a superuser (needed to create roles and SET
    ROLE to them on PostgreSQL 16+; RLS itself is tested through SET ROLE);
  * the public schema must have no tables;
  * the connecting role must have no default privileges for tables in public
    (the test adds some and restores them to exactly "none");
  * the helper owner role must not already exist.

Cleanup: downgrade if application tables remain; drop alembic_version (it
did not exist before) ONLY if no application tables remain, so a failed
downgrade keeps its revision record; remove the default privileges the test
added; drop ONLY the roles this run created (each is printed when created).
Pre-existing roles are never dropped. The owner-path and privilege
experiments run inside a transaction that is always rolled back.

Manual cleanup after a killed run (a rerun refuses until this is done). As a
superuser on the disposable server, connected to a different database:
    DROP DATABASE nexus_rls_test;    -- removes its tables and default privileges
    CREATE DATABASE nexus_rls_test;  -- if you want to reuse it
    DROP ROLE IF EXISTS nexus_rls_test_owner;
    DROP ROLE anon;                  -- ONLY if the run printed that it created it
    DROP ROLE authenticated;         -- likewise

Exit codes: 0 = checks passed and cleanup complete (or skipped: variable unset),
            1 = refused, or checks failed (cleanup complete or not needed),
            2 = cleanup incomplete (manual cleanup needed; details printed).
"""
from __future__ import annotations

import ipaddress
import os
import sys
import uuid
from collections.abc import Mapping

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection, Engine, make_url
from sqlalchemy.exc import DBAPIError
from sqlalchemy.pool import NullPool

from app.core.config import is_loopback_host, sync_database_url
from app.database.models import Base
from app.database.session import BACKEND_DIR

API_ROLES = ("anon", "authenticated")
OWNER_ROLE = "nexus_rls_test_owner"  # non-superuser stand-in for Supabase's postgres
APP_TABLES = sorted(Base.metadata.tables)
VERSION_TABLE = "alembic_version"
SECURED_TABLES = sorted([*APP_TABLES, VERSION_TABLE])
TABLE_PRIVILEGES = "SELECT,INSERT,UPDATE,DELETE,TRUNCATE,REFERENCES,TRIGGER"
INSUFFICIENT_PRIVILEGE = "42501"  # permission denied / RLS violation

# URL query parameters that cannot redirect the connection elsewhere.
ALLOWED_QUERY_PARAMS = frozenset({"sslmode", "connect_timeout"})
# libpq environment variables that can redirect a connection to another server.
LIBPQ_REDIRECT_ENV = ("PGHOSTADDR", "PGSERVICE", "PGSERVICEFILE")


class Refused(Exception):
    """Preconditions for a safe, local, fully reversible run are not met."""


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def is_disposable_name(name: str) -> bool:
    name = name.lower()
    return name.startswith("test_") or name.endswith("_test")


def is_loopback_address(addr: str | None) -> bool:
    """Server address as reported by host(inet_server_addr()); NULL = socket."""
    if addr is None:
        return True
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        return False
    mapped = getattr(ip, "ipv4_mapped", None)  # ::ffff:127.0.0.1
    return (mapped or ip).is_loopback


def check_target(url: str, environ: Mapping[str, str]) -> None:
    """Pre-connection refusals. Raises Refused; reports names, never values."""
    redirect = [name for name in LIBPQ_REDIRECT_ENV if name in environ]
    if redirect:
        raise Refused(f"unset {', '.join(redirect)}: libpq could use it to reach another server")
    parsed = make_url(url)
    extra = sorted(set(parsed.query) - ALLOWED_QUERY_PARAMS)
    if extra:
        raise Refused(f"URL query parameter(s) not allowed: {', '.join(extra)} "
                      f"(allowed: {', '.join(sorted(ALLOWED_QUERY_PARAMS))})")
    host = (parsed.host or "").lower()
    if "supabase" in host or not is_loopback_host(host):
        raise Refused("host is not loopback; use a disposable LOCAL database (never Supabase)")
    if not is_disposable_name(parsed.database or ""):
        raise Refused('database name must start with "test_" or end with "_test"')


def scalar(c: Connection, sql: str, **params):
    return c.execute(text(sql), params).scalar()


def public_tables(c: Connection) -> list[str]:
    return list(c.execute(text(
        "SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY 1"
    )).scalars())


def role_exists(c: Connection, role: str) -> bool:
    return bool(scalar(c, "SELECT count(*) FROM pg_roles WHERE rolname = :r", r=role))


def default_acl_rows(c: Connection) -> int:
    """Default privileges the connecting role defines for tables in public."""
    return scalar(c,
        "SELECT count(*) FROM pg_default_acl "
        "WHERE defaclrole = (SELECT oid FROM pg_roles WHERE rolname = current_user) "
        "AND defaclnamespace = 'public'::regnamespace AND defaclobjtype = 'r'")


def is_denied(c: Connection, sql: str, **params) -> bool:
    """True if the statement fails with insufficient_privilege (in a savepoint)."""
    sp = c.begin_nested()
    try:
        c.execute(text(sql), params)
    except DBAPIError as exc:
        sp.rollback()
        return getattr(exc.orig, "sqlstate", None) == INSUFFICIENT_PRIVILEGE
    sp.rollback()
    return False


def preflight(engine: Engine) -> list[str]:
    """Read-only post-connection refusals; return the roles to create."""
    with engine.connect() as c:
        # First query of the run: prove where we actually are.
        if not is_loopback_address(scalar(c, "SELECT host(inet_server_addr())")):
            raise Refused("the connected server is not on a loopback address; nothing was changed")
        if not scalar(c, "SELECT rolsuper FROM pg_roles WHERE rolname = current_user"):
            raise Refused("connecting role must be a superuser on the disposable server")
        if public_tables(c):
            raise Refused("the public schema is not empty")
        if default_acl_rows(c):
            raise Refused("connecting role already has default privileges for tables in "
                          "public; they could not be restored exactly")
        if role_exists(c, OWNER_ROLE):
            raise Refused(f"role {OWNER_ROLE} already exists (left over from a killed run? "
                          "see the manual cleanup in this module's docstring)")
        return [r for r in API_ROLES if not role_exists(c, r)] + [OWNER_ROLE]


def run_checks(engine: Engine, cfg: Config) -> None:
    command.upgrade(cfg, "head")

    with engine.connect() as c:
        rows = c.execute(text(
            "SELECT relname, relrowsecurity, relforcerowsecurity FROM pg_class "
            "WHERE relnamespace = 'public'::regnamespace AND relkind = 'r' "
            "AND relname = ANY(:t)"), {"t": SECURED_TABLES}).all()
        check(sorted(r[0] for r in rows) == SECURED_TABLES, "not all six tables were found")
        for name, rls, forced in rows:
            check(rls, f"RLS is not enabled on {name}")
            check(not forced, f"FORCE ROW LEVEL SECURITY is set on {name}")
        check(scalar(c, "SELECT count(*) FROM pg_policies WHERE schemaname = 'public'") == 0,
              "unexpected RLS policies exist")
        for role in API_ROLES:
            for t in SECURED_TABLES:
                check(not scalar(c, "SELECT has_table_privilege(:r, :t, :p)",
                                 r=role, t=f'public."{t}"', p=TABLE_PRIVILEGES),
                      f"{role} still has privileges on {t}")

    # Behavioural checks in a transaction that is ALWAYS rolled back: the
    # ownership change, test rows and the temporary GRANT never persist.
    with engine.connect() as c:
        tx = c.begin()
        try:
            for role in API_ROLES:
                c.execute(text(f'SET LOCAL ROLE "{role}"'))
                for t in SECURED_TABLES:
                    check(is_denied(c, f'SELECT 1 FROM "{t}" LIMIT 1'),
                          f"{role} was not denied on {t}")
                c.execute(text("RESET ROLE"))

            # A non-superuser owner without BYPASSRLS, as on Supabase.
            for t in APP_TABLES:
                c.execute(text(f'ALTER TABLE "{t}" OWNER TO "{OWNER_ROLE}"'))
            check(not scalar(c, "SELECT rolbypassrls FROM pg_roles WHERE rolname = :r",
                             r=OWNER_ROLE), "owner role unexpectedly has BYPASSRLS")
            uid, cid = uuid.uuid4(), uuid.uuid4()
            c.execute(text(f'SET LOCAL ROLE "{OWNER_ROLE}"'))
            c.execute(text("INSERT INTO users (id, created_at, updated_at) "
                           "VALUES (:u, now(), now())"), {"u": uid})
            c.execute(text("INSERT INTO conversations (id, user_id, title, created_at, updated_at) "
                           "VALUES (:c, :u, 't', now(), now())"), {"c": cid, "u": uid})
            c.execute(text("INSERT INTO messages (id, conversation_id, position, role, content, created_at) "
                           "VALUES (:m, :c, 1, 'user', 'hi', now())"), {"m": uuid.uuid4(), "c": cid})
            check(scalar(c, "SELECT count(*) FROM messages") == 1,
                  "the table owner could not read its rows (RLS was not bypassed)")
            c.execute(text("RESET ROLE"))

            # Second layer on its own: re-grant a privilege; no policy => no access.
            c.execute(text(f'GRANT SELECT, INSERT ON users TO "{API_ROLES[0]}"'))
            c.execute(text(f'SET LOCAL ROLE "{API_ROLES[0]}"'))
            check(scalar(c, "SELECT count(*) FROM users") == 0, "RLS leaked rows to anon")
            check(is_denied(c, "INSERT INTO users (id, created_at, updated_at) "
                               "VALUES (:u, now(), now())", u=uuid.uuid4()),
                  "anon insert was not blocked by RLS")
            c.execute(text("RESET ROLE"))

            c.execute(text(f'SET LOCAL ROLE "{OWNER_ROLE}"'))
            c.execute(text("DELETE FROM users WHERE id = :u"), {"u": uid})
            check(scalar(c, "SELECT count(*) FROM conversations") == 0
                  and scalar(c, "SELECT count(*) FROM messages") == 0,
                  "ON DELETE CASCADE did not remove dependent rows")
        finally:
            tx.rollback()

    command.downgrade(cfg, "base")
    with engine.connect() as c:
        remaining = public_tables(c)
        check(remaining == [VERSION_TABLE], f"downgrade left unexpected tables: {remaining}")
        check(scalar(c, f'SELECT count(*) FROM "{VERSION_TABLE}"') == 0,
              "alembic_version still records a revision after downgrade")
        check(scalar(c, "SELECT relrowsecurity FROM pg_class WHERE relname = :t "
                        "AND relnamespace = 'public'::regnamespace", t=VERSION_TABLE),
              "alembic_version lost RLS after downgrade")


def cleanup(engine: Engine, cfg: Config, created_roles: list[str], acl_changed: bool) -> list[str]:
    """Restore the pre-run state. Returns problems (empty list = clean)."""
    problems: list[str] = []

    def step(desc: str, fn) -> None:
        try:
            fn()
        except Exception as exc:  # report and keep cleaning up
            problems.append(f"{desc} failed ({type(exc).__name__})")

    def leftover_app_tables() -> list[str]:
        with engine.connect() as c:
            return sorted(set(public_tables(c)) & set(APP_TABLES))

    def downgrade_if_needed() -> None:  # e.g. a check failed before the downgrade
        if leftover_app_tables():
            command.downgrade(cfg, "base")

    def drop_version_table() -> None:  # preflight proved it did not exist before
        with engine.begin() as c:
            c.execute(text(f'DROP TABLE IF EXISTS "{VERSION_TABLE}"'))

    def revoke_defaults() -> None:
        with engine.begin() as c:
            roles = ", ".join(f'"{r}"' for r in API_ROLES)
            c.execute(text(f"ALTER DEFAULT PRIVILEGES IN SCHEMA public REVOKE ALL ON TABLES FROM {roles}"))

    def drop_role(role: str) -> None:
        with engine.begin() as c:
            c.execute(text(f'DROP ROLE "{role}"'))

    step("downgrading leftover application tables", downgrade_if_needed)

    # Keep alembic_version unless the application tables are known to be gone:
    # with tables left behind, its revision record is what allows a downgrade.
    try:
        leftover: list[str] | None = leftover_app_tables()
    except Exception as exc:
        leftover = None
        problems.append(f"checking for application tables failed ({type(exc).__name__})")
    if leftover == []:
        step("dropping alembic_version", drop_version_table)
    else:
        problems.append("application tables remain; alembic_version was kept so its "
                        "revision record is preserved for a later downgrade")

    if acl_changed:
        step("restoring default privileges", revoke_defaults)
    for role in reversed(created_roles):  # only roles this run created
        step(f"dropping role {role}", lambda r=role: drop_role(r))

    try:
        with engine.connect() as c:
            if leftover == [] and public_tables(c):
                problems.append("tables remain in the public schema")
            if default_acl_rows(c):
                problems.append("default privileges were not restored to none")
            for role in created_roles:
                if role_exists(c, role):
                    problems.append(f"role {role} still exists")
    except Exception as exc:
        problems.append(f"verifying cleanup failed ({type(exc).__name__})")
    return problems


def report(checks_passed: bool, failure: str, cleanup_needed: bool, problems: list[str]) -> int:
    """Print checks and cleanup separately; PASS only when both succeeded."""
    if checks_passed:
        print("Checks:  PASSED -- RLS on all six tables, no policies, API roles revoked, "
              "non-superuser owner unaffected, downgrade clean.")
    else:
        print(f"Checks:  FAILED -- {failure or 'unknown error'}")
    if problems:
        print("Cleanup: INCOMPLETE -- fix manually on the disposable server "
              "(see this module's docstring):")
        for p in problems:
            print(f"  - {p}")
        print("Result:  CLEANUP INCOMPLETE" + (" (checks passed)" if checks_passed else " (checks failed)"))
        return 2
    print("Cleanup: complete -- server restored to its pre-run state."
          if cleanup_needed else "Cleanup: nothing to restore.")
    print("Result:  PASS" if checks_passed else "Result:  FAIL")
    return 0 if checks_passed else 1


def main() -> int:
    raw = os.environ.get("NEXUS_TEST_PG_URL", "")
    if not raw:
        print("SKIP: NEXUS_TEST_PG_URL is not set.")
        return 0
    url = sync_database_url(raw)
    try:
        check_target(url, os.environ)  # before an engine exists or anything connects
    except Refused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1

    engine = create_engine(url, poolclass=NullPool)
    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))  # ConfigParser escaping

    created_roles: list[str] = []
    acl_changed = False
    checks_passed, failure = False, ""
    problems: list[str] = []
    try:
        try:
            to_create = preflight(engine)
        except Refused as exc:
            print(f"REFUSED: {exc}", file=sys.stderr)
            return 1

        for role in to_create:  # one transaction each: record only what was created
            with engine.begin() as c:
                c.execute(text(f'CREATE ROLE "{role}" NOLOGIN'))
            created_roles.append(role)
            print(f"Created role {role} (will be dropped at cleanup).")
        with engine.begin() as c:
            # Imitate Supabase's expected default grants so the REVOKE is exercised.
            roles = ", ".join(f'"{r}"' for r in API_ROLES)
            c.execute(text(f"ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO {roles}"))
        acl_changed = True

        run_checks(engine, cfg)
        checks_passed = True
    except AssertionError as exc:
        failure = str(exc)
    except DBAPIError as exc:  # type name only: driver messages can carry connection details
        failure = f"database error ({type(exc.orig).__name__})"
    except Exception as exc:
        failure = type(exc).__name__
    except KeyboardInterrupt:  # still clean up and report
        failure = "interrupted"
    finally:
        cleanup_needed = bool(created_roles or acl_changed)
        if cleanup_needed:
            problems = cleanup(engine, cfg, created_roles, acl_changed)
        engine.dispose()

    return report(checks_passed, failure, cleanup_needed, problems)


if __name__ == "__main__":
    sys.exit(main())
