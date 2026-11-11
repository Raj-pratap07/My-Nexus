"""message_turn_status

Adds explicit failed-turn tracking to ``messages``:
  * status      VARCHAR(16) NOT NULL DEFAULT 'completed'  (completed | failed)
  * error_code  VARCHAR(32) NULL                          (set iff failed)
plus three CHECK constraints (see app/database/models.py).

Existing rows: every message already stored is a completed turn, so the
server default backfills status='completed' and error_code stays NULL.

PostgreSQL: plain ALTER TABLE statements on ``messages`` only. ADD COLUMN with
a constant DEFAULT backfills existing rows; no table is recreated, no foreign
key is touched, and no new table is created, so the RLS / privilege DDL of
0001 is unaffected. SQLite: batch mode recreates ``messages`` (SQLite cannot
add constraints with ALTER). Alembic's migration engine does not enable
PRAGMA foreign_keys, so tool_executions.message_id (ON DELETE SET NULL) is not
cleared by the copy; app/database/test_database.py verifies that.

Downgrade first deletes failed-turn markers (fixed, safe text; the user
message of each failed turn is kept), restoring 0001's invariant that every
assistant message is a model answer, then drops the constraints and columns.

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-11
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = '0002'
down_revision = '0001'
branch_labels = None
depends_on = None

CONSTRAINTS = (
    ("ck_messages_status_allowed", "status IN ('completed', 'failed')"),
    ("ck_messages_failed_only_assistant", "status = 'completed' OR role = 'assistant'"),
    ("ck_messages_error_code_iff_failed",
     "(status = 'failed' AND error_code IS NOT NULL)"
     " OR (status = 'completed' AND error_code IS NULL)"),
)


def upgrade() -> None:
    with op.batch_alter_table("messages") as batch:
        batch.add_column(sa.Column("status", sa.String(length=16), nullable=False,
                                   server_default="completed"))
        batch.add_column(sa.Column("error_code", sa.String(length=32), nullable=True))
        for name, condition in CONSTRAINTS:
            batch.create_check_constraint(op.f(name), condition)


def downgrade() -> None:
    op.execute("DELETE FROM messages WHERE status = 'failed'")
    with op.batch_alter_table("messages") as batch:
        for name, _ in reversed(CONSTRAINTS):
            batch.drop_constraint(op.f(name), type_="check")
        batch.drop_column("error_code")
        batch.drop_column("status")
