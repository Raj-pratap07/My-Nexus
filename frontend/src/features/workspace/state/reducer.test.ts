import { describe, expect, it } from "vitest";
import { messageFixture } from "../../../test/fixtures";
import { analyzeTurn, initialState, maxPosition, NEW, reducer, unansweredTail } from "./reducer";

const u = (position: number) => messageFixture({ id: `u${position}`, role: "user", position });
const a = (position: number, status = "completed") =>
  messageFixture({ id: `a${position}`, role: "assistant", position, status, content: "x" });
const conv = (id: string, title = id) => ({
  id,
  title,
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
});

describe("analyzeTurn", () => {
  const before = [u(0), a(1)];
  it("not-stored when nothing new exists after the baseline", () => {
    expect(analyzeTurn(before, maxPosition(before))).toEqual({ kind: "not-stored" });
  });
  it("awaiting when only the new user message is stored", () => {
    expect(analyzeTurn([...before, u(2)], 1)).toMatchObject({ kind: "awaiting", user: { id: "u2" } });
  });
  it("replied when an assistant message (completed or failed) follows it", () => {
    expect(analyzeTurn([...before, u(2), a(3, "failed")], 1)).toMatchObject({ kind: "replied", assistant: { id: "a3" } });
  });
  it("without a baseline, judges by the trailing message", () => {
    expect(analyzeTurn([u(0)], null).kind).toBe("awaiting");
    expect(analyzeTurn(before, null).kind).toBe("replied");
    expect(analyzeTurn([], null).kind).toBe("not-stored");
  });
  it("unansweredTail finds only a trailing user message", () => {
    expect(unansweredTail([u(0), a(1), u(2)])?.id).toBe("u2");
    expect(unansweredTail(before)).toBeNull();
  });
});

describe("reducer", () => {
  it("moves the draft and turn from the new-conversation key to the created id", () => {
    let s = reducer(initialState, { type: "draft/set", key: NEW, text: "hello" });
    s = reducer(s, { type: "turn/start", key: NEW, content: "hello", startedAt: 1 });
    expect(s.drafts[NEW]).toBe("");
    s = reducer(s, { type: "turn/conversation", id: "c1" });
    expect(s.turn).toMatchObject({ key: "c1", conversationId: "c1", content: "hello" });
    expect(NEW in s.drafts).toBe(false);
  });

  it("appends by id and keeps messages ordered by position", () => {
    let s = reducer(initialState, { type: "thread/loaded", id: "c", messages: [u(0), a(1)], truncated: false });
    s = reducer(s, { type: "thread/append", id: "c", messages: [a(3), u(2), a(1)] });
    expect(s.threads.c!.messages.map((m) => m.id)).toEqual(["u0", "a1", "u2", "a3"]);
  });

  it("removing a conversation clears its thread, draft, notice and awaiting state", () => {
    let s = reducer(initialState, { type: "conversation/upsert", conversation: conv("c"), toTop: true });
    s = reducer(s, { type: "draft/set", key: "c", text: "d" });
    s = reducer(s, {
      type: "awaiting/set",
      id: "c",
      awaiting: { baseline: null, userStored: true, content: "q", since: 0, checkingSince: 0, deadline: 1, phase: "polling" },
    });
    s = reducer(s, { type: "conversation/remove", id: "c" });
    expect(s.conversations.items).toEqual([]);
    expect(s.drafts.c).toBeUndefined();
    expect(s.awaiting.c).toBeUndefined();
  });

  it("keeps an existing conversation's place unless moved to the top; appends a new one", () => {
    let s = reducer(initialState, { type: "list/loaded", items: [conv("a"), conv("b")], append: false, hasMore: false });
    s = reducer(s, { type: "conversation/upsert", conversation: conv("b", "B2"), toTop: false });
    expect(s.conversations.items.map((c) => c.title)).toEqual(["a", "B2"]);
    s = reducer(s, { type: "conversation/upsert", conversation: conv("z"), toTop: false });
    expect(s.conversations.items.map((c) => c.id)).toEqual(["a", "b", "z"]);
    s = reducer(s, { type: "conversation/upsert", conversation: conv("b"), toTop: true });
    expect(s.conversations.items.map((c) => c.id)).toEqual(["b", "a", "z"]);
  });
});
