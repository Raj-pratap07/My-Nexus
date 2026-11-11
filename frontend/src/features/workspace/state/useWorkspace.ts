/**
 * Workspace controller: all backend calls and the turn lifecycle.
 *
 * Turn rules (Milestone 2B amendment 1):
 *  - A message is sent only by an explicit user action. Nothing is retried
 *    automatically, and no new user message is ever created because an
 *    earlier request timed out or lost its connection.
 *  - A network failure, proxy timeout or browser disconnect does not prove
 *    the backend stopped. The conversation enters reconciliation: stored
 *    messages are re-read on a timer until a reply appears or the window
 *    expires, after which the outcome is reported as unknown.
 *  - "Send again" exists only for turns the backend recorded as failed.
 */
import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from "react";
import { ApiError } from "../../../api/client";
import { api, listAllMessages, validateMessage } from "../../../api/conversations";
import { parseTimestamp } from "../../../api/decode";
import { LIMITS } from "../../../api/types";
import { navigate, paths } from "../../../app/router";
import { isSetupError } from "../errors";
import {
  analyzeTurn,
  initialState,
  maxPosition,
  NEW,
  reducer,
  unansweredTail,
  type Awaiting,
  type Key,
  type WorkspaceState,
} from "./reducer";

/** Above the worst-case backend turn (5 model calls x 180 s) plus queueing. */
export const TURN_WINDOW_MS = 20 * 60 * 1000;
/** A stored user message appears within milliseconds of the request arriving. */
export const NOT_STORED_WINDOW_MS = 30 * 1000;
/** The first check comes quickly: a stored user message appears almost at once. */
export const FIRST_CHECK_MS = 1_000;
export const POLL_FAST_MS = 5_000;
export const POLL_SLOW_MS = 15_000;
const FAST_PHASE_MS = 60_000;

const toApiError = (e: unknown): ApiError =>
  e instanceof ApiError
    ? e
    : new ApiError({ kind: "network", code: "network_error", message: "The request did not complete.", source: "client" });

export interface WorkspaceApi {
  state: WorkspaceState;
  selectedKey: Key;
  announcement: string;
  /** Why sending is blocked for the selected conversation, or null. */
  sendBlockedReason: string | null;
  loadMore(): void;
  reloadList(): void;
  refreshThread(id: string): void;
  send(text: string): Promise<void>;
  sendAgain(failedMessageId: string): Promise<void>;
  deleteConversation(id: string): Promise<ApiError | null>;
  setDraft(key: Key, text: string): void;
  dismissNotice(key: Key): void;
}

export function useWorkspace(conversationId: string | null): WorkspaceApi {
  const [state, dispatch] = useReducer(reducer, initialState);
  const [announcement, setAnnouncement] = useState("");
  const stateRef = useRef(state);
  stateRef.current = state;
  const selectedKey: Key = conversationId ?? NEW;
  const selectedRef = useRef(selectedKey);
  selectedRef.current = selectedKey;
  const timers = useRef(new Map<string, number>());
  const removed = useRef(new Set<string>());
  const mounted = useRef(true);

  useEffect(() => {
    mounted.current = true;
    const pending = timers.current;
    return () => {
      mounted.current = false;
      for (const t of pending.values()) window.clearTimeout(t);
      pending.clear();
    };
  }, []);

  const noteSetup = useCallback((e: ApiError) => {
    if (isSetupError(e)) dispatch({ type: "setup/set", error: e });
  }, []);
  const clearSetup = useCallback(() => {
    if (stateRef.current.setup) dispatch({ type: "setup/set", error: null });
  }, []);

  const conversationExists = (id: string) =>
    stateRef.current.conversations.items.some((c) => c.id === id) || stateRef.current.threads[id]?.status === "ready";

  const removeConversation = useCallback((id: string) => {
    removed.current.add(id);
    const t = timers.current.get(id);
    if (t !== undefined) window.clearTimeout(t);
    timers.current.delete(id);
    dispatch({ type: "conversation/remove", id });
    dispatch({ type: "thread/missing", id });
    if (selectedRef.current === id) navigate(paths.workspace, { replace: true });
  }, []);

  // ----- conversation list -------------------------------------------------
  const loadList = useCallback(async (more: boolean) => {
    dispatch({ type: "list/loading", more });
    const offset = more ? stateRef.current.conversations.items.length : 0;
    try {
      const items = await api.listConversations(offset);
      clearSetup();
      dispatch({ type: "list/loaded", items, append: more, hasMore: items.length === LIMITS.conversationsPage });
    } catch (e) {
      const err = toApiError(e);
      noteSetup(err);
      dispatch({ type: "list/failed", error: err });
    }
  }, [clearSetup, noteSetup]);

  useEffect(() => {
    void loadList(false);
  }, [loadList]);

  // ----- reconciliation ----------------------------------------------------
  const schedule = useCallback((id: string, delay: number, run: () => void) => {
    const prev = timers.current.get(id);
    if (prev !== undefined) window.clearTimeout(prev);
    timers.current.set(id, window.setTimeout(() => {
      timers.current.delete(id);
      run();
    }, delay));
  }, []);

  const check = useCallback(async (id: string, manual = false): Promise<void> => {
    const current = stateRef.current.awaiting[id];
    if (!current || !mounted.current) return;
    let messages;
    try {
      const result = await listAllMessages(id);
      messages = result.messages;
      clearSetup();
      dispatch({ type: "thread/loaded", id, messages, truncated: result.truncated });
    } catch (e) {
      const err = toApiError(e);
      if (err.kind === "http" && err.status === 404) {
        removeConversation(id);
        return;
      }
      noteSetup(err);
      messages = null; // keep the state; try again on the next tick
    }
    const a = stateRef.current.awaiting[id];
    if (!a || !mounted.current || removed.current.has(id)) return;
    const now = Date.now();
    let next: Awaiting = a;
    if (messages) {
      const outcome = analyzeTurn(messages, a.baseline);
      if (outcome.kind === "replied") {
        dispatch({ type: "awaiting/set", id, awaiting: null });
        setAnnouncement(outcome.assistant.status === "failed" ? "The turn failed." : "Reply received.");
        return;
      }
      if (outcome.kind === "awaiting" && !a.userStored) next = { ...a, userStored: true };
      if (outcome.kind === "not-stored" && a.baseline === null) {
        // Nothing unanswered (e.g. no messages at all): nothing to wait for.
        dispatch({ type: "awaiting/set", id, awaiting: null });
        return;
      }
    }
    const limit = next.userStored ? next.deadline : Math.min(next.deadline, next.checkingSince + NOT_STORED_WINDOW_MS);
    if (now >= limit) {
      if (next.phase !== "expired") setAnnouncement("The outcome of the last message is unknown.");
      next = { ...next, phase: "expired" };
      dispatch({ type: "awaiting/set", id, awaiting: next });
      return;
    }
    if (manual && next.phase === "expired") {
      dispatch({ type: "awaiting/set", id, awaiting: next });
      return;
    }
    next = { ...next, phase: "polling" };
    dispatch({ type: "awaiting/set", id, awaiting: next });
    const delay = now - next.checkingSince < FAST_PHASE_MS ? POLL_FAST_MS : POLL_SLOW_MS;
    schedule(id, delay, () => void check(id));
  }, [clearSetup, noteSetup, removeConversation, schedule]);

  const startReconcile = useCallback((id: string, init: { baseline: number | null; content: string; since: number }) => {
    const now = Date.now();
    const awaiting: Awaiting = {
      baseline: init.baseline,
      userStored: init.baseline === null,
      content: init.content,
      since: init.since,
      checkingSince: now,
      deadline: init.since + TURN_WINDOW_MS,
      phase: now >= init.since + TURN_WINDOW_MS ? "expired" : "polling",
    };
    dispatch({ type: "awaiting/set", id, awaiting });
    if (awaiting.phase === "polling") schedule(id, FIRST_CHECK_MS, () => void check(id));
  }, [check, schedule]);

  // ----- open the selected conversation (also restores state after reload) --
  const openThread = useCallback(async (id: string) => {
    dispatch({ type: "thread/loading", id });
    const [conv, msgs] = await Promise.allSettled([api.getConversation(id), listAllMessages(id)]);
    for (const r of [conv, msgs]) {
      if (r.status === "rejected") {
        const err = toApiError(r.reason);
        if (err.kind === "http" && err.status === 404) {
          dispatch({ type: "conversation/remove", id });
          dispatch({ type: "thread/missing", id });
          return;
        }
      }
    }
    if (conv.status === "fulfilled") dispatch({ type: "conversation/upsert", conversation: conv.value, toTop: false });
    if (msgs.status === "rejected") {
      const err = toApiError(msgs.reason);
      noteSetup(err);
      dispatch({ type: "thread/failed", id, error: err });
      return;
    }
    clearSetup();
    const { messages, truncated } = msgs.value;
    dispatch({ type: "thread/loaded", id, messages, truncated });
    const s = stateRef.current;
    const tail = unansweredTail(messages);
    if (tail && s.turn?.conversationId !== id && !s.awaiting[id]) {
      startReconcile(id, { baseline: null, content: tail.content, since: parseTimestamp(tail.created_at).getTime() });
    }
  }, [clearSetup, noteSetup, startReconcile]);

  useEffect(() => {
    if (!conversationId) return;
    const thread = stateRef.current.threads[conversationId];
    if (!thread || thread.status === "error") void openThread(conversationId);
  }, [conversationId, openThread]);

  const refreshThread = useCallback((id: string) => {
    if (stateRef.current.awaiting[id]) void check(id, true);
    else void openThread(id);
  }, [check, openThread]);

  // ----- sending -------------------------------------------------------------
  const pollingAnywhere = Object.values(state.awaiting).some((a) => a.phase === "polling");
  const sendBlockedReason = useMemo(() => {
    if (state.turn) {
      return state.turn.key === selectedKey
        ? "Waiting for the reply to your last message."
        : "Another conversation is waiting for the model. One message is processed at a time.";
    }
    const own = state.awaiting[selectedKey];
    if (own?.phase === "polling") return "Checking whether the last message was answered.";
    if (pollingAnywhere) return "Another conversation is still being checked for a reply. One message is processed at a time.";
    const thread = selectedKey === NEW ? null : state.threads[selectedKey];
    if (thread && thread.status !== "ready") return "This conversation is not loaded.";
    return null;
  }, [state.turn, state.awaiting, state.threads, selectedKey, pollingAnywhere]);

  const handleSendError = useCallback(async (
    id: string,
    content: string,
    baseline: number,
    startedAt: number,
    err: ApiError,
  ) => {
    const restoreDraft = (key: Key = id) => dispatch({ type: "draft/set", key, text: content });
    const notice = (key: Key = id) => dispatch({ type: "notice/set", key, notice: { error: err } });
    noteSetup(err);

    // Outcome unknown: the request may have reached the backend, which may still be working.
    const uncertain =
      err.kind === "network" ||
      (err.kind === "http" && err.source === "proxy" && (err.code === "proxy_timeout" || err.code === "backend_connection_lost"));
    if (uncertain) {
      notice();
      startReconcile(id, { baseline, content, since: startedAt });
      return;
    }
    if (err.kind === "contract") {
      notice();
      void openThread(id);
      return;
    }
    if (err.status === 404) {
      removeConversation(id);
      restoreDraft(NEW);
      notice(NEW);
      return;
    }
    // Rejected before anything was stored (backend_unreachable: no connection was ever made).
    if (err.status === 401 || err.status === 413 || err.status === 422 || (err.source === "proxy" && err.status === 403) ||
        err.code === "proxy_not_configured" || err.code === "backend_unreachable") {
      restoreDraft();
      notice();
      return;
    }
    if (err.code === "conversation_busy") {
      // Our message was not stored; an earlier turn is still running.
      restoreDraft();
      notice();
      await openThread(id);
      return;
    }
    // 503 agent_unavailable / 500 internal_error / anything else: the request
    // has finished. Show what the backend stored (a failed assistant message
    // if it could record one).
    notice();
    try {
      const { messages, truncated } = await listAllMessages(id);
      dispatch({ type: "thread/loaded", id, messages, truncated });
      if (analyzeTurn(messages, baseline).kind === "not-stored") restoreDraft();
    } catch (e) {
      dispatch({ type: "thread/failed", id, error: toApiError(e) });
    }
  }, [noteSetup, openThread, removeConversation, startReconcile]);

  const sendContent = useCallback(async (key: Key, raw: string) => {
    const content = raw.trim();
    if (validateMessage(content) || stateRef.current.turn) return;
    const before = key === NEW ? [] : (stateRef.current.threads[key]?.messages ?? []);
    const baseline = maxPosition(before);
    const startedAt = Date.now();
    dispatch({ type: "turn/start", key, content, startedAt });
    setAnnouncement("Message sent. Waiting for the model.");

    let id = key === NEW ? null : key;
    if (id === null) {
      try {
        const conv = await api.createConversation();
        clearSetup();
        id = conv.id;
        dispatch({ type: "conversation/upsert", conversation: conv, toTop: true });
        dispatch({ type: "thread/loaded", id, messages: [], truncated: false });
        dispatch({ type: "turn/conversation", id });
        if (selectedRef.current === NEW) navigate(paths.conversation(id), { replace: true });
      } catch (e) {
        const err = toApiError(e);
        noteSetup(err);
        dispatch({ type: "turn/end" });
        dispatch({ type: "draft/set", key: NEW, text: content });
        dispatch({ type: "notice/set", key: NEW, notice: { error: err } });
        setAnnouncement("The conversation could not be created.");
        return;
      }
    }

    try {
      const result = await api.sendMessage(id, content);
      clearSetup();
      dispatch({ type: "turn/end" });
      if (!conversationExists(id)) return; // deleted while waiting
      dispatch({ type: "thread/append", id, messages: [result.user_message, result.assistant_message] });
      setAnnouncement(result.assistant_message.status === "failed" ? "The turn failed." : "Reply received.");
      api.getConversation(id).then(
        (conversation) => dispatch({ type: "conversation/upsert", conversation, toTop: true }),
        () => {},
      );
    } catch (e) {
      dispatch({ type: "turn/end" });
      if (!mounted.current) return;
      setAnnouncement("The message did not get a reply.");
      await handleSendError(id, content, baseline, startedAt, toApiError(e));
    }
  }, [clearSetup, handleSendError, noteSetup]);

  const send = useCallback(async (text: string) => {
    if (sendBlockedReason) return;
    await sendContent(selectedKey, text);
  }, [selectedKey, sendBlockedReason, sendContent]);

  /** Re-sends the user message before a turn the backend recorded as FAILED. */
  const sendAgain = useCallback(async (failedMessageId: string) => {
    if (sendBlockedReason || selectedKey === NEW) return;
    const messages = stateRef.current.threads[selectedKey]?.messages ?? [];
    const i = messages.findIndex((m) => m.id === failedMessageId);
    const failed = messages[i];
    const user = messages[i - 1];
    if (!failed || failed.status !== "failed" || i !== messages.length - 1 || user?.role !== "user") return;
    await sendContent(selectedKey, user.content);
  }, [selectedKey, sendBlockedReason, sendContent]);

  const deleteConversation = useCallback(async (id: string): Promise<ApiError | null> => {
    if (stateRef.current.turn?.conversationId === id) {
      return new ApiError({ kind: "http", code: "conversation_busy", message: "Wait for the reply before deleting.", source: "client" });
    }
    try {
      await api.deleteConversation(id);
    } catch (e) {
      const err = toApiError(e);
      if (!(err.kind === "http" && err.status === 404)) {
        noteSetup(err);
        return err;
      }
    }
    removeConversation(id);
    setAnnouncement("Conversation deleted.");
    return null;
  }, [noteSetup, removeConversation]);

  return {
    state,
    selectedKey,
    announcement,
    sendBlockedReason,
    loadMore: () => void loadList(true),
    reloadList: () => void loadList(false),
    refreshThread,
    send,
    sendAgain,
    deleteConversation,
    setDraft: (key, text) => dispatch({ type: "draft/set", key, text }),
    dismissNotice: (key) => dispatch({ type: "notice/set", key, notice: null }),
  };
}
