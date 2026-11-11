import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "../../App";
import { err, FakeBackend } from "../../test/fakeBackend";
import { NOT_STORED_WINDOW_MS, TURN_WINDOW_MS } from "./state/useWorkspace";

let backend: FakeBackend;

beforeEach(() => {
  backend = new FakeBackend();
  vi.stubGlobal("fetch", vi.fn(backend.fetch));
});
afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  window.history.replaceState(null, "", "/");
});

function renderAt(path: string) {
  window.history.replaceState(null, "", path);
  const user = userEvent.setup({
    advanceTimers: (ms) => {
      if (vi.isFakeTimers()) vi.advanceTimersByTime(ms);
    },
  });
  render(<App />);
  return user;
}

const composer = () => screen.getByRole("textbox", { name: "Message" });
const sendButton = () => screen.getByRole("button", { name: "Send" });
const sends = () => backend.count("POST", /\/messages$/);
const messageList = () => screen.getByRole("list", { name: "Messages" });

async function typeAndSend(user: ReturnType<typeof userEvent.setup>, text: string) {
  await user.click(composer());
  await user.keyboard(text.replace(/[{[]/g, (c) => c + c));
  await user.keyboard("{Enter}");
}

describe("conversation list and history", () => {
  it("lists conversations and loads a selected conversation with tool records", async () => {
    const c = backend.addConversation("Papers on RAG", [
      { role: "user", content: "Find papers on RAG" },
      {
        role: "assistant",
        content: "Two papers found.",
        tool_executions: [
          { tool_name: "research:search_works", status: "succeeded", created_at: "2026-11-11T09:00:05Z" },
          { tool_name: "github:get_file_contents", status: "rejected", created_at: "2026-11-11T09:00:06Z" },
        ],
      },
    ]);
    const user = renderAt("/nexus");
    const nav = await screen.findByRole("navigation", { name: "Conversations" });
    await user.click(await within(nav).findByRole("link", { name: /Papers on RAG/ }));
    expect(window.location.pathname).toBe(`/nexus/c/${c.id}`);

    expect(await screen.findByText("Two papers found.")).toBeInTheDocument();
    const tools = screen.getByRole("list", { name: "Tool executions" });
    expect(within(tools).getByText("research:search_works")).toBeInTheDocument();
    expect(within(tools).getByText("succeeded")).toBeInTheDocument();
    expect(within(tools).getByText("rejected")).toBeInTheDocument();
    const times = tools.querySelectorAll("time");
    expect(times[0]).toHaveAttribute("dateTime", "2026-11-11T09:00:05Z");
    expect(times[0]!.getAttribute("title")).toBe("2026-11-11T09:00:05.000Z");
  });

  it("restores the conversation from the URL after a reload", async () => {
    const c = backend.addConversation("Saved", [
      { role: "user", content: "Q" },
      { role: "assistant", content: "Stored answer" },
    ]);
    renderAt(`/nexus/c/${c.id}`);
    expect(await screen.findByText("Stored answer")).toBeInTheDocument();
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("Saved");
  });

  it("renders message content as plain text, never as HTML", async () => {
    const c = backend.addConversation("XSS", [
      { role: "user", content: "hi" },
      { role: "assistant", content: '<img src=x onerror="alert(1)"> **bold**' },
    ]);
    renderAt(`/nexus/c/${c.id}`);
    expect(await screen.findByText('<img src=x onerror="alert(1)"> **bold**')).toBeInTheDocument();
    expect(document.querySelector("main img")).toBeNull();
  });

  it("shows 'not found' for a deleted conversation (404) and for a malformed id without calling the API for it", async () => {
    renderAt("/nexus/c/3f1c1e9a-7d1b-4c3e-9a51-0d6b9f1e2a99");
    expect(await screen.findByRole("heading", { name: "Conversation not found" })).toBeInTheDocument();
    expect(screen.queryByRole("textbox", { name: "Message" })).not.toBeInTheDocument();

    window.history.replaceState(null, "", "/nexus/c/not-a-uuid");
    window.dispatchEvent(new PopStateEvent("popstate"));
    expect(await screen.findByRole("heading", { name: "Conversation not found" })).toBeInTheDocument();
    expect(backend.calls.some((c) => c.path.includes("not-a-uuid"))).toBe(false);
  });

  it("loads more conversations with offset pagination", async () => {
    for (let i = 0; i < 51; i++) backend.addConversation(`Conversation ${i}`);
    const user = renderAt("/nexus");
    await user.click(await screen.findByRole("button", { name: "Load more" }));
    expect(await screen.findByText("Conversation 0")).toBeInTheDocument();
    expect(backend.calls.some((c) => c.path === "/v1/conversations?limit=50&offset=50")).toBe(true);
  });
});

describe("sending", () => {
  it("creates the conversation on first send (D2), shows the reply and the backend's title", async () => {
    backend.sends.push({ kind: "reply", content: "Here you go.", tools: [{ tool_name: "research:search_works", status: "succeeded" }] });
    const user = renderAt("/nexus");
    await screen.findByRole("heading", { name: "Start a conversation" });
    expect(backend.count("POST", /^\/v1\/conversations$/)).toBe(0);

    await typeAndSend(user, "Find papers on retrieval");
    expect(await screen.findByText("Here you go.")).toBeInTheDocument();
    expect(backend.count("POST", /^\/v1\/conversations$/)).toBe(1);
    expect(sends()).toBe(1);
    const created = backend.conversations[0]!;
    expect(window.location.pathname).toBe(`/nexus/c/${created.id}`);
    expect(backend.calls.find((c) => c.method === "POST" && c.path.endsWith("/messages"))!.body).toEqual({
      content: "Find papers on retrieval",
    });
    await waitFor(() => expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("Find papers on retrieval"));
    expect(within(messageList()).getByText("research:search_works")).toBeInTheDocument();
    expect(composer()).toHaveValue("");
  });

  it("shows real elapsed time while waiting and allows only one send at a time across conversations (D6)", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    const a = backend.addConversation("A", [{ role: "user", content: "q" }, { role: "assistant", content: "a" }]);
    const b = backend.addConversation("B", [{ role: "user", content: "q" }, { role: "assistant", content: "b" }]);
    backend.sends.push({ kind: "hang" });
    const user = renderAt(`/nexus/c/${a.id}`);
    await screen.findByText("a");
    await typeAndSend(user, "Slow question");

    expect(await screen.findByText("Sending…")).toBeInTheDocument();
    const pending = screen.getByTestId("pending-reply");
    expect(within(pending).getByText("Waiting for the model")).toBeInTheDocument();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(65_000);
    });
    expect(within(pending).getByLabelText("Elapsed time").textContent).toMatch(/^1:0[5-6]$/);
    expect(sendButton()).toBeDisabled();
    expect(screen.getByText("Waiting for the reply to your last message.")).toBeInTheDocument();

    await user.click(screen.getByRole("link", { name: /^B/ }));
    await screen.findByText("b");
    await user.click(composer());
    await user.keyboard("Another");
    expect(sendButton()).toBeDisabled();
    expect(screen.getByText(/Another conversation is waiting for the model/)).toBeInTheDocument();

    await user.click(screen.getByRole("link", { name: /^A/ }));
    await act(async () => backend.finishHanging(a.id, "Finally."));
    expect(await screen.findByText("Finally.")).toBeInTheDocument();
    expect(screen.queryByTestId("pending-reply")).not.toBeInTheDocument();
    expect(sends()).toBe(1);
    void b;
  });

  it("blocks over-long messages before sending", async () => {
    const user = renderAt("/nexus");
    await screen.findByRole("heading", { name: "Start a conversation" });
    await user.click(composer());
    await user.paste("x".repeat(8001));
    expect(sendButton()).toBeDisabled();
    expect(screen.getByText("Messages are limited to 8,000 characters.")).toBeInTheDocument();
  });
});

describe("failed turns and errors", () => {
  async function inConversation() {
    const c = backend.addConversation("Conv", [{ role: "user", content: "earlier" }, { role: "assistant", content: "earlier answer" }]);
    const user = renderAt(`/nexus/c/${c.id}`);
    await screen.findByText("earlier answer");
    return { c, user };
  }

  it("503 agent_unavailable: shows the stored failed message; 'Send again' re-sends only on request", async () => {
    backend.sends.push({ kind: "agent-failure", code: "model_timeout" });
    const { user } = await inConversation();
    await typeAndSend(user, "Will time out");

    const failed = await screen.findByRole("article", { name: "Nexus, failed" });
    expect(within(failed).getByText("The language model did not respond in time.")).toBeInTheDocument();
    expect(within(failed).getByText("Failed · Timed out")).toBeInTheDocument();
    expect(screen.getByRole("alert")).toHaveTextContent("No answer from the model");
    expect(screen.getByRole("alert")).toHaveTextContent("req503");
    expect(sends()).toBe(1);

    backend.sends.push({ kind: "reply", content: "Second try worked." });
    await user.click(within(failed).getByRole("button", { name: "Send again" }));
    expect(await screen.findByText("Second try worked.")).toBeInTheDocument();
    expect(sends()).toBe(2);
    const bodies = backend.calls.filter((c) => c.method === "POST" && c.path.endsWith("/messages")).map((c) => c.body);
    expect(bodies).toEqual([{ content: "Will time out" }, { content: "Will time out" }]);
  });

  it("shows failed messages loaded from history, with Send again only on the latest failed turn", async () => {
    const c = backend.addConversation("History", [
      { role: "user", content: "one" },
      { role: "assistant", content: "The language model could not be reached.", status: "failed", error_code: "model_unreachable" },
      { role: "user", content: "two" },
      { role: "assistant", content: "The language model returned an error.", status: "failed", error_code: "model_error" },
    ]);
    renderAt(`/nexus/c/${c.id}`);
    const failed = await screen.findAllByRole("article", { name: "Nexus, failed" });
    expect(failed).toHaveLength(2);
    expect(within(failed[0]!).getByText("Failed · Model unreachable")).toBeInTheDocument();
    expect(within(failed[0]!).queryByRole("button", { name: "Send again" })).toBeNull();
    expect(within(failed[1]!).getByRole("button", { name: "Send again" })).toBeInTheDocument();
  });

  it("500 internal_error: shows what the backend stored", async () => {
    backend.sends.push({ kind: "internal-error" });
    const { user } = await inConversation();
    await typeAndSend(user, "Boom");
    const failed = await screen.findByRole("article", { name: "Nexus, failed" });
    expect(within(failed).getByText("Failed · Internal error")).toBeInTheDocument();
    expect(screen.getByRole("alert")).toHaveTextContent("internal error");
  });

  it.each([
    [409, "conversation_busy", "Still processing an earlier message"],
    [422, "credential_rejected", "looks like it contains a secret"],
    [422, "validation_error", "rejected the message"],
    [413, "request_too_large", "Message too large"],
  ])("%i %s: nothing stored, draft restored, explained", async (status, code, text) => {
    backend.sends.push({ kind: "error", status, code });
    const { user } = await inConversation();
    await typeAndSend(user, "my draft text");
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(text);
    expect(alert).toHaveTextContent(`HTTP ${status}`);
    expect(composer()).toHaveValue("my draft text");
    expect(within(messageList()).queryByText("my draft text")).toBeNull();
    expect(sends()).toBe(1);
  });

  it("401: shows the setup banner and keeps the draft", async () => {
    backend.sends.push({ kind: "error", status: 401, code: "unauthenticated" });
    const { user } = await inConversation();
    await typeAndSend(user, "hello");
    expect(await screen.findByTestId("setup-banner")).toHaveTextContent("rejected the proxy's credential");
    expect(composer()).toHaveValue("hello");
  });

  it("proxy_not_configured: explains how to supply the credential", async () => {
    backend.override = () => err(503, "proxy_not_configured", "no token", true);
    renderAt("/nexus");
    const banner = await screen.findByTestId("setup-banner");
    expect(banner).toHaveTextContent("no backend credential");
    expect(banner).toHaveTextContent("npm run dev:secure");
    expect(await screen.findByText("Proxy not configured")).toBeInTheDocument();
  });
});

describe("uncertain turns (amendment 1)", () => {
  it("network loss after the message was stored: polls stored messages and shows the reply; never re-sends", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    backend.sends.push({ kind: "network", stores: "user" });
    const c = backend.addConversation("Conv", [{ role: "user", content: "q" }, { role: "assistant", content: "a" }]);
    const user = renderAt(`/nexus/c/${c.id}`);
    await screen.findByText("a");
    await typeAndSend(user, "Lost connection");

    expect(await screen.findByText("Connection lost")).toBeInTheDocument();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(5_100);
    });
    const panel = await screen.findByTestId("awaiting-panel");
    expect(panel).toHaveTextContent("Waiting for a stored reply");
    expect(within(messageList()).getByText("Lost connection")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Send again" })).toBeNull();
    expect(sendButton()).toBeDisabled();
    expect(composer()).toHaveValue("");

    backend.storeMessage(c.id, { role: "assistant", content: "Reply that finished anyway." });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(5_100);
    });
    expect(await screen.findByText("Reply that finished anyway.")).toBeInTheDocument();
    expect(screen.queryByTestId("awaiting-panel")).toBeNull();
    expect(sends()).toBe(1);
  });

  it("proxy backend_connection_lost: reconciles (outcome uncertain) without a setup banner", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    backend.sends.push({ kind: "proxy", status: 502, code: "backend_connection_lost", storesUser: true });
    const c = backend.addConversation("Conv", [{ role: "user", content: "q" }, { role: "assistant", content: "a" }]);
    const user = renderAt(`/nexus/c/${c.id}`);
    await screen.findByText("a");
    await typeAndSend(user, "Dropped mid-turn");
    expect(await screen.findByText("Connection to the backend lost")).toBeInTheDocument();
    expect(screen.queryByTestId("setup-banner")).toBeNull();
    const panel = await screen.findByTestId("awaiting-panel");
    expect(panel).toHaveTextContent("Dropped mid-turn"); // quoted until confirmed stored
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1_100);
    });
    await waitFor(() => expect(screen.getByTestId("awaiting-panel")).toHaveTextContent("Waiting for a stored reply"));
    expect(composer()).toHaveValue("");
    expect(sends()).toBe(1);
  });

  it("proxy backend_unreachable (connection refused): nothing was sent, so the draft is restored", async () => {
    backend.sends.push({ kind: "proxy", status: 502, code: "backend_unreachable", storesUser: false });
    const c = backend.addConversation("Conv", [{ role: "user", content: "q" }, { role: "assistant", content: "a" }]);
    const user = renderAt(`/nexus/c/${c.id}`);
    await screen.findByText("a");
    await typeAndSend(user, "Backend is down");
    expect(await screen.findByTestId("setup-banner")).toHaveTextContent("not reachable");
    expect(composer()).toHaveValue("Backend is down");
    expect(screen.queryByTestId("awaiting-panel")).toBeNull();
  });

  it("no reply within the window: reports the outcome as unknown with Refresh and Copy, never Send again", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    backend.sends.push({ kind: "network", stores: "user" });
    const c = backend.addConversation("Conv", [{ role: "user", content: "q" }, { role: "assistant", content: "a" }]);
    const user = renderAt(`/nexus/c/${c.id}`);
    await screen.findByText("a");
    await typeAndSend(user, "Never answered");
    await act(async () => {
      await vi.advanceTimersByTimeAsync(TURN_WINDOW_MS + 30_000);
    });
    const panel = await screen.findByTestId("awaiting-panel");
    await waitFor(() => expect(panel).toHaveAttribute("data-phase", "expired"));
    expect(panel).toHaveTextContent("Outcome unknown");
    expect(within(panel).getByRole("button", { name: "Refresh" })).toBeInTheDocument();
    expect(within(panel).getByRole("button", { name: "Copy message" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Send again" })).toBeNull();
    expect(screen.getByText(/unanswered message stays in the history/)).toBeInTheDocument();
    expect(sends()).toBe(1);

    const polls = backend.count("GET", /\/messages\?/);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(60_000);
    });
    expect(backend.count("GET", /\/messages\?/)).toBe(polls); // polling stopped

    backend.storeMessage(c.id, { role: "assistant", content: "Arrived much later." });
    await user.click(within(panel).getByRole("button", { name: "Refresh" }));
    expect(await screen.findByText("Arrived much later.")).toBeInTheDocument();
    expect(screen.queryByTestId("awaiting-panel")).toBeNull();
  });

  it("network loss before anything was stored: says it cannot be confirmed; does not restore or re-send", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    backend.sends.push({ kind: "network", stores: "nothing" });
    const c = backend.addConversation("Conv", [{ role: "user", content: "q" }, { role: "assistant", content: "a" }]);
    const user = renderAt(`/nexus/c/${c.id}`);
    await screen.findByText("a");
    await typeAndSend(user, "Dropped");
    await act(async () => {
      await vi.advanceTimersByTimeAsync(NOT_STORED_WINDOW_MS + 20_000);
    });
    const panel = await screen.findByTestId("awaiting-panel");
    await waitFor(() => expect(panel).toHaveAttribute("data-phase", "expired"));
    expect(panel).toHaveTextContent("does not appear to have been stored");
    expect(screen.queryByRole("button", { name: "Send again" })).toBeNull();
    expect(composer()).toHaveValue("");
    expect(sendButton()).toBeDisabled(); // empty draft
    expect(sends()).toBe(1);
  });

  it("copy message puts the text on the clipboard", async () => {
    vi.setSystemTime(Date.parse("2026-11-11T12:00:00Z"));
    const c = backend.addConversation("Old orphan", [{ role: "user", content: "Orphaned question" }]);
    const user = renderAt(`/nexus/c/${c.id}`);
    const panel = await screen.findByTestId("awaiting-panel");
    await user.click(within(panel).getByRole("button", { name: "Copy message" }));
    // user-event installs a clipboard stub on navigator.clipboard.
    expect(await navigator.clipboard.readText()).toBe("Orphaned question");
    expect(await within(panel).findByRole("button", { name: "Copied" })).toBeInTheDocument();
    vi.useRealTimers();
  });

  it("reload with a recent unanswered user message: polls for the reply", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    vi.setSystemTime(Date.parse("2026-11-11T09:00:30Z"));
    const c = backend.addConversation("In progress", [{ role: "user", content: "Still thinking?" }]);
    renderAt(`/nexus/c/${c.id}`);
    const panel = await screen.findByTestId("awaiting-panel");
    expect(panel).toHaveAttribute("data-phase", "polling");
    expect(sendButton()).toBeDisabled();
    backend.storeMessage(c.id, { role: "assistant", content: "Done thinking." });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(5_100);
    });
    expect(await screen.findByText("Done thinking.")).toBeInTheDocument();
  });

  it("reload with an old orphaned user message: outcome unknown immediately, composer usable with a warning", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    vi.setSystemTime(Date.parse("2026-11-11T12:00:00Z"));
    const c = backend.addConversation("Orphan", [{ role: "user", content: "Server restarted mid-turn" }]);
    const user = renderAt(`/nexus/c/${c.id}`);
    const panel = await screen.findByTestId("awaiting-panel");
    expect(panel).toHaveAttribute("data-phase", "expired");
    expect(panel).toHaveTextContent("Outcome unknown");
    await user.click(composer());
    await user.keyboard("New question");
    expect(sendButton()).toBeEnabled();
    expect(screen.getByText(/unanswered message stays in the history/)).toBeInTheDocument();
  });
});

describe("deleting", () => {
  it("asks for confirmation, deletes, and leaves the deleted conversation", async () => {
    const c = backend.addConversation("Delete me", [{ role: "user", content: "q" }, { role: "assistant", content: "a" }]);
    backend.addConversation("Keep me");
    const user = renderAt(`/nexus/c/${c.id}`);
    await screen.findByText("a");

    await user.click(screen.getByRole("button", { name: "Delete conversation: Delete me" }));
    const dialog = screen.getByRole("alertdialog", { name: "Delete this conversation?" });
    expect(within(dialog).getByRole("button", { name: "Cancel" })).toHaveFocus();
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("alertdialog")).toBeNull();
    expect(backend.count("DELETE", /./)).toBe(0);

    await user.click(screen.getByRole("button", { name: "Delete conversation: Delete me" }));
    await user.click(within(screen.getByRole("alertdialog")).getByRole("button", { name: "Delete" }));
    await waitFor(() => expect(window.location.pathname).toBe("/nexus"));
    expect(backend.count("DELETE", new RegExp(c.id))).toBe(1);
    expect(screen.queryByRole("link", { name: /Delete me/ })).toBeNull();
    expect(screen.getByRole("link", { name: /Keep me/ })).toBeInTheDocument();
  });

  it("shows a delete error inside the dialog", async () => {
    const c = backend.addConversation("Stuck");
    backend.override = null;
    const user = renderAt("/nexus");
    await screen.findByRole("link", { name: /Stuck/ });
    const original = backend.fetch;
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) =>
      init?.method === "DELETE" ? err(500, "internal_error", "Internal server error.") : original(input, init),
    ));
    await user.click(screen.getByRole("button", { name: "Delete conversation: Stuck" }));
    await user.click(within(screen.getByRole("alertdialog")).getByRole("button", { name: "Delete" }));
    expect(await within(screen.getByRole("alertdialog")).findByRole("alert")).toHaveTextContent("internal error");
    void c;
  });
});

describe("workspace architecture", () => {
  it("does not render the 3D scene", async () => {
    renderAt("/nexus");
    await screen.findByRole("navigation", { name: "Conversations" });
    expect(screen.queryByTestId("scene-layer")).toBeNull();
    expect(document.querySelector("canvas")).toBeNull();
  });
});
