import { describe, expect, it } from "vitest";
import { parseRoute, paths } from "./router";

describe("router", () => {
  it("parses the three routes and rejects malformed conversation ids", () => {
    expect(parseRoute("/")).toEqual({ name: "landing" });
    expect(parseRoute("/nexus")).toEqual({ name: "workspace", conversationId: null });
    expect(parseRoute("/nexus/")).toEqual({ name: "workspace", conversationId: null });
    const id = "3F1C1E9A-7D1B-4C3E-9A51-0D6B9F1E2A01";
    expect(parseRoute(`/nexus/c/${id}`)).toEqual({ name: "workspace", conversationId: id.toLowerCase() });
    expect(parseRoute("/nexus/c/123")).toEqual({ name: "workspace", conversationId: null, invalidId: true });
    expect(parseRoute("/nexus/c/a/b")).toEqual({ name: "not-found" });
    expect(parseRoute("/elsewhere")).toEqual({ name: "not-found" });
    expect(paths.conversation("a b")).toBe("/nexus/c/a%20b");
  });
});
