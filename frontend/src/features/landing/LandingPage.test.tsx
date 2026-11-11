import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { App } from "../../App";
import { available, planned } from "./content";

// No WebGL in jsdom: this is the page exactly as a browser without WebGL sees it.
describe("landing page without WebGL", () => {
  it("renders the full content with landmarks and a single h1", () => {
    render(<App />);
    expect(screen.getByRole("banner")).toBeInTheDocument();
    expect(screen.getByRole("main")).toBeInTheDocument();
    expect(screen.getByRole("contentinfo")).toBeInTheDocument();
    expect(screen.getByRole("navigation", { name: "Primary" })).toBeInTheDocument();
    expect(screen.getAllByRole("heading", { level: 1 })).toHaveLength(1);
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(
      "A research assistant that runs on your own machine.",
    );
  });

  it("shows the static fallback instead of a canvas", () => {
    render(<App />);
    expect(screen.getByTestId("scene-layer")).toHaveAttribute("data-scene", "unsupported");
    expect(screen.getByTestId("scene-fallback")).toBeInTheDocument();
    expect(document.querySelector("canvas")).toBeNull();
  });

  it("keeps the decorative scene out of the accessibility tree", () => {
    render(<App />);
    expect(screen.getByTestId("scene-layer")).toHaveAttribute("aria-hidden", "true");
  });

  it("is keyboard navigable, starting with the skip link", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.tab();
    const skip = screen.getByRole("link", { name: "Skip to content" });
    expect(skip).toHaveFocus();
    expect(skip).toHaveAttribute("href", "#main");
    await user.tab();
    expect(screen.getByRole("link", { name: "My-Nexus home" })).toHaveFocus();
  });

  it("CTAs: 'Enter Nexus' links to the workspace; the secondary CTA to an in-page section", () => {
    render(<App />);
    expect(screen.getByRole("link", { name: "Enter Nexus" })).toHaveAttribute("href", "/nexus");
    const link = screen.getByRole("link", { name: "See what works today" });
    const target = link.getAttribute("href")!.slice(1);
    expect(document.getElementById(target)).not.toBeNull();
  });
});

describe("entering the Nexus", () => {
  afterEach(() => window.history.replaceState(null, "", "/"));

  it("navigates client-side to /nexus, unmounts the 3D scene layer and shows the workspace", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("[]", { status: 200 })));
    // Reduced motion: navigate without the fade delay.
    vi.spyOn(window, "matchMedia").mockImplementation(
      (q: string) => ({ matches: q.includes("reduce"), media: q, addEventListener() {}, removeEventListener() {} }) as unknown as MediaQueryList,
    );
    const user = userEvent.setup();
    render(<App />);
    expect(screen.getByTestId("scene-layer")).toBeInTheDocument();
    await user.click(screen.getByRole("link", { name: "Enter Nexus" }));
    expect(window.location.pathname).toBe("/nexus");
    expect(await screen.findByRole("navigation", { name: "Conversations" })).toBeInTheDocument();
    expect(screen.queryByTestId("scene-layer")).not.toBeInTheDocument();
    vi.unstubAllGlobals();
  });
});

describe("landing page copy is accurate", () => {
  it("labels every available item 'Available now' and every planned item 'Planned'", () => {
    render(<App />);
    const availableList = within(document.getElementById("available")!).getByRole("list");
    const plannedList = within(document.getElementById("planned")!).getByRole("list");
    expect(within(availableList).getAllByRole("listitem")).toHaveLength(available.length);
    expect(within(availableList).getAllByText("Available now")).toHaveLength(available.length);
    expect(within(plannedList).getAllByRole("listitem")).toHaveLength(planned.length);
    expect(within(plannedList).getAllByText("Planned")).toHaveLength(planned.length);
  });

  it("lists memory, automation and the desktop app as planned, not available", () => {
    const plannedText = planned.map((p) => `${p.title} ${p.body}`).join(" ").toLowerCase();
    for (const word of ["memory", "automation", "desktop"]) {
      expect(plannedText).toContain(word);
    }
    const availableText = available.map((a) => `${a.title} ${a.body}`).join(" ").toLowerCase();
    for (const word of ["memory", "automation", "desktop", "tauri", "streaming responses"]) {
      expect(availableText).not.toContain(word);
    }
  });

  it("describes the browser workspace accurately: available, blocking, no streaming", () => {
    const workspace = available.find((a) => a.title === "A browser workspace");
    expect(workspace?.body).toMatch(/one piece/);
    expect(workspace?.body).toMatch(/no streaming/);
    expect(planned.some((p) => /chat|web app/i.test(p.title))).toBe(false);
  });

  it("makes no claims the backend cannot back up", () => {
    const all = JSON.stringify([available, planned]).toLowerCase();
    for (const claim of ["write access", "any repository", "abstracts are included", "cloud"]) {
      expect(all).not.toContain(claim);
    }
    expect(all).toContain("read-only");
    expect(all).toContain("abstracts are not included");
  });
});
