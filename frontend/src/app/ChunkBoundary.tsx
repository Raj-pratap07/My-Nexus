import { Component, type ReactNode } from "react";

/** Shows a recoverable message if a lazily loaded page fails to load or render. */
export class ChunkBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  componentDidCatch(error: unknown) {
    console.warn("[app] page failed to load:", error instanceof Error ? error.name : typeof error);
  }

  render() {
    if (!this.state.failed) return this.props.children;
    return (
      <main className="grid min-h-[100svh] place-items-center px-5 text-center">
        <div className="max-w-md">
          <h1 className="font-display text-2xl font-semibold text-ink">This page could not be loaded.</h1>
          <p className="mt-3 text-ink-muted">
            The app may have been updated or the dev server stopped. Reloading usually fixes it.
          </p>
          <button
            type="button"
            onClick={() => window.location.reload()}
            className="mt-6 rounded-full bg-champagne px-5 py-2.5 text-sm font-semibold text-abyss"
          >
            Reload
          </button>
        </div>
      </main>
    );
  }
}
