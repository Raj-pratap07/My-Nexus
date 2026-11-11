import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props {
  fallback: ReactNode;
  onError?: (error: Error) => void;
  children: ReactNode;
}

interface State {
  failed: boolean;
}

/**
 * Contains every render-time scene failure (a throwing component, a failed
 * scene-chunk import, a renderer that cannot be created) so the rest of the
 * page keeps working. Logs the error TYPE only.
 */
export class SceneBoundary extends Component<Props, State> {
  override state: State = { failed: false };

  static getDerivedStateFromError(): State {
    return { failed: true };
  }

  override componentDidCatch(error: Error, _info: ErrorInfo): void {
    console.warn(`3D scene disabled (${error.name}); showing the static fallback.`);
    this.props.onError?.(error);
  }

  override render(): ReactNode {
    return this.state.failed ? this.props.fallback : this.props.children;
  }
}
