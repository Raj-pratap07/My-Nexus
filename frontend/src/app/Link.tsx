import type { AnchorHTMLAttributes, MouseEvent } from "react";
import { isPlainClick, navigate } from "./router";

type Props = AnchorHTMLAttributes<HTMLAnchorElement> & { href: string };

/** A real <a href> that navigates client-side on a plain click. */
export function Link({ href, onClick, ...rest }: Props) {
  const handle = (e: MouseEvent<HTMLAnchorElement>) => {
    onClick?.(e);
    if (!isPlainClick(e)) return;
    e.preventDefault();
    navigate(href);
  };
  return <a href={href} onClick={handle} {...rest} />;
}
