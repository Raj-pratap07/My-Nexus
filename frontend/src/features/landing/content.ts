/**
 * Landing-page copy. Every "available" item describes something the backend
 * does today (see the project README); everything else is listed under
 * "planned" and labelled as such. Keep it that way when editing.
 */

export const brand = {
  name: "My-Nexus",
  tagline: "Local-first AI research assistant",
} as const;

export const hero = {
  eyebrow: "Early development · Local use only",
  title: "A research assistant that runs on your own machine.",
  body:
    "My-Nexus pairs a local language model in Ollama with read-only tools for GitHub " +
    "files and scholarly search, behind a backend that runs on your computer. " +
    "No paid model API is involved.",
  primaryCta: { label: "Enter Nexus" },
  secondaryCta: { label: "See what works today", href: "#available" },
} as const;

export interface Capability {
  title: string;
  body: string;
}

export const available: readonly Capability[] = [
  {
    title: "A local model",
    body:
      "Answers come from a model running in Ollama on your computer " +
      "(qwen3:4b-instruct by default).",
  },
  {
    title: "Read-only GitHub file access",
    body:
      "Reads files from the My-Nexus repository through the GitHub MCP server. " +
      "Read-only, and the repository and paths are checked in code.",
  },
  {
    title: "Scholarly search",
    body:
      "Searches OpenAlex and Crossref and returns titles and metadata. " +
      "Abstracts are not included.",
  },
  {
    title: "A browser workspace",
    body:
      "Create, open and delete conversations and send messages to the local backend. " +
      "Each answer appears in one piece when the model finishes; there is no streaming.",
  },
  {
    title: "Conversation history",
    body:
      "Conversations, messages and tool activity are stored in your database. " +
      "A failed answer is recorded, not lost.",
  },
  {
    title: "Guardrails in code",
    body:
      "A default-deny tool allowlist, a limit on tool calls per request and a filter " +
      "that rejects messages containing credentials.",
  },
];

export const planned: readonly Capability[] = [
  {
    title: "Long-term memory",
    body: "The database schema exists; remembering things across conversations is not implemented.",
  },
  {
    title: "Automation",
    body: "Scheduled or multi-step tasks are not implemented.",
  },
  {
    title: "Desktop app",
    body: "A Tauri desktop app sharing this backend is planned.",
  },
];

export const footer = {
  note: "My-Nexus is in early development and intended for local use only.",
} as const;
