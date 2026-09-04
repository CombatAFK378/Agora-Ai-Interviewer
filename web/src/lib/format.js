// Small display helpers. No app state, no side effects.

export const pct = (v) => Math.round((Number(v) || 0) * 100);

// competency keys arrive as snake_case; the backend also sends human names on
// live ledger rows, so prefer those and fall back to this.
export const prettyKey = (k) =>
  (k || "").replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());

// Recommendation and override enums are SCREAMING_SNAKE on the wire.
export const prettyEnum = (v) => (v || "").replace(/_/g, " ");

export const shortTime = (ts) =>
  ts ? new Date(ts * 1000).toLocaleString(undefined, {
    day: "numeric", month: "short", hour: "2-digit", minute: "2-digit",
  }) : "";
