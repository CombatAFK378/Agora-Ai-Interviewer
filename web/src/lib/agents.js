// Interviewer identity.
//
// The previous build coloured tiles from an array indexed by position, so an
// interviewer's colour changed depending on who else was on the panel. Colour
// is identity here, so it is keyed to the agent id and never moves.
//
// These hues are deliberately desaturated and deliberately NOT green, amber or
// red: those three are semantic in this app (live, flagged, declined) and an
// identity colour must never be mistaken for a state.

const IDENTITY = {
  hiring_manager: { hue: "#8B9DC3", icon: "ph-briefcase" },
  technical:      { hue: "#5FA8B8", icon: "ph-terminal-window" },
  product:        { hue: "#B08FC7", icon: "ph-compass" },
  customer:       { hue: "#D89A7A", icon: "ph-chat-circle" },
  coding:         { hue: "#9AAE7B", icon: "ph-code" },
  orchestrator:   { hue: "#7E8794", icon: "ph-broadcast" },
};

// Stable fallback for any agent id the dossier introduces that we have no
// entry for: hash the id so the same id always lands on the same hue.
const FALLBACK = ["#8B9DC3", "#5FA8B8", "#B08FC7", "#D89A7A", "#9AAE7B", "#7E8794"];

function hashIndex(id, len) {
  let h = 0;
  for (let i = 0; i < (id || "").length; i++) h = (h * 31 + id.charCodeAt(i)) >>> 0;
  return h % len;
}

export function agentHue(id) {
  return (IDENTITY[id] || {}).hue || FALLBACK[hashIndex(id, FALLBACK.length)];
}

export function agentIcon(id) {
  return (IDENTITY[id] || {}).icon || "ph-user";
}

export function initials(name) {
  return (name || "?").slice(0, 1).toUpperCase();
}

// Chart series colour, indexed by position.
//
// Competency lines must NOT use agentHue(): those keys are competency names,
// not agent ids, so they fall through to the hash and collide. With the shipped
// competency set that put three of six series on the same stroke, and drew
// "coding" in Liam's personal colour. Series identity is positional, so index it.
const SERIES = [
  "#4FB8D9", "#C4A05E", "#9C7BC7", "#5FA8B8", "#D08A6A", "#7E9E6B",
  "#B0899B", "#6E8BC3", "#C77B7B", "#8AA9A0",
];

export function seriesColor(i) {
  return SERIES[i % SERIES.length];
}
