import { agentHue } from "../lib/agents";

const mmss = (s) => `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;

// The interview as a scrubbable strip.
//
// Every frame here is built from events the client already receives; nothing
// new crosses the wire. Each block is one turn, coloured by whoever held the
// floor, with a mark for every claim that landed on it. Dragging the playhead
// rewinds the whole stage to what the panel knew at that moment.
//
// The visible blocks are decoration over a real <input type="range">, which is
// what actually owns the interaction: that gives drag, keyboard arrows, Home,
// End and a screen-reader-announced value for free.
export default function Timeline({ timeline, scrubIndex, onScrub, onLive }) {
  if (timeline.length < 2) return null;

  const last = timeline.length - 1;
  const at = scrubIndex == null ? last : scrubIndex;
  const live = scrubIndex == null;
  const frame = timeline[at];

  return (
    <div className={"timeline" + (live ? "" : " scrubbed")}>
      <div className="tl-head">
        <span className="sec-label">
          {live ? "Interview timeline" : `Rewound to ${mmss(frame.t)}`}
        </span>
        {!live && (
          <button className="ghost tl-live" onClick={onLive}>
            <i className="ph-fill ph-broadcast" aria-hidden="true" />
            Back to live
          </button>
        )}
      </div>

      <div className="tl-track">
        <div className="tl-blocks" aria-hidden="true">
          {timeline.map((f, i) => (
            <span
              key={f.id}
              className={
                "tl-block" +
                (f.kind === "ask" ? " ask" : " answer") +
                (i === at ? " at" : "") +
                (i > at ? " ahead" : "")
              }
              style={{ "--hue": f.kind === "ask" ? agentHue(f.agentId) : "var(--text-faint)" }}
            >
              {f.claimCount > 0 && (
                <span className="tl-marks">
                  {Array.from({ length: Math.min(f.claimCount, 4) }).map((_, m) => (
                    <span className="tl-mark" key={m} />
                  ))}
                </span>
              )}
            </span>
          ))}
        </div>

        <input
          className="tl-range"
          type="range"
          min={0}
          max={last}
          step={1}
          value={at}
          onChange={(e) => {
            const v = Number(e.target.value);
            onScrub(v === last ? null : v);
          }}
          aria-label={`Interview timeline, turn ${at + 1} of ${timeline.length}`}
        />
      </div>

      <div className="tl-foot">
        <span>{mmss(timeline[0].t)}</span>
        <span className="tl-hint">
          {live
            ? "drag to rewind and see what the panel knew"
            : `${frame.kind === "ask" ? frame.name : "Candidate"} at turn ${frame.turn}`}
        </span>
        <span>{mmss(timeline[last].t)}</span>
      </div>
    </div>
  );
}
