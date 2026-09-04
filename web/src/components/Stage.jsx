import { agentHue, agentIcon } from "../lib/agents";
import { pct } from "../lib/format";

const mmss = (s) => `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;

// The live interview.
//
// Once the candidate joins, the setup chrome goes away and the panel becomes
// the object on screen. The speaking interviewer is physically larger and
// carries a ring driven by real voice amplitude, and the caption is the
// largest text in the interface, because during an interview it is the thing
// you are actually reading.
export default function Stage({
  stageRef, agents, speaking, thinking, activePanel, finished,
  agentCap, candCap, elapsed, turnCount,
  coverage, claims, contradictions,
  onInterrupt, onFinish, onLeave, scoring, status, busy,
  rewound, timeline,
}) {
  const live = agents.filter(
    (a) => !finished.includes(a.id) && (activePanel.length === 0 || activePanel.includes(a.id))
  );
  // A contradicted claim is not solid. Counting it as one made the headline
  // read "3 of 3 solid" directly above a red CONFLICT row.
  const solid = claims.filter((c) => c.status === "SOLID" && !c.contradicts).length;
  const weakest = coverage.length
    ? coverage.reduce((a, b) => (a.value <= b.value ? a : b))
    : null;

  return (
    <div className={"stage" + (rewound ? " rewound" : "")} ref={stageRef}>
      <div className="stage-bar">
        <span className={"onair" + (rewound ? " past" : "")}>
          <span className="onair-dot" />
          {rewound ? "Rewound" : "Live"}
        </span>
        <span className="stage-clock">{mmss(elapsed)}</span>
        <span className="stage-meta">
          turn {turnCount}
          {claims.length > 0 && ` / ${solid} of ${claims.length} claims solid`}
          {weakest && ` / thinnest: ${weakest.name}`}
        </span>
        {!rewound && (
          <span className="stage-mic" title="Your microphone level">
            <i className="ph-fill ph-microphone" aria-hidden="true" />
            <span className="mic-track"><span className="mic-fill" /></span>
          </span>
        )}
        <div className="stage-actions">
          <button className="btn-flag" onClick={onInterrupt}>
            <i className="ph ph-hand-palm" aria-hidden="true" />
            Interrupt
          </button>
          <button className="btn-live" onClick={onFinish} disabled={scoring}>
            <i className="ph ph-flag-checkered" aria-hidden="true" />
            {scoring ? "Scoring" : "Finish and score"}
          </button>
          <button className="ghost" onClick={onLeave}>Leave</button>
        </div>
      </div>

      <div className="seats">
        {live.map((a) => {
          const isSpeaking = speaking === a.id;
          return (
            <div
              key={a.id}
              className={"seat" + (isSpeaking ? " on" : thinking ? " waiting" : "")}
              style={{ "--hue": agentHue(a.id) }}
            >
              <div className="seat-ring">
                <div className="seat-face">
                  <i className={`ph-fill ${agentIcon(a.id)}`} aria-hidden="true" />
                  <span className="icon-fallback">{(a.name || "?").slice(0, 1)}</span>
                </div>
              </div>
              <div className="seat-name">{a.name}</div>
              <div className="seat-role">{a.title}</div>
            </div>
          );
        })}
      </div>

      {/* Each interviewer is scoring privately and cannot read the others until
          the record locks. That is the central claim of the system, so it gets
          a physical representation instead of a line of prose. */}
      <div className="sealed">
        <span className="sealed-label">
          <i className="ph ph-lock-simple" aria-hidden="true" />
          Scores sealed until the record locks
        </span>
        <div className="sealed-row">
          {live.map((a) => (
            <span className="sheet" key={a.id} style={{ "--hue": agentHue(a.id) }}
                  title={`${a.name} is scoring privately`}>
              <span className="sheet-line" />
              <span className="sheet-line" />
              <span className="sheet-line" />
            </span>
          ))}
        </div>
      </div>

      <div className="stage-body">
        <div className="floor">
          {thinking && (
            <div className="floor-thinking">
              <span className="tri" /><span className="tri" /><span className="tri" />
              the panel is deciding who asks next
            </div>
          )}

          {/* Hue comes from the caption's own agent, not from `speaking`:
              `speaking` clears on idle while the caption stays up, and
              agentHue(null) hashes to the hiring manager's colour, so the
              stripe used to change identity after every single turn. */}
          {agentCap && (
            <div className="said" style={{ "--hue": agentHue(agentCap.id) }}>
              <div className="said-who">
                {agentCap.name}
                <span>{agentCap.title}</span>
              </div>
              <p className="said-text">{agentCap.text}</p>
            </div>
          )}

          {candCap && (
            <div className="said you">
              <div className="said-who">You</div>
              <p className="said-text">{candCap}</p>
            </div>
          )}

          {!agentCap && !candCap && !thinking && (
            <p className="empty">Waiting for the panel to open.</p>
          )}
        </div>

        <aside className="rail">
          <div className="rail-block">
            <div className="panelcard-head">
              <span className="sec-label">Coverage</span>
            </div>
            {coverage.length === 0 ? (
              <p className="empty">Fills as you answer.</p>
            ) : (
              coverage.map((c) => (
                <div className="cov-row" key={c.key}>
                  <span className="cov-name" title={c.name}>{c.name}</span>
                  <span className="cov-bar">
                    <span className="cov-fill" style={{ width: `${pct(c.value)}%` }} />
                  </span>
                  <span className="cov-val">{pct(c.value)}%</span>
                </div>
              ))
            )}
          </div>

          <div className="rail-block rail-grow">
            <div className="panelcard-head">
              <span className="sec-label">Evidence</span>
              {contradictions > 0 && (
                <span className="flag-count">
                  <i className="ph ph-warning-diamond" aria-hidden="true" />
                  {contradictions} conflict{contradictions > 1 ? "s" : ""}
                </span>
              )}
            </div>
            {claims.length === 0 ? (
              <p className="empty">No claims yet.</p>
            ) : (
              <div className="claimlist">
                {claims.slice().reverse().map((cl, i) => (
                  <div
                    key={`${cl.turn}-${cl.competency}-${(cl.text || "").slice(0, 24)}-${i}`}
                    className={
                      "claim landed " +
                      (cl.contradicts ? "contra" : cl.status === "VAGUE" ? "vague" : "solid")
                    }
                  >
                    <div className="claim-top">
                      <span className="claim-status">
                        {cl.contradicts ? "CONFLICT" : cl.status}
                      </span>
                      <span className="claim-text">{cl.text}</span>
                    </div>
                    <div className="claim-meta">
                      {cl.competency} / {pct(cl.strength)}% / turn {cl.turn}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </aside>
      </div>

      {timeline}

      {/* This is the only status surface during a live interview: the control
          strip is not rendered here, so interrupt feedback and the
          share-your-whole-screen rejection have nowhere else to land. */}
      {status && (
        <div className="stage-status">
          {busy && <i className="ph ph-circle-notch spin" aria-hidden="true" />}
          {status}
        </div>
      )}
    </div>
  );
}
