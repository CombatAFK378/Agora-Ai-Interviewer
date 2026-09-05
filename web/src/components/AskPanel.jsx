import { agentHue } from "../lib/agents";
import { prettyEnum, shortTime } from "../lib/format";

export default function AskPanel({
  report, nameOf, talking, speaking, onJoinVoice, onToggleMic, micMuted,
  qa, askTarget, setAskTarget, askQ, setAskQ, asking, onAsk,
  cfAgent, setCfAgent, cfTurn, setCfTurn, cfHypo, setCfHypo, onCounterfactual,
  ovDecision, setOvDecision, ovReason, setOvReason, onOverride, overrides,
}) {
  const scores = report.scores || [];

  return (
    <section className="rep-section askpanel">
      <div className="rep-section-head">
        <span className="sec-label">Ask the panel</span>
        <span className="sec-count">answers come from the locked record only</span>
      </div>

      <div className="voice-join">
        {!talking ? (
          <button className="btn-primary" onClick={onJoinVoice}>
            <i className="ph-fill ph-microphone" aria-hidden="true" />
            Join and talk to the panel
          </button>
        ) : (
          <>
            <span className="talking-note">
              <i className="ph-fill ph-microphone" aria-hidden="true" />
              In voice with the panel, just speak.
              {speaking ? ` ${nameOf(speaking)} is answering.` : ""}
            </span>
            <button
              className={"btn-flag" + (micMuted ? " muted" : "")}
              onClick={onToggleMic}
              aria-pressed={micMuted}
              title={micMuted ? "Microphone is off — click to unmute" : "Mute your microphone"}
            >
              <i className={"ph-fill " + (micMuted ? "ph-microphone-slash" : "ph-microphone")}
                 aria-hidden="true" />
              {micMuted ? "Mic off" : "Mic on"}
            </button>
          </>
        )}
        <span className="voice-or">or type</span>
      </div>

      {qa.map((x, i) => (
        <div className="qa" key={i} style={{ "--hue": agentHue(x.agentId) }}>
          {(x.mode || x.ts) && (
            <div className="qa-meta">
              {x.mode === "addressed" ? "addressed" : x.mode ? "open" : ""}
              {x.ts ? ` ${shortTime(x.ts)}` : ""}
            </div>
          )}
          <div className="qa-q">{x.q}</div>
          <div className="qa-a"><b>{x.by}</b> {x.a}</div>
        </div>
      ))}

      <div className="ask-row">
        <select value={askTarget} onChange={(e) => setAskTarget(e.target.value)}
                aria-label="Who to ask">
          <option value="">Open, answered by the host</option>
          {scores.map((s) => (
            <option key={s.agent_id} value={s.agent_id}>{nameOf(s.agent_id)}</option>
          ))}
        </select>
        <input
          value={askQ}
          placeholder="Why did you flag this candidate?"
          onChange={(e) => setAskQ(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && onAsk()}
          aria-label="Question for the panel"
        />
        <button className="btn-primary" onClick={onAsk} disabled={asking}>
          {asking ? "Asking" : "Ask"}
        </button>
      </div>

      <div className="ask-sub">Counterfactual, what if they had said this</div>
      <div className="ask-row">
        <select value={cfAgent} onChange={(e) => setCfAgent(e.target.value)}
                aria-label="Interviewer to re-score">
          <option value="">Interviewer</option>
          {scores.map((s) => (
            <option key={s.agent_id} value={s.agent_id}>{nameOf(s.agent_id)}</option>
          ))}
        </select>
        <input className="cf-turn" type="number" value={cfTurn} placeholder="turn"
               onChange={(e) => setCfTurn(e.target.value)} aria-label="Turn number" />
        <input value={cfHypo} placeholder="A better answer they could have given"
               onChange={(e) => setCfHypo(e.target.value)} aria-label="Hypothetical answer" />
        <button onClick={onCounterfactual} disabled={asking}>Re-score</button>
      </div>

      <div className="ask-sub">Override the recommendation</div>
      <div className="ask-row">
        <select value={ovDecision} onChange={(e) => setOvDecision(e.target.value)}
                aria-label="Override decision">
          <option value="">Decision</option>
          {["PROCEED", "PROCEED_FLAGGED", "INSUFFICIENT_SIGNAL", "DECLINE"].map((d) => (
            <option key={d} value={d}>{prettyEnum(d)}</option>
          ))}
        </select>
        <input value={ovReason} placeholder="Reason, kept on the record"
               onChange={(e) => setOvReason(e.target.value)} aria-label="Override reason" />
        <button className="btn-flag" onClick={onOverride}>Log override</button>
      </div>

      {overrides.map((o, i) => (
        <div className="ovr" key={i}>
          <b>{prettyEnum(o.original_recommendation)}</b> overridden to{" "}
          <b>{prettyEnum(o.decision)}</b>
          {o.reason ? `. ${o.reason}` : ""}
          <span className="ovr-note">
            The original recommendation stays on the record.
            {o.ts ? ` Logged ${shortTime(o.ts)}.` : ""}
          </span>
        </div>
      ))}
    </section>
  );
}
