import { agentHue } from "../lib/agents";

// The live coding round. Two paths exist server-side (Gemini Live, and a
// snapshot fallback); this renders whichever one is active.
export default function CodingRound({
  mode, task, active, sharing, screenRead,
  onStartGemini, onStopGemini, onShareScreen, onStopSharing, codingName, status,
}) {
  const isGemini = mode === "gemini";
  const running = isGemini ? active : sharing;

  return (
    <div className="coding" style={{ "--hue": agentHue("coding") }}>
      <div className="coding-head">
        <strong>
          <i className="ph ph-code" aria-hidden="true" /> Live coding round with {codingName}
        </strong>
        {running ? (
          <button className="ghost" onClick={isGemini ? onStopGemini : onStopSharing}>
            {isGemini ? "End coding round" : "Stop sharing"}
          </button>
        ) : (
          <button className="btn-primary" onClick={isGemini ? onStartGemini : onShareScreen}>
            <i className="ph ph-monitor-arrow-up" aria-hidden="true" />
            Share entire screen and start
          </button>
        )}
      </div>

      <div className="coding-task">{task}</div>

      <p className="coding-note">
        {running
          ? isGemini
            ? `Live with ${codingName}. He can see your screen and hear you, so talk through your code as you go.`
            : `Sharing your whole screen. ${codingName} can see your editor and will react to your code.`
          : `Use headphones to avoid echo. You must share your ENTIRE screen, a window or tab share is rejected, so ${codingName} can watch you work. Think out loud as you go.`}
      </p>

      {/* A rejected share sets status and, during a live interview, the control
          strip that used to show it is not rendered. Surface it beside the
          button that caused it. */}
      {status && <p className="coding-note">{status}</p>}

      {/* The vision model's running description of the editor. It used to go
          to the console log only. */}
      {screenRead && (
        <div className="coding-see">
          <i className="ph ph-eye" aria-hidden="true" />
          <span>{screenRead}</span>
        </div>
      )}
    </div>
  );
}

// Shown once the round closes. The summary is produced by the coding model and
// posted by this client, and until now was displayed nowhere: a flagged round
// looked identical to a clean one.
export function CodingVerdict({ verdict, summary, codingName }) {
  if (!verdict) return null;
  const flagged = verdict === "cheating";
  return (
    <div className={"verdict " + (flagged ? "flagged" : "ok")}>
      <div className="verdict-head">
        <i className={"ph-fill " + (flagged ? "ph-warning-diamond" : "ph-check-circle")}
           aria-hidden="true" />
        {flagged
          ? `${codingName} flagged the coding round`
          : `Coding round complete`}
      </div>
      {summary && <p className="verdict-text">{summary}</p>}
    </div>
  );
}
