import { agentHue } from "../lib/agents";

export default function Captions({ agentCap, candCap }) {
  if (!agentCap && !candCap) return null;
  return (
    <div className="captions">
      {/* Hue from the caption's own agent: `speaking` clears on idle while the
          caption remains, and agentHue(null) resolves to another
          interviewer's colour. */}
      {agentCap && (
        <div className="cap agent" style={{ "--hue": agentHue(agentCap.id) }}>
          <div className="who">
            {agentCap.name}, {agentCap.title}
          </div>
          <div className="txt">{agentCap.text}</div>
        </div>
      )}
      {candCap && (
        <div className="cap cand">
          <div className="who">You said</div>
          <div className="txt">{candCap}</div>
        </div>
      )}
    </div>
  );
}
