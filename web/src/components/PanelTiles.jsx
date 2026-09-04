import { agentHue, agentIcon, initials } from "../lib/agents";

// The five interviewers. Colour is identity and comes from the agent id, so a
// person keeps their colour whatever the dossier does to the panel.
export default function PanelTiles({ agents, speaking, thinking, activePanel, finished }) {
  return (
    <div className="tiles">
      {agents.map((a) => {
        const done = finished.includes(a.id);
        const offPanel = !done && activePanel.length > 0 && !activePanel.includes(a.id);
        const dim = offPanel || done;
        const isSpeaking = !dim && speaking === a.id;
        const cls =
          "tile" +
          (dim ? " off-panel" : "") +
          (isSpeaking ? " speaking" : !dim && thinking ? " thinking" : "");

        return (
          <div key={a.id} className={cls} style={{ "--hue": agentHue(a.id) }}>
            <div className="avatar">
              <i className={`ph-fill ${agentIcon(a.id)}`} aria-hidden="true" />
              <span className="icon-fallback">{initials(a.name)}</span>
            </div>
            <div className="name">{a.name}</div>
            <div className="role">{a.title}</div>
            <div className="tile-state">
              {isSpeaking ? (
                <><span className="live-dot" />speaking</>
              ) : done ? (
                "round complete"
              ) : offPanel ? (
                "not on this panel"
              ) : (
                ""
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
