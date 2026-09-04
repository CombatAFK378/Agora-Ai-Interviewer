import { prettyEnum, shortTime } from "../lib/format";

export default function Dashboard({ interviews, onOpen }) {
  return (
    <div className="dash">
      <div className="rep-section-head">
        <span className="sec-label">Past interviews</span>
        <span className="sec-count">{interviews.length}</span>
      </div>

      {interviews.length === 0 && (
        <p className="empty">
          No interviews yet. Run one from the Live interview tab.
        </p>
      )}

      {interviews.map((it) => (
        <button className="dash-row" key={it.interview_id} onClick={() => onOpen(it)}>
          <div className="dash-main">
            <span className="dash-name">{it.candidate_name || "Unnamed candidate"}</span>
            <span className="dash-role">{it.role || "Role not recorded"}</span>
            {/* The one-line panel summary ships on every row and used to be
                dropped by the renderer. */}
            {it.headline && <span className="dash-headline">{it.headline}</span>}
          </div>
          <div className="dash-meta">
            <span className={"rec rec-" + it.recommendation}>
              {prettyEnum(it.recommendation)}
            </span>
            {it.override && (
              <span className={"rec rec-" + it.override}>
                <i className="ph ph-arrow-right" aria-hidden="true" />
                {prettyEnum(it.override)}
              </span>
            )}
            <span className="dash-date">{shortTime(it.created_at)}</span>
          </div>
        </button>
      ))}
    </div>
  );
}
