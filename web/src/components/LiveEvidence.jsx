import { pct } from "../lib/format";

// Competency coverage and the evidence ledger.
//
// These were previously behind a collapsed toggle labelled "Panel internals"
// and styled as a debug drawer. They are the visible proof that the panel is
// reasoning from evidence rather than improvising, so they sit in the room.
export default function LiveEvidence({ coverage, claims, contradictions }) {
  // Contradicted claims are not solid; see Stage.jsx.
  const solidCount = claims.filter((c) => c.status === "SOLID" && !c.contradicts).length;

  return (
    <div className="evidence">
      <section className="panelcard">
        <div className="panelcard-head">
          <span className="sec-label">Competency coverage</span>
          <span className="sec-count">
            {coverage.length ? `${coverage.length} tracked` : ""}
          </span>
        </div>
        {coverage.length === 0 ? (
          <p className="empty">Nothing measured yet. Coverage fills as the candidate answers.</p>
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
      </section>

      <section className="panelcard">
        <div className="panelcard-head">
          <span className="sec-label">Evidence ledger</span>
          <span className="sec-count">
            {claims.length > 0 && `${solidCount} solid of ${claims.length}`}
            {contradictions > 0 && (
              <span className="flag-count" style={{ marginLeft: 10 }}>
                <i className="ph ph-warning-diamond" aria-hidden="true" />
                {contradictions} contradiction{contradictions > 1 ? "s" : ""}
              </span>
            )}
          </span>
        </div>
        {claims.length === 0 ? (
          <p className="empty">No claims extracted yet.</p>
        ) : (
          <div className="claimlist">
            {claims.slice().reverse().map((cl, i) => (
              <div
                key={`${cl.turn}-${cl.competency}-${(cl.text || "").slice(0, 24)}-${i}`}
                className={
                  "claim " +
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
      </section>
    </div>
  );
}
