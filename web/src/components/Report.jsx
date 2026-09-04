import ConfidenceChart from "./ConfidenceChart";
import { agentHue } from "../lib/agents";
import { pct, prettyKey, prettyEnum } from "../lib/format";

function ScoreCard({ s, nameOf, titleOf, i, reveal }) {
  const hue = agentHue(s.agent_id);
  const evidence = s.evidence || [];
  const turns = evidence
    .filter((e) => typeof e === "string" && e.toLowerCase().startsWith("turn"))
    .map((e) => e.replace(/[^0-9]/g, ""))
    .filter(Boolean);
  const claimRefs = evidence.length - turns.length;

  return (
    <div
      className={"scorecard" + (reveal ? " reveal" : "")}
      style={{ "--hue": hue, "--i": i }}
    >
      <div className="sc-head">
        <span className="sc-name">{nameOf(s.agent_id)}</span>
        <span className={"chip " + (s.conviction === "STRONG" ? "strong" : "neutral")}>
          {s.conviction}
        </span>
      </div>
      <div className="sc-role">{titleOf(s.agent_id)}</div>
      <div className="sc-overall">
        {pct(s.overall)}<span>/100</span>
      </div>

      {Object.entries(s.competency_scores || {}).map(([k, v]) => (
        <div className="sc-comp" key={k}>
          <span className="sc-comp-name" title={prettyKey(k)}>{prettyKey(k)}</span>
          <span className="sc-bar">
            <span className="sc-fill" style={{ width: `${pct(v)}%` }} />
          </span>
          <span className="sc-val">{pct(v)}</span>
        </div>
      ))}

      {s.rationale && <div className="sc-rat">{s.rationale}</div>}

      {/* Evidence refs ship on every score and were never drawn. This is the
          backlink that makes a number auditable. */}
      {evidence.length > 0 && (
        <div className="sc-ev">
          {claimRefs > 0 && (
            <span className="evchip">{claimRefs} claim{claimRefs > 1 ? "s" : ""}</span>
          )}
          {turns.length > 0 && (
            <span className="evchip">turns {turns.slice(0, 6).join(", ")}</span>
          )}
        </div>
      )}
    </div>
  );
}

export default function Report({ report, overrides, nameOf, titleOf, topBar, fresh, children }) {
  const conclusion = report.conclusion || {};
  const coverage = report.coverage || {};
  const coverageKeys = Object.keys(coverage);

  // debate_statement carries a round, and scoring puts the widest-diverging
  // pair first. The old view flattened all of it into one list.
  const rounds = (report.debate || []).reduce((acc, d) => {
    const r = typeof d.round === "number" ? d.round : 1;
    (acc[r] = acc[r] || []).push(d);
    return acc;
  }, {});
  const roundKeys = Object.keys(rounds).sort((a, b) => a - b);

  return (
    <div className="report">
      {topBar}

      <div className="rec-line">
        <span className={"rec rec-" + conclusion.recommendation}>
          {prettyEnum(conclusion.recommendation)}
        </span>
        {overrides.length > 0 && (
          <span className={"rec rec-" + overrides[overrides.length - 1].decision}>
            <i className="ph ph-arrow-right" aria-hidden="true" />
            overridden to {prettyEnum(overrides[overrides.length - 1].decision)}
          </span>
        )}
      </div>

      <p className="headline">{conclusion.headline}</p>

      <section className="rep-section">
        <div className="rep-section-head">
          <span className="sec-label">Panel scores, locked independently</span>
          <span className="sec-count">
            {(report.scores || []).length} interviewers, none saw another's sheet
          </span>
        </div>
        <div className="score-grid">
          {(report.scores || []).map((s, i) => (
            <ScoreCard key={s.agent_id} s={s} nameOf={nameOf} titleOf={titleOf}
                       i={i} reveal={fresh} />
          ))}
        </div>
      </section>

      {/* report.coverage ships on both conclude and open, and had zero
          references in the previous build, so a reopened interview showed
          no coverage at all. */}
      {coverageKeys.length > 0 && (
        <section className="rep-section">
          <div className="rep-section-head">
            <span className="sec-label">Final evidence coverage</span>
          </div>
          <div className="panelcard">
            {coverageKeys.map((k) => (
              <div className="cov-row" key={k}>
                <span className="cov-name">{prettyKey(k)}</span>
                <span className="cov-bar">
                  <span className="cov-fill" style={{ width: `${pct(coverage[k])}%` }} />
                </span>
                <span className="cov-val">{pct(coverage[k])}%</span>
              </div>
            ))}
          </div>
        </section>
      )}

      {report.trajectory && report.trajectory.length >= 2 && (
        <section className="rep-section">
          <div className="rep-section-head">
            <span className="sec-label">Evidence coverage across the interview</span>
          </div>
          <ConfidenceChart trajectory={report.trajectory} />
        </section>
      )}

      {(report.debate || []).length > 0 && (
      <section className="rep-section">
        <div className="rep-section-head">
          <span className="sec-label">Debate</span>
          <span className="sec-count">
            {roundKeys.length > 1 ? `${roundKeys.length} rounds` : ""}
          </span>
        </div>
        {roundKeys.map((r) => (
          <div key={r}>
            {roundKeys.length > 1 && <div className="round-label">Round {r}</div>}
            {rounds[r].map((d, i) => (
              <div className="deb" key={i}>
                <span className={"deb-act " + (d.rejected ? "rej" : (d.action || "").toLowerCase())}>
                  {d.rejected ? "held" : (d.action || "").toLowerCase()}
                </span>
                <div>
                  <span className="deb-who" style={{ "--hue": agentHue(d.agent_id) }}>
                    <span className="pill-dot" />
                    {nameOf(d.agent_id)}
                    {d.action === "MOVE" && !d.rejected && (
                      <span className="deb-move">
                        {pct(d.score_before)} to {pct(d.score_after)}
                      </span>
                    )}
                    {d.rejected && (
                      <span className="sec-count">conviction STRONG, move rejected</span>
                    )}
                  </span>
                  <span className="deb-text">{d.statement}</span>
                </div>
              </div>
            ))}
          </div>
        ))}
      </section>
      )}

      <section className="rep-section">
        <div className="rep-section-head">
          <span className="sec-label">Conclusion</span>
        </div>
        <p className="rep-reason">{conclusion.reasoning}</p>
        {(conclusion.unresolved || []).length > 0 && (
          <>
            <div className="ask-sub">Unresolved, a human should verify</div>
            {conclusion.unresolved.map((u, i) => (
              <div className="unres" key={i}>
                <span className="unres-item">{u.item}</span>
                <span className="unres-ev">{u.evidence}</span>
              </div>
            ))}
          </>
        )}
      </section>

      <div className="rep-hash">
        <i className="ph ph-lock-simple" aria-hidden="true" />
        locked record, SHA-256 {(report.locked_hash || "").slice(0, 32)}
      </div>

      {children}
    </div>
  );
}
