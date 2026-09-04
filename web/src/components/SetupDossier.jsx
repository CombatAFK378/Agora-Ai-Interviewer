import { agentHue } from "../lib/agents";
import { pct, prettyKey } from "../lib/format";

export default function SetupDossier({
  jd, setJd, resume, setResume, jdMeta, setJdMeta, resumeMeta, setResumeMeta,
  uploading, uploadPdf, parsing, previewDossier, dossier, joined, nameOf,
}) {
  return (
    <div className="setup">
      <div className="setup-head">
        <strong>Interview dossier</strong>
        <p className="setup-sub">
          Upload the job description and the candidate's resume as PDFs, or paste the text.
          The panel, the competency weights and the questions all adapt to the role.
          Leave both blank for a generic panel.
        </p>
      </div>

      <div className="setup-cols">
        <div className="setup-col">
          <label className="filebtn">
            <input type="file" accept="application/pdf,.pdf"
                   onChange={(e) => uploadPdf("jd", e.target)}
                   disabled={joined || uploading === "jd"} />
            <i className="ph ph-paperclip" aria-hidden="true" />
            {uploading === "jd"
              ? "Reading"
              : jdMeta
              ? `${jdMeta.filename}, ${jdMeta.pages}p`
              : "Upload job description"}
          </label>
          <textarea
            className="setup-ta"
            placeholder="or paste the job description here"
            value={jd}
            onChange={(e) => { setJd(e.target.value); setJdMeta(null); }}
            disabled={joined}
            aria-label="Job description"
          />
        </div>

        <div className="setup-col">
          <label className="filebtn">
            <input type="file" accept="application/pdf,.pdf"
                   onChange={(e) => uploadPdf("resume", e.target)}
                   disabled={joined || uploading === "resume"} />
            <i className="ph ph-paperclip" aria-hidden="true" />
            {uploading === "resume"
              ? "Reading"
              : resumeMeta
              ? `${resumeMeta.filename}, ${resumeMeta.pages}p`
              : "Upload resume"}
          </label>
          <textarea
            className="setup-ta"
            placeholder="or paste the candidate resume here"
            value={resume}
            onChange={(e) => { setResume(e.target.value); setResumeMeta(null); }}
            disabled={joined}
            aria-label="Candidate resume"
          />
        </div>
      </div>

      <div className="setup-actions">
        <button onClick={previewDossier} disabled={parsing || (!jd.trim() && !resume.trim())}>
          <i className="ph ph-scan" aria-hidden="true" />
          {parsing ? "Reading" : "Preview dossier"}
        </button>
      </div>

      {dossier && (
        <div className="dossier">
          <div className="dossier-role">
            {dossier.seniority} {dossier.role}
            {dossier.candidate_name && (
              <span className="dossier-sum"> / candidate: {dossier.candidate_name}</span>
            )}
          </div>
          {dossier.summary && <p className="setup-sub">{dossier.summary}</p>}

          {(dossier.focus || []).length > 0 && (
            <div className="dossier-block">
              <span className="sec-label">Role focus</span>
              <div className="pillrow">
                {dossier.focus.map((f, i) => <span className="pill" key={i}>{f}</span>)}
              </div>
            </div>
          )}

          <div className="dossier-block">
            <span className="sec-label">Panel for this role</span>
            <div className="pillrow">
              {(dossier.panel || []).map((a) => (
                <span className="pill" key={a} style={{ "--hue": agentHue(a) }}>
                  <span className="pill-dot" />
                  {nameOf(a)}
                </span>
              ))}
            </div>
          </div>

          {Object.keys(dossier.competency_weights || {}).length > 0 && (
            <div className="dossier-block">
              <span className="sec-label">Competency weights</span>
              <div className="pillrow">
                {Object.entries(dossier.competency_weights).map(([k, v]) => (
                  <span className="pill pill-w" key={k}>
                    {prettyKey(k)} <b>{pct(v)}%</b>
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Per-interviewer rubric: what STRONG means for this specific role.
              The dossier has always returned this and it was never displayed. */}
          {Object.keys(dossier.rubrics || {}).length > 0 && (
            <div className="dossier-block">
              <span className="sec-label">What each interviewer is looking for</span>
              <div className="rubric-list">
                {Object.entries(dossier.rubrics).map(([id, text]) => (
                  <div className="rubric" key={id} style={{ "--hue": agentHue(id) }}>
                    <span className="rubric-who">
                      <span className="pill-dot" />
                      {nameOf(id)}
                    </span>
                    <span className="rubric-text">{text}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {(dossier.resume_claims || []).length > 0 && (
            <div className="dossier-block">
              <span className="sec-label">
                Resume claims the panel will test ({dossier.resume_claims.length})
              </span>
              <div className="claimcheck">
                {dossier.resume_claims.slice(0, 8).map((c, i) => (
                  <div className="claimcheck-row" key={i}>
                    <b>{c.text}</b> <i>{c.competency}</i>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
