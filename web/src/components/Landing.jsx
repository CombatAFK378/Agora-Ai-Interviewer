import { useEffect, useRef, useState } from "react";
import { agentHue, agentIcon } from "../lib/agents";

// The front door.
//
// Everything on this page is the real interface, not a picture of it: the same
// seat component, the same claim rows, the same tokens. The only thing standing
// in is the voice amplitude, which is a local loop here rather than a live
// microphone, because there is no interview running on a landing page.

const PANEL = [
  { id: "hiring_manager", name: "Maya",   title: "Hiring Manager" },
  { id: "technical",      name: "Ethan",  title: "Technical" },
  { id: "product",        name: "Sophia", title: "Product" },
  { id: "customer",       name: "Nina",   title: "Customer" },
  { id: "coding",         name: "Liam",   title: "Coding" },
];

// Representative of what the panel actually asks. Not a transcript.
const BEATS = [
  { who: "technical", ask: "You said the retry is safe. Safe for who?" },
  { who: "product",   ask: "That works. What does it cost the customer when it fires twice?" },
  { who: "customer",  ask: "I have had three duplicate charges this week. Talk to me." },
  { who: "hiring_manager", ask: "Who did you have to convince, and what did they say no to?" },
  { who: "coding",    ask: "Walk me through the branch you did not take." },
];

const SCORES = [
  { id: "hiring_manager", n: 72, conviction: "STRONG" },
  { id: "technical",      n: 81, conviction: "STRONG" },
  { id: "product",        n: 44, conviction: "NEUTRAL" },
  { id: "customer",       n: 51, conviction: "NEUTRAL" },
  { id: "coding",         n: 77, conviction: "STRONG" },
];

const BIDS = [
  { id: "hiring_manager", interest: 0.31, gap: 0.22, recency: 1.0 },
  { id: "technical",      interest: 0.88, gap: 0.14, recency: 0.5 },
  { id: "product",        interest: 0.74, gap: 0.81, recency: 1.0 },
  { id: "customer",       interest: 0.42, gap: 0.66, recency: 0.8 },
  { id: "coding",         interest: 0.19, gap: 0.35, recency: 1.0 },
];
const LAMBDA = 1.2;
const priority = (b) => b.interest * (1 + LAMBDA * b.gap) * b.recency;

const CLAIMS = [
  { status: "SOLID", text: "Named the p99 target before being asked", meta: "system_design / 77% / turn 6" },
  { status: "VAGUE", text: "Would “probably loop in whoever owns billing”", meta: "ownership / 29% / turn 7" },
  { status: "CONFLICT", text: "Says the retry is safe, earlier said charges are not idempotent", meta: "technical_depth / 61% / turn 8" },
  { status: "SOLID", text: "Adds an idempotency key on the charge path", meta: "technical_depth / 84% / turn 9" },
];

// Reveal on scroll, once, and only when the viewer has not asked for less motion.
function useReveal() {
  const ref = useRef(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      el.classList.add("in");
      return;
    }
    const io = new IntersectionObserver(
      (entries) => entries.forEach((e) => { if (e.isIntersecting) { el.classList.add("in"); io.disconnect(); } }),
      { threshold: 0.25 }
    );
    io.observe(el);
    return () => io.disconnect();
  }, []);
  return ref;
}

function Section({ label, title, body, children, wide }) {
  const ref = useReveal();
  return (
    <section className={"lp-section" + (wide ? " wide" : "")} ref={ref}>
      <div className="lp-section-copy">
        <span className="lp-label">{label}</span>
        <h2 className="lp-h2">{title}</h2>
        <p className="lp-body">{body}</p>
      </div>
      <div className="lp-section-visual">{children}</div>
    </section>
  );
}

export default function Landing({ onEnter }) {
  const [beat, setBeat] = useState(0);
  const stageRef = useRef(null);

  // Rotate the floor between interviewers, the way the orchestrator does.
  useEffect(() => {
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduce) return;
    const t = setInterval(() => setBeat((b) => (b + 1) % BEATS.length), 3600);
    return () => clearInterval(t);
  }, []);

  // Stand-in for the analyser. In the room this is a real microphone.
  useEffect(() => {
    const el = stageRef.current;
    if (!el) return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      el.style.setProperty("--level", "0.4");
      return;
    }
    let raf = 0, t = 0;
    const tick = () => {
      t += 0.055;
      const v = Math.max(0, Math.sin(t) * 0.42 + Math.sin(t * 3.3) * 0.3 + Math.sin(t * 7.9) * 0.16);
      el.style.setProperty("--level", v.toFixed(3));
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  const active = BEATS[beat];
  const winner = BIDS.reduce((a, b) => (priority(a) >= priority(b) ? a : b));
  const maxP = priority(winner);
  const nameOf = (id) => (PANEL.find((p) => p.id === id) || {}).name || id;

  return (
    <div className="lp">
      <header className="lp-nav">
        <span className="lp-mark"><b>968</b>ms</span>
        <button className="btn-primary" onClick={onEnter}>
          Start an interview
          <i className="ph ph-arrow-right" aria-hidden="true" />
        </button>
      </header>

      <section className="lp-hero">
        <h1 className="lp-h1">
          Five interviewers<br />who disagree about you.
        </h1>
        <p className="lp-lede">
          A live voice panel with separate objectives. They score blind, and only
          when the record locks do they see each other and argue it out.
        </p>

        <div className="lp-stage" ref={stageRef}>
          <div className="seats">
            {PANEL.map((a) => {
              const on = active.who === a.id;
              return (
                <div key={a.id} className={"seat" + (on ? " on" : "")}
                     style={{ "--hue": agentHue(a.id) }}>
                  <div className="seat-ring">
                    <div className="seat-face">
                      <i className={`ph-fill ${agentIcon(a.id)}`} aria-hidden="true" />
                      <span className="icon-fallback">{a.name.slice(0, 1)}</span>
                    </div>
                  </div>
                  <div className="seat-name">{a.name}</div>
                  <div className="seat-role">{a.title}</div>
                </div>
              );
            })}
          </div>

          <div className="lp-said" key={beat} style={{ "--hue": agentHue(active.who) }}>
            <div className="said-who">
              {nameOf(active.who)}
              <span>has the floor</span>
            </div>
            <p className="said-text">{active.ask}</p>
          </div>
        </div>
      </section>

      <Section
        label="Turn taking"
        title="Nobody decides who speaks next."
        body="Every interviewer bids on every answer. The floor goes to interest, times how much of their competency is still uncovered, times a penalty for having just spoken. It is arithmetic, it is written to the audit log, and replaying it four days later returns the same winner."
      >
        <div className="lp-bids">
          <div className="lp-formula">
            priority = interest &times; (1 + &lambda; &middot; gap) &times; recency
          </div>
          {BIDS.map((b) => {
            const p = priority(b);
            const won = b.id === winner.id;
            return (
              <div className={"lp-bid" + (won ? " won" : "")} key={b.id}
                   style={{ "--hue": agentHue(b.id) }}>
                <span className="lp-bid-name">{nameOf(b.id)}</span>
                <span className="lp-bid-bar">
                  <span className="lp-bid-fill" style={{ width: `${(p / maxP) * 100}%` }} />
                </span>
                <span className="lp-bid-val">{p.toFixed(2)}</span>
              </div>
            );
          })}
          <div className="lp-bid-note">
            Sophia wins on a lower bid than Ethan, because two thirds of her
            competency is still unmeasured and he just spoke.
          </div>
        </div>
      </Section>

      <Section
        label="Blind scoring"
        title="They cannot read each other's sheet."
        body="For the whole interview each interviewer keeps a private score and private notes. No agent can see another's, and the prompt builder raises if one tries. The sheets turn face up only when the record locks."
      >
        <div className="lp-sheets">
          {SCORES.map((s, i) => (
            <div className="lp-sheet" key={s.id} style={{ "--hue": agentHue(s.id), "--i": i }}>
              <div className="lp-sheet-back">
                <span className="sheet-line" /><span className="sheet-line" /><span className="sheet-line" />
              </div>
              <div className="lp-sheet-front">
                <span className="lp-sheet-name">{nameOf(s.id)}</span>
                <span className="lp-sheet-score">{s.n}</span>
                <span className={"chip " + (s.conviction === "STRONG" ? "strong" : "neutral")}>
                  {s.conviction}
                </span>
              </div>
            </div>
          ))}
        </div>
      </Section>

      <Section
        label="Evidence"
        title="Every number has a receipt."
        body="Claims are pulled out of the candidate's answers as they speak, marked solid or vague, and tied to the turn that produced them. When an answer contradicts something said earlier, or something on the resume, the panel is told."
      >
        <div className="lp-claims">
          {CLAIMS.map((c, i) => (
            <div className={"claim " + c.status.toLowerCase().replace("conflict", "contra")}
                 key={i} style={{ "--i": i }}>
              <div className="claim-top">
                <span className="claim-status">{c.status}</span>
                <span className="claim-text">{c.text}</span>
              </div>
              <div className="claim-meta">{c.meta}</div>
            </div>
          ))}
        </div>
      </Section>

      <Section
        label="The record"
        title="Sealed, and answerable months later."
        body="Transcript, claims, scores and convictions are hashed at lock. A recruiter can rejoin afterwards and cross-examine any interviewer about its reasoning, and ask what a different answer would have changed. None of it can move the locked score."
      >
        <div className="lp-locked">
          <div className="rep-hash">
            <i className="ph ph-lock-simple" aria-hidden="true" />
            locked record, SHA-256 8f2ae41cb7d095e6104b3fa27c88d1e0
          </div>
          <div className="qa">
            <div className="qa-q">Why did you score customer impact so low?</div>
            <div className="qa-a">
              <b>Sophia</b> Because across fourteen turns he described what the
              system does four times, and never once what a person experiences
              when it fails.
            </div>
          </div>
        </div>
      </Section>

      <section className="lp-cta">
        <h2 className="lp-h2">Give it a job description.</h2>
        <p className="lp-body">
          The panel, the competency weights and the questions are all built from
          the role. Bring a resume too and it will check what was written against
          what gets said.
        </p>
        <button className="btn-primary lp-cta-btn" onClick={onEnter}>
          Start an interview
          <i className="ph ph-arrow-right" aria-hidden="true" />
        </button>
      </section>

      <footer className="lp-foot">
        <span>968ms</span>
        <span>Every interviewer is AI, and every candidate is told so before a word is spoken.</span>
      </footer>
    </div>
  );
}
