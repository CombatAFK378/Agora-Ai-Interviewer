import { seriesColor } from "../lib/agents";
import { prettyKey } from "../lib/format";

// Mean evidence coverage over the interview, with faint per-competency lines.
// Pure inline SVG, no chart library.
//
// Two fixes over the previous version: the X axis is the real turn number that
// ships on every trajectory point (it used to be the array index), and the
// claim volume that also ships on every point is drawn behind the line.
export default function ConfidenceChart({ trajectory }) {
  if (!trajectory || trajectory.length < 2) return null;

  const W = 720, H = 220, padL = 34, padR = 16, padT = 14, padB = 26;
  const n = trajectory.length;

  const turns = trajectory.map((p, i) => (typeof p.turn === "number" ? p.turn : i));
  const tMin = turns[0];
  const tMax = turns[n - 1] === tMin ? tMin + 1 : turns[n - 1];

  const x = (i) => padL + ((turns[i] - tMin) / (tMax - tMin)) * (W - padL - padR);
  const y = (v) => H - padB - Math.max(0, Math.min(1, v)) * (H - padT - padB);

  const keys = Object.keys(trajectory[n - 1].coverage || {});
  const line = (getter) => trajectory.map((p, i) => `${x(i)},${y(getter(p))}`).join(" ");

  // One point without a mean would put NaN into the path and silently erase
  // the line, the fill and every dot while the legend still claims otherwise.
  const meanPts = line((p) => (typeof p.mean === "number" ? p.mean : 0));
  const areaPts = `${x(0)},${H - padB} ${meanPts} ${x(n - 1)},${H - padB}`;

  // Claim volume, normalised against its own peak so it reads as texture
  // behind the measurement rather than competing with it.
  const maxClaims = Math.max(1, ...trajectory.map((p) => p.claims || 0));
  const volPts =
    `${x(0)},${H - padB} ` +
    trajectory.map((p, i) => `${x(i)},${y(((p.claims || 0) / maxClaims) * 0.55)}`).join(" ") +
    ` ${x(n - 1)},${H - padB}`;

  return (
    <div className="chart">
      <svg viewBox={`0 0 ${W} ${H}`} className="chart-svg" role="img"
           aria-label="Evidence coverage across the interview">
        {[0, 0.25, 0.5, 0.75, 1].map((g) => (
          <g key={g}>
            <line x1={padL} y1={y(g)} x2={W - padR} y2={y(g)} className="grid" />
            <text x={6} y={y(g) + 3} className="axis">{Math.round(g * 100)}</text>
          </g>
        ))}

        <polygon points={volPts} className="chart-vol" />

        {keys.map((k, i) => (
          <polyline
            key={k}
            points={line((p) => (p.coverage || {})[k] ?? 0)}
            fill="none"
            stroke={seriesColor(i)}
            strokeWidth="1"
            opacity="0.5"
          />
        ))}

        <polygon points={areaPts} className="chart-area" />
        <polyline points={meanPts} className="chart-mean" fill="none" />
        {trajectory.map((p, i) => (
          <circle key={i} cx={x(i)} cy={y(typeof p.mean === "number" ? p.mean : 0)}
                  r="2.5" className="chart-dot" />
        ))}

        <text x={padL} y={H - 8} className="axis">turn {tMin}</text>
        <text x={W - padR} y={H - 8} className="axis" textAnchor="end">turn {tMax}</text>
      </svg>

      <div className="chart-legend">
        <span className="lg lg-mean">
          <i style={{ background: "var(--accent)", height: 3 }} />
          panel mean
        </span>
        {keys.map((k, i) => (
          <span className="lg" key={k}>
            <i style={{ background: seriesColor(i) }} />
            {prettyKey(k)}
          </span>
        ))}
        <span className="lg">
          <i style={{ background: "rgba(152,161,175,.35)", height: 6 }} />
          claim volume
        </span>
      </div>
    </div>
  );
}
