import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import gsap from "gsap";
import traceData from "../data/model-traces.json";

const CLASS_ORDER = traceData.meta.classOrder;
const CLASS_COLORS = {
  CANDIDATE: "#c9e66f",
  CONFIRMED: "#7fb28c",
  "FALSE POSITIVE": "#f18d59",
};
const MODEL_COLORS = { lightgbm: "#6cad68", catboost: "#ee9b4a" };
const EMBEDDED = new URLSearchParams(window.location.search).has("embed");
const STEPS = [
  { at: 0, label: "Observation", title: "A KOI becomes one structured measurement field.", note: "The models see numbers describing a transit-like signal and its host star—not an image of a planet." },
  { at: 0.15, label: "Contract", title: "The answer-like fields lose access to the core.", note: "Only the exact 98-feature conservative contract continues. Missing measurements remain blank for native tree handling." },
  { at: 0.31, label: "Forest", title: "Eighteen hundred boosted trees activate.", note: "LightGBM creates 1,350 class-specific trees across 450 rounds; CatBoost contributes 450 symmetric trees. Each sends this KOI to one real held-out leaf." },
  { at: 0.57, label: "Belief", title: "Two forests become two probability distributions.", note: "Each model estimates Candidate, Confirmed, and False Positive independently." },
  { at: 0.74, label: "Soft vote", title: "The two beliefs acquire their locked weights.", note: "LightGBM contributes 0.6364. CatBoost contributes 0.3636. The vectors combine class by class." },
  { at: 0.9, label: "Reality", title: "The strongest combined probability becomes the prediction.", note: "The held-out prediction is compared with the catalog disposition; confidence remains evidence, not certainty." },
];

const clamp = (value, min = 0, max = 1) => Math.min(max, Math.max(min, value));
const phase = (progress, start, end) => clamp((progress - start) / (end - start));
const smooth = (value) => value * value * (3 - 2 * value);
const pct = (value) => `${(value * 100).toFixed(1)}%`;
const metric = (value) => value === null || value === undefined ? "missing" : Number(value).toLocaleString(undefined, { maximumFractionDigits: 3 });

function drawTrace(canvas, trace, progress) {
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  const w = rect.width;
  const h = rect.height;
  const cx = w / 2;
  const cy = h * 0.48;
  const compact = w < 720;
  const radius = Math.min(w, h) * (compact ? 0.095 : 0.115);
  const inputP = smooth(phase(progress, 0, 0.18));
  const gateP = smooth(phase(progress, 0.14, 0.31));
  const forestP = smooth(phase(progress, 0.29, 0.58));
  const beliefP = smooth(phase(progress, 0.54, 0.75));
  const voteP = smooth(phase(progress, 0.72, 0.91));
  const resultP = smooth(phase(progress, 0.88, 1));

  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#152019";
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = "rgba(244,240,230,.08)";
  ctx.lineWidth = 1;
  for (let x = 0; x < w; x += 48) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke(); }
  for (let y = 0; y < h; y += 48) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }

  const leavesLeft = trace.models.lightgbm.leafIndices;
  const leavesRight = trace.models.catboost.leafIndices;
  const forestSpan = compact ? w * 0.37 : w * 0.41;
  const trunkLeft = cx - radius * 0.72;
  const trunkRight = cx + radius * 0.72;
  ctx.globalAlpha = forestP * (1 - resultP * 0.55);
  for (let i = 0; i < leavesLeft.length; i += 1) {
    const band = ((i * 37) % 1349) / 1348 - 0.5;
    const y = cy + band * h * (compact ? 0.48 : 0.66);
    const leftDepth = 0.38 + (leavesLeft[i] % 31) / 46;
    const lx = cx - forestSpan * leftDepth;
    const alpha = 0.035 + ((i * 13) % 17) / 390;
    ctx.strokeStyle = `rgba(131,180,119,${alpha})`;
    ctx.beginPath(); ctx.moveTo(trunkLeft, cy); ctx.quadraticCurveTo(cx - forestSpan * 0.35, cy + band * 40, lx, y); ctx.stroke();
    if (i % 31 === 0) {
      const travel = (forestP * 2.2 + i / leavesLeft.length) % 1;
      ctx.fillStyle = MODEL_COLORS.lightgbm;
      ctx.beginPath(); ctx.arc(trunkLeft + (lx - trunkLeft) * travel, cy + (y - cy) * travel, 1.5 + forestP * 1.5, 0, Math.PI * 2); ctx.fill();
    }
  }
  for (let i = 0; i < leavesRight.length; i += 1) {
    const band = ((i * 37) % 449) / 448 - 0.5;
    const y = cy + band * h * (compact ? 0.48 : 0.66);
    const rightDepth = 0.38 + (leavesRight[i] % 64) / 92;
    const rx = cx + forestSpan * rightDepth;
    const alpha = 0.035 + ((i * 13) % 17) / 390;
    ctx.strokeStyle = `rgba(228,154,79,${alpha})`;
    ctx.beginPath(); ctx.moveTo(trunkRight, cy); ctx.quadraticCurveTo(cx + forestSpan * 0.35, cy - band * 40, rx, y); ctx.stroke();
    if (i % 13 === 0) {
      const travel = (forestP * 2.2 + i / leavesRight.length) % 1;
      ctx.fillStyle = MODEL_COLORS.catboost;
      ctx.beginPath(); ctx.arc(trunkRight + (rx - trunkRight) * travel, cy + (y - cy) * travel, 1.5 + forestP * 1.5, 0, Math.PI * 2); ctx.fill();
    }
  }
  ctx.globalAlpha = 1;

  const retained = trace.input.retainedFeatureCount;
  const excluded = trace.input.excludedFieldCount;
  for (let i = 0; i < retained + excluded; i += 1) {
    const kept = i < retained;
    const a = (i / (retained + excluded)) * Math.PI * 2 + progress * 0.7;
    const orbit = Math.min(w, h) * (kept ? 0.31 : 0.4);
    const targetR = kept ? radius * 1.45 : orbit * (1 + gateP * 1.5);
    const r = orbit + (targetR - orbit) * (kept ? inputP : gateP);
    const x = cx + Math.cos(a) * r * (compact ? 0.75 : 1);
    const y = cy + Math.sin(a) * r * 0.72;
    const isMissing = kept && i < trace.input.missingFeatureCount;
    ctx.globalAlpha = kept ? 0.25 + (1 - gateP) * 0.55 : (1 - gateP) * 0.52;
    ctx.beginPath(); ctx.arc(x, y, kept ? 2.1 : 1.7, 0, Math.PI * 2);
    if (isMissing) { ctx.strokeStyle = "#f4f0e6"; ctx.stroke(); } else { ctx.fillStyle = kept ? "#c9e66f" : "#f18d59"; ctx.fill(); }
  }
  ctx.globalAlpha = 1;

  const chosen = trace.predicted;
  const chosenColor = CLASS_COLORS[chosen];
  const pulse = 1 + Math.sin(progress * Math.PI * 18) * 0.035;
  const coreScale = 0.72 + inputP * 0.28 + voteP * 0.12;
  ctx.save();
  ctx.translate(cx, cy);
  ctx.scale(coreScale * pulse, coreScale * pulse);
  for (let ring = 3; ring >= 1; ring -= 1) {
    ctx.beginPath();
    ctx.arc(0, 0, radius * (0.45 + ring * 0.33), 0, Math.PI * 2);
    ctx.strokeStyle = resultP > 0 ? `${chosenColor}${Math.round((0.18 + resultP * 0.55) * 255).toString(16).padStart(2, "0")}` : `rgba(244,240,230,${0.1 + ring * 0.06})`;
    ctx.lineWidth = ring === 1 ? 2.5 : 1;
    ctx.stroke();
  }
  const gradient = ctx.createRadialGradient(0, 0, 0, 0, 0, radius);
  gradient.addColorStop(0, resultP > 0.05 ? chosenColor : "#f4f0e6");
  gradient.addColorStop(0.35, resultP > 0.05 ? `${chosenColor}aa` : "#c9e66faa");
  gradient.addColorStop(1, "rgba(21,32,25,0)");
  ctx.fillStyle = gradient;
  ctx.beginPath(); ctx.arc(0, 0, radius, 0, Math.PI * 2); ctx.fill();
  ctx.restore();

  if (beliefP > 0) {
    const distributions = [trace.models.lightgbm.probabilities, trace.models.catboost.probabilities];
    distributions.forEach((distribution, modelIndex) => {
      const side = modelIndex === 0 ? -1 : 1;
      const anchorX = cx + side * (compact ? w * 0.28 : w * 0.34);
      CLASS_ORDER.forEach((label, classIndex) => {
        const value = distribution[label];
        const y = cy + (classIndex - 1) * 26;
        ctx.globalAlpha = beliefP * 0.85 * (1 - voteP * 0.55);
        ctx.fillStyle = "rgba(244,240,230,.13)";
        ctx.fillRect(anchorX - 42, y, 84, 5);
        ctx.fillStyle = CLASS_COLORS[label];
        ctx.fillRect(side < 0 ? anchorX + 42 - 84 * value : anchorX - 42, y, 84 * value, 5);
      });
    });
  }
  ctx.globalAlpha = 1;

  if (voteP > 0) {
    CLASS_ORDER.forEach((label, index) => {
      const value = trace.ensemble.rawProbabilities[label];
      const angle = -Math.PI / 2 + index * (Math.PI * 2 / 3);
      const startX = cx + Math.cos(angle) * radius * 2.5;
      const startY = cy + Math.sin(angle) * radius * 2.5;
      ctx.strokeStyle = CLASS_COLORS[label];
      ctx.lineWidth = 1 + value * 7 * voteP;
      ctx.globalAlpha = voteP * (0.25 + value * 0.75);
      ctx.beginPath(); ctx.moveTo(startX, startY); ctx.lineTo(cx + Math.cos(angle) * radius * 1.2, cy + Math.sin(angle) * radius * 1.2); ctx.stroke();
    });
  }
  ctx.globalAlpha = 1;
}

function TraceCanvas({ trace, progress }) {
  const canvasRef = useRef(null);
  useLayoutEffect(() => {
    const canvas = canvasRef.current;
    const render = () => drawTrace(canvas, trace, progress);
    render();
    window.addEventListener("resize", render);
    return () => window.removeEventListener("resize", render);
  }, [trace, progress]);
  return <canvas ref={canvasRef} className="trace-canvas" role="img" aria-label={`Animated exact inference trace for ${trace.objectId}. ${trace.models.lightgbm.treeCount} LightGBM class trees and ${trace.models.catboost.treeCount} CatBoost trees produce ${trace.predicted}.`} />;
}

function ProbabilityVector({ title, color, probabilities, weight, visible }) {
  return (
    <div className="probability-vector" style={{ "--vector-color": color, opacity: visible ? 1 : 0, transform: `translateY(${visible ? 0 : 12}px)` }}>
      <div><span>{title}</span>{weight ? <small>weight {weight.toFixed(4)}</small> : null}</div>
      {CLASS_ORDER.map((label) => <div className="probability-line" key={label}><span>{label}</span><i><b style={{ width: `${probabilities[label] * 100}%`, background: CLASS_COLORS[label] }} /></i><strong>{pct(probabilities[label])}</strong></div>)}
    </div>
  );
}

function ExactDrawer({ trace, open, onClose }) {
  const [model, setModel] = useState("lightgbm");
  const [pathIndex, setPathIndex] = useState(0);
  const selectedPath = trace.models[model].samplePaths[pathIndex];
  useEffect(() => setPathIndex(0), [trace, model]);
  return (
    <aside className={`trace-drawer ${open ? "is-open" : ""}`} aria-hidden={!open} aria-labelledby="drawerTitle">
      <button className="drawer-close" type="button" onClick={onClose}>Close ×</button>
      <span className="drawer-kicker">Exact trace inspector</span>
      <h2 id="drawerTitle">One active leaf in every tree.</h2>
      <p>The full animation draws all 1,800 held-out leaf assignments. Twelve trees from either model can be expanded here without pretending that one tree represents the ensemble.</p>
      <div className="model-switch" role="group" aria-label="Choose model path"><button type="button" className={model === "lightgbm" ? "active" : ""} onClick={() => setModel("lightgbm")}>LightGBM</button><button type="button" className={model === "catboost" ? "active" : ""} onClick={() => setModel("catboost")}>CatBoost</button></div>
      <label className="tree-select">Expanded tree<select value={pathIndex} onChange={(event) => setPathIndex(Number(event.target.value))}>{trace.models[model].samplePaths.map((path, index) => <option key={path.tree} value={index}>Tree {path.tree} · leaf {path.leaf}</option>)}</select></label>
      <div className="path-stack">
        {model === "lightgbm" ? selectedPath.path.map((node, index) => node.leaf !== undefined ? <div className="leaf-stop" key={`leaf-${node.leaf}`}><span>LEAF</span><strong>{node.leaf}</strong><small>tree value {metric(node.value)}</small></div> : <div className="path-stop" key={`${node.feature}-${index}`}><span>{String(index + 1).padStart(2, "0")}</span><strong>{node.label}</strong><small>{node.observed === null ? "missing" : metric(node.observed)} · {node.direction} branch</small></div>) : selectedPath.decisionBits.map((bit, index) => <div className="path-stop" key={`${selectedPath.tree}-${index}`}><span>{String(index + 1).padStart(2, "0")}</span><strong>Symmetric split</strong><small>decision bit {bit}</small></div>)}
      </div>
      <div className="drawer-truth"><strong>What is not shown</strong><p>No imputation, scaling, Domain V2 features, neural network, or leakage field exists in this final inference path.</p></div>
    </aside>
  );
}

function App() {
  const [objectIndex, setObjectIndex] = useState(0);
  const [progress, setProgress] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [exact, setExact] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const tweenRef = useRef(null);
  const trace = traceData.objects[objectIndex];
  const activeStep = useMemo(() => STEPS.reduce((current, step, index) => progress >= step.at ? index : current, 0), [progress]);

  function stop() {
    tweenRef.current?.kill();
    tweenRef.current = null;
    setPlaying(false);
  }

  function play(from = progress) {
    stop();
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) { setProgress(1); return; }
    const state = { value: from >= 0.999 ? 0 : from };
    setProgress(state.value);
    setPlaying(true);
    tweenRef.current = gsap.to(state, { value: 1, duration: 14 * (1 - state.value), ease: "none", onUpdate: () => setProgress(state.value), onComplete: () => { setProgress(1); setPlaying(false); } });
  }

  function seek(value) {
    stop();
    setProgress(value);
  }

  function selectObject(event) {
    stop();
    setObjectIndex(Number(event.target.value));
    setProgress(0);
    setDrawerOpen(false);
  }

  useEffect(() => () => tweenRef.current?.kill(), []);
  useEffect(() => { const timer = window.setTimeout(() => play(0), 450); return () => window.clearTimeout(timer); }, [objectIndex]);

  const step = STEPS[activeStep];
  const resultVisible = progress >= 0.88;
  return (
    <main id="prototype" className={`prototype-shell ${EMBEDDED ? "is-embedded" : ""}`}>
      <header className="prototype-header">
        <div className="prototype-brand"><span>K</span><div><strong>KOI INFERENCE CORE</strong><small>ISOLATED TRACE PROTOTYPE</small></div></div>
        <div className="truth-seal"><i /> REAL FOLD {trace.fold} TREES · REAL OOF PROBABILITIES</div>
      </header>

      <section className="trace-intro">
        <span>SELECT ONE HELD-OUT OBJECT</span>
        <select aria-label="Choose a real KOI trace" value={objectIndex} onChange={selectObject}>{traceData.objects.map((item, index) => <option key={item.rowid} value={index}>{item.example} · {item.objectId}</option>)}</select>
        <div className="object-facts"><span>KOI <strong>{trace.objectId}</strong></span><span>ACTUAL <strong>{trace.actual}</strong></span><span>FOLD <strong>{trace.fold}</strong></span></div>
      </section>

      <section className="inference-arena" aria-labelledby="stageTitle">
        <TraceCanvas trace={trace} progress={progress} />
        <div className="model-label lightgbm-label" style={{ opacity: phase(progress, 0.29, 0.42) }}><span>{trace.models.lightgbm.treeCount.toLocaleString()} CLASS TREES</span><strong>LIGHTGBM</strong></div>
        <div className="model-label catboost-label" style={{ opacity: phase(progress, 0.29, 0.42) }}><span>{trace.models.catboost.treeCount.toLocaleString()} TREES</span><strong>CATBOOST</strong></div>

        <div className="measurement-orbit" style={{ opacity: 1 - phase(progress, 0.25, 0.38) }}>
          {trace.input.values.slice(0, 8).map((feature, index) => <button type="button" key={feature.name} style={{ "--i": index }} data-missing={feature.missing}><span>{feature.label}</span><strong>{feature.missing ? "missing" : metric(feature.value)}{feature.unit ? ` ${feature.unit}` : ""}</strong></button>)}
        </div>

        <div className="gate-readout" style={{ opacity: phase(progress, 0.14, 0.21) * (1 - phase(progress, 0.31, 0.42)) }}><span>{trace.input.retainedFeatureCount} RETAINED</span><i /><span>{trace.input.excludedFieldCount} EXCLUDED</span><i /><span>{trace.input.missingFeatureCount} MISSING</span></div>

        <div className="vectors">
          <ProbabilityVector title="LIGHTGBM" color={MODEL_COLORS.lightgbm} probabilities={trace.models.lightgbm.probabilities} weight={trace.ensemble.weights.lightgbm} visible={progress >= 0.56 && progress < 0.94} />
          <ProbabilityVector title="CATBOOST" color={MODEL_COLORS.catboost} probabilities={trace.models.catboost.probabilities} weight={trace.ensemble.weights.catboost} visible={progress >= 0.56 && progress < 0.94} />
        </div>

        <div className={`result-reveal ${resultVisible ? "is-visible" : ""}`}>
          <span>HELD-OUT PREDICTION</span><strong style={{ color: CLASS_COLORS[trace.predicted] }}>{trace.predicted}</strong><div>{pct(trace.ensemble.rawProbabilities[trace.predicted])} raw confidence · margin {pct(trace.ensemble.margin)}</div><small className={trace.correct ? "correct" : "incorrect"}>{trace.correct ? "MATCHES CATALOG DISPOSITION" : `${trace.actual} IN CATALOG · INCORRECT`}</small>
        </div>

        <div className="stage-caption"><span>{String(activeStep + 1).padStart(2, "0")} / 06 · {step.label}</span><h1 id="stageTitle">{step.title}</h1><p>{step.note}</p></div>
      </section>

      <section className="trace-controls" aria-label="Animation controls">
        <div className="play-controls"><button type="button" className="primary-control" onClick={() => playing ? stop() : play()}>{playing ? "Pause trace" : progress >= 0.999 ? "Replay trace" : "Play trace"}</button><button type="button" onClick={() => seek(0)}>Reset</button><button type="button" aria-pressed={exact} onClick={() => setExact(!exact)}>{exact ? "Hide exact values" : "Show exact values"}</button><button type="button" onClick={() => setDrawerOpen(true)}>Inspect tree paths</button></div>
        <label className="scrubber"><span>TRACE POSITION</span><input type="range" min="0" max="1" step="0.001" value={progress} onChange={(event) => seek(Number(event.target.value))} /></label>
        <div className="step-rail">{STEPS.map((item, index) => <button type="button" key={item.label} className={activeStep === index ? "active" : ""} onClick={() => seek(item.at)}><span>{String(index + 1).padStart(2, "0")}</span>{item.label}</button>)}</div>
      </section>

      {exact ? <section className="exact-equation" aria-label="Exact ensemble calculation"><div><span>LIGHTGBM</span><strong>{trace.ensemble.weights.lightgbm.toFixed(4)}</strong></div><b>×</b><div className="equation-vector">{CLASS_ORDER.map((label) => <span key={label}>{label}<strong>{trace.models.lightgbm.probabilities[label].toFixed(6)}</strong></span>)}</div><b>+</b><div><span>CATBOOST</span><strong>{trace.ensemble.weights.catboost.toFixed(4)}</strong></div><b>×</b><div className="equation-vector">{CLASS_ORDER.map((label) => <span key={label}>{label}<strong>{trace.models.catboost.probabilities[label].toFixed(6)}</strong></span>)}</div><b>=</b><div className="equation-vector final-vector">{CLASS_ORDER.map((label) => <span key={label}>{label}<strong>{trace.ensemble.rawProbabilities[label].toFixed(6)}</strong></span>)}</div></section> : null}

      <section className="static-summary" aria-label="Readable model trace summary"><span>STATIC SUMMARY</span><p>{trace.objectId} entered held-out fold {trace.fold} with {trace.input.missingFeatureCount} missing measurements. Leakage and metadata fields were excluded before inference. LightGBM evaluated 1,350 class-specific trees across 450 boosting rounds and CatBoost evaluated 450 trees, then their probabilities combined using weights 0.6364 and 0.3636. The ensemble predicted {trace.predicted}; the catalog label is {trace.actual}.</p></section>

      <ExactDrawer trace={trace} open={drawerOpen} onClose={() => setDrawerOpen(false)} />
      {drawerOpen ? <button className="drawer-backdrop" type="button" aria-label="Close trace inspector" onClick={() => setDrawerOpen(false)} /> : null}
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
