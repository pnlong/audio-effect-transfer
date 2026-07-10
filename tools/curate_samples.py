#!/usr/bin/env python3
"""
WorldAFX audio sample curation tool.

SETUP (one-time):
    uv pip install flask

STEP 1 — Browse and classify samples in your browser:
    cd project_page
    python tools/curate_samples.py browse [--port 8765]

    If on a remote server, forward the port first:
        ssh -L 8765:localhost:8765 cpu-190

STEP 2 — Export randomly-selected good samples to project_page/audio/:
    python tools/curate_samples.py export --main 6 --dsp 4 --reverb 4 [--seed 42]
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

STUDY_BASE   = Path("/storage/phillip/neural_audio_effects/tests/listening_study")
TOOLS_DIR    = Path(__file__).parent.resolve()
RESULTS_FILE = TOOLS_DIR / "curation_results.json"
AUDIO_OUT    = TOOLS_DIR.parent / "audio"

# ── Sample catalogue ───────────────────────────────────────────────────────────

def _sorted_subdirs(path: Path, prefix: str) -> list[str]:
    """Return sorted list of subdirectory names matching a prefix."""
    if not path.exists():
        return []
    return sorted(d.name for d in path.iterdir()
                  if d.is_dir() and d.name.startswith(prefix))

def build_catalogue() -> dict:
    """Return {category: [sample_id, ...]} for all three study types."""
    return {
        "main":    _sorted_subdirs(STUDY_BASE / "full_pre_gan", "pair_"),
        "dsp":     _sorted_subdirs(STUDY_BASE / "ablation_dsp_training", "sample_"),
        "reverb":  _sorted_subdirs(STUDY_BASE / "ablation_reverb", "sample_"),
    }

def load_results() -> dict:
    if RESULTS_FILE.exists():
        return json.loads(RESULTS_FILE.read_text())
    return {"main": {}, "dsp": {}, "reverb": {}}

def save_results(results: dict) -> None:
    RESULTS_FILE.write_text(json.dumps(results, indent=2))

def _prefer_mp3(directory: Path, stem: str) -> str | None:
    """Return the best available audio file path (MP3 preferred over WAV)."""
    for ext in ("mp3", "wav"):
        p = directory / f"{stem}.{ext}"
        if p.exists():
            return str(p)
    return None

def _load_metadata(directory: Path) -> dict:
    meta_file = directory / "metadata.json"
    if meta_file.exists():
        try:
            return json.loads(meta_file.read_text())
        except Exception:
            pass
    return {}

def sample_files(category: str, sample_id: str) -> dict:
    """Return {label: absolute_path} for all audio conditions in a sample."""
    if category == "main":
        pre_dir  = STUDY_BASE / "full_pre_gan"  / sample_id
        post_dir = STUDY_BASE / "full_post_gan" / sample_id
        return {
            "dry":       _prefer_mp3(pre_dir, "dry_b"),
            "reference": _prefer_mp3(pre_dir, "wet_b"),
            "pre_gan":   _prefer_mp3(pre_dir, "wet_b_predicted"),
            "post_gan":  _prefer_mp3(post_dir, "wet_b_predicted"),
        }
    elif category == "dsp":
        d = STUDY_BASE / "ablation_dsp_training" / sample_id
        return {
            "dry":           _prefer_mp3(d, "dry"),
            "reference":     _prefer_mp3(d, "wet_real"),
            "separate":      _prefer_mp3(d, "wet_separate"),
            "joint_ft":      _prefer_mp3(d, "wet_joint_finetune"),
            "joint_scratch": _prefer_mp3(d, "wet_joint_scratch"),
        }
    elif category == "reverb":
        d = STUDY_BASE / "ablation_reverb" / sample_id
        return {
            "dry":       _prefer_mp3(d, "dry"),
            "reference": _prefer_mp3(d, "wet_real"),
            "param_mlp": _prefer_mp3(d, "wet_param_mlp"),
            "ir_gan":    _prefer_mp3(d, "wet_ir_gan"),
        }
    return {}

def sample_metadata(category: str, sample_id: str) -> dict:
    if category == "main":
        return _load_metadata(STUDY_BASE / "full_pre_gan" / sample_id)
    elif category == "dsp":
        return _load_metadata(STUDY_BASE / "ablation_dsp_training" / sample_id)
    elif category == "reverb":
        return _load_metadata(STUDY_BASE / "ablation_reverb" / sample_id)
    return {}

# ── Flask browse mode ──────────────────────────────────────────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>WorldAFX Sample Curator</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg: #0f172a; --surface: #1e293b; --border: #334155;
    --text: #e2e8f0; --muted: #94a3b8; --accent: #3b82f6;
    --good: #22c55e; --bad: #ef4444; --skip: #f59e0b; --none: #475569;
  }
  body { font-family: system-ui, sans-serif; background: var(--bg); color: var(--text);
         display: flex; flex-direction: column; height: 100vh; overflow: hidden; }

  /* header */
  header { display: flex; align-items: center; gap: 1.5rem; padding: .75rem 1.5rem;
           background: var(--surface); border-bottom: 1px solid var(--border);
           flex-shrink: 0; }
  header h1 { font-size: 1rem; font-weight: 700; }
  .stats { display: flex; gap: 1rem; font-size: .8rem; margin-left: auto; }
  .stat { display: flex; align-items: center; gap: .4rem; }
  .dot { width: 8px; height: 8px; border-radius: 50%; }
  .dot-good { background: var(--good); }
  .dot-bad  { background: var(--bad); }
  .dot-skip { background: var(--skip); }
  .dot-none { background: var(--none); }

  /* layout */
  .body { display: flex; flex: 1; overflow: hidden; }

  /* sidebar */
  aside { width: 220px; flex-shrink: 0; overflow-y: auto;
          background: var(--surface); border-right: 1px solid var(--border); padding: .5rem 0; }
  .cat-label { font-size: .65rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase;
               color: var(--muted); padding: .6rem 1rem .3rem; }
  .sample-item { display: flex; align-items: center; gap: .5rem; padding: .35rem 1rem;
                 font-size: .78rem; cursor: pointer; color: var(--muted);
                 border-left: 3px solid transparent; transition: background .1s; }
  .sample-item:hover { background: rgba(255,255,255,.05); }
  .sample-item.active { background: rgba(59,130,246,.15); border-left-color: var(--accent);
                        color: var(--text); }
  .sample-item .mark { font-size: .7rem; margin-left: auto; }
  .mark-good { color: var(--good); }
  .mark-bad  { color: var(--bad); }
  .mark-skip { color: var(--skip); }

  /* main view */
  main { flex: 1; overflow-y: auto; padding: 1.5rem 2rem; }

  .sample-header { margin-bottom: 1.25rem; }
  .sample-id { font-size: 1.25rem; font-weight: 700; }
  .sample-meta { display: flex; flex-wrap: wrap; gap: .5rem; margin-top: .5rem; }
  .tag { display: inline-block; padding: .2rem .6rem; border-radius: 9999px;
         font-size: .72rem; font-weight: 600; }
  .tag-cat { background: rgba(59,130,246,.2); color: #93c5fd; }
  .tag-src { background: rgba(148,163,184,.15); color: var(--muted); }
  .tag-score { background: rgba(34,197,94,.15); color: #86efac; }

  .effect-desc { font-size: .8rem; color: var(--muted); margin-top: .5rem;
                 background: rgba(255,255,255,.04); border-radius: 6px;
                 padding: .5rem .75rem; border-left: 3px solid var(--border); }
  .effect-desc li { margin: .15rem 0; }

  /* audio grid */
  .audio-grid { display: grid; gap: 1rem; margin: 1.25rem 0;
                grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); }
  .audio-card { background: var(--surface); border: 2px solid var(--border);
                border-radius: 8px; padding: .75rem 1rem; cursor: pointer;
                transition: border-color .12s, background .12s; }
  .audio-card:hover { border-color: #475569; }
  .audio-card.active-audio { border-color: var(--accent); background: rgba(59,130,246,.1); }
  .audio-card-label { font-size: .72rem; font-weight: 700; text-transform: uppercase;
                      letter-spacing: .06em; color: var(--muted); margin-bottom: .5rem; }
  .audio-card.active-audio .audio-card-label { color: #93c5fd; }
  .audio-card audio { width: 100%; }

  /* buttons */
  .actions { display: flex; gap: .75rem; margin-top: 1.5rem; flex-wrap: wrap; }
  .btn { padding: .6rem 1.5rem; border: none; border-radius: 8px; font-size: .9rem;
         font-weight: 700; cursor: pointer; transition: opacity .15s, transform .1s; }
  .btn:active { transform: scale(.97); }
  .btn-good { background: var(--good); color: #fff; }
  .btn-bad  { background: var(--bad);  color: #fff; }
  .btn-skip { background: var(--skip); color: #000; }
  .btn-nav  { background: var(--surface); color: var(--text);
              border: 1px solid var(--border); padding: .5rem 1rem; font-size: .8rem; }
  .btn:hover { opacity: .85; }

  .kbd { display: inline-block; background: var(--surface); border: 1px solid var(--border);
         border-radius: 4px; padding: .1rem .4rem; font-size: .7rem; font-family: monospace; }

  .current-label { margin-top: 1rem; font-size: .85rem; color: var(--muted); }
  .current-label span { font-weight: 700; }
  .current-good { color: var(--good); }
  .current-bad  { color: var(--bad); }
  .current-skip { color: var(--skip); }

  .nav-row { display: flex; gap: .5rem; margin-top: 1rem; align-items: center; }
  .nav-row .position { font-size: .8rem; color: var(--muted); margin: 0 .5rem; }
</style>
</head>
<body>

<header>
  <h1>WorldAFX Sample Curator</h1>
  <div class="stats">
    <div class="stat"><div class="dot dot-good"></div> <span id="cnt-good">0</span> good</div>
    <div class="stat"><div class="dot dot-bad"></div>  <span id="cnt-bad">0</span> bad</div>
    <div class="stat"><div class="dot dot-skip"></div> <span id="cnt-skip">0</span> skip</div>
    <div class="stat"><div class="dot dot-none"></div> <span id="cnt-none">0</span> unrated</div>
  </div>
</header>

<div class="body">
  <aside id="sidebar"></aside>
  <main id="main-view">
    <p style="color:var(--muted);margin-top:2rem;">Loading samples…</p>
  </main>
</div>

<script>
let catalogue = [];   // [{category, id, label}, ...]
let current = 0;

const LABELS = { good: '✓ Good', bad: '✗ Bad', skip: '⤳ Skip', null: '—' };
const MARK   = { good: '<span class="mark mark-good">✓</span>',
                 bad:  '<span class="mark mark-bad">✗</span>',
                 skip: '<span class="mark mark-skip">⤳</span>',
                 null: '' };

const CAT_NAMES = { main: 'Main Results', dsp: 'DSP Ablation', reverb: 'Reverb Ablation' };
const COND_NAMES = {
  dry: 'Dry (Input)', reference: 'Reference', pre_gan: 'Pre-GAN', post_gan: 'Post-GAN',
  separate: 'Separate', joint_ft: 'Joint Finetune', joint_scratch: 'Joint Scratch',
  param_mlp: 'Param MLP', ir_gan: 'IR GAN',
};

async function fetchJSON(url, opts={}) {
  const r = await fetch(url, opts);
  return r.json();
}

async function init() {
  const data = await fetchJSON('/api/catalogue');
  catalogue = data.samples;
  renderSidebar();
  const firstUnrated = findNext(0);
  showSample(firstUnrated !== null ? firstUnrated : 0);
  updateStats();
}

function renderSidebar() {
  const el = document.getElementById('sidebar');
  let html = '';
  let lastCat = null;
  catalogue.forEach((s, i) => {
    if (s.category !== lastCat) {
      html += `<div class="cat-label">${CAT_NAMES[s.category]}</div>`;
      lastCat = s.category;
    }
    const mark = MARK[s.label] || '';
    html += `<div class="sample-item ${i===current?'active':''}" id="si-${i}" onclick="showSample(${i})">
      ${s.id} ${mark}
    </div>`;
  });
  el.innerHTML = html;
}

function sidebarRefreshItem(idx) {
  const s = catalogue[idx];
  const el = document.getElementById(`si-${idx}`);
  if (!el) return;
  const mark = MARK[s.label] || '';
  el.innerHTML = `${s.id} ${mark}`;
  el.className = `sample-item ${idx===current?'active':''}`;
}

async function showSample(idx) {
  current = idx;
  const s = catalogue[idx];

  // update sidebar active state
  document.querySelectorAll('.sample-item').forEach((el,i) => {
    el.classList.toggle('active', i===idx);
  });
  document.getElementById(`si-${idx}`)?.scrollIntoView({block:'nearest'});

  const data = await fetchJSON(`/api/sample/${s.category}/${s.id}`);

  const condNames = Object.entries(data.files).map(([k,v], i) =>
    `<div class="audio-card" id="card-${i}" onclick="selectAudio(${i})">
      <div class="audio-card-label">${COND_NAMES[k] || k}</div>
      <audio controls src="/audio${v}"></audio>
    </div>`
  ).join('');

  const meta = data.metadata;
  const tags = [
    meta.category ? `<span class="tag tag-cat">${meta.category}</span>` : '',
    meta.source_dataset ? `<span class="tag tag-src">${meta.source_dataset}</span>` : '',
    meta.effect_score != null ? `<span class="tag tag-score">effect ${meta.effect_score?.toFixed(2)}</span>` : '',
  ].filter(Boolean).join('');

  const descItems = (meta.wet_config_changes || [])
    .map(c => `<li>${c}</li>`).join('');
  const descBlock = descItems
    ? `<ul class="effect-desc">${descItems}</ul>` : '';

  const labelClass = s.label ? `current-${s.label}` : '';
  const labelText  = s.label ? LABELS[s.label] : 'Unrated';

  const position = `${idx+1} / ${catalogue.length}`;

  document.getElementById('main-view').innerHTML = `
    <div class="sample-header">
      <div class="sample-id">${CAT_NAMES[s.category]} — ${s.id}</div>
      <div class="sample-meta">${tags}</div>
      ${descBlock}
    </div>

    <div class="audio-grid">${condNames}</div>

    <div class="actions">
      <button class="btn btn-good" onclick="classify('good')">👍 Good <span class="kbd">G</span></button>
      <button class="btn btn-bad"  onclick="classify('bad')">👎 Bad <span class="kbd">B</span></button>
      <button class="btn btn-skip" onclick="classify('skip')">⤳ Skip <span class="kbd">S</span></button>
    </div>

    <div class="current-label">
      Current: <span class="${labelClass}">${labelText}</span>
    </div>

    <div class="nav-row">
      <button class="btn btn-nav" onclick="navigate(-1)">← Prev</button>
      <span class="position">${position}</span>
      <button class="btn btn-nav" onclick="navigate(1)">Next →</button>
    </div>

    <div style="margin-top:.75rem;font-size:.72rem;color:var(--muted);line-height:1.8;">
      <span class="kbd">G</span> good &nbsp;
      <span class="kbd">B</span> bad &nbsp;
      <span class="kbd">S</span> skip &nbsp;&nbsp;
      <span class="kbd">Space</span> play/pause &nbsp;
      <span class="kbd">[</span><span class="kbd">]</span> cycle conditions &nbsp;
      <span class="kbd">←</span><span class="kbd">→</span> prev/next sample
    </div>
  `;

  selectAudio(0);
}

let activeAudioIdx = 0;

function pauseAll() {
  document.querySelectorAll('audio').forEach(a => a.pause());
}

function selectAudio(idx, autoplay = false) {
  activeAudioIdx = idx;
  const cards  = [...document.querySelectorAll('.audio-card')];
  const audios = [...document.querySelectorAll('.audio-card audio')];
  cards.forEach((c, i) => c.classList.toggle('active-audio', i === idx));
  if (autoplay && audios[idx]) {
    pauseAll();
    audios[idx].play();
  }
}

function cycleAudio(delta) {
  const cards = document.querySelectorAll('.audio-card');
  if (!cards.length) return;
  const next = (activeAudioIdx + delta + cards.length) % cards.length;
  selectAudio(next, true);
}

function toggleActiveAudio() {
  const audios = [...document.querySelectorAll('.audio-card audio')];
  const a = audios[activeAudioIdx];
  if (!a) return;
  if (a.paused) a.play(); else a.pause();
}

// pause others when one starts playing (handles clicks on native controls too)
document.addEventListener('play', e => {
  if (e.target.tagName !== 'AUDIO') return;
  const audios = [...document.querySelectorAll('.audio-card audio')];
  const idx = audios.indexOf(e.target);
  if (idx !== -1) selectAudio(idx);
  audios.forEach(a => { if (a !== e.target) a.pause(); });
}, true);

async function classify(label) {
  const s = catalogue[current];
  s.label = label;
  sidebarRefreshItem(current);
  updateStats();
  await fetchJSON(`/api/classify/${s.category}/${s.id}`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({label}),
  });
  // auto-advance to next unrated sample
  const nextUnrated = findNext(current + 1);
  if (nextUnrated !== null) showSample(nextUnrated);
}

function findNext(from) {
  for (let i = from; i < catalogue.length; i++) {
    if (!catalogue[i].label) return i;
  }
  // wrap
  for (let i = 0; i < from; i++) {
    if (!catalogue[i].label) return i;
  }
  return null; // all rated
}

function navigate(delta) {
  const next = Math.max(0, Math.min(catalogue.length - 1, current + delta));
  showSample(next);
}

function updateStats() {
  const counts = {good:0, bad:0, skip:0, null:0};
  catalogue.forEach(s => counts[s.label || 'null']++);
  document.getElementById('cnt-good').textContent = counts.good;
  document.getElementById('cnt-bad').textContent  = counts.bad;
  document.getElementById('cnt-skip').textContent = counts.skip;
  document.getElementById('cnt-none').textContent = counts.null;
}

document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  if (e.key === 'g' || e.key === 'G') { classify('good'); return; }
  if (e.key === 'b' || e.key === 'B') { classify('bad');  return; }
  if (e.key === 's' || e.key === 'S') { classify('skip'); return; }
  if (e.key === 'ArrowRight') { navigate(1);  return; }
  if (e.key === 'ArrowLeft')  { navigate(-1); return; }
  if (e.key === '[') { e.preventDefault(); cycleAudio(-1); return; }
  if (e.key === ']') { e.preventDefault(); cycleAudio(1);  return; }
  if (e.key === ' ') { e.preventDefault(); toggleActiveAudio(); }
});

init();
</script>
</body>
</html>
"""


def run_browse(port: int) -> None:
    try:
        from flask import Flask, jsonify, request, send_file, abort
    except ImportError:
        sys.exit("Flask not found. Install it with:\n  uv pip install flask")

    app = Flask(__name__)
    catalogue = build_catalogue()

    @app.get("/")
    def index():
        return HTML_TEMPLATE, 200, {"Content-Type": "text/html"}

    @app.get("/api/catalogue")
    def api_catalogue():
        results = load_results()
        samples = []
        for cat, ids in catalogue.items():
            for sid in ids:
                label = results.get(cat, {}).get(sid)
                samples.append({"category": cat, "id": sid, "label": label})
        return jsonify({"samples": samples})

    @app.get("/api/sample/<category>/<sample_id>")
    def api_sample(category: str, sample_id: str):
        if category not in catalogue or sample_id not in catalogue[category]:
            abort(404)
        files = {k: v for k, v in sample_files(category, sample_id).items() if v}
        meta  = sample_metadata(category, sample_id)
        return jsonify({"files": files, "metadata": meta})

    @app.post("/api/classify/<category>/<sample_id>")
    def api_classify(category: str, sample_id: str):
        if category not in catalogue:
            abort(404)
        label = request.json.get("label")
        if label not in ("good", "bad", "skip"):
            abort(400)
        results = load_results()
        if category not in results:
            results[category] = {}
        results[category][sample_id] = label
        save_results(results)
        return jsonify({"ok": True})

    @app.get("/audio/<path:filepath>")
    def serve_audio(filepath: str):
        # filepath is the absolute path passed from the frontend
        p = Path("/" + filepath)  # restore leading /
        if not p.exists():
            abort(404)
        return send_file(str(p))

    print(f"\n  WorldAFX Sample Curator — http://localhost:{port}")
    print(f"  Remote? Forward the port:  ssh -L {port}:localhost:{port} cpu-190\n")
    app.run(host="127.0.0.1", port=port, debug=False)


# ── Export mode ────────────────────────────────────────────────────────────────

EXPORT_MAP = {
    "main": {
        "dry":       ("full_pre_gan",  "{id}", "dry_b"),
        "reference": ("full_pre_gan",  "{id}", "wet_b"),
        "pre_gan":   ("full_pre_gan",  "{id}", "wet_b_predicted"),
        "post_gan":  ("full_post_gan", "{id}", "wet_b_predicted"),
    },
    "dsp": {
        "dry":           ("ablation_dsp_training", "{id}", "dry"),
        "reference":     ("ablation_dsp_training", "{id}", "wet_real"),
        "separate":      ("ablation_dsp_training", "{id}", "wet_separate"),
        "joint_finetune":("ablation_dsp_training", "{id}", "wet_joint_finetune"),
        "joint_scratch": ("ablation_dsp_training", "{id}", "wet_joint_scratch"),
    },
    "reverb": {
        "dry":       ("ablation_reverb", "{id}", "dry"),
        "reference": ("ablation_reverb", "{id}", "wet_real"),
        "param_mlp": ("ablation_reverb", "{id}", "wet_param_mlp"),
        "ir_gan":    ("ablation_reverb", "{id}", "wet_ir_gan"),
    },
}

# Destination folder name within AUDIO_OUT
CAT_OUT_DIR = {"main": "main", "dsp": "dsp_ablation", "reverb": "reverb_ablation"}
# Destination subfolder prefix (pair_00, sample_00, …)
CAT_PREFIX  = {"main": "pair", "dsp": "sample", "reverb": "sample"}


def _to_mp3(src: Path, dst: Path) -> None:
    """Copy src to dst as MP3, converting WAV if needed."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix == ".mp3":
        shutil.copy2(src, dst)
        return
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-codec:a", "libmp3lame", "-q:a", "2", str(dst)],
        check=True, capture_output=True,
    )


def run_export(n_main: int, n_dsp: int, n_reverb: int, seed: int) -> None:
    results = load_results()
    rng = random.Random(seed)

    requests = {"main": n_main, "dsp": n_dsp, "reverb": n_reverb}

    for cat, n in requests.items():
        good = [sid for sid, lbl in results.get(cat, {}).items() if lbl == "good"]
        if len(good) < n:
            print(f"[{cat}] WARNING: only {len(good)} good samples but {n} requested — using all of them.")
        chosen = rng.sample(good, min(n, len(good)))
        print(f"\n[{cat}] Selected {len(chosen)}/{len(good)} good samples:")

        for out_idx, sample_id in enumerate(chosen):
            out_subdir = f"{CAT_PREFIX[cat]}_{out_idx:02d}"
            out_path   = AUDIO_OUT / CAT_OUT_DIR[cat] / out_subdir
            out_path.mkdir(parents=True, exist_ok=True)
            print(f"  {sample_id}  →  audio/{CAT_OUT_DIR[cat]}/{out_subdir}/")

            for out_name, (study_dir, id_tmpl, stem) in EXPORT_MAP[cat].items():
                src_dir = STUDY_BASE / study_dir / sample_id
                src     = Path(_prefer_mp3(src_dir, stem) or "")
                if not src.exists():
                    print(f"    WARNING: missing {stem}.{{mp3,wav}} — skipping")
                    continue
                dst = out_path / f"{out_name}.mp3"
                _to_mp3(src, dst)

            # write a small metadata sidecar so we know where this came from
            meta = sample_metadata(cat, sample_id)
            meta["source_sample_id"] = sample_id
            (out_path / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nDone. Files written to {AUDIO_OUT}")
    print("Remember to update the badge-effect / badge-content labels in index.html.")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="WorldAFX audio sample curation tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_browse = sub.add_parser("browse", help="Start web UI for classifying samples")
    p_browse.add_argument("--port", type=int, default=8765, help="Local port (default: 8765)")

    p_export = sub.add_parser("export", help="Copy good samples into project_page/audio/")
    p_export.add_argument("--main",   type=int, default=6,  help="# main samples (default: 6)")
    p_export.add_argument("--dsp",    type=int, default=4,  help="# DSP ablation samples (default: 4)")
    p_export.add_argument("--reverb", type=int, default=4,  help="# reverb ablation samples (default: 4)")
    p_export.add_argument("--seed",   type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()

    if args.cmd == "browse":
        run_browse(args.port)
    elif args.cmd == "export":
        run_export(args.main, args.dsp, args.reverb, args.seed)


if __name__ == "__main__":
    main()
