# Project Page Tools

## `curate_samples.py` — Audio Sample Curation

Two-step workflow for selecting the best audio examples to feature on the project page.

---

### Prerequisites

Flask must be installed in the project venv (it already is):

```bash
# from the repo root
source .venv/bin/activate
```

---

### Step 1: Browse and classify samples

Start the web server:

```bash
# from the repo root (neural-audio-effects/)
source .venv/bin/activate
python project_page/tools/curate_samples.py browse
# optional: --port 8765 (default)
```

**If running on a remote server (cpu-190)**, open a separate terminal on your local machine and forward the port:

```bash
ssh -L 8765:localhost:8765 cpu-190
```

Then open **http://localhost:8765** in your browser.

#### What you'll see

126 samples total (60 main results, 36 DSP ablation, 30 reverb ablation). For each sample:

- Effect category, source dataset, effect score, and a human-readable description of what the plugin did (e.g. "darker −850 Hz centroid")
- Audio players for every condition side by side (Dry / Reference / model outputs)
- Rate buttons and keyboard shortcuts:

| Key | Action |
|-----|--------|
| `G` | Mark **good** — advance to next unrated |
| `B` | Mark **bad** — advance to next unrated |
| `S` | **Skip** — advance to next unrated |
| `←` / `→` | Navigate freely without rating |

Ratings are saved automatically to `project_page/tools/curation_results.json` after each click. You can close the server and resume later — already-rated samples keep their label.

---

### Step 2: Export selected samples to `project_page/audio/`

Once you've rated samples, run export to randomly pick N good ones per category and copy them into the audio directory the HTML page expects:

```bash
python project_page/tools/curate_samples.py export --main 6 --dsp 4 --reverb 4
# optional: --seed 42 (controls random selection; change to get a different draw)
```

This will:
1. Read `curation_results.json`
2. Randomly select N good-labeled samples per category
3. Copy MP3s into `project_page/audio/` using the naming convention expected by `index.html`:
   - `audio/main/pair_00/` — `dry.mp3`, `reference.mp3`, `pre_gan.mp3`, `post_gan.mp3`
   - `audio/dsp_ablation/sample_00/` — `dry.mp3`, `reference.mp3`, `separate.mp3`, `joint_finetune.mp3`, `joint_scratch.mp3`
   - `audio/reverb_ablation/sample_00/` — `dry.mp3`, `reference.mp3`, `param_mlp.mp3`, `ir_gan.mp3`
4. Write a `meta.json` sidecar in each folder so you can trace which original sample it came from

After export, **update the `badge-effect` and `badge-content` labels** in `index.html` to match the actual effect type and content type for each example (the `meta.json` files have this info).

---

### Re-running export with different samples

Change `--seed` to get a different random draw from your good-labeled pool, or re-run `browse` to update ratings first:

```bash
python project_page/tools/curate_samples.py export --main 6 --dsp 4 --reverb 4 --seed 7
```

The export overwrites whatever is currently in `project_page/audio/`.
