/* Camera-trap species demo, running entirely in the browser (onnxruntime-web, WebAssembly).
 *
 * Same pipeline as src/video.py: sample frames -> MegaDetector v6 (ONNX) -> best box + 10% margin, letterboxed
 * to 224x224 -> DINOv2-small + linear probe (ONNX) -> drop frames the probe calls "empty" -> average = clip verdict.
 * Models, config and example clips are loaded from a Hugging Face model repo (data-base on the script tag).
 *
 * Usage: <div id="ct-demo"></div>
 *        <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.wasm.min.js"></script>
 *        <script src="camera-trap-demo.js" data-base="https://huggingface.co/<user>/<repo>/resolve/main/"></script>
 * Add ?selftest to the page URL to re-run the first example live and compare with its stored results.
 */
(() => {
  "use strict";
  const SCRIPT = document.currentScript;
  const BASE = (SCRIPT && SCRIPT.dataset.base) || "https://huggingface.co/ray0987/camera-trap-demo-models/resolve/main/";
  const ORT_DIST = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";
  const MAX_SECONDS = 30;
  const CACHE = "camera-trap-demo-v1";

  let cfg = null, examples = [], det = null, cls = null, modelsLoading = null;
  let current = null;           // { frames, fps, url }
  const $ = (tag, attrs = {}, ...kids) => {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") el.className = v; else if (k.startsWith("on")) el.addEventListener(k.slice(2), v); else el.setAttribute(k, v);
    }
    for (const c of kids) el.append(c);
    return el;
  };
  const pct = (x) => `${Math.round(x * 100)}%`;
  const errText = (e) => (e && e.message) || (typeof e === "string" ? e : JSON.stringify(e)) || String(e);

  // ---------- loading ----------
  async function fetchBytes(url, onProgress) {
    let cache = null;
    try { cache = await caches.open(CACHE); const hit = await cache.match(url); if (hit) return new Uint8Array(await hit.arrayBuffer()); } catch (e) { cache = null; }
    const res = await fetch(url);
    if (!res.ok) throw new Error(`download failed (${res.status}) for ${url}`);
    const total = +res.headers.get("content-length") || 0;
    const reader = res.body.getReader();
    const parts = []; let got = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      parts.push(value); got += value.length;
      if (onProgress && total) onProgress(got / total);
    }
    const out = new Uint8Array(got); let o = 0;
    for (const p of parts) { out.set(p, o); o += p.length; }
    try { if (cache) await cache.put(url, new Response(out.slice())); } catch (e) { /* storage full or blocked: fine */ }
    return out;
  }

  function loadModels(setStatus) {
    if (modelsLoading) return modelsLoading;
    modelsLoading = (async () => {
      ort.env.wasm.wasmPaths = ORT_DIST;
      ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
      const opts = { executionProviders: ["wasm"], graphOptimizationLevel: "all" };
      const d = await fetchBytes(BASE + cfg.detector, (p) => setStatus(`Downloading the detector… ${pct(p)} (first visit only)`, p * 0.7));
      const c = await fetchBytes(BASE + cfg.classifier, (p) => setStatus(`Downloading the classifier… ${pct(p)} (first visit only)`, 0.7 + p * 0.25));
      setStatus("Starting the models…", 0.97);
      det = await ort.InferenceSession.create(d, opts);
      cls = await ort.InferenceSession.create(c, opts);
    })();
    modelsLoading.catch(() => { modelsLoading = null; });
    return modelsLoading;
  }

  // ---------- pipeline (mirrors src/video.py and src/preprocess.py) ----------
  // PIL-compatible resampling (Pillow's Resample.c: separable filter, support widened when shrinking = antialias,
  // horizontal pass then vertical, rounding to 8 bits in between). Matches PIL resize(BILINEAR / BICUBIC).
  const FILTERS = {
    bilinear: { support: 1, f: (x) => { x = Math.abs(x); return x < 1 ? 1 - x : 0; } },
    bicubic: { support: 2, f: (x) => { const a = -0.5; x = Math.abs(x);
      return x < 1 ? ((a + 2) * x - (a + 3)) * x * x + 1 : x < 2 ? (((x - 5) * x + 8) * x - 4) * a : 0; } },
  };
  function coeffs(inSize, outSize, kind) {
    const { support, f } = FILTERS[kind], scale = inSize / outSize, fs = Math.max(scale, 1), sup = support * fs;
    const out = [];
    for (let i = 0; i < outSize; i++) {
      const center = (i + 0.5) * scale;
      const lo = Math.max(Math.trunc(center - sup + 0.5), 0), hi = Math.min(Math.trunc(center + sup + 0.5), inSize);
      const w = []; let ww = 0;
      for (let j = lo; j < hi; j++) { const v = f((j - center + 0.5) / fs); w.push(v); ww += v; }
      out.push({ lo, w: w.map((v) => (ww ? v / ww : 0)) });
    }
    return out;
  }
  const clip8 = (v) => (v <= 0 ? 0 : v >= 255 ? 255 : Math.round(v));
  // src: RGBA Uint8ClampedArray of width sw; region (rx, ry, rw, rh) -> RGB Uint8Array of size dw x dh
  function resizePIL(src, sw, rx, ry, rw, rh, dw, dh, kind) {
    const cx = coeffs(rw, dw, kind), cy = coeffs(rh, dh, kind);
    const tmp = new Uint8Array(dw * rh * 3);
    for (let y = 0; y < rh; y++) {
      const row = ((ry + y) * sw + rx) * 4;
      for (let x = 0; x < dw; x++) {
        const { lo, w } = cx[x]; let r = 0, g = 0, b = 0;
        for (let k = 0; k < w.length; k++) { const o = row + (lo + k) * 4; r += src[o] * w[k]; g += src[o + 1] * w[k]; b += src[o + 2] * w[k]; }
        const t = (y * dw + x) * 3; tmp[t] = clip8(r); tmp[t + 1] = clip8(g); tmp[t + 2] = clip8(b);
      }
    }
    const out = new Uint8Array(dw * dh * 3);
    for (let y = 0; y < dh; y++) {
      const { lo, w } = cy[y];
      for (let x = 0; x < dw; x++) {
        let r = 0, g = 0, b = 0;
        for (let k = 0; k < w.length; k++) { const o = ((lo + k) * dw + x) * 3; r += tmp[o] * w[k]; g += tmp[o + 1] * w[k]; b += tmp[o + 2] * w[k]; }
        const t = (y * dw + x) * 3; out[t] = clip8(r); out[t + 1] = clip8(g); out[t + 2] = clip8(b);
      }
    }
    return out;
  }
  function rgbToCHW(rgb, n, mean, std) {
    const out = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) {
      const v = rgb[i * 3 + c] / 255;
      out[c * n + i] = mean ? (v - mean[c]) / std[c] : v;
    }
    return out;
  }

  async function detect(frame) {                // frame = { px: RGBA, W, H }
    const { px, W, H } = frame, S = cfg.det_size;
    const rgb = resizePIL(px, W, 0, 0, W, H, S, S, "bilinear");   // torchvision Resize((640, 640)) on a PIL image
    const images = new ort.Tensor("float32", rgbToCHW(rgb, S * S), [1, 3, S, S]);
    const sizes = new ort.Tensor("int64", BigInt64Array.from([BigInt(W), BigInt(H)]), [1, 2]);
    const out = await det.run({ images, orig_target_sizes: sizes });
    const boxes = out.boxes.data, scores = out.scores.data;
    let best = null;
    for (let i = 0; i < scores.length; i++) {
      const s = scores[i];
      if (s < cfg.det_conf) continue;
      const x0 = Math.max(0, boxes[i * 4]), y0 = Math.max(0, boxes[i * 4 + 1]);
      const x1 = Math.min(W, boxes[i * 4 + 2]), y1 = Math.min(H, boxes[i * 4 + 3]);
      if (x1 <= x0 || y1 <= y0) continue;
      if (!best || s > best.conf) best = { conf: s, bbox: [x0 / W, y0 / H, (x1 - x0) / W, (y1 - y0) / H] };
    }
    return best;
  }

  async function classify(frame, bbox) {
    const { px, W, H } = frame, m = cfg.crop_margin, S = cfg.crop_size;
    const [x, y, w, h] = bbox;                    // src/preprocess.py crop_box: +10% margin, integer box
    const x0 = x * W, y0 = y * H, x1 = (x + w) * W, y1 = (y + h) * H;
    const mx = (x1 - x0) * m, my = (y1 - y0) * m;
    const bx0 = Math.max(0, Math.trunc(x0 - mx)), by0 = Math.max(0, Math.trunc(y0 - my));
    const bx1 = Math.min(W, Math.trunc(x1 + mx)), by1 = Math.min(H, Math.trunc(y1 + my));
    const cw = bx1 - bx0, ch = by1 - by0, s = S / Math.max(cw, ch);   // letterbox_pad_square: long side -> 224, pad
    const nw = Math.max(1, Math.round(cw * s)), nh = Math.max(1, Math.round(ch * s));
    const small = resizePIL(px, W, bx0, by0, cw, ch, nw, nh, "bicubic");
    const canvas = new Uint8Array(S * S * 3), [fr, fg, fb] = cfg.crop_fill;
    for (let i = 0; i < S * S; i++) { canvas[i * 3] = fr; canvas[i * 3 + 1] = fg; canvas[i * 3 + 2] = fb; }
    const ox = (S - nw) >> 1, oy = (S - nh) >> 1;
    for (let yy = 0; yy < nh; yy++) canvas.set(small.subarray(yy * nw * 3, (yy + 1) * nw * 3), ((oy + yy) * S + ox) * 3);
    const x_ = new ort.Tensor("float32", rgbToCHW(canvas, S * S, cfg.mean, cfg.std), [1, 3, S, S]);
    const out = await cls.run({ pixel_values: x_ });
    return Array.from(out.probs.data);
  }

  const argmax = (a) => a.reduce((bi, v, i) => (v > a[bi] ? i : bi), 0);
  const isAnimal = (f) => f.probs && cfg.classes[argmax(f.probs)] !== "empty";

  function vote(frames) {
    const P = frames.filter(isAnimal).map((f) => f.probs);
    if (!P.length) return { pred: "empty", conf: null, n: frames.length, na: 0 };
    const m = P[0].map((_, k) => P.reduce((s, p) => s + p[k], 0) / P.length);
    const order = m.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]);
    return { pred: cfg.classes[order[0][1]], conf: order[0][0], second: cfg.classes[order[1][1]], second_conf: order[1][0],
             n: frames.length, na: P.length };
  }

  function seek(video, t) {
    return new Promise((resolve, reject) => {
      const ok = () => { cleanup(); resolve(); }, bad = () => { cleanup(); reject(new Error("could not read this video")); };
      const cleanup = () => { video.removeEventListener("seeked", ok); video.removeEventListener("error", bad); };
      video.addEventListener("seeked", ok); video.addEventListener("error", bad);
      video.currentTime = t;
    });
  }

  // Seek and report the timestamp of the frame actually shown (requestVideoFrameCallback), or null if unknown.
  function seekTimed(video, t) {
    return new Promise((resolve, reject) => {
      let done = false, shown = null;
      const finish = () => { if (!done) { done = true; cleanup(); resolve(shown); } };
      const onSeeked = () => setTimeout(finish, 120);          // give the frame callback a moment to report
      const onError = () => { done = true; cleanup(); reject(new Error("could not read this video")); };
      const cleanup = () => { video.removeEventListener("seeked", onSeeked); video.removeEventListener("error", onError); };
      if (video.requestVideoFrameCallback) video.requestVideoFrameCallback((_, meta) => { shown = meta.mediaTime; finish(); });
      video.addEventListener("seeked", onSeeked); video.addEventListener("error", onError);
      video.currentTime = t;
    });
  }

  // Show the first frame at or after time t, like src/video.py's iter_frames (a plain seek shows the frame before t).
  async function seekFrame(video, t) {
    if (!video.requestVideoFrameCallback) { await seek(video, t + 0.02); return; }
    for (const d of [0, 0.012, 0.025, 0.04, 0.06, 0.09]) {
      const shown = await seekTimed(video, t + d);
      if (shown == null || shown >= t - 1e-3) return;
    }
  }

  async function analyse(video, fps, setStatus) {
    const dur = Math.min(video.duration || 0, MAX_SECONDS);
    const times = [];
    for (let t = 0; t < dur - 1e-3; t += 1 / fps) times.push(t);
    const fc = document.createElement("canvas");
    fc.width = video.videoWidth; fc.height = video.videoHeight;
    const g = fc.getContext("2d", { willReadFrequently: true });
    const frames = [];
    for (let i = 0; i < times.length; i++) {
      setStatus(`Analysing frame ${i + 1} of ${times.length}…`, i / times.length);
      await seekFrame(video, times[i]);
      g.drawImage(video, 0, 0, fc.width, fc.height);
      const frame = { px: g.getImageData(0, 0, fc.width, fc.height).data, W: fc.width, H: fc.height };
      const d = await detect(frame);
      const f = { t: times[i], det_conf: d ? d.conf : 0 };
      if (d) { f.bbox = d.bbox; f.probs = await classify(frame, d.bbox); }
      frames.push(f);
    }
    await seek(video, 0);
    return { frames, fps, cut: (video.duration || 0) > MAX_SECONDS + 0.5 };
  }

  // ---------- UI ----------
  function build(root) {
    const status = $("p", { class: "ct-status", role: "status" }, "Pick an example, or upload a trail-camera video.");
    const bar = $("div", { class: "ct-bar" }, $("span"));
    const fpsSel = $("select", { "aria-label": "Frames analysed per second" },
      $("option", { value: "0.5" }, "0.5 frame/s"), $("option", { value: "1", selected: "" }, "1 frame/s"), $("option", { value: "2" }, "2 frames/s"));
    const file = $("input", { type: "file", accept: "video/*", class: "ct-file", id: "ct-file" });
    const exRow = $("div", { class: "ct-examples", role: "group", "aria-label": "Example videos" });
    const video = $("video", { class: "ct-video", controls: "", playsinline: "", muted: "", crossorigin: "anonymous" });
    const overlay = $("canvas", { class: "ct-overlay", "aria-hidden": "true" });
    const verdict = $("div", { class: "ct-verdict", "aria-live": "polite" });
    const tbody = $("tbody");
    const table = $("details", { class: "ct-table" }, $("summary", {}, "Per-frame results"),
      $("table", {}, $("thead", {}, $("tr", {}, $("th", {}, "Time"), $("th", {}, "Detector"), $("th", {}, "Species"), $("th", {}, "Confidence"))), tbody));
    root.append(
      $("div", { class: "ct-head" }, $("p", { class: "ct-label" }, "Try an example"), exRow),
      $("div", { class: "ct-controls" },
        $("label", { class: "ct-upload", for: "ct-file" }, "Upload your video"), file, fpsSel,
        $("span", { class: "ct-hint" }, `Up to ${MAX_SECONDS} s. Runs in your browser; nothing is uploaded.`)),
      $("div", { class: "ct-progress" }, status, bar),
      $("div", { class: "ct-stage" }, video, overlay),
      verdict, table);

    const setStatus = (msg, p) => { status.textContent = msg; bar.firstChild.style.width = p == null ? "0" : `${Math.round(p * 100)}%`; };
    let busy = false;

    function show(result, note) {
      current = result;
      const v = vote(result.frames);
      verdict.replaceChildren();
      if (v.pred === "empty") {
        verdict.append($("h3", {}, "No animal found"), $("p", {}, `No animal in any of the ${v.n} analysed frames.`));
      } else {
        verdict.append($("h3", {}, "Verdict: ", $("strong", {}, v.pred), ` (${pct(v.conf)})`),
          $("p", {}, `Runner-up: ${v.second} (${pct(v.second_conf)}). Animal seen in ${v.na} of ${v.n} analysed frames.`));
        if (v.conf < cfg.low_conf) verdict.append($("p", { class: "ct-warn" },
          "Low confidence. This may be a species the model was never trained on; it can only answer with one of its 13 labels."));
      }
      if (result.cut) verdict.append($("p", { class: "ct-note" }, `Only the first ${MAX_SECONDS} s were analysed.`));
      if (note) verdict.append($("p", { class: "ct-note" }, note));
      tbody.replaceChildren(...result.frames.map((f) => {
        const a = isAnimal(f), k = f.probs ? argmax(f.probs) : -1;
        return $("tr", {}, $("td", {}, `${f.t.toFixed(1)} s`), $("td", {}, f.det_conf ? pct(f.det_conf) : "–"),
          $("td", {}, a ? cfg.classes[k] : "no animal"), $("td", {}, a ? pct(f.probs[k]) : "–"));
      }));
      video.play().catch(() => {});
    }

    // overlay: draw the result of the analysed frame nearest to the playback time
    function draw() {
      const r = video.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
      if (overlay.width !== Math.round(r.width * dpr) || overlay.height !== Math.round(r.height * dpr)) {
        overlay.width = Math.round(r.width * dpr); overlay.height = Math.round(r.height * dpr);
      }
      const g = overlay.getContext("2d");
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      g.clearRect(0, 0, r.width, r.height);
      if (current && video.videoWidth) {
        const s = Math.min(r.width / video.videoWidth, r.height / video.videoHeight);
        const vw = video.videoWidth * s, vh = video.videoHeight * s, ox = (r.width - vw) / 2, oy = (r.height - vh) / 2;
        let f = null, bd = Infinity;
        for (const x of current.frames) { const d = Math.abs(x.t - video.currentTime); if (d < bd) { bd = d; f = x; } }
        if (f && bd <= 1 / current.fps && isAnimal(f)) {
          const [x, y, w, h] = f.bbox, k = argmax(f.probs);
          const X = ox + x * vw, Y = oy + y * vh, Wd = w * vw, Ht = h * vh;
          g.lineWidth = 3; g.strokeStyle = "#3cc83c"; g.strokeRect(X, Y, Wd, Ht);
          const tag = (text, tx, ty, bg) => {
            g.font = "600 14px 'DM Sans', system-ui, sans-serif";
            const tw = g.measureText(text).width + 12;
            tx = Math.min(Math.max(ox, tx), ox + vw - tw); ty = Math.min(Math.max(oy + 22, ty), oy + vh);
            g.fillStyle = bg; g.fillRect(tx, ty - 22, tw, 22);
            g.fillStyle = "#fff"; g.fillText(text, tx + 6, ty - 6);
          };
          tag(`${cfg.classes[k]} ${pct(f.probs[k])}`, X, Y, "#2f8f3a");
          const below = Y + Ht + 24 > oy + vh - 4 ? Y + 46 : Y + Ht + 24;
          tag(`detector ${pct(f.det_conf)}`, X, below, "rgba(40,40,40,.85)");
        }
      }
      requestAnimationFrame(draw);
    }
    requestAnimationFrame(draw);

    async function runExample(e) {
      if (busy) return;
      setStatus(`Example: ${e.label}. Results computed in advance with the same models.`, null);
      const r = await (await fetch(BASE + e.result)).json();
      video.src = BASE + e.video;
      show({ frames: r.frames, fps: r.fps, cut: false }, `Clip: ${e.title} (${e.credit}, ${e.license}).`);
    }

    async function runFile(f) {
      if (busy || !f) return;
      busy = true; root.classList.add("ct-busy");
      try {
        await loadModels(setStatus);
        video.removeAttribute("crossorigin");
        video.src = URL.createObjectURL(f);
        await new Promise((res, rej) => { video.onloadedmetadata = res; video.onerror = () => rej(new Error("this browser cannot play that video format")); });
        const result = await analyse(video, +fpsSel.value, setStatus);
        setStatus("Done. Play the video to see the boxes.", 1);
        show(result);
      } catch (err) {
        setStatus(`Sorry, that did not work: ${errText(err)}`, null); console.error(err);
      } finally {
        busy = false; root.classList.remove("ct-busy"); video.setAttribute("crossorigin", "anonymous");
      }
    }

    file.addEventListener("change", () => runFile(file.files[0]));
    examples.forEach((e) => exRow.append($("button", { type: "button", class: "ct-ex", onclick: () => runExample(e) }, e.label)));

    // optional self-test: re-run the first example live and compare with its stored results
    if (new URLSearchParams(location.search).has("selftest") && examples.length) (async () => {
      const want = new URLSearchParams(location.search).get("selftest");
      const e = examples.find((x) => x.id === want) || examples[0];
      await loadModels(setStatus);
      const stored = await (await fetch(BASE + e.result)).json();
      video.src = BASE + e.video;
      await new Promise((res) => { video.onloadedmetadata = res; });
      const live = await analyse(video, stored.fps, setStatus);
      let maxDiff = 0, same = 0; const bad = [];
      live.frames.forEach((f, i) => {
        const s = stored.frames[i];
        if (s && f.probs && s.probs) { maxDiff = Math.max(maxDiff, ...f.probs.map((p, k) => Math.abs(p - s.probs[k]))); }
        if (s && (!!f.probs === !!s.probs) && (!f.probs || argmax(f.probs) === argmax(s.probs))) same++;
        else if (s) bad.push(`t=${f.t.toFixed(2)} live ${f.probs ? cfg.classes[argmax(f.probs)] : "-"}@${f.det_conf.toFixed(2)} stored ${s.probs ? cfg.classes[argmax(s.probs)] : "-"}@${s.det_conf.toFixed(2)}`);
      });
      const msg = `SELFTEST ${e.id}: ${same}/${live.frames.length} frames agree, max prob diff ${maxDiff.toFixed(4)}, ` +
                  `live verdict ${vote(live.frames).pred} vs stored ${vote(stored.frames).pred}` + (bad.length ? ` | differ: ${bad.join('; ')}` : "");
      console.log(msg); setStatus(msg, 1); document.body.dataset.selftest = msg;
      show(live);
    })().catch((err) => { const m = `SELFTEST failed: ${errText(err)}`; console.error(m); setStatus(m, null); document.body.dataset.selftest = m; });
  }

  async function init() {
    const root = document.getElementById("ct-demo");
    if (!root) return;
    try {
      if (typeof ort === "undefined") throw new Error("onnxruntime-web did not load");
      [cfg, examples] = await Promise.all([fetch(BASE + "config.json").then((r) => r.json()),
                                           fetch(BASE + "examples/examples.json").then((r) => r.json())]);
      build(root);
    } catch (err) {
      root.textContent = `The demo could not start (${errText(err)}).`;
    }
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
