/*
 * frontend/assets/anim-runtime.js
 * 教学动画运行时：在沙箱 iframe 中为 LLM 生成的 p5.js sketch 提供稳定 API。
 *
 * LLM 生成的代码只调用全局 defineAnimation({ width, height, title, scenes:[{name,duration,draw}] })。
 * 时间轴、播放控制、标题/进度绘制、缓动与公式渲染全部由本运行时接管。
 *
 * 工具集 u 的所有绘图方法都【不需要】传入 p5 实例作为首参（运行时内部持有 _p）。
 * 为兼容旧代码，若首参恰好是 p5 实例会被自动忽略。
 */
(function () {
  "use strict";

  var PALETTE = {
    bg: "#FFFFFF", primary: "#4F6EF7", accent: "#C77B3C",
    text: "#1F2937", muted: "#9CA3AF", ok: "#10B981", warn: "#EF4444"
  };

  // 布局常量：标题占顶部，进度条占底部，中间为内容安全区
  var TOP = 56, BOTTOM = 30;

  var _p = null; // 当前 p5 实例，由 setup 注入

  // 若首参是 p5 实例则剥离（兼容旧的 u.fn(p, ...) 写法）
  function _args(a) {
    if (a.length && a[0] && typeof a[0] === "object" && typeof a[0].ellipse === "function") {
      return Array.prototype.slice.call(a, 1);
    }
    return Array.prototype.slice.call(a);
  }

  // ---- 缓动与插值 ----
  function ease(t) {
    t = Math.max(0, Math.min(1, t));
    return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
  }
  function lerp(a, b, t) { return a + (b - a) * t; }
  function lerpColor() {
    var a = _args(arguments), c1 = a[0], c2 = a[1], t = a[2];
    try {
      return _p.lerpColor(_p.color(c1), _p.color(c2), Math.max(0, Math.min(1, t == null ? 0 : t)));
    } catch (e) {
      try { return _p.color(PALETTE.primary); } catch (e2) { return PALETTE.primary; }
    }
  }
  // 安全设色：非法颜色降级为默认色而非抛错崩溃
  function safeFill() {
    var a = _args(arguments);
    try { _p.fill.apply(_p, a); } catch (e) { _p.fill(PALETTE.text); }
  }
  function safeStroke() {
    var a = _args(arguments);
    try { _p.stroke.apply(_p, a); } catch (e) { _p.stroke(PALETTE.text); }
  }

  // ---- 绘图辅助（均不需传 p）----
  function arrow() {
    var a = _args(arguments), x1 = a[0], y1 = a[1], x2 = a[2], y2 = a[3];
    _p.push();
    _p.stroke(PALETTE.text); _p.strokeWeight(2); _p.fill(PALETTE.text);
    _p.line(x1, y1, x2, y2);
    var ang = Math.atan2(y2 - y1, x2 - x1), s = 9;
    _p.translate(x2, y2); _p.rotate(ang);
    _p.triangle(0, 0, -s, -s / 2.2, -s, s / 2.2);
    _p.pop();
  }
  function label() {
    var a = _args(arguments), text = a[0], x = a[1], y = a[2], size = a[3];
    _p.push();
    _p.noStroke(); _p.fill(PALETTE.text);
    _p.textAlign(_p.CENTER, _p.CENTER);
    var fs = size || 18;
    _p.textSize(fs);
    // 自动缩小 + 钳制，避免居中文字超出画布左右边缘
    var CW = TOOLS.W || 800, M = 10;
    var tw = _p.textWidth(String(text));
    if (tw > CW - 2 * M && fs > 10) {
      fs = Math.max(10, Math.floor(fs * (CW - 2 * M) / tw));
      _p.textSize(fs);
      tw = _p.textWidth(String(text));
    }
    var half = tw / 2;
    var cx = Math.max(M + half, Math.min(x, CW - M - half));
    _p.text(text, cx, y);
    _p.pop();
  }

  // ---- LaTeX 叠加层管理 ----
  // KaTeX 渲染为 DOM，绝对定位覆盖在 canvas 上方（字体可靠生效）。
  var latexLayer = null;
  var latexPool = {}; // key -> {el, used}
  function ensureLatexLayer() {
    if (latexLayer) return;
    latexLayer = document.createElement("div");
    latexLayer.style.position = "absolute";
    latexLayer.style.left = "0";
    latexLayer.style.top = "0";
    latexLayer.style.pointerEvents = "none";
    latexLayer.style.transformOrigin = "top left";
    var holder = document.getElementById("anim-canvas-holder") || document.getElementById("anim-root");
    holder.appendChild(latexLayer);
  }
  function latexBeginFrame() { for (var k in latexPool) latexPool[k].used = false; }
  function latexEndFrame() {
    for (var k in latexPool) {
      if (!latexPool[k].used) latexPool[k].el.style.display = "none";
    }
  }
  // drawLatex(tex, x, y, size)：x,y 视为公式【中心点】，便于居中布局
  function drawLatex() {
    var a = _args(arguments), tex = a[0], x = a[1], y = a[2], size = a[3];
    ensureLatexLayer();
    var fontSize = size || 22;
    var key = fontSize + "::" + tex;
    var entry = latexPool[key];
    if (!entry) {
      var el = document.createElement("div");
      el.style.position = "absolute";
      el.style.whiteSpace = "nowrap";
      el.style.color = PALETTE.text;
      try {
        if (window.katex) window.katex.render(tex, el, { throwOnError: false, displayMode: false });
        else el.textContent = tex;
      } catch (e) { el.textContent = tex; }
      latexLayer.appendChild(el);
      entry = latexPool[key] = { el: el, used: false };
    }
    entry.el.style.fontSize = fontSize + "px";
    entry.el.style.display = "block";
    // 居中：用元素自身尺寸把 (x,y) 当作中心
    var w = entry.el.offsetWidth || 0, h = entry.el.offsetHeight || 0;
    var CW = TOOLS.W || 800, CH = TOOLS.H || 460, M = 8;
    // 若公式比画布还宽，自动缩小字号直到放得下（最低 10px）
    if (w > CW - 2 * M && fontSize > 10) {
      var scaled = Math.max(10, Math.floor(fontSize * (CW - 2 * M) / w));
      entry.el.style.fontSize = scaled + "px";
      w = entry.el.offsetWidth || w;
      h = entry.el.offsetHeight || h;
    }
    // 钳制位置：元素整体不超出画布边缘（留 M 边距）
    var left = x - w / 2, top = y - h / 2;
    left = Math.max(M, Math.min(left, CW - w - M));
    top = Math.max(M, Math.min(top, CH - h - M));
    entry.el.style.left = left + "px";
    entry.el.style.top = top + "px";
    entry.used = true;
  }

  var TOOLS = {
    ease: ease, lerp: lerp, lerpColor: lerpColor,
    arrow: arrow, label: label, drawLatex: drawLatex,
    fill: safeFill, stroke: safeStroke,
    palette: PALETTE,
    // 布局辅助（在 setup 后由运行时填充实际值）
    cx: 400, cy: 258, W: 800, H: 460,
    safeTop: TOP, safeBottom: 0
  };

  window.__ANIM_TOOLS__ = TOOLS;
  window.__ANIM_PALETTE__ = PALETTE;
  window.__animSetP = function (p) { _p = p; };
  window.__animLatexHooks = { begin: latexBeginFrame, end: latexEndFrame };

  // ====================================================================
  // 运行时引擎：接收 defineAnimation 配置，驱动 p5 时间轴与播放控制
  // ====================================================================
  var MIN_DURATION = 3500;  // 每场景最短时长（毫秒），防止一闪而过
  var HOLD_MS = 900;        // 场景内容绘制完成后的静止观察期（毫秒）

  var STATE = {
    config: null, p5i: null,
    playing: true, startMs: 0, elapsedMs: 0, totalMs: 0,
    bounds: [],  // 每个场景的 [startMs, endMs]（含 hold）
    holds: [],   // 每个场景的 holdStartMs（该点后 t 固定为 1）
  };

  function computeBounds(scenes) {
    var bounds = [], holds = [], acc = 0;
    for (var i = 0; i < scenes.length; i++) {
      var raw = Math.max(MIN_DURATION, scenes[i].duration || MIN_DURATION);
      var total = raw + HOLD_MS;          // 实际时长 = 动画期 + 静止观察期
      bounds.push([acc, acc + total]);
      holds.push(acc + raw);              // 到达此点后 t 锁定为 1（静止）
      acc += total;
    }
    STATE.totalMs = acc;
    STATE.holds = holds;
    return bounds;
  }

  function fmtErr(msg) {
    var root = document.getElementById("anim-root");
    if (root) {
      root.innerHTML =
        '<div style="padding:24px;color:#EF4444;font-family:sans-serif;font-size:14px;line-height:1.6">' +
        "⚠️ 动画运行出错：" + String(msg) + "</div>";
    }
  }

  window.defineAnimation = function (config) {
    if (!config || !Array.isArray(config.scenes) || config.scenes.length === 0) {
      fmtErr("动画配置无效（缺少 scenes）");
      return;
    }
    STATE.config = config;
    STATE.bounds = computeBounds(config.scenes);

    var W = config.width || 800, H = config.height || 460;
    var TITLE = config.title || "教学动画";
    // 填充布局辅助：内容区中心
    TOOLS.W = W; TOOLS.H = H;
    TOOLS.cx = W / 2;
    TOOLS.cy = TOP + (H - TOP - BOTTOM) / 2;
    TOOLS.safeBottom = H - BOTTOM;

    var sketch = function (p) {
      p.setup = function () {
        var c = p.createCanvas(W, H);
        c.parent("anim-canvas-holder");
        p.textFont("system-ui, -apple-system, 'Microsoft YaHei', sans-serif");
        if (window.__animSetP) window.__animSetP(p);
        STATE.startMs = p.millis();
      };

      p.draw = function () {
        if (STATE.playing) {
          STATE.elapsedMs = p.millis() - STATE.startMs;
          if (STATE.elapsedMs >= STATE.totalMs) {
            STATE.elapsedMs = STATE.totalMs;
            STATE.playing = false;
            syncButtons();
          }
        }
        var now = Math.min(STATE.elapsedMs, STATE.totalMs - 1);

        // 找到当前场景
        var idx = 0;
        for (var i = 0; i < STATE.bounds.length; i++) {
          if (now >= STATE.bounds[i][0] && now < STATE.bounds[i][1]) { idx = i; break; }
          idx = i;
        }
        var sc = STATE.config.scenes[idx];
        var b = STATE.bounds[idx];
        var holdStart = STATE.holds[idx];
        // 动画期内 t 从 0→1，进入 hold 后锁定为 1（静止观察）
        var animSpan = Math.max(1, holdStart - b[0]);
        var t = Math.min(1, (now - b[0]) / animSpan);

        // 背景 + 标题 + 步骤名
        p.background(PALETTE.bg);
        latexBeginFrame();
        p.push();
        p.noStroke(); p.fill(PALETTE.text);
        p.textAlign(p.LEFT, p.TOP); p.textSize(16); p.textStyle(p.BOLD);
        p.text(TITLE, 20, 14);
        p.textStyle(p.NORMAL); p.fill(PALETTE.muted); p.textSize(12);
        p.text("步骤 " + (idx + 1) + "/" + STATE.config.scenes.length + "　" + (sc.name || ""), 20, 36);
        p.pop();

        // 场景内容（单帧异常不崩整个动画：停在错误提示）
        try {
          if (typeof sc.draw === "function") sc.draw(p, t, TOOLS);
        } catch (e) {
          fmtErr("场景「" + (sc.name || (idx + 1)) + "」绘制异常：" + e.message);
          p.noLoop();
          return;
        }

        // 进度条（底部）
        var pr = now / Math.max(1, STATE.totalMs);
        p.push();
        p.noStroke(); p.fill("#EEF0F6");
        p.rect(20, H - 16, W - 40, 5, 3);
        p.fill(PALETTE.primary);
        p.rect(20, H - 16, (W - 40) * pr, 5, 3);
        p.pop();

        latexEndFrame();
      };
    };

    try {
      STATE.p5i = new p5(sketch);
    } catch (e) {
      fmtErr("p5 初始化失败：" + e.message);
    }
    wireControls();
  };

  // ---- 播放控制 ----
  function syncButtons() {
    var btn = document.getElementById("anim-playpause");
    if (btn) btn.textContent = STATE.playing ? "⏸ 暂停" : "▶ 播放";
  }
  function wireControls() {
    var pp = document.getElementById("anim-playpause");
    var rp = document.getElementById("anim-replay");
    if (pp) pp.onclick = function () {
      if (!STATE.playing && STATE.elapsedMs >= STATE.totalMs) return replay();
      STATE.playing = !STATE.playing;
      if (STATE.playing) STATE.startMs = STATE.p5i.millis() - STATE.elapsedMs;
      syncButtons();
    };
    if (rp) rp.onclick = replay;
    syncButtons();
  }
  function replay() {
    if (!STATE.p5i) return;
    STATE.elapsedMs = 0;
    STATE.startMs = STATE.p5i.millis();
    STATE.playing = true;
    STATE.p5i.loop();
    syncButtons();
  }
})();
