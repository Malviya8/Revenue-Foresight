/*
 * Motion layer for Revenue Foresight.
 *
 * This file is injected into the top-level app document (not the component
 * iframe), so the observers below survive Streamlit reruns.
 */
(function () {
  "use strict";

  if (window.__rfMotionReady) return;
  window.__rfMotionReady = true;

  var doc = document;
  var root = doc.documentElement;

  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    root.classList.add("rf-booted");
    return;
  }

  var REVEAL = [
    '[data-testid="stMetric"]',
    '[data-testid="stPlotlyChart"]',
    '[data-testid="stDataFrame"]',
    ".rf-card",
    ".rf-issue"
  ].join(",");

  var SPOTLIGHT = '.rf-card, .rf-hero, [data-testid="stMetric"]';

  var booted = false;

  /* ----------------------------------------------------------------------
   * Scroll reveal
   * -------------------------------------------------------------------- */

  var io = new IntersectionObserver(
    function (entries) {
      for (var i = 0; i < entries.length; i++) {
        if (!entries[i].isIntersecting) continue;
        entries[i].target.classList.add("rf-in");
        io.unobserve(entries[i].target);
      }
    },
    { rootMargin: "0px 0px -6% 0px", threshold: 0.06 }
  );

  function inView(el) {
    var r = el.getBoundingClientRect();
    if (!r.height && !r.width) return false;
    var vh = window.innerHeight || root.clientHeight;
    return r.top < vh * 0.94 && r.bottom > 0;
  }

  /* Side-by-side columns cascade left to right instead of arriving together. */
  function staggerDelay(el) {
    var col = el.closest('[data-testid="stColumn"], [data-testid="column"]');
    if (!col || !col.parentElement) return 0;
    var sibs = Array.prototype.slice.call(col.parentElement.children);
    var i = sibs.indexOf(col);
    return i > 0 ? Math.min(i, 5) * 70 : 0;
  }

  function armReveal(el) {
    if (el.dataset.rfReveal) return;
    el.dataset.rfReveal = "1";
    // Once booted, anything already on screen is left alone, so a rerun never
    // replays an entrance the user has already watched.
    if (booted && inView(el)) return;
    var delay = staggerDelay(el);
    if (delay) el.style.transitionDelay = delay + "ms";
    el.classList.add("rf-reveal");
    io.observe(el);
  }

  /* ----------------------------------------------------------------------
   * Cursor spotlight
   * -------------------------------------------------------------------- */

  var frame = 0;
  var pending = null;

  function applySpotlight() {
    frame = 0;
    if (!pending) return;
    var el = pending.el;
    var r = el.getBoundingClientRect();
    if (r.width && r.height) {
      el.style.setProperty("--rf-x", (((pending.x - r.left) / r.width) * 100).toFixed(2) + "%");
      el.style.setProperty("--rf-y", (((pending.y - r.top) / r.height) * 100).toFixed(2) + "%");
    }
    pending = null;
  }

  doc.addEventListener(
    "mousemove",
    function (event) {
      var node = event.target;
      if (!node || node.nodeType !== 1 || !node.closest) return;
      var el = node.closest(SPOTLIGHT);
      if (!el) return;
      pending = { el: el, x: event.clientX, y: event.clientY };
      if (!frame) frame = window.requestAnimationFrame(applySpotlight);
    },
    { passive: true }
  );

  /* ----------------------------------------------------------------------
   * Metric count-up
   *
   * Keyed on the metric label rather than the node, because Streamlit throws
   * the node away on every rerun. That way an unchanged number stays still
   * and a changed number tweens from its previous value.
   * -------------------------------------------------------------------- */

  var seen = new Map();
  var NUMBER = /-?\d[\d,]*(?:\.\d+)?/;

  function group(text) {
    var parts = text.split(".");
    parts[0] = parts[0].replace(/\B(?=(\d{3})+(?!\d))/g, ",");
    return parts.join(".");
  }

  function countUp(el) {
    var text = el.textContent;
    if (!text) return;

    // Ranges such as "$12,000 - $30,000" hold two numbers; tweening them is noise.
    var all = text.match(/-?\d[\d,]*(?:\.\d+)?/g);
    if (!all || all.length !== 1) return;

    var match = text.match(NUMBER);
    var raw = match[0];
    var target = parseFloat(raw.replace(/,/g, ""));
    if (!isFinite(target)) return;

    var box = el.closest('[data-testid="stMetric"]');
    var label = box && box.querySelector('[data-testid="stMetricLabel"]');
    var key = label && label.textContent ? label.textContent.trim() : text;

    var prev = seen.get(key);
    seen.set(key, target);
    if (prev !== undefined && Math.abs(prev - target) < 1e-9) return;

    var decimals = raw.indexOf(".") >= 0 ? raw.split(".")[1].length : 0;
    var grouped = raw.indexOf(",") >= 0;
    var prefix = text.slice(0, match.index);
    var suffix = text.slice(match.index + raw.length);
    var from = prev === undefined ? 0 : prev;
    var start = 0;

    function step(now) {
      if (!start) start = now;
      var p = Math.min(1, (now - start) / 900);
      var eased = 1 - Math.pow(1 - p, 3);
      var value = from + (target - from) * eased;
      var shown = decimals ? value.toFixed(decimals) : String(Math.round(value));
      el.textContent = prefix + (grouped ? group(shown) : shown) + suffix;
      if (p < 1) {
        window.requestAnimationFrame(step);
      } else {
        el.textContent = text;
      }
    }

    window.requestAnimationFrame(step);
  }

  /* ----------------------------------------------------------------------
   * Wiring
   * -------------------------------------------------------------------- */

  function scan() {
    var targets = doc.querySelectorAll(REVEAL);
    for (var i = 0; i < targets.length; i++) armReveal(targets[i]);

    var metrics = doc.querySelectorAll('[data-testid="stMetricValue"]');
    for (var j = 0; j < metrics.length; j++) {
      if (metrics[j].dataset.rfCount) continue;
      metrics[j].dataset.rfCount = "1";
      countUp(metrics[j]);
    }
  }

  var queued = false;
  function queueScan() {
    if (queued) return;
    queued = true;
    window.requestAnimationFrame(function () {
      queued = false;
      scan();
    });
  }

  new MutationObserver(queueScan).observe(doc.body, { childList: true, subtree: true });

  scan();
  window.setTimeout(function () {
    booted = true;
    root.classList.add("rf-booted");
  }, 1500);
})();
