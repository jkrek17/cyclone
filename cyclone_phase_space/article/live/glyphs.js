/* Cyclone Phase Space, live: the low markers.
   Four ways to draw a low center (S.marker), chosen in the layers panel:
     class  the class color dot, larger below DEEP_HPA (the original)
     split  A: a split disc, the lower core in the bottom half and the upper
            core in the top, red warm and blue cold; a pale tail toward the
            warm side when asymmetric
     shape  B: the lower core's color; a triangle when asymmetric, a circle
            when symmetric; filled when deep (the upper core agrees with the
            lower), a hollow ring when shallow
     combo  C: B's circle and triangle with A's split fill
   Hart's three axes come from the class shown (clsOf): the two cores from
   its cell, asymmetry from its B wall, and for class 6, which spans both
   sides of the wall, from B itself (bOf).

   A's tail needs a direction. Hart's B is the thickness right of the
   motion minus left of it (times -1 south of the equator), so the warm side
   is 90 degrees right of the heading when B is positive in the north. Only
   tracked lows with B from their own motion have a heading (hdg on the
   track point); an asymmetric low without one gets a second, pale ring in
   place of the tail. Tail and ring are pale with a dark edge, so they read
   on the pale ocean basemap and over the dark globe alike.

   Each glyph is a list of SVG paths, drawn to canvas images for the map
   (Path2D) and inlined as SVG in the legend, so the two always match.
   Uses the globals from app.js and storms.js. */
'use strict';

const MARKERS = ['class', 'split', 'shape', 'combo'];
const MARKER_NAMES = { class: 'Class color', split: 'A: split disc', shape: 'B: color, shape, fill', combo: 'C: shapes, split fill' };
const GLYPH = { warm: '#e0362c', cold: '#2f74e0', ink: '#0b1220', hollow: '#f2f5f9', r: 7, ratio: 2 };
// Each class's lower and upper core, warm (1) or cold (0), and whether it is
// asymmetric (class 6 by its own B).
const CORES = [[1, 1], [1, 0], [1, 1], [1, 0], [0, 0], [0, 0], [0, 1]];
const ASYM = [false, false, true, true, true, false, null];

const coreHex = (w) => (w ? GLYPH.warm : GLYPH.cold);

// The three axes of a low, from the class shown: { lo, up, asym }, or null.
function axesOf(c) {
  const k = clsOf(c);
  if (k == null || blank(k) || !CORES[+k]) return null;
  const [lo, up] = CORES[+k];
  const b = bOf(c);
  const asym = ASYM[+k] ?? (!blank(b) && +b > FIG1.onset);
  return { lo, up, asym };
}

// The bearing of the warm side (degrees clockwise from north), for a tracked
// low with B from its own motion, else null.
function warmBearing(c) {
  if (!trackB(c)) return null;
  const e = stormOf(c)?.at[S.i];
  const b = +bOf(c);
  if (!e || blank(e.hdg) || !b) return null;
  const side = Math.sign(b) * (c.lat < 0 ? -1 : 1);
  return (((+e.hdg + 90 * side) % 360) + 360) % 360;
}

/* ---------- geometry, centered on 0,0, in CSS px ---------- */

const f2 = (v) => +v.toFixed(2);
const circ = (r) => `M ${f2(-r)} 0 A ${f2(r)} ${f2(r)} 0 1 0 ${f2(r)} 0 A ${f2(r)} ${f2(r)} 0 1 0 ${f2(-r)} 0 Z`;
const poly = (pts) => `M ${pts.map(([x, y]) => `${f2(x)} ${f2(y)}`).join(' L ')} Z`;
// The triangle, sat so its two halves above and below y = 0 have equal area:
// apex at -1.06 R, base at +0.44 R, and 0.612 R half-wide at y = 0.
const TRI = (R) => ({
  all: poly([[0, -1.06 * R], [0.866 * R, 0.44 * R], [-0.866 * R, 0.44 * R]]),
  top: poly([[0, -1.06 * R], [0.612 * R, 0], [-0.612 * R, 0]]),
  bot: poly([[-0.612 * R, 0], [0.612 * R, 0], [0.866 * R, 0.44 * R], [-0.866 * R, 0.44 * R]]),
  half: 0.612 * R,
});

function glyphPaths(mode, { lo, up, asym }, ringless = false) {
  const r = GLYPH.r;
  const ink = GLYPH.ink;
  if (mode === 'split') {
    const out = [];
    if (asym && !ringless) out.push({ d: circ(r + 2.6), stroke: ink, w: 3.4 }, { d: circ(r + 2.6), stroke: GLYPH.hollow, w: 1.6 });
    out.push({ d: `M ${-r} 0 A ${r} ${r} 0 0 1 ${r} 0 Z`, fill: coreHex(up) },
      { d: `M ${-r} 0 A ${r} ${r} 0 0 0 ${r} 0 Z`, fill: coreHex(lo) },
      { d: `M ${-r} 0 L ${r} 0`, stroke: ink, w: 0.8 },
      { d: circ(r), stroke: ink, w: 1.1 });
    return out;
  }
  const t = TRI(r * 1.3);
  if (mode === 'shape') {
    const deep = lo === up;
    const d = asym ? t.all : circ(deep ? r : r - 0.6);
    return deep ? [{ d, fill: coreHex(lo), stroke: ink, w: 1.1 }] : [{ d, fill: GLYPH.hollow, stroke: coreHex(lo), w: 2.8 }];
  }
  // combo
  if (asym) {
    return [{ d: t.top, fill: coreHex(up) }, { d: t.bot, fill: coreHex(lo) },
      { d: `M ${f2(-t.half)} 0 L ${f2(t.half)} 0`, stroke: ink, w: 0.8 }, { d: t.all, stroke: ink, w: 1.1 }];
  }
  return glyphPaths('split', { lo, up, asym: false });
}

// A's tail, pointing north from the disc's edge; the map turns it to the warm side.
function tailPaths() {
  const r = GLYPH.r;
  const w = 0.62;
  return [{ d: poly([[-r * Math.sin(w), -r * Math.cos(w)], [0, -2.4 * r], [r * Math.sin(w), -r * Math.cos(w)]]), fill: GLYPH.hollow, stroke: GLYPH.ink, w: 1.1 }];
}

const glyphName = (mode, a, ringless) => `g-${mode}-${a.lo}${a.up}${a.asym ? 1 : 0}${ringless ? 't' : ''}`;
const GLYPH_HALF = 18;  // the drawn box, half its side in CSS px (A's tail reaches 16.8)

/* ---------- map images ---------- */

function glyphImage(paths) {
  const n = GLYPH_HALF * 2 * GLYPH.ratio;
  const cv = document.createElement('canvas');
  cv.width = cv.height = n;
  const ctx = cv.getContext('2d');
  ctx.setTransform(GLYPH.ratio, 0, 0, GLYPH.ratio, n / 2, n / 2);
  ctx.lineJoin = 'round';
  for (const p of paths) {
    const path = new Path2D(p.d);
    if (p.fill) { ctx.fillStyle = p.fill; ctx.fill(path); }
    if (p.stroke) { ctx.strokeStyle = p.stroke; ctx.lineWidth = p.w; ctx.stroke(path); }
  }
  return ctx.getImageData(0, 0, n, n);
}

// Every glyph of every mode into the map's sprite, once per style.
function addGlyphImages(m) {
  const add = (name, paths) => { if (!m.hasImage(name)) m.addImage(name, glyphImage(paths), { pixelRatio: GLYPH.ratio }); };
  for (const mode of MARKERS.slice(1)) {
    for (const lo of [0, 1]) for (const up of [0, 1]) for (const asym of [false, true]) {
      const a = { lo, up, asym };
      add(glyphName(mode, a), glyphPaths(mode, a));
      if (mode === 'split' && asym) add(glyphName(mode, a, true), glyphPaths(mode, a, true));
    }
  }
  add('g-tail', tailPaths());
}

// The marker properties of a low for the lows source: the glyph's image,
// the tail's bearing, and the halo's color. Null in class mode.
function glyphProps(c) {
  if (S.marker === 'class') return null;
  const a = axesOf(c);
  if (!a) return null;
  const tail = S.marker === 'split' && a.asym ? warmBearing(c) : null;
  const out = { glyph: glyphName(S.marker, a, tail != null), halo: coreHex(a.lo) };
  if (tail != null) out.tail = tail;
  return out;
}

// How far a low's glyph reaches from its center on screen at zoom z, in
// CSS px, and whether A's tail points into the right half, where the
// pressure goes: { r, tailRight }, or null for the class dot.
const GLYPH_SIZE = [1, 1.3];  // icon-size at zoom 0 and from LABEL_ZOOM (app.js)
function glyphReach(c, z) {
  const g = c.terrain ? null : glyphProps(c);
  if (!g) return null;
  const k = GLYPH_SIZE[0] + (GLYPH_SIZE[1] - GLYPH_SIZE[0]) * Math.min(1, Math.max(0, z / LABEL_ZOOM));
  const asym = axesOf(c).asym;
  let r = GLYPH.r;
  if (S.marker === 'split') { if (asym && g.tail == null) r += 4.3; }  // the second ring
  else if (asym) r *= 1.25;  // the triangle, between its base corners and its apex
  r *= k;
  const tailRight = g.tail != null && Math.sin((g.tail - map.getBearing()) * Math.PI / 180) > 0.3;
  return { r: Math.ceil(r), tailRight };
}

/* ---------- legend ---------- */

function glyphSVG(mode, a, { tail = null, size = 28 } = {}) {
  const h = GLYPH_HALF;
  const el = (p) => `<path d="${p.d}" fill="${p.fill || 'none'}"${p.stroke ? ` stroke="${p.stroke}" stroke-width="${p.w}" stroke-linejoin="round"` : ''}/>`;
  const t = tail == null ? '' : `<g transform="rotate(${tail})">${tailPaths().map(el).join('')}</g>`;
  return `<svg class="glyph" viewBox="${-h} ${-h} ${2 * h} ${2 * h}" width="${size}" height="${size}" aria-hidden="true">${t}${glyphPaths(mode, a, tail != null).map(el).join('')}</svg>`;
}

// One glyph per class, for the legend; class 6 drawn symmetric.
function glyphLegendRows(mode) {
  return [...S.classes.values()].map((c) => {
    const [lo, up] = CORES[c.code] || [0, 0];
    const a = { lo, up, asym: !!ASYM[c.code] };
    const tail = mode === 'split' && a.asym ? 60 : null;
    return `<li title="${esc(c.name)}">${glyphSVG(mode, a, { tail })}<b>${c.code}</b>${esc(SHORT[c.code] ?? c.name)}</li>`;
  }).join('');
}

const MARKER_HELP = {
  split: 'Bottom half the lower core, top half the upper core: red warm, blue cold. A tail points to the warm side of an asymmetric low; a second ring marks one with no motion to point from.',
  shape: 'Color is the lower core, red warm and blue cold. Triangle asymmetric, circle symmetric. Filled is deep (the upper core agrees); a hollow ring is shallow.',
  combo: 'Triangle asymmetric, circle symmetric. Bottom half the lower core, top half the upper core: red warm, blue cold.',
};

/* ---------- the switch ---------- */

function setMarker(m) {
  if (!MARKERS.includes(m)) return;
  S.marker = m;
  document.querySelectorAll('[data-marker]').forEach((b) => b.setAttribute('aria-checked', String(b.dataset.marker === m)));
  renderLegend();
  if (!S.ready) return;
  drawLows();
  writeHash();
}
