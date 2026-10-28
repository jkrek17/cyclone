/* Cyclone Phase Space, live: the low markers.
   Every low center is drawn by Hart's three axes, one visual channel each:
   color the lower core (red warm, blue cold); a triangle when asymmetric,
   a circle when symmetric; filled when deep (the upper core agrees with the
   lower), a hollow ring when shallow. The axes come from the class shown
   (clsOf): the two cores from its cell, asymmetry from its B wall, and for
   class 6, which spans both sides of the wall, from B itself (bOf).

   Each glyph is a list of SVG paths, drawn to canvas images for the map
   (Path2D), straight to a canvas in the phase space (glyphDraw) and
   inlined as SVG in the legend and the panels, so they always match.
   Everything else that marks a low by its class follows the same code:
   tracks, footprints and the cells of the phase space take the lower
   core's color (coreHexOf), a deep cell stronger than a shallow one.
   Uses the globals from app.js and storms.js. */
'use strict';

const GLYPH = { warm: '#e0362c', cold: '#2f74e0', ink: '#0b1220', hollow: '#f2f5f9', r: 7, ratio: 2 };
const GLYPH_SIZE = [1, 1.3];  // icon-size at zoom 0 and from LABEL_ZOOM (app.js)
const GLYPH_HALF = 12;        // the drawn box, half its side in CSS px
// Each class's lower and upper core, warm (1) or cold (0), and whether it is
// asymmetric (class 6 by its own B).
const CORES = [[1, 1], [1, 0], [1, 1], [1, 0], [0, 0], [0, 0], [0, 1]];
const ASYM = [false, false, true, true, true, false, null];
const GLYPH_ALT = '#8b97a8';  // the thin ring where the field's class differs
const GLYPH_HELP = 'Color is the lower core, red warm and blue cold. Triangle asymmetric, circle symmetric. Filled is deep (the upper core agrees); a hollow ring is shallow.';

const coreHex = (w) => (w ? GLYPH.warm : GLYPH.cold);
// The lower core's color of a class code, gray where it is missing.
const coreHexOf = (k) => (k == null || blank(k) || !CORES[+k] ? '#8a8a86' : coreHex(CORES[+k][0]));
// Whether a class code is deep (its upper core agrees with the lower).
const deepCls = (k) => !!CORES[+k] && CORES[+k][0] === CORES[+k][1];

// The three axes of class k with B b: { lo, up, asym }, or null.
function axesOfCls(k, b) {
  if (k == null || blank(k) || !CORES[+k]) return null;
  const [lo, up] = CORES[+k];
  const asym = ASYM[+k] ?? (!blank(b) && +b > FIG1.onset);
  return { lo, up, asym };
}

// The three axes of a low, from the class shown, or null.
const axesOf = (c) => axesOfCls(clsOf(c), bOf(c));

/* ---------- geometry, centered on 0,0, in CSS px ---------- */

const f2 = (v) => +v.toFixed(2);
const circ = (r) => `M ${f2(-r)} 0 A ${f2(r)} ${f2(r)} 0 1 0 ${f2(r)} 0 A ${f2(r)} ${f2(r)} 0 1 0 ${f2(-r)} 0 Z`;
// The triangle, centered on its centroid, R from it to each corner.
const tri = (R) => `M 0 ${f2(-R)} L ${f2(0.866 * R)} ${f2(0.5 * R)} L ${f2(-0.866 * R)} ${f2(0.5 * R)} Z`;

function glyphPaths({ lo, up, asym }) {
  const r = GLYPH.r;
  const deep = lo === up;
  const d = asym ? tri(r * 1.3) : circ(deep ? r : r - 0.6);
  return deep ? [{ d, fill: coreHex(lo), stroke: GLYPH.ink, w: 1.1 }] : [{ d, fill: GLYPH.hollow, stroke: coreHex(lo), w: 2.8 }];
}

const glyphName = ({ lo, up, asym }) => `g-${lo}${lo === up ? 1 : 0}${asym ? 1 : 0}`;

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

// Every glyph into the map's sprite, once per style.
function addGlyphImages(m) {
  for (const lo of [0, 1]) for (const up of [0, 1]) for (const asym of [false, true]) {
    const a = { lo, up, asym };
    const name = glyphName(a);
    if (!m.hasImage(name)) m.addImage(name, glyphImage(glyphPaths(a)), { pixelRatio: GLYPH.ratio });
  }
}

// The marker properties of a low for the lows source: the glyph's image and
// the halo's color, or null where the class is missing.
function glyphProps(c) {
  const a = axesOf(c);
  return a ? { glyph: glyphName(a), halo: coreHex(a.lo) } : null;
}

// How far a low's glyph reaches from its center on screen at zoom z, in CSS
// px, so the labels clear it, or null where the low has no glyph.
function glyphReach(c, z) {
  const a = c.terrain ? null : axesOf(c);
  if (!a) return null;
  const k = GLYPH_SIZE[0] + (GLYPH_SIZE[1] - GLYPH_SIZE[0]) * Math.min(1, Math.max(0, z / LABEL_ZOOM));
  return Math.ceil(GLYPH.r * (a.asym ? 1.35 : 1) * k);  // the triangle reaches 1.3 r to its corners
}

// A glyph on a 2-D canvas at x, y, scaled so its circle has radius r
// (CSS px), at opacity alpha.
function glyphDraw(ctx, a, x, y, r, alpha = 1) {
  const k = r / GLYPH.r;
  ctx.save();
  ctx.translate(x, y);
  ctx.scale(k, k);
  ctx.globalAlpha = alpha;
  ctx.lineJoin = 'round';
  for (const p of glyphPaths(a)) {
    const path = new Path2D(p.d);
    if (p.fill) { ctx.fillStyle = p.fill; ctx.fill(path); }
    if (p.stroke) { ctx.strokeStyle = p.stroke; ctx.lineWidth = p.w; ctx.stroke(path); }
  }
  ctx.restore();
}

/* ---------- legend and panels ---------- */

function glyphSVG(a, size = 20) {
  const h = GLYPH_HALF;
  const el = (p) => `<path d="${p.d}" fill="${p.fill || 'none'}"${p.stroke ? ` stroke="${p.stroke}" stroke-width="${p.w}" stroke-linejoin="round"` : ''}/>`;
  return `<svg class="glyph" viewBox="${-h} ${-h} ${2 * h} ${2 * h}" width="${size}" height="${size}" aria-hidden="true">${glyphPaths(a).map(el).join('')}</svg>`;
}

// A low's glyph inline in the panels and tooltips, with the thin gray ring
// and its title where the field's class differs (classDiffers).
function glyphDot(e, size = 18) {
  const a = axesOf(e);
  const svg = a ? glyphSVG(a, size) : `<svg class="glyph" viewBox="-12 -12 24 24" width="${size}" height="${size}" aria-hidden="true"><path d="${circ(GLYPH.r - 1)}" fill="#8a8a86"/></svg>`;
  return classDiffers(e) ? `<span class="glyph-alt" title="${esc(diffTitle(e))}">${svg}</span>` : svg;
}

// The field's class (steering flow) as a glyph, where it differs.
const fieldGlyph = (e, size = 14) => { const a = axesOfCls(e.cls, e.hb); return a ? glyphSVG(a, size) : ''; };

// One glyph per class, for the legend; class 6 drawn symmetric.
function glyphLegendRows() {
  return [...S.classes.values()].map((c) => {
    const [lo, up] = CORES[c.code] || [0, 0];
    return `<li title="${esc(c.name)}">${glyphSVG({ lo, up, asym: !!ASYM[c.code] })}<b>${c.code}</b>${esc(SHORT[c.code] ?? c.name)}</li>`;
  }).join('');
}
