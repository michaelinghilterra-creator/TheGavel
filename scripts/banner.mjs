// Regenerates assets/thegavel.svg: node scripts/banner.mjs assets/thegavel.svg
// Pixel-art banner: the name, a tagline, and one rating's probability over the digits 0 to 5.
import { writeFileSync, mkdirSync } from 'node:fs';
import { dirname } from 'node:path';

const OUT = process.argv[2] || 'assets/thegavel.svg';
const W = 1200, H = 400, CELL = 8;

const FONT = {
  T: ['11111', '00100', '00100', '00100', '00100', '00100', '00100'],
  H: ['10001', '10001', '10001', '11111', '10001', '10001', '10001'],
  E: ['11111', '10000', '10000', '11110', '10000', '10000', '11111'],
  G: ['01110', '10001', '10000', '10111', '10001', '10001', '01110'],
  A: ['01110', '10001', '10001', '11111', '10001', '10001', '10001'],
  V: ['10001', '10001', '10001', '10001', '01010', '01010', '00100'],
  L: ['10000', '10000', '10000', '10000', '10000', '10000', '11111'],
};

const NAME = 'THEGAVEL';
const X0 = 64, Y0 = 72;
let cx = 0;
const px = [];
for (const ch of NAME) {
  FONT[ch].forEach((row, r) => [...row].forEach((b, c) => {
    if (b === '1') px.push(`<rect x="${X0 + (cx + c) * CELL}" y="${Y0 + r * CELL}" width="7" height="7"/>`);
  }));
  cx += 6;
}
const nameW = (cx - 1) * CELL;

// One rating, read as a probability over the digits (an invented example).
const probs = [0.01, 0.02, 0.06, 0.17, 0.56, 0.18];
const ev = probs.reduce((s, p, v) => s + p * v, 0);
const sd = Math.sqrt(probs.reduce((s, p, v) => s + p * (v - ev) ** 2, 0));
const BX = 760, BASE = 300, MAXH = 176, BW = 48, BG = 20;
const colors = ['#30363d', '#3d444d', '#4b5563', '#2dd4bf', '#38bdf8', '#a78bfa'];
const bars = probs.map((p, v) => {
  const h = Math.max(CELL, Math.round((p * MAXH) / CELL / 0.56) * CELL);
  const x = BX + v * (BW + BG);
  const cells = [];
  for (let y = BASE - h; y < BASE; y += CELL) for (let xx = x; xx < x + BW; xx += CELL) cells.push(`<rect x="${xx}" y="${y}" width="7" height="7"/>`);
  return `<g class="bar b${v}" fill="${colors[v]}">${cells.join('')}</g>` +
    `<text class="mono digit" x="${x + BW / 2}" y="${BASE + 30}" text-anchor="middle">${v}</text>` +
    `<text class="mono pct" x="${x + BW / 2}" y="${BASE - h - 10}" text-anchor="middle">${Math.round(p * 100)}%</text>`;
});
const evX = BX + ev * (BW + BG) + BW / 2;

const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-labelledby="t d">
<title id="t">TheGavel</title>
<desc id="d">Pixel-art banner. TheGavel reads the probability of each rating digit instead of letting the model write the score. Example: one rating with most probability on 4, expected value ${ev.toFixed(2)}.</desc>
<style>
  .mono { font-family: ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, 'Liberation Mono', monospace; }
  .sub { font-size: 22px; fill: #c9d1d9; }
  .dim { font-size: 17px; fill: #8b949e; }
  .digit { font-size: 20px; font-weight: 700; fill: #c9d1d9; }
  .pct { font-size: 14px; fill: #8b949e; }
  .ev { font-size: 16px; fill: #e6edf3; }
  .bar { opacity: .5; animation: read 4.5s ease-in-out infinite; }
  .b1 { animation-delay: .15s; } .b2 { animation-delay: .3s; } .b3 { animation-delay: .45s; } .b4 { animation-delay: .6s; } .b5 { animation-delay: .75s; }
  @keyframes read { 0%, 45%, 100% { opacity: .5; } 15% { opacity: 1; } }
  .marker { animation: settle 4.5s ease-out infinite; }
  @keyframes settle { 0%, 20% { opacity: 0; } 32%, 85% { opacity: 1; } 100% { opacity: 0; } }
  .cursor { animation: blink 1.1s steps(1) infinite; }
  @keyframes blink { 50% { opacity: 0; } }
  @media (prefers-reduced-motion: reduce) { .bar, .marker, .cursor { animation: none; } .bar, .marker { opacity: 1; } }
</style>
<rect width="${W}" height="${H}" rx="16" fill="#0d1117"/>
<g fill="#e6edf3">${px.join('')}</g>
<rect class="cursor" x="${X0 + nameW + 16}" y="${Y0}" width="7" height="56" fill="#2dd4bf"/>
<text class="mono sub" x="${X0}" y="${Y0 + 104}">read the probabilities.</text>
<text class="mono sub" x="${X0}" y="${Y0 + 136}">don't let the judge talk.</text>
<text class="mono dim" x="${X0}" y="${Y0 + 196}">28 tokens, not 579</text>
<text class="mono dim" x="${X0}" y="${Y0 + 224}">0 malformed outputs</text>
<text class="mono dim" x="${X0}" y="${Y0 + 252}">byte-identical runs</text>
${bars.join('\n')}
<line x1="${BX - 12}" y1="${BASE}" x2="${BX + 6 * (BW + BG) - BG + 12}" y2="${BASE}" stroke="#30363d" stroke-width="2"/>
<g class="marker">
  <line x1="${evX.toFixed(1)}" y1="${BASE - MAXH - 40}" x2="${evX.toFixed(1)}" y2="${BASE}" stroke="#e6edf3" stroke-width="2" stroke-dasharray="4 6"/>
  <text class="mono ev" x="${evX.toFixed(1)}" y="${BASE - MAXH - 50}" text-anchor="middle">ev ${ev.toFixed(2)} · sd ${sd.toFixed(2)}</text>
</g>
</svg>
`;
mkdirSync(dirname(OUT), { recursive: true });
writeFileSync(OUT, svg);
console.log(`wrote ${OUT} (ev ${ev.toFixed(2)}, sd ${sd.toFixed(2)})`);
