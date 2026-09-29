/**
 * Deterministic Procedural Acoustic Sonar SVG Generators
 * Replicating side-scan and multibeam sonar displays from the original dashboard.
 */

export function boxLabel(x, y, w, h, text, color) {
  const cx = x + w / 2;
  const cy = y + h / 2;
  const labelWidth = Math.max(w, text.length * 5.6);
  return `
    <g class="sonar-bbox" style="transform-origin:${cx}px ${cy}px;">
      <rect x="${x}" y="${y}" width="${w}" height="${h}" fill="none" stroke="${color}" stroke-width="1.6"/>
      <rect x="${x}" y="${y - 14}" width="${labelWidth}" height="13" fill="${color}"/>
      <text x="${x + 3}" y="${y - 4}" font-family="IBM Plex Mono" font-size="8.5" fill="#04211D" font-weight="600">${text}</text>
    </g>
  `;
}

export function sonarNoiseSVG(seed, withBoxes) {
  let bands = '';
  let rng = seed;
  const rand = () => {
    rng = (rng * 9301 + 49297) % 233280;
    return rng / 233280;
  };

  for (let i = 0; i < 70; i++) {
    const y = rand() * 190;
    const h = 1 + rand() * 2.4;
    const x = rand() * 300;
    const w = 20 + rand() * 160;
    const op = 0.05 + rand() * 0.35;
    const c = rand() > 0.5 ? '#2DD4C4' : '#0F3A46';
    bands += `<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="${c}" opacity="${op}"/>`;
  }

  let boxes = '';
  if (withBoxes) {
    boxes += boxLabel(58, 52, 70, 44, 'GHOST NET 93%', '#E8604C');
    boxes += boxLabel(168, 96, 58, 36, 'METAL 88%', '#F0A93B');
    boxes += boxLabel(206, 26, 44, 30, 'DEBRIS 81%', '#F0A93B');
  }

  return `
    <rect width="300" height="190" fill="#020a0f"/>
    <rect width="300" height="190" fill="url(#scan${seed})" opacity=".5"/>
    ${bands}
    <line x1="0" y1="95" x2="300" y2="95" stroke="#123244" stroke-width="1"/>
    ${boxes}
    <defs>
      <linearGradient id="scan${seed}" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#0B2131"/>
        <stop offset="100%" stop-color="#020a0f"/>
      </linearGradient>
    </defs>
  `;
}

export function bandsFor(W, H, seed) {
  let rng = seed * 7;
  const rand = () => {
    rng = (rng * 9301 + 49297) % 233280;
    return rng / 233280;
  };
  let out = '';
  for (let i = 0; i < 90; i++) {
    const y = rand() * H;
    const h = 1 + rand() * 2.6;
    const x = rand() * W;
    const w = 20 + rand() * (W * 0.5);
    const op = 0.05 + rand() * 0.3;
    const c = rand() > 0.5 ? '#2DD4C4' : '#0F3A46';
    out += `<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="${c}" opacity="${op}"/>`;
  }
  return out;
}

export function reviewSVG() {
  return `
    <rect width="480" height="320" fill="#020a0f"/>
    ${bandsFor(480, 320, 4)}
    <g class="sonar-bbox" style="transform-origin: 215px 155px;">
      <rect x="140" y="110" width="150" height="90" fill="none" stroke="#E8604C" stroke-width="2.2"/>
      <rect x="140" y="94" width="150" height="16" fill="#E8604C"/>
      <text x="146" y="106" font-family="IBM Plex Mono" font-size="10" fill="#04211D" font-weight="600">GHOST NET · 93%</text>
    </g>
    <line x1="0" y1="160" x2="480" y2="160" stroke="#123244" stroke-width="1"/>
  `;
}
