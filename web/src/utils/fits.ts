import type { PreviewCube } from '../types';

const BLOCK = 2880;
const MAX_PIXELS = 60_000_000;

export interface FitsPreview {
  width: number;
  height: number;
  frames: Float32Array[];
}

function readCards(text: string): { cards: Record<string, string>; endBlock: number } {
  const cards: Record<string, string> = {};
  const blockCount = Math.ceil(text.length / BLOCK);
  for (let block = 0; block < blockCount; block += 1) {
    for (let cardIndex = 0; cardIndex < BLOCK / 80; cardIndex += 1) {
      const start = block * BLOCK + cardIndex * 80;
      const card = text.slice(start, start + 80);
      const key = card.slice(0, 8).trim();
      if (key === 'END') return { cards, endBlock: block };
      if (!key || key === 'COMMENT' || key === 'HISTORY') continue;
      const raw = card.slice(10).trim();
      const value = raw.includes('\'')
        ? raw.slice(raw.indexOf('\''), raw.lastIndexOf('\'') + 1).replace(/'/g, '').trim()
        : raw.split(/\s+\//)[0].trim();
      cards[key] = value;
    }
  }
  throw new Error('FITS header has no END card.');
}

function numberCard(cards: Record<string, string>, key: string, fallback: number): number {
  const value = Number.parseInt(cards[key] ?? '', 10);
  return Number.isFinite(value) ? value : fallback;
}

/**
 * Minimal primary-HDU FITS reader for local previews. Supports 2D/3D images
 * with BITPIX 8/16/32/-32/-64 and optional BZERO/BSCALE scaling. Data is sent
 * to the server as the original bytes; this parser only feeds the blink view.
 */
export function parseFitsPreview(buffer: ArrayBuffer): FitsPreview {
  if (buffer.byteLength < BLOCK) throw new Error('FITS file is shorter than one header block.');
  const text = new TextDecoder('ascii').decode(
    new Uint8Array(buffer, 0, Math.min(buffer.byteLength, 200 * BLOCK)),
  );
  const { cards, endBlock } = readCards(text);
  const bitpix = numberCard(cards, 'BITPIX', -32);
  const naxis = numberCard(cards, 'NAXIS', 0);
  if (naxis < 2 || naxis > 3) throw new Error(`Unsupported FITS NAXIS=${naxis} (expected 2 or 3).`);

  const width = numberCard(cards, 'NAXIS1', 0);
  const height = numberCard(cards, 'NAXIS2', 0);
  const frames = naxis === 3 ? numberCard(cards, 'NAXIS3', 1) : 1;
  if (width <= 0 || height <= 0 || frames <= 0) throw new Error('Invalid FITS image dimensions.');
  if (width * height * frames > MAX_PIXELS) throw new Error('FITS cube is too large for an in-browser preview.');

  const bzero = Number(cards['BZERO'] ?? '0') || 0;
  const bscale = Number(cards['BSCALE'] ?? '1') || 1;
  const bytesPerPixel = Math.abs(bitpix) / 8;
  if (![1, 2, 4, 8].includes(bytesPerPixel)) throw new Error(`Unsupported FITS BITPIX=${bitpix}.`);

  const planePixels = width * height;
  const dataOffset = (endBlock + 1) * BLOCK;
  const needed = dataOffset + planePixels * frames * bytesPerPixel;
  if (buffer.byteLength < needed) throw new Error('FITS data section is truncated.');

  const view = new DataView(buffer, dataOffset, planePixels * frames * bytesPerPixel);
  const framesOut: Float32Array[] = new Array(frames);

  for (let f = 0; f < frames; f += 1) {
    const frame = new Float32Array(planePixels);
    const base = f * planePixels * bytesPerPixel;
    for (let i = 0; i < planePixels; i += 1) {
      const offset = base + i * bytesPerPixel;
      let value: number;
      switch (bitpix) {
        case -32:
          value = view.getFloat32(offset, false);
          break;
        case -64:
          value = view.getFloat64(offset, false);
          break;
        case 8:
          value = view.getUint8(offset);
          break;
        case 16:
          value = view.getInt16(offset, false);
          break;
        case 32:
          value = view.getInt32(offset, false);
          break;
        case 64: {
          const big = view.getBigInt64(offset, false);
          value = Number(big);
          break;
        }
        default:
          throw new Error(`Unsupported FITS BITPIX=${bitpix}.`);
      }
      frame[i] = value * bscale + bzero;
    }
    framesOut[f] = frame;
  }

  return { width, height, frames: framesOut };
}

/** Deterministic, small PRNG for the synthetic preview. */
function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(random: () => number): number {
  let u = random();
  if (u < 1e-12) u = 1e-12;
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * random());
}

/**
 * The simulate endpoint returns truth + metadata, not raw cube pixels. Build
 * a deterministic noise + PSF preview so the blink comparator still shows a
 * faithful, labelled *approximation* of the simulated frames.
 */
export function synthesizePreview(
  width: number,
  height: number,
  nFrames: number,
  snrFrame: number,
  seed: number,
  truth: { t: number; x: number; y: number }[],
): PreviewCube {
  const random = mulberry32(seed || 1);
  const pixels = width * height;
  const frames: Float32Array[] = [];
  const psfRadius = 4;
  const psfSigma2 = 2 * 1.1 * 1.1;
  const stars = Array.from({ length: Math.min(30, Math.round(pixels / 300)) }, () => ({
    x: random() * width,
    y: random() * height,
    flux: 8 + random() * 45,
  }));

  for (let t = 0; t < nFrames; t += 1) {
    const frame = new Float32Array(pixels);
    for (let i = 0; i < pixels; i += 1) frame[i] = gaussian(random);

    for (const star of stars) {
      const x0 = Math.max(0, Math.floor(star.x - psfRadius));
      const x1 = Math.min(width - 1, Math.ceil(star.x + psfRadius));
      const y0 = Math.max(0, Math.floor(star.y - psfRadius));
      const y1 = Math.min(height - 1, Math.ceil(star.y + psfRadius));
      for (let y = y0; y <= y1; y += 1) {
        for (let x = x0; x <= x1; x += 1) {
          const r2 = (x + 0.5 - star.x) ** 2 + (y + 0.5 - star.y) ** 2;
          frame[y * width + x] += star.flux * Math.exp(-r2 / psfSigma2);
        }
      }
    }

    for (const point of truth) {
      if (Math.round(point.t) !== t) continue;
      const x0 = Math.max(0, Math.floor(point.x - psfRadius));
      const x1 = Math.min(width - 1, Math.ceil(point.x + psfRadius));
      const y0 = Math.max(0, Math.floor(point.y - psfRadius));
      const y1 = Math.min(height - 1, Math.ceil(point.y + psfRadius));
      for (let y = y0; y <= y1; y += 1) {
        for (let x = x0; x <= x1; x += 1) {
          const r2 = (x + 0.5 - point.x) ** 2 + (y + 0.5 - point.y) ** 2;
          frame[y * width + x] += snrFrame * Math.exp(-r2 / psfSigma2);
        }
      }
    }
    frames.push(frame);
  }

  return { width, height, frames };
}
