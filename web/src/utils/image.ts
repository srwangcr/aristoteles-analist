/** Shared canvas rendering helpers for frame blinking and stack heatmaps. */

const MAGMA_STOPS: [number, number, number][] = [
  [0, 0, 4],
  [28, 16, 68],
  [81, 18, 124],
  [134, 33, 122],
  [183, 55, 121],
  [229, 95, 101],
  [251, 136, 97],
  [254, 188, 120],
  [252, 253, 191],
];

export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

export function percentile(values: Float32Array, percent: number, sampleLimit = 20_000): number {
  if (values.length === 0) return 0;
  const stride = Math.max(1, Math.floor(values.length / sampleLimit));
  const sampled: number[] = [];
  for (let i = 0; i < values.length; i += stride) sampled.push(values[i]);
  sampled.sort((a, b) => a - b);
  const index = clamp(Math.round((percent / 100) * (sampled.length - 1)), 0, sampled.length - 1);
  return sampled[index] ?? 0;
}

export function stats(values: Float32Array): { min: number; max: number; mean: number } {
  if (values.length === 0) return { min: 0, max: 0, mean: 0 };
  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  for (let i = 0; i < values.length; i += 1) {
    const v = values[i];
    if (v < min) min = v;
    if (v > max) max = v;
    sum += v;
  }
  return { min, max, mean: sum / values.length };
}

export function magma(t: number): [number, number, number] {
  const position = clamp(t, 0, 1) * (MAGMA_STOPS.length - 1);
  const lower = Math.floor(position);
  const upper = Math.min(MAGMA_STOPS.length - 1, lower + 1);
  const fraction = position - lower;
  const a = MAGMA_STOPS[lower];
  const b = MAGMA_STOPS[upper];
  return [
    Math.round(a[0] + (b[0] - a[0]) * fraction),
    Math.round(a[1] + (b[1] - a[1]) * fraction),
    Math.round(a[2] + (b[2] - a[2]) * fraction),
  ];
}

export function magmaGradientCss(): string {
  const stops = MAGMA_STOPS.map((color, index) => {
    const t = index / (MAGMA_STOPS.length - 1);
    return `rgb(${color[0]},${color[1]},${color[2]}) ${(t * 100).toFixed(0)}%`;
  });
  return `linear-gradient(to top, ${stops.join(', ')})`;
}

function prepareCanvas(canvas: HTMLCanvasElement, width: number, height: number): CanvasRenderingContext2D | null {
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const context = canvas.getContext('2d');
  if (!context) return null;
  context.clearRect(0, 0, width, height);
  return context;
}

export function drawGrayscale(
  canvas: HTMLCanvasElement,
  values: Float32Array,
  width: number,
  height: number,
  loPercentile: number,
  hiPercentile: number,
): { lo: number; hi: number } | null {
  const context = prepareCanvas(canvas, width, height);
  if (!context) return null;
  const image = context.createImageData(width, height);
  let lo = percentile(values, loPercentile);
  let hi = percentile(values, hiPercentile);
  if (hi <= lo) hi = lo + 1e-6;
  const scale = 255 / (hi - lo);
  for (let i = 0; i < values.length; i += 1) {
    const gray = clamp(Math.round((values[i] - lo) * scale), 0, 255);
    const offset = i * 4;
    image.data[offset] = gray;
    image.data[offset + 1] = gray;
    image.data[offset + 2] = gray;
    image.data[offset + 3] = 255;
  }
  context.putImageData(image, 0, 0);
  return { lo, hi };
}

export function drawHeatmap(
  canvas: HTMLCanvasElement,
  values: Float32Array,
  width: number,
  height: number,
  min: number,
  max: number,
): boolean {
  const context = prepareCanvas(canvas, width, height);
  if (!context) return false;
  const image = context.createImageData(width, height);
  const span = max - min || 1;
  for (let i = 0; i < values.length; i += 1) {
    const [r, g, b] = magma((values[i] - min) / span);
    const offset = i * 4;
    image.data[offset] = r;
    image.data[offset + 1] = g;
    image.data[offset + 2] = b;
    image.data[offset + 3] = 255;
  }
  context.putImageData(image, 0, 0);
  return true;
}

/** Map a pointer event to intrinsic canvas pixel coordinates (object-contain). */
export function canvasCoordinates(
  event: { clientX: number; clientY: number },
  canvas: HTMLCanvasElement,
): { x: number; y: number } | null {
  const rect = canvas.getBoundingClientRect();
  if (rect.width === 0 || rect.height === 0 || canvas.width === 0 || canvas.height === 0) return null;
  const scale = Math.min(rect.width / canvas.width, rect.height / canvas.height);
  const offsetX = rect.left + (rect.width - canvas.width * scale) / 2;
  const offsetY = rect.top + (rect.height - canvas.height * scale) / 2;
  const x = Math.floor((event.clientX - offsetX) / scale);
  const y = Math.floor((event.clientY - offsetY) / scale);
  if (x < 0 || y < 0 || x >= canvas.width || y >= canvas.height) return null;
  return { x, y };
}

export function drawMarker(
  context: CanvasRenderingContext2D,
  x: number,
  y: number,
  color: string,
  label?: string,
): void {
  const radius = Math.max(4, context.canvas.width / 120);
  context.save();
  context.strokeStyle = color;
  context.lineWidth = Math.max(1, context.canvas.width / 400);
  context.beginPath();
  context.arc(x + 0.5, y + 0.5, radius, 0, Math.PI * 2);
  context.moveTo(x + 0.5 - radius * 1.6, y + 0.5);
  context.lineTo(x + 0.5 - radius * 0.55, y + 0.5);
  context.moveTo(x + 0.5 + radius * 0.55, y + 0.5);
  context.lineTo(x + 0.5 + radius * 1.6, y + 0.5);
  context.moveTo(x + 0.5, y + 0.5 - radius * 1.6);
  context.lineTo(x + 0.5, y + 0.5 - radius * 0.55);
  context.moveTo(x + 0.5, y + 0.5 + radius * 0.55);
  context.lineTo(x + 0.5, y + 0.5 + radius * 1.6);
  context.stroke();
  if (label) {
    context.font = `${Math.max(9, Math.round(context.canvas.width / 40))}px monospace`;
    context.fillStyle = color;
    context.fillText(label, x + radius * 1.8, y - radius);
  }
  context.restore();
}
