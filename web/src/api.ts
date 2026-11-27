import type {
  CandidateFilters,
  CandidatesResponse,
  DetectionRequest,
  DetectionResponse,
  ExportFormat,
  FitsResponse,
  JobMessage,
  JobResponse,
  JobSocketHandlers,
  SimulateRequest,
  SimulateResponse,
} from './types';

export const API_BASE = (import.meta.env.VITE_API_URL ?? 'http://localhost:8080').replace(/\/$/, '');

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status?: number,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

export function errorMessage(error: unknown): string {
  if (error instanceof Error) return error.message;
  if (typeof error === 'string') return error;
  return 'Unexpected error';
}

async function readError(response: Response): Promise<string> {
  const text = await response.text().catch(() => '');
  try {
    const parsed: unknown = JSON.parse(text);
    if (parsed && typeof parsed === 'object') {
      const record = parsed as Record<string, unknown>;
      const detail = record.detail ?? record.message ?? record.error;
      if (typeof detail === 'string') return detail;
    }
  } catch {
    // Not JSON — fall through to the raw body.
  }
  return text.trim() || `Request failed with status ${response.status}`;
}

async function jsonRequest<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, init);
  } catch {
    throw new ApiError(`Cannot reach the API at ${API_BASE}. Is the server running?`);
  }
  if (!response.ok) {
    throw new ApiError(await readError(response), response.status);
  }
  try {
    return (await response.json()) as T;
  } catch {
    throw new ApiError('The API returned a malformed JSON response.');
  }
}

export function simulateCube(payload: SimulateRequest): Promise<SimulateResponse> {
  return jsonRequest<SimulateResponse>('/api/simulate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

export function uploadFits(file: File): Promise<FitsResponse> {
  return jsonRequest<FitsResponse>('/api/fits', {
    method: 'POST',
    headers: { 'Content-Type': 'application/octet-stream' },
    body: file,
  });
}

export function startDetection(payload: DetectionRequest): Promise<DetectionResponse> {
  return jsonRequest<DetectionResponse>('/api/detect', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

export function getJob(jobId: string): Promise<JobResponse> {
  return jsonRequest<JobResponse>(`/api/jobs/${encodeURIComponent(jobId)}`);
}

export function getCandidates(
  jobId: string,
  filters: Pick<CandidateFilters, 'minSnr' | 'maxTrackRms'>,
): Promise<CandidatesResponse> {
  const query = new URLSearchParams({
    min_snr: String(filters.minSnr),
    max_rms: String(filters.maxTrackRms),
  });
  return jsonRequest<CandidatesResponse>(`/api/candidates/${encodeURIComponent(jobId)}?${query}`);
}

export async function getStack(
  jobId: string,
  vx: number,
  vy: number,
): Promise<{ data: Float32Array; height: number; width: number }> {
  const query = new URLSearchParams({ vx: String(vx), vy: String(vy) });
  let response: Response;
  try {
    response = await fetch(
      `${API_BASE}/api/stack/${encodeURIComponent(jobId)}?${query}`,
      { headers: { Accept: 'application/octet-stream' } },
    );
  } catch {
    throw new ApiError(`Cannot reach the API at ${API_BASE}. Is the server running?`);
  }
  if (!response.ok) {
    throw new ApiError(await readError(response), response.status);
  }
  const height = Number(response.headers.get('X-Height'));
  const width = Number(response.headers.get('X-Width'));
  const buffer = await response.arrayBuffer();
  if (!Number.isFinite(height) || !Number.isFinite(width) || height <= 0 || width <= 0) {
    throw new ApiError('Stack response is missing valid X-Height / X-Width headers.');
  }
  const expected = height * width;
  if (buffer.byteLength < expected * 4) {
    throw new ApiError(
      `Stack response is too short: got ${buffer.byteLength} bytes, expected ${expected * 4}.`,
    );
  }
  // The API specifies little-endian float32; DataView gives that explicitly,
  // independent of the host architecture.
  const view = new DataView(buffer);
  const data = new Float32Array(expected);
  for (let i = 0; i < expected; i += 1) {
    data[i] = view.getFloat32(i * 4, true);
  }
  return { data, height, width };
}

export async function getExport(jobId: string, format: ExportFormat): Promise<{ blob: Blob; filename: string }> {
  const query = new URLSearchParams({ format });
  let response: Response;
  try {
    response = await fetch(
      `${API_BASE}/api/export/${encodeURIComponent(jobId)}?${query}`,
      { headers: { Accept: 'application/octet-stream' } },
    );
  } catch {
    throw new ApiError(`Cannot reach the API at ${API_BASE}. Is the server running?`);
  }
  if (!response.ok) {
    throw new ApiError(await readError(response), response.status);
  }
  const disposition = response.headers.get('Content-Disposition') ?? '';
  const match = /filename\*?=(?:UTF-8'')?"?([^";]+)"?/i.exec(disposition);
  const filename = match?.[1]?.trim() || `catalog_${jobId}.${format}`;
  return { blob: await response.blob(), filename };
}

function socketUrl(jobId: string): string {
  const wsBase = API_BASE.replace(/^http:\/\//, 'ws://').replace(/^https:\/\//, 'wss://');
  return `${wsBase}/api/jobs/${encodeURIComponent(jobId)}/ws`;
}

/**
 * Subscribe to a job's WebSocket channel. Terminal messages are de-duplicated;
 * if the socket drops, retries use exponential backoff and a REST poll keeps
 * the UI in sync if a progress event was missed.
 */
export function connectJobSocket(jobId: string, handlers: JobSocketHandlers): () => void {
  let socket: WebSocket | null = null;
  let disposed = false;
  let terminal = false;
  let attempt = 0;
  let retryTimer: number | undefined;
  let pollTimer: number | undefined;

  const finish = (kind: 'done' | 'error', message: string, elapsedS = 0) => {
    if (terminal || disposed) return;
    terminal = true;
    if (retryTimer !== undefined) window.clearTimeout(retryTimer);
    if (pollTimer !== undefined) window.clearInterval(pollTimer);
    // A terminal job is not a reconnect; keep the last progress message.
    handlers.onConnectionChange?.(false, 0);
    if (kind === 'done') handlers.onDone?.(elapsedS);
    else handlers.onError?.(message);
  };

  const pollJob = () => {
    if (disposed || terminal) return;
    void getJob(jobId)
      .then((job) => {
        if (disposed || terminal) return;
        const status = job.status.toLowerCase();
        if (status === 'done' || status === 'completed' || status === 'finished' || status === 'succeeded') {
          finish('done', job.message, job.elapsed_s ?? 0);
        } else if (status === 'error' || status === 'failed') {
          finish('error', job.message || 'Detection failed.');
        }
      })
      .catch(() => {
        // Transient poll failures are retried on the next interval.
      });
  };

  const connect = () => {
    if (disposed || terminal) return;
    try {
      socket = new WebSocket(socketUrl(jobId));
    } catch {
      scheduleRetry();
      return;
    }

    socket.onopen = () => {
      if (disposed) return;
      attempt = 0;
      handlers.onConnectionChange?.(true, attempt);
      pollJob();
    };

    socket.onmessage = (event: MessageEvent<string>) => {
      if (disposed || terminal) return;
      let message: JobMessage;
      try {
        message = JSON.parse(event.data) as JobMessage;
      } catch {
        return;
      }
      if (message.type === 'progress') {
        handlers.onProgress?.(message.value, message.message);
      } else if (message.type === 'done') {
        finish('done', `Detected ${message.candidates} candidates`, message.elapsed_s);
      } else if (message.type === 'error') {
        finish('error', message.message || 'Detection failed.');
      }
    };

    socket.onerror = () => {
      // onclose always follows; reconnection is handled there.
    };

    socket.onclose = () => {
      socket = null;
      if (disposed || terminal) return;
      handlers.onConnectionChange?.(false, attempt);
      pollJob();
      scheduleRetry();
    };
  };

  const scheduleRetry = () => {
    if (disposed || terminal || retryTimer !== undefined) return;
    attempt += 1;
    const delay = Math.min(8000, 400 * 2 ** Math.min(attempt - 1, 5));
    retryTimer = window.setTimeout(() => {
      retryTimer = undefined;
      connect();
    }, delay);
  };

  connect();
  pollTimer = window.setInterval(pollJob, 4000);

  return () => {
    disposed = true;
    if (retryTimer !== undefined) window.clearTimeout(retryTimer);
    if (pollTimer !== undefined) window.clearInterval(pollTimer);
    if (socket) {
      socket.onclose = null;
      socket.close();
      socket = null;
    }
  };
}
