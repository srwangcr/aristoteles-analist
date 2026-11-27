export type Backend = 'auto' | 'cpu' | 'gpu';

export type ExportFormat = 'json' | 'mpc' | 'parquet';

export interface TruthPoint {
  object_id: string;
  t: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
  snr_frame: number;
}

export interface SimulateRequest {
  snr_frame: number;
  seed: number;
  n_frames?: number;
  shape?: [number, number];
  vx?: number;
  vy?: number;
}

export interface SimulateResponse {
  cube_id: string;
  truth: TruthPoint[];
}

export interface FitsResponse {
  cube_id: string;
  frames: number;
  height: number;
  width: number;
}

export interface CubeMeta {
  cubeId: string;
  frames: number;
  height: number;
  width: number;
  truth: TruthPoint[];
  origin: 'fits' | 'simulation';
  sourceName?: string;
}

/** Local frame buffers used only for the visual blink preview. */
export interface PreviewCube {
  width: number;
  height: number;
  frames: Float32Array[];
}

export interface DetectionParams {
  backend: Backend;
  velocityMax: number;
  velocityStep: number;
  detectionSigma: number;
}

export interface DetectionRequest {
  cube_id: string;
  backend?: Backend;
  velocity_max?: number;
  velocity_step?: number;
  detection_sigma?: number;
}

export interface DetectionResponse {
  job_id: string;
}

export interface JobResponse {
  status: string;
  progress: number;
  message: string;
  elapsed_s?: number;
}

export type JobMessage =
  | { type: 'progress'; value: number; message: string }
  | { type: 'done'; candidates: number; elapsed_s: number }
  | { type: 'error'; message: string };

export interface JobSocketHandlers {
  onProgress?: (value: number, message: string) => void;
  onDone?: (elapsedS: number) => void;
  onError?: (message: string) => void;
  onConnectionChange?: (connected: boolean, attempt: number) => void;
}

export type JobStatus = 'idle' | 'queued' | 'running' | 'done' | 'error';

export interface JobViewState {
  status: JobStatus;
  progress: number;
  message: string;
  elapsedS?: number;
  socketConnected: boolean;
  error?: string;
}

export interface Candidate {
  x: number;
  y: number;
  vx: number;
  vy: number;
  snr: number;
  stacked_flux: number;
  peak: number;
  track_rms: number;
  photometric_cv: number;
  cosmic_fraction: number;
}

export interface CandidateFilters {
  minSnr: number;
  maxTrackRms: number;
  maxCosmicFraction: number;
}

export interface CandidatesResponse {
  candidates: Candidate[];
}

export interface StackMap {
  data: Float32Array;
  height: number;
  width: number;
  vx: number;
  vy: number;
  min: number;
  max: number;
  peak: number;
}
