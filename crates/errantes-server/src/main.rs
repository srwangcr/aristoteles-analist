use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use axum::body::Bytes;
use axum::extract::{Path, Query, State, WebSocketUpgrade};
use axum::http::{header, StatusCode};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use errantes_core::config::{PipelineConfig, SimulatorConfig, WandererSpec};
use errantes_core::detect_candidates_with_progress;
use errantes_core::{ComputeBackend, ShiftStackEngine};
use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;
use tower_http::cors::{Any, CorsLayer};
use uuid::Uuid;

#[derive(Clone)]
struct AppState {
    cubes: Arc<Mutex<HashMap<String, CubeData>>>,
    jobs: Arc<Mutex<HashMap<String, JobState>>>,
}

#[derive(Clone)]
struct CubeData {
    n_frames: usize,
    height: usize,
    width: usize,
    frames: Vec<f64>,
    _truth: Vec<TruthRow>,
}

#[derive(Clone, Serialize, Deserialize)]
struct TruthRow {
    object_id: String,
    t: usize,
    x: f64,
    y: f64,
    vx: f64,
    vy: f64,
    snr_frame: f64,
}

#[derive(Clone, Serialize)]
struct CandidateRow {
    x: f64,
    y: f64,
    vx: f64,
    vy: f64,
    snr: f64,
    stacked_flux: f64,
    peak: f64,
    track_rms: f64,
    photometric_cv: f64,
    cosmic_fraction: f64,
}

struct JobState {
    status: String,
    progress: f64,
    message: String,
    elapsed_s: Option<f64>,
    cube_id: String,
    candidates: Vec<CandidateRow>,
    /// Cubo saneado usado por la detección, reutilizado por `/api/stack`.
    cleaned: Option<Vec<f64>>,
    tx: broadcast::Sender<String>,
}

#[derive(Deserialize)]
struct SimulateReq {
    snr_frame: f64,
    seed: u64,
    n_frames: Option<usize>,
    shape: Option<[usize; 2]>,
    vx: Option<f64>,
    vy: Option<f64>,
}

#[derive(Deserialize)]
struct DetectReq {
    cube_id: String,
    backend: Option<String>,
    velocity_max: Option<f64>,
    velocity_step: Option<f64>,
    detection_sigma: Option<f64>,
    /// Velocidad mínima (px/frame); 0.08 por defecto.
    v_min: Option<f64>,
    /// Radio de NMS conjunta en (x, y, vx, vy), en píxeles.
    r_nms: Option<f64>,
    /// Umbral MAD para el máscara de estrellas estacionarias.
    star_sigma: Option<f64>,
    /// Radio de dilatación del halo estelar.
    star_mask_radius: Option<usize>,
    /// Contraste mínimo del núcleo estelar (evita enmascarar fuentes lentas).
    star_core_sigma: Option<f64>,
    /// Activar/desactivar la sustracción de la mediana temporal.
    temporal_median: Option<bool>,
}

#[derive(Deserialize)]
struct StackQuery { vx: f64, vy: f64 }
#[derive(Deserialize)]
struct CandidatesQuery { min_snr: Option<f64>, max_rms: Option<f64> }
#[derive(Deserialize)]
struct ExportQuery { format: Option<String> }

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt().with_env_filter("info").init();
    let state = AppState {
        cubes: Arc::new(Mutex::new(HashMap::new())),
        jobs: Arc::new(Mutex::new(HashMap::new())),
    };
    let app = Router::new()
        .route("/api/simulate", post(simulate))
        .route("/api/fits", post(upload_fits))
        .route("/api/detect", post(detect))
        .route("/api/jobs/{job_id}", get(job_status))
        .route("/api/jobs/{job_id}/ws", get(job_ws))
        .route("/api/candidates/{job_id}", get(candidates))
        .route("/api/stack/{job_id}", post(stack_map))
        .route("/api/export/{job_id}", get(export))
        .layer(CorsLayer::new().allow_origin(Any).allow_methods(Any).allow_headers(Any))
        .with_state(state);
    let addr = "0.0.0.0:8080";
    tracing::info!("errantes-server listening on {addr}");
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn simulate(State(state): State<AppState>, Json(req): Json<SimulateReq>) -> impl IntoResponse {
    let mut cfg = SimulatorConfig::default();
    cfg.n_frames = req.n_frames.unwrap_or(cfg.n_frames);
    if let Some([h, w]) = req.shape { cfg.height = h; cfg.width = w; }
    cfg.rng_seed = req.seed;
    let mut wspec = WandererSpec::default();
    wspec.snr_frame = req.snr_frame;
    if let Some(vx) = req.vx { wspec.vx = vx; }
    if let Some(vy) = req.vy { wspec.vy = vy; }

    let (frames, truth) = simulate_frames(&cfg, &wspec);
    let cube_id = Uuid::new_v4().to_string();
    state.cubes.lock().unwrap().insert(cube_id.clone(), CubeData {
        n_frames: cfg.n_frames, height: cfg.height, width: cfg.width, frames, _truth: truth.clone(),
    });
    Json(serde_json::json!({"cube_id": cube_id, "truth": truth}))
}

fn simulate_frames(cfg: &SimulatorConfig, w: &WandererSpec) -> (Vec<f64>, Vec<TruthRow>) {
    // Simulador ligero compatible con la API del frontend: fondo + PSF + Poisson + lectura.
    let size = cfg.height * cfg.width;
    let mut frames = vec![0.0_f64; cfg.n_frames * size];
    let sigma = cfg.psf.sigma;
    let psf_peak = 1.0 / (2.0 * std::f64::consts::PI * sigma * sigma);
    let sigma_local = (cfg.background + cfg.read_noise * cfg.read_noise).sqrt();
    let flux = w.snr_frame * sigma_local / psf_peak;
    let mut truth = Vec::new();
    let mut seed = cfg.rng_seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let mut next_u = move || -> f64 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 33) as f64 / (u32::MAX as f64)
    };
    for t in 0..cfg.n_frames {
        let frame = &mut frames[t * size..(t + 1) * size];
        for px in frame.iter_mut() { *px = cfg.background; }
        let x = w.x0 + w.vx * t as f64;
        let y = w.y0 + w.vy * t as f64;
        truth.push(TruthRow { object_id: w.object_id.clone(), t, x, y, vx: w.vx, vy: w.vy, snr_frame: w.snr_frame });
        let r = (4.0 * sigma).ceil() as i32;
        let xc = x.round() as i32; let yc = y.round() as i32;
        for dy in -r..=r { for dx in -r..=r {
            let xx = xc + dx; let yy = yc + dy;
            if xx < 0 || yy < 0 || xx >= cfg.width as i32 || yy >= cfg.height as i32 { continue; }
            let ddx = xx as f64 - x; let ddy = yy as f64 - y;
            let val = flux * (-0.5 * (ddx*ddx + ddy*ddy) / (sigma*sigma)).exp() * psf_peak;
            frame[yy as usize * cfg.width + xx as usize] += val;
        }}
        for px in frame.iter_mut() {
            let lam = px.max(0.0);
            let poisson = if lam < 30.0 {
                let l = (-lam).exp(); let mut k=0u32; let mut p=1.0;
                while p > l { k+=1; p *= next_u().max(1e-12); if k>5000 {break;} }
                (k.saturating_sub(1)) as f64
            } else {
                let u1 = next_u().max(1e-12); let u2 = next_u();
                let g = (-2.0*u1.ln()).sqrt()*(2.0*std::f64::consts::PI*u2).cos();
                (lam + lam.sqrt()*g).max(0.0).round()
            };
            let u1 = next_u().max(1e-12); let u2 = next_u();
            let g = (-2.0*u1.ln()).sqrt()*(2.0*std::f64::consts::PI*u2).cos();
            *px = poisson + cfg.read_noise * g;
        }
    }
    (frames, truth)
}

async fn upload_fits(State(state): State<AppState>, body: Bytes) -> impl IntoResponse {
    // FITS parser mínimo: lee cubo primario float32/float64 3D.
    match parse_fits_cube(&body) {
        Ok((n_frames, height, width, frames)) => {
            let cube_id = Uuid::new_v4().to_string();
            state.cubes.lock().unwrap().insert(cube_id.clone(), CubeData { n_frames, height, width, frames, _truth: vec![] });
            Json(serde_json::json!({"cube_id":cube_id,"frames":n_frames,"height":height,"width":width})).into_response()
        }
        Err(e) => (StatusCode::BAD_REQUEST, e).into_response(),
    }
}

fn parse_fits_cube(bytes: &[u8]) -> Result<(usize, usize, usize, Vec<f64>), String> {
    if bytes.len() < 2880 || &bytes[0..6] != b"SIMPLE" { return Err("FITS inválido".into()); }
    let mut offset2 = 0usize;
    let mut bitpix = 0i32;
    let mut naxis = 0usize;
    let mut dims: Vec<usize> = vec![];
    'outer: loop {
        if offset2 + 2880 > bytes.len() { return Err("FITS truncado".into()); }
        let block = &bytes[offset2..offset2+2880]; offset2 += 2880;
        for card in block.chunks_exact(80) {
            let key = std::str::from_utf8(&card[0..8]).unwrap_or("").trim();
            let raw = std::str::from_utf8(&card[10..]).unwrap_or("");
            let val = raw.split('/').next().unwrap_or("").trim().trim_matches('\'').trim();
            match key {
                "BITPIX" => bitpix = val.replace("=", "").trim().parse().unwrap_or(0),
                "NAXIS" => naxis = val.replace("=", "").trim().parse().unwrap_or(0),
                k if k.starts_with("NAXIS") && k.len()>5 => {
                    if let Ok(n) = k[5..].parse::<usize>() { while dims.len()<n { dims.push(1);} dims[n-1]=val.trim().parse().unwrap_or(1); }
                }
                "END" => break 'outer,
                _ => {}
            }
        }
    }
    if naxis < 2 { return Err("se esperaba cubo 2D/3D".into()); }
    let width = *dims.get(0).ok_or("sin NAXIS1")?;
    let height = *dims.get(1).ok_or("sin NAXIS2")?;
    let n_frames = if naxis >= 3 { *dims.get(2).ok_or("sin NAXIS3")? } else { 1 };
    let count = n_frames * height * width;
    let bytes_per = (bitpix.abs()/8) as usize;
    let data_len = count * bytes_per;
    if offset2 + data_len > bytes.len() { return Err("datos FITS truncados".into()); }
    let mut frames = Vec::with_capacity(count);
    let data = &bytes[offset2..offset2+data_len];
    for i in 0..count {
        let b = &data[i*bytes_per..(i+1)*bytes_per];
        let v = match bitpix {
            -32 => f32::from_be_bytes(b.try_into().unwrap()) as f64,
            -64 => f64::from_be_bytes(b.try_into().unwrap()),
            16 => i16::from_be_bytes(b.try_into().unwrap()) as f64,
            32 => i32::from_be_bytes(b.try_into().unwrap()) as f64,
            _ => return Err(format!("BITPIX {bitpix} no soportado")),
        };
        frames.push(v);
    }
    Ok((n_frames, height, width, frames))
}

async fn detect(State(state): State<AppState>, Json(req): Json<DetectReq>) -> impl IntoResponse {
    let cube = { state.cubes.lock().unwrap().get(&req.cube_id).cloned() };
    let Some(cube) = cube else { return (StatusCode::NOT_FOUND, "cube_id desconocido").into_response(); };
    let job_id = Uuid::new_v4().to_string();
    let (tx, _) = broadcast::channel(64);
    state.jobs.lock().unwrap().insert(job_id.clone(), JobState {
        status: "running".into(), progress: 0.0, message: "iniciando".into(), elapsed_s: None,
        cube_id: req.cube_id.clone(), candidates: vec![], cleaned: None, tx: tx.clone(),
    });
    let jobs = state.jobs.clone();
    let jid = job_id.clone();
    tokio::spawn(async move {
        let start = Instant::now();
        let _ = tx.send(serde_json::json!({"type":"progress","value":0.05,"message":"preparando"}).to_string());
        let mut cfg = PipelineConfig::default();
        if let Some(b) = req.backend.as_deref() {
            cfg.backend = match b { "cpu" => ComputeBackend::Cpu, "gpu" => ComputeBackend::Gpu, _ => ComputeBackend::Auto };
        }
        if let Some(v) = req.velocity_max { cfg.velocity_max = v; }
        if let Some(v) = req.velocity_step { cfg.velocity_step = v; }
        if let Some(v) = req.detection_sigma { cfg.detection_sigma = v; }
        if let Some(v) = req.v_min { cfg.v_min = v.max(0.0); }
        if let Some(v) = req.r_nms { cfg.r_nms = v.max(0.0); }
        if let Some(v) = req.star_sigma { cfg.star_sigma = v.max(0.0); }
        if let Some(v) = req.star_mask_radius { cfg.star_mask_radius = v; }
        if let Some(v) = req.star_core_sigma { cfg.star_core_sigma = v.max(0.0); }
        if let Some(v) = req.temporal_median { cfg.temporal_median = v; }

        let progress = |p: f64| {
            let _ = tx.send(serde_json::json!({"type":"progress","value":p,"message":"barrido de velocidades"}).to_string());
            if let Some(job) = jobs.lock().unwrap().get_mut(&jid) {
                job.progress = p;
                job.message = "barrido de velocidades".into();
            }
        };
        let result = detect_candidates_with_progress(
            &cube.frames, cube.n_frames, cube.height, cube.width, &cfg, None, &progress,
        );
        let result = match result {
            Ok(r) => r,
            Err(e) => {
                let _ = tx.send(serde_json::json!({"type":"error","message":e}).to_string());
                if let Some(job) = jobs.lock().unwrap().get_mut(&jid) { job.status="error".into(); job.message=e; }
                return;
            }
        };
        let best: Vec<CandidateRow> = result.candidates.iter().map(|c| CandidateRow {
            x: c.x, y: c.y, vx: c.vx, vy: c.vy, snr: c.snr,
            stacked_flux: c.stacked_flux, peak: c.peak,
            track_rms: finite_or_zero(c.track_residual_rms),
            photometric_cv: finite_or_zero(c.photometric_cv),
            cosmic_fraction: finite_or_zero(c.cosmic_fraction),
        }).collect();
        let cleaned = result.processed_frames;
        let elapsed = start.elapsed().as_secs_f64();
        let _ = tx.send(serde_json::json!({"type":"done","candidates":best.len(),"elapsed_s":elapsed}).to_string());
        if let Some(job) = jobs.lock().unwrap().get_mut(&jid) {
            job.status="done".into(); job.progress=1.0; job.message="completado".into();
            job.elapsed_s=Some(elapsed); job.candidates=best; job.cleaned=Some(cleaned);
        }
    });
    Json(serde_json::json!({"job_id": job_id})).into_response()
}

/// `serde_json` no serializa NaN/Inf; el catálogo siempre expone números finitos.
fn finite_or_zero(value: f64) -> f64 {
    if value.is_finite() { value } else { 0.0 }
}

async fn job_status(State(state): State<AppState>, Path(job_id): Path<String>) -> impl IntoResponse {
    let jobs = state.jobs.lock().unwrap();
    let Some(job) = jobs.get(&job_id) else { return (StatusCode::NOT_FOUND, "job no encontrado").into_response(); };
    Json(serde_json::json!({"status":job.status,"progress":job.progress,"message":job.message,"elapsed_s":job.elapsed_s})).into_response()
}

async fn job_ws(ws: WebSocketUpgrade, State(state): State<AppState>, Path(job_id): Path<String>) -> impl IntoResponse {
    let tx = { state.jobs.lock().unwrap().get(&job_id).map(|j| j.tx.subscribe()) };
    ws.on_upgrade(move |socket| async move {
        let Some(mut rx) = tx else { return; };
        let (mut sender, mut _recv) = socket.split();
        while let Ok(msg) = rx.recv().await {
            if sender.send(axum::extract::ws::Message::Text(msg.into())).await.is_err() { break; }
        }
    })
}

async fn candidates(State(state): State<AppState>, Path(job_id): Path<String>, Query(q): Query<CandidatesQuery>) -> impl IntoResponse {
    let jobs = state.jobs.lock().unwrap();
    let Some(job) = jobs.get(&job_id) else { return (StatusCode::NOT_FOUND, "job no encontrado").into_response(); };
    let mut rows: Vec<CandidateRow> = job.candidates.clone();
    if let Some(min_snr) = q.min_snr { rows.retain(|c| c.snr >= min_snr); }
    if let Some(max_rms) = q.max_rms { rows.retain(|c| c.track_rms <= max_rms); }
    Json(serde_json::json!({"candidates": rows})).into_response()
}

async fn stack_map(State(state): State<AppState>, Path(job_id): Path<String>, Query(q): Query<StackQuery>) -> impl IntoResponse {
    let (cube, cleaned, backend) = {
        let jobs = state.jobs.lock().unwrap();
        let Some(job) = jobs.get(&job_id) else { return (StatusCode::NOT_FOUND, "job no encontrado").into_response(); };
        let cube_id = job.cube_id.clone();
        let cleaned = job.cleaned.clone();
        drop(jobs);
        let cubes = state.cubes.lock().unwrap();
        (cubes.get(&cube_id).cloned(), cleaned, ComputeBackend::Auto)
    };
    let Some(cube) = cube else { return (StatusCode::NOT_FOUND, "cubo no encontrado").into_response(); };
    let frames = cleaned.unwrap_or_else(|| cube.frames.clone());
    let mut engine = match ShiftStackEngine::new(frames, cube.n_frames, cube.height, cube.width, None, backend, 1) {
        Ok(e) => e, Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, e).into_response(),
    };
    match engine.stack(q.vx, q.vy) {
        Ok(map) => {
            let mut bytes = Vec::with_capacity(map.len()*4);
            for v in map { bytes.extend_from_slice(&(v as f32).to_le_bytes()); }
            let mut headers = axum::http::HeaderMap::new();
            headers.insert("X-Height", cube.height.to_string().parse().unwrap());
            headers.insert("X-Width", cube.width.to_string().parse().unwrap());
            (headers, bytes).into_response()
        }
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e).into_response(),
    }
}

async fn export(State(state): State<AppState>, Path(job_id): Path<String>, Query(q): Query<ExportQuery>) -> impl IntoResponse {
    let jobs = state.jobs.lock().unwrap();
    let Some(job) = jobs.get(&job_id) else { return (StatusCode::NOT_FOUND, "job no encontrado").into_response(); };
    let fmt = q.format.as_deref().unwrap_or("json");
    match fmt {
        "json" => {
            let body = serde_json::to_vec(&job.candidates).unwrap_or_default();
            ([(header::CONTENT_TYPE, "application/json"), (header::CONTENT_DISPOSITION, "attachment; filename=candidates.json")], body).into_response()
        }
        "mpc" => {
            let mut out = String::from("# MPC placeholder\n");
            for c in &job.candidates { out.push_str(&format!("{:.4} {:.4} {:.4} {:.4} snr={:.2}\n", c.x,c.y,c.vx,c.vy,c.snr)); }
            ([(header::CONTENT_TYPE, "text/plain"), (header::CONTENT_DISPOSITION, "attachment; filename=candidates.mpc")], out.into_bytes()).into_response()
        }
        _ => {
            let body = serde_json::to_vec(&job.candidates).unwrap_or_default();
            ([(header::CONTENT_TYPE, "application/octet-stream"), (header::CONTENT_DISPOSITION, "attachment; filename=candidates.parquet")], body).into_response()
        }
    }
}
