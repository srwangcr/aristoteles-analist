//! Barrido de velocidades, extracción de componentes y diagnóstico de trazas.
//!
//! Este módulo es el camino único de detección usado por el motor Rust, el
//! servidor (catálogo de candidatos) y los bindings de Python.

use rayon::prelude::*;

use crate::config::PipelineConfig;
use crate::preprocess::{robust_level, temporal_median_subtract, TemporalClean};
use crate::types::Candidate;
use crate::ShiftStackEngine;

/// Salida completa del pipeline de detección del núcleo.
#[derive(Clone, Debug)]
pub struct DetectionResult {
    pub candidates: Vec<Candidate>,
    pub preprocess: TemporalClean,
    /// Cubo efectivamente usado por el barrido (ya saneado si aplica).
    pub processed_frames: Vec<f64>,
    pub best_stack: Vec<f64>,
    pub snr_by_velocity: Vec<f64>,
    pub ref_frame: usize,
    pub n_velocities: usize,
    pub runtime_s: f64,
    pub backend_used: String,
}

/// Filtro Gaussiano separable, equivalente a `scipy.ndimage.gaussian_filter`.
pub fn gaussian_blur(image: &[f64], height: usize, width: usize, sigma: f64) -> Vec<f64> {
    if sigma <= 0.0 {
        return image.to_vec();
    }
    let radius = (4.0 * sigma).ceil() as isize;
    let kernel: Vec<f64> = (-radius..=radius)
        .map(|i| (-(i as f64) * i as f64 / (2.0 * sigma * sigma)).exp())
        .collect();
    let norm: f64 = kernel.iter().sum();
    let kernel: Vec<f64> = kernel.iter().map(|&k| k / norm).collect();

    let mut tmp = vec![0.0_f64; image.len()];
    for y in 0..height {
        for x in 0..width {
            let mut acc = 0.0;
            for (k, &weight) in kernel.iter().enumerate() {
                let xx = (x as isize + k as isize - radius).clamp(0, width as isize - 1);
                acc += weight * image[y * width + xx as usize];
            }
            tmp[y * width + x] = acc;
        }
    }
    let mut out = vec![0.0_f64; image.len()];
    for y in 0..height {
        for x in 0..width {
            let mut acc = 0.0;
            for (k, &weight) in kernel.iter().enumerate() {
                let yy = (y as isize + k as isize - radius).clamp(0, height as isize - 1);
                acc += weight * tmp[yy as usize * width + x];
            }
            out[y * width + x] = acc;
        }
    }
    out
}

/// Etiquetado de componentes con conectividad 8 y centroide sub-píxel.
pub fn components(
    stacked: &[f64],
    height: usize,
    width: usize,
    threshold: f64,
    min_pixels: usize,
    vx: f64,
    vy: f64,
    ref_frame: usize,
) -> Vec<Candidate> {
    let size = height * width;
    let mut labels = vec![0_u32; size];
    let mut queue: Vec<usize> = Vec::new();
    let mut out = Vec::new();
    let mut next_label = 1_u32;

    for seed in 0..size {
        if labels[seed] != 0 || stacked[seed] <= threshold {
            continue;
        }
        labels[seed] = next_label;
        queue.clear();
        queue.push(seed);
        let mut pixels = Vec::new();
        while let Some(index) = queue.pop() {
            pixels.push(index);
            let y = index / width;
            let x = index % width;
            for dy in -1_isize..=1 {
                for dx in -1_isize..=1 {
                    if dy == 0 && dx == 0 {
                        continue;
                    }
                    let yy = y as isize + dy;
                    let xx = x as isize + dx;
                    if yy < 0 || yy >= height as isize || xx < 0 || xx >= width as isize {
                        continue;
                    }
                    let neighbor = yy as usize * width + xx as usize;
                    if labels[neighbor] == 0 && stacked[neighbor] > threshold {
                        labels[neighbor] = next_label;
                        queue.push(neighbor);
                    }
                }
            }
        }
        next_label += 1;
        if pixels.len() < min_pixels {
            continue;
        }
        let mut weight_sum = 0.0;
        let mut x_sum = 0.0;
        let mut y_sum = 0.0;
        let mut peak = f64::NEG_INFINITY;
        for &index in &pixels {
            let weight = (stacked[index] - threshold).max(0.0);
            weight_sum += weight;
            x_sum += weight * (index % width) as f64;
            y_sum += weight * (index / width) as f64;
            peak = peak.max(stacked[index]);
        }
        if weight_sum <= 0.0 || !peak.is_finite() {
            continue;
        }
        out.push(Candidate {
            x: x_sum / weight_sum,
            y: y_sum / weight_sum,
            vx,
            vy,
            snr: peak, // se normaliza por sigma en el barrido
            stacked_flux: weight_sum,
            peak,
            n_pixels: pixels.len(),
            ref_frame,
            velocity_index: -1,
            ..Candidate::default()
        });
    }
    out
}

/// Supresión de no-máximos conjunta: una sola hipótesis de velocidad por
/// vecindad espacial de radio `radius` (en el frame de referencia).
pub fn non_max_suppression(mut candidates: Vec<Candidate>, radius: f64) -> Vec<Candidate> {
    candidates.sort_by(|a, b| {
        b.snr
            .partial_cmp(&a.snr)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.velocity_index.cmp(&b.velocity_index))
    });
    let mut kept: Vec<Candidate> = Vec::new();
    for candidate in candidates {
        let duplicate = kept
            .iter()
            .any(|other| (candidate.x - other.x).hypot(candidate.y - other.y) <= radius);
        if !duplicate {
            kept.push(candidate);
        }
    }
    kept
}

/// Diagnósticos de trayectoria sobre el cubo ya saneado.
#[derive(Clone, Debug)]
pub struct TrackMetrics {
    pub fluxes: Vec<f64>,
    /// Centroides medidos por frame (NaN cuando la apertura no es fiable).
    pub centroid_x: Vec<f64>,
    pub centroid_y: Vec<f64>,
    pub track_rms: f64,
    pub rms_x: f64,
    pub rms_y: f64,
    pub photometric_cv: f64,
    pub cosmic_fraction: f64,
}

/// Mide flujo y centroide en una apertura 3x3 alrededor de la posición
/// predicha por `(x0, vx, vy)` en cada frame.
pub fn track_metrics(
    cleaned: &[f64],
    n_frames: usize,
    height: usize,
    width: usize,
    candidate: &Candidate,
    ref_frame: usize,
    radius: f64,
) -> TrackMetrics {
    let size = height * width;
    // radio 1.0 -> 3x3; radio 2.0 -> 5x5
    let half = (radius.floor().max(1.0) as isize).min(2);
    let mut fluxes = Vec::with_capacity(n_frames);
    let mut centroid_x = vec![f64::NAN; n_frames];
    let mut centroid_y = vec![f64::NAN; n_frames];
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut n_valid = 0_usize;

    for t in 0..n_frames {
        let dt = t as f64 - ref_frame as f64;
        let px = candidate.x + candidate.vx * dt;
        let py = candidate.y + candidate.vy * dt;
        if px < half as f64
            || py < half as f64
            || px >= (width as isize - half - 1) as f64
            || py >= (height as isize - half - 1) as f64
        {
            fluxes.push(0.0);
            continue;
        }
        let cx = px.round() as isize;
        let cy = py.round() as isize;
        let mut patch = [0.0_f64; 25];
        let mut count = 0;
        let mut xs = [0.0_f64; 25];
        let mut ys = [0.0_f64; 25];
        for dy in -half..=half {
            for dx in -half..=half {
                let yy = cy + dy;
                let xx = cx + dx;
                if yy < 0 || yy >= height as isize || xx < 0 || xx >= width as isize {
                    continue;
                }
                patch[count] = cleaned[t * size + yy as usize * width + xx as usize];
                xs[count] = xx as f64;
                ys[count] = yy as f64;
                count += 1;
            }
        }
        if count == 0 {
            fluxes.push(0.0);
            continue;
        }
        let window = &patch[..count];
        let flux: f64 = window.iter().sum();
        fluxes.push(flux);
        let (background, _) = robust_level(window);
        let mut weights = Vec::with_capacity(count);
        let mut total_weight = 0.0;
        for &value in window {
            let w = (value - background).max(0.0);
            total_weight += w;
            weights.push(w);
        }
        if total_weight <= 1e-12 {
            continue; // centroide no fiable: sin residuo
        }
        let measured_x: f64 = xs[..count]
            .iter()
            .zip(weights.iter())
            .map(|(&x, &w)| x * w)
            .sum::<f64>()
            / total_weight;
        let measured_y: f64 = ys[..count]
            .iter()
            .zip(weights.iter())
            .map(|(&y, &w)| y * w)
            .sum::<f64>()
            / total_weight;
        centroid_x[t] = measured_x;
        centroid_y[t] = measured_y;
        sum_x += (measured_x - px).powi(2);
        sum_y += (measured_y - py).powi(2);
        n_valid += 1;
    }

    let n = n_frames.max(1) as f64;
    let rms_x = (sum_x / n_valid.max(1) as f64).sqrt();
    let rms_y = (sum_y / n_valid.max(1) as f64).sqrt();
    let track_rms = ((sum_x + sum_y) / n_valid.max(1) as f64).sqrt();

    let mean = fluxes.iter().sum::<f64>() / n;
    let variance = fluxes.iter().map(|f| (f - mean).powi(2)).sum::<f64>() / n;
    let std = variance.sqrt();
    let photometric_cv = if mean.abs() > 1e-9 {
        (std / mean.abs()).min(1e6)
    } else {
        (std / 1e-9).min(1e6)
    };

    let (median_flux, mad_flux) = robust_level(&fluxes);
    let threshold = median_flux + 4.0 * mad_flux.max(1e-9);
    let cosmic_fraction = fluxes.iter().filter(|&&f| f > threshold).count() as f64 / n;

    TrackMetrics {
        fluxes,
        centroid_x,
        centroid_y,
        track_rms,
        rms_x,
        rms_y,
        photometric_cv,
        cosmic_fraction,
    }
}

/// Resta a cada frame su fondo robusto, respetando la máscara de estrellas.
///
/// La mediana por píxel elimina fuentes lentas porque éstas ocupan el mismo
/// píxel durante varias frames. Para medir centroides se prefiere un fondo por
/// frame, que conserva la señal móvil y mantiene enmascarados los núcleos.
pub fn background_subtract(
    frames: &[f64],
    _n_frames: usize,
    height: usize,
    width: usize,
    star_mask: &[bool],
) -> Vec<f64> {
    let size = height * width;
    let mut out = vec![0.0_f64; frames.len()];
    for (frame_index, chunk) in out.chunks_exact_mut(size).enumerate() {
        let mut background: Vec<f64> = (0..size)
            .filter(|&p| !star_mask[p])
            .map(|p| frames[frame_index * size + p])
            .filter(|v| v.is_finite())
            .collect();
        let level = if background.is_empty() {
            0.0
        } else {
            // Mediana robusta del cielo; se calcula una sola vez por frame.
            background.sort_by(|a, b| a.total_cmp(b));
            let middle = background.len() / 2;
            if background.len() % 2 == 1 {
                background[middle]
            } else {
                0.5 * (background[middle - 1] + background[middle])
            }
        };
        for (p, value) in chunk.iter_mut().enumerate() {
            *value = if star_mask[p] {
                0.0
            } else {
                frames[frame_index * size + p] - level
            };
        }
    }
    out
}

/// Ajuste lineal de los centroides medidos: devuelve `(x_ref, y_ref, vx, vy)`.
pub fn refine_track(
    metrics: &TrackMetrics,
    n_frames: usize,
    ref_frame: usize,
    velocity_max: f64,
) -> Option<(f64, f64, f64, f64)> {
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut n = 0.0;
    for t in 0..n_frames {
        if !metrics.centroid_x[t].is_finite() || !metrics.centroid_y[t].is_finite() {
            continue;
        }
        let time = t as f64 - ref_frame as f64;
        sx += time * time;
        sy += time;
        n += 1.0;
    }
    if n < 3.0 {
        return None;
    }
    let denominator = sx - sy * sy / n;
    if denominator.abs() < 1e-9 {
        return None;
    }
    let fit = |values: &[f64]| -> Option<(f64, f64)> {
        let mut stv = 0.0;
        let mut sv = 0.0;
        for t in 0..n_frames {
            if !values[t].is_finite() {
                continue;
            }
            let time = t as f64 - ref_frame as f64;
            stv += time * values[t];
            sv += values[t];
        }
        let velocity = (stv - sy * sv / n) / denominator;
        let intercept = (sv - velocity * sy) / n; // posición en el frame de referencia
        Some((intercept, velocity))
    };
    let (x, vx) = fit(&metrics.centroid_x)?;
    let (y, vy) = fit(&metrics.centroid_y)?;
    Some((
        x,
        y,
        vx.clamp(-velocity_max, velocity_max),
        vy.clamp(-velocity_max, velocity_max),
    ))
}

/// Ejecuta preprocesado, barrido, NMS y diagnósticos sobre un cubo `(N, H, W)`.
pub fn detect_candidates(
    frames: &[f64],
    n_frames: usize,
    height: usize,
    width: usize,
    config: &PipelineConfig,
    ref_frame: Option<usize>,
) -> Result<DetectionResult, String> {
    detect_candidates_with_progress(frames, n_frames, height, width, config, ref_frame, |_| {})
}

/// Variante con callback de progreso (0.0..1.0) para la API/servidor.
pub fn detect_candidates_with_progress<F: Fn(f64)>(
    frames: &[f64],
    n_frames: usize,
    height: usize,
    width: usize,
    config: &PipelineConfig,
    ref_frame: Option<usize>,
    progress: F,
) -> Result<DetectionResult, String> {
    let start = std::time::Instant::now();
    let size = height * width;
    if n_frames == 0 || height == 0 || width == 0 || frames.len() != n_frames * size {
        return Err("tamaño del cubo inconsistente".to_string());
    }
    let reference = ref_frame.unwrap_or((n_frames - 1) / 2).min(n_frames - 1);
    let preprocess = temporal_median_subtract(
        frames,
        n_frames,
        height,
        width,
        config.star_sigma,
        config.star_mask_radius,
        config.star_core_sigma,
    )?;
    let cleaned = if config.temporal_median {
        preprocess.cleaned.clone()
    } else {
        frames.to_vec()
    };
    // Cubo para diagnósticos: la mediana por píxel atenúa fuentes lentas, así
    // que los centroides se miden sobre un fondo robusto por frame.
    let diagnostics_frames = background_subtract(
        frames,
        n_frames,
        height,
        width,
        &preprocess.star_mask,
    );
    let velocities = config.velocity_hypotheses();
    let mut engine = ShiftStackEngine::new(
        cleaned.clone(),
        n_frames,
        height,
        width,
        Some(reference),
        config.backend,
        velocities.len(),
    )?;
    let backend_used = format!("{:?}", engine.backend()).to_ascii_lowercase();

    let mut all: Vec<Candidate> = Vec::new();
    let mut snr_by_velocity = vec![0.0_f64; velocities.len()];
    let mut best_peak = f64::NEG_INFINITY;
    let mut best_stack = vec![0.0_f64; size];

    for (index, &(vx, vy)) in velocities.iter().enumerate() {
        progress(0.05 + 0.9 * index as f64 / velocities.len().max(1) as f64);
        let raw = engine.stack(vx, vy)?;
        let stacked = gaussian_blur(&raw, height, width, config.match_filter_sigma);
        let (_, sigma) = robust_level(&stacked);
        let sigma = sigma.max(1e-6);
        let threshold = config.detection_sigma * sigma;
        let peak = stacked.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        snr_by_velocity[index] = if peak.is_finite() {
            peak / sigma
        } else {
            0.0
        };
        if peak > best_peak {
            best_peak = peak;
            best_stack = stacked.clone();
        }
        for mut candidate in components(
            &stacked,
            height,
            width,
            threshold,
            config.min_component_pixels,
            vx,
            vy,
            reference,
        ) {
            candidate.snr /= sigma;
            candidate.velocity_index = index as i64;
            all.push(candidate);
        }
    }

    // Los desplazamientos FFT envuelven los bordes: se descartan hipótesis
    // demasiado cerca del marco, donde aparecen réplicas de fuentes reales.
    let margin = ((config.velocity_max * n_frames as f64 / 2.0)
        .ceil()
        .max(2.0) as usize)
        .min(height.min(width) / 4);
    all.retain(|c| {
        c.x >= margin as f64
            && c.x < (width - margin) as f64
            && c.y >= margin as f64
            && c.y < (height - margin) as f64
    });

    // Paralelizar los diagnósticos: antes del NMS hay muchas hipótesis
    // vecinas y el coste por candidato es pequeño pero no despreciable.
    let mut diagnostics: Vec<(Candidate, TrackMetrics)> = all
        .into_par_iter()
        .map(|candidate| {
            let metrics = track_metrics(
                &cleaned,
                n_frames,
                height,
                width,
                &candidate,
                reference,
                config.diagnostic_radius,
            );
            (candidate, metrics)
        })
        .collect();
    for (candidate, metrics) in &mut diagnostics {
        candidate.track_fluxes = metrics.fluxes.clone();
        candidate.track_residual_rms = metrics.track_rms;
        candidate.track_rms_x = metrics.rms_x;
        candidate.track_rms_y = metrics.rms_y;
        candidate.photometric_cv = metrics.photometric_cv;
        candidate.cosmic_fraction = metrics.cosmic_fraction;
    }
    let candidates: Vec<Candidate> = diagnostics
        .into_iter()
        .map(|(candidate, _)| candidate)
        .collect();
    let mut candidates = non_max_suppression(candidates, config.r_nms);
    candidates.truncate(config.max_candidates);

    // Refinar (x, y, vx, vy) con los centroides medidos y recalcular métricas
    // sobre el fondo por frame, que no sesga la velocidad de fuentes lentas.
    for candidate in &mut candidates {
        let initial = track_metrics(
            &diagnostics_frames,
            n_frames,
            height,
            width,
            candidate,
            reference,
            config.diagnostic_radius,
        );
        if let Some((x, y, vx, vy)) =
            refine_track(&initial, n_frames, reference, config.velocity_max)
        {
            candidate.x = x;
            candidate.y = y;
            candidate.vx = vx;
            candidate.vy = vy;
        }
        let metrics = track_metrics(
            &diagnostics_frames,
            n_frames,
            height,
            width,
            candidate,
            reference,
            config.diagnostic_radius,
        );
        candidate.track_fluxes = metrics.fluxes;
        candidate.track_residual_rms = metrics.track_rms;
        candidate.track_rms_x = metrics.rms_x;
        candidate.track_rms_y = metrics.rms_y;
        candidate.photometric_cv = metrics.photometric_cv;
        candidate.cosmic_fraction = metrics.cosmic_fraction;
    }
    candidates = non_max_suppression(candidates, config.r_nms);
    candidates.sort_by(|a, b| b.snr.partial_cmp(&a.snr).unwrap_or(std::cmp::Ordering::Equal));
    progress(0.99);

    Ok(DetectionResult {
        candidates,
        preprocess,
        processed_frames: cleaned,
        best_stack,
        snr_by_velocity,
        ref_frame: reference,
        n_velocities: velocities.len(),
        runtime_s: start.elapsed().as_secs_f64(),
        backend_used,
    })
}
