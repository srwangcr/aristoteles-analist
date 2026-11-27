//! Preprocesado temporal: mediana por píxel y máscara de estrellas estacionarias.
//!
//! Se ejecuta *antes* del shift-and-stack. La mediana temporal elimina el cielo
//! y cualquier fuente que no se mueva; los núcleos saturados se enmascaran y se
//! dilatan para cubrir el halo, de modo que sus residuos no dominen el catálogo.

use rayon::prelude::*;

/// Contraste mínimo para considerar núcleo estelar: por debajo de él, una
/// fuente móvil o un píxel ruidoso también eleva la mediana temporal.
const MIN_STAR_CONTRAST_SIGMA: f64 = 60.0;

/// Resultado del saneado temporal de un cubo `(N, H, W)`.
#[derive(Clone, Debug)]
pub struct TemporalClean {
    /// Cubo limpio, en el mismo orden que la entrada. Los píxeles enmascarados valen 0.
    pub cleaned: Vec<f64>,
    /// Mediana temporal de cada píxel, de tamaño `H * W`.
    pub median_frame: Vec<f64>,
    /// Máscara de estrellas dilatada, de tamaño `H * W`.
    pub star_mask: Vec<bool>,
    /// Mediana global del mapa de medianas.
    pub level: f64,
    /// Dispersión MAD robusta (1.4826 * MAD) del mapa de medianas.
    pub sigma: f64,
    /// Número de píxeles enmascarados tras la dilatación.
    pub n_masked: usize,
}

fn median(values: &mut Vec<f64>) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        0.5 * (values[n / 2 - 1] + values[n / 2])
    }
}

/// Mediana robusta y MAD escalado de un conjunto de valores finitos.
pub fn robust_level(values: &[f64]) -> (f64, f64) {
    let mut finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    let center = median(&mut finite);
    let deviations: Vec<f64> = finite.iter().map(|v| (v - center).abs()).collect();
    let mut abs_dev = deviations;
    let mad = median(&mut abs_dev);
    (center, 1.4826 * mad)
}

/// Dilatación morfológica con un disco de radio `radius` píxeles.
pub fn dilate(mask: &[bool], height: usize, width: usize, radius: usize) -> Vec<bool> {
    let r = radius as isize;
    let offsets: Vec<(isize, isize)> = (-r..=r)
        .flat_map(|dy| (-r..=r).map(move |dx| (dy, dx)))
        .filter(|&(dy, dx)| dx * dx + dy * dy <= r * r)
        .collect();
    let mut out = vec![false; mask.len()];
    for y in 0..height {
        for x in 0..width {
            if !mask[y * width + x] {
                continue;
            }
            for &(dy, dx) in &offsets {
                let yy = y as isize + dy;
                let xx = x as isize + dx;
                if yy >= 0 && yy < height as isize && xx >= 0 && xx < width as isize {
                    out[yy as usize * width + xx as usize] = true;
                }
            }
        }
    }
    out
}

/// Resta la mediana temporal por píxel y enmascara estrellas brillantes.
///
/// `star_sigma` multiplo de MAD por encima del cual un píxel de la mediana se
/// considera núcleo estelar; `mask_radius` es el radio de dilatación del halo.
pub fn temporal_median_subtract(
    frames: &[f64],
    n_frames: usize,
    height: usize,
    width: usize,
    star_sigma: f64,
    mask_radius: usize,
    core_sigma: f64,
) -> Result<TemporalClean, String> {
    let size = height * width;
    if n_frames == 0 || height == 0 || width == 0 {
        return Err("el cubo no puede estar vacío".to_string());
    }
    if frames.len() != n_frames * size {
        return Err("tamaño del cubo inconsistente".to_string());
    }

    // Mediana temporal de cada píxel. Se ignoran NaN/Inf para no contaminar
    // la imagen maestra con píxeles defectuosos del detector.
    let mut median_frame = vec![0.0_f64; size];
    let mut min_frame = vec![f64::INFINITY; size];
    for pixel in 0..size {
        let mut column: Vec<f64> = (0..n_frames)
            .map(|t| frames[t * size + pixel])
            .filter(|v| v.is_finite())
            .collect();
        for &v in &column {
            min_frame[pixel] = min_frame[pixel].min(v);
        }
        median_frame[pixel] = median(&mut column);
        if column.is_empty() {
            min_frame[pixel] = 0.0;
        }
    }

    let (level, sigma) = robust_level(&median_frame);
    let sigma = if sigma > 1e-9 { sigma } else { 1e-6 };
    let threshold = level + star_sigma.max(0.0) * sigma;
    // El segundo criterio evita enmascarar fuentes lentas: su mediana temporal
    // también es alta, pero el píxel no está brillante en *todos* los frames.
    let core_threshold = level + core_sigma.max(0.0) * sigma;
    let contrast_threshold = level + MIN_STAR_CONTRAST_SIGMA * sigma;
    let raw_mask: Vec<bool> = median_frame
        .iter()
        .zip(min_frame.iter())
        .map(|(&v, &minimum)| {
            v > threshold
                && v > contrast_threshold
                && (minimum > core_threshold
                    || v - minimum <= star_sigma.max(0.0) * sigma)
        })
        .collect();
    let star_mask = dilate(&raw_mask, height, width, mask_radius);
    let n_masked = star_mask.iter().filter(|&&m| m).count();

    let cleaned: Vec<f64> = vec![0.0; frames.len()];
    let mut cleaned = cleaned;
    cleaned
        .par_chunks_exact_mut(size)
        .enumerate()
        .for_each(|(t, frame)| {
            for (pixel, value) in frame.iter_mut().enumerate() {
                if star_mask[pixel] {
                    *value = 0.0;
                } else {
                    let original = frames[t * size + pixel];
                    *value = if original.is_finite() {
                        original - median_frame[pixel]
                    } else {
                        0.0
                    };
                }
            }
            debug_assert!(t < n_frames);
        });

    Ok(TemporalClean {
        cleaned,
        median_frame,
        star_mask,
        level,
        sigma,
        n_masked,
    })
}
