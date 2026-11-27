//! PSF, conversiones flujo/SNR y ruido Poisson-Gaussiano.

use crate::config::PsfConfig;

pub fn psf_value(dy: f64, dx: f64, model: PsfConfig) -> f64 {
    let r2 = dx * dx + dy * dy;
    match model.kind {
        crate::config::PsfKind::Gaussian => (-0.5 * r2 / model.sigma.powi(2)).exp(),
        crate::config::PsfKind::Moffat => (1.0 + r2 / model.alpha().powi(2)).powf(-model.beta),
    }
}

pub fn flux_for_snr(snr: f64, sigma_local: f64, model: PsfConfig) -> f64 {
    snr * sigma_local / model.peak()
}

pub fn flux_to_magnitude(flux: f64, zeropoint: f64) -> f64 {
    zeropoint - 2.5 * flux.max(1e-9).log10()
}

/// Renderiza fuentes sub-pixel en un recorte local; devuelve una imagen row-major.
pub fn render_sources(
    height: usize,
    width: usize,
    positions: &[(f64, f64)],
    fluxes: &[f64],
    model: PsfConfig,
) -> Vec<f64> {
    let mut image = vec![0.0_f64; height * width];
    let scale = if model.kind == crate::config::PsfKind::Gaussian {
        model.sigma
    } else {
        model.alpha()
    };
    let mut radius = (5.0 * scale).ceil() as isize;
    radius = radius.min(height.max(width) as isize);
    for (&(x, y), &flux) in positions.iter().zip(fluxes.iter()) {
        let fx = x.floor() as isize;
        let fy = y.floor() as isize;
        let x0 = (fx - radius).max(0) as usize;
        let x1 = (fx + radius + 2).min(width as isize) as usize;
        let y0 = (fy - radius).max(0) as usize;
        let y1 = (fy + radius + 2).min(height as isize) as usize;
        if x1 <= x0 || y1 <= y0 {
            continue;
        }
        let mut psf = Vec::with_capacity((y1 - y0) * (x1 - x0));
        let mut total = 0.0;
        for yy in y0..y1 {
            for xx in x0..x1 {
                let value = psf_value(yy as f64 - y, xx as f64 - x, model);
                psf.push(value);
                total += value;
            }
        }
        if total <= 0.0 {
            continue;
        }
        for (index, &value) in psf.iter().enumerate() {
            let yy = y0 + index / (x1 - x0);
            let xx = x0 + index % (x1 - x0);
            image[yy * width + xx] += flux * value / total;
        }
    }
    image
}
