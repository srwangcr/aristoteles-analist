//! Configuraciones serializables que espejean la API de Python.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ComputeBackend {
    Auto,
    Cpu,
    Gpu,
}

impl Default for ComputeBackend {
    fn default() -> Self {
        Self::Auto
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum PsfKind {
    Gaussian,
    Moffat,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct PsfConfig {
    pub kind: PsfKind,
    pub sigma: f64,
    pub beta: f64,
}

impl Default for PsfConfig {
    fn default() -> Self {
        Self {
            kind: PsfKind::Gaussian,
            sigma: 1.35,
            beta: 3.5,
        }
    }
}

impl PsfConfig {
    pub fn alpha(self) -> f64 {
        self.sigma * 2.0 * (2.0_f64.powf(1.0 / self.beta) - 1.0).sqrt()
    }

    pub fn peak(self) -> f64 {
        match self.kind {
            PsfKind::Gaussian => 1.0 / (2.0 * std::f64::consts::PI * self.sigma.powi(2)),
            PsfKind::Moffat => (self.beta - 1.0) / (std::f64::consts::PI * self.alpha().powi(2)),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimulatorConfig {
    pub n_frames: usize,
    pub height: usize,
    pub width: usize,
    pub background: f64,
    pub gradient: [f64; 2],
    pub background_drift: f64,
    pub read_noise: f64,
    pub gain: f64,
    pub psf: PsfConfig,
    pub seeing_jitter: f64,
    pub guide_jitter: f64,
    pub n_stars: usize,
    pub star_flux_median: f64,
    pub star_flux_sigma: f64,
    pub cosmic_rate: f64,
    pub cosmic_amplitude: [f64; 2],
    pub n_defective_columns: usize,
    pub magnitude_zeropoint: f64,
    pub rng_seed: u64,
}

impl Default for SimulatorConfig {
    fn default() -> Self {
        Self {
            n_frames: 12,
            height: 96,
            width: 96,
            background: 120.0,
            gradient: [0.35, -0.20],
            background_drift: 0.6,
            read_noise: 5.0,
            gain: 1.0,
            psf: PsfConfig::default(),
            seeing_jitter: 0.10,
            guide_jitter: 0.12,
            n_stars: 30,
            star_flux_median: 1600.0,
            star_flux_sigma: 0.65,
            cosmic_rate: 3.0,
            cosmic_amplitude: [12.0, 35.0],
            n_defective_columns: 2,
            magnitude_zeropoint: 25.0,
            rng_seed: 1234,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WandererSpec {
    pub object_id: String,
    pub x0: f64,
    pub y0: f64,
    pub vx: f64,
    pub vy: f64,
    pub snr_frame: f64,
    pub ref_frame: usize,
}

impl Default for WandererSpec {
    fn default() -> Self {
        Self {
            object_id: "W01".to_string(),
            x0: 36.0,
            y0: 41.0,
            vx: 0.37,
            vy: -0.21,
            snr_frame: 2.0,
            ref_frame: 5,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PipelineConfig {
    pub mesh_size: usize,
    pub background_sigma: f64,
    pub background_iterations: usize,
    pub align: bool,
    pub alignment_upsample: usize,
    pub normalize_flux: bool,
    pub match_seeing: bool,
    pub velocity_max: f64,
    pub velocity_step: f64,
    pub detection_sigma: f64,
    pub match_filter_sigma: f64,
    pub min_component_pixels: usize,
    pub dedup_radius: f64,
    pub aperture_radius: f64,
    pub max_track_rms: f64,
    pub max_cosmic_fraction: f64,
    pub max_photometric_cv: f64,
    pub min_detection_frames: usize,
    pub backend: ComputeBackend,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            mesh_size: 16,
            background_sigma: 3.0,
            background_iterations: 4,
            align: true,
            alignment_upsample: 8,
            normalize_flux: true,
            match_seeing: true,
            velocity_max: 1.5,
            velocity_step: 0.25,
            detection_sigma: 5.0,
            match_filter_sigma: 0.8,
            min_component_pixels: 2,
            dedup_radius: 3.0,
            aperture_radius: 2.2,
            max_track_rms: 0.9,
            max_cosmic_fraction: 0.55,
            max_photometric_cv: 0.65,
            min_detection_frames: 3,
            backend: ComputeBackend::Auto,
        }
    }
}

impl PipelineConfig {
    pub fn velocities(&self) -> Vec<(f64, f64)> {
        let mut axis = Vec::new();
        let steps = (2.0 * self.velocity_max / self.velocity_step).round() as usize + 1;
        for i in 0..steps {
            axis.push(-self.velocity_max + i as f64 * self.velocity_step);
        }
        let mut out = Vec::with_capacity(axis.len() * axis.len());
        for &vy in &axis {
            for &vx in &axis {
                out.push((vx, vy));
            }
        }
        out
    }

    pub fn n_velocities(&self) -> usize {
        let axis = (2.0 * self.velocity_max / self.velocity_step).round() as usize + 1;
        axis * axis
    }
}
