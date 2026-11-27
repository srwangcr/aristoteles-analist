//! Tipos de datos del motor Rust.

use serde::{Deserialize, Serialize};

use crate::config::SimulatorConfig;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Wanderer {
    pub object_id: String,
    pub x0: f64,
    pub y0: f64,
    pub vx: f64,
    pub vy: f64,
    pub snr_frame: f64,
    pub flux: f64,
    pub magnitude: f64,
    pub ref_frame: usize,
}

impl Wanderer {
    pub fn position(&self, t: usize) -> (f64, f64) {
        (self.x0 + self.vx * t as f64, self.y0 + self.vy * t as f64)
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TruthRow {
    pub object_id: String,
    pub t: usize,
    pub x: f64,
    pub y: f64,
    pub vx: f64,
    pub vy: f64,
    pub flux_e: f64,
    pub magnitude: f64,
    pub snr_frame: f64,
    pub psf_sigma: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GroundTruth {
    pub objects: Vec<Wanderer>,
    pub positions: Vec<TruthRow>,
}

#[derive(Clone, Debug)]
pub struct Cube {
    /// Cubo en orden row-major: frame + y * width + x.
    pub frames: Vec<f64>,
    pub n_frames: usize,
    pub height: usize,
    pub width: usize,
    pub truth: GroundTruth,
    pub config: SimulatorConfig,
    pub star_positions: Vec<(f64, f64)>,
    pub star_fluxes: Vec<f64>,
    pub frame_psf_sigma: Vec<f64>,
    pub defect_columns: Vec<usize>,
}

impl Cube {
    pub fn frame(&self, t: usize) -> &[f64] {
        let size = self.height * self.width;
        &self.frames[t * size..(t + 1) * size]
    }

    pub fn frame_mut(&mut self, t: usize) -> &mut [f64] {
        let size = self.height * self.width;
        &mut self.frames[t * size..(t + 1) * size]
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Candidate {
    pub x: f64,
    pub y: f64,
    pub vx: f64,
    pub vy: f64,
    pub snr: f64,
    pub stacked_flux: f64,
    pub peak: f64,
    pub n_pixels: usize,
    pub ref_frame: usize,
    pub track_fluxes: Vec<f64>,
    pub track_residual_rms: f64,
    pub track_rms_x: f64,
    pub track_rms_y: f64,
    pub photometric_cv: f64,
    pub cosmic_fraction: f64,
    pub velocity_index: i64,
}

impl Default for Candidate {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            vx: 0.0,
            vy: 0.0,
            snr: 0.0,
            stacked_flux: 0.0,
            peak: 0.0,
            n_pixels: 0,
            ref_frame: 0,
            track_fluxes: Vec::new(),
            track_residual_rms: f64::NAN,
            track_rms_x: f64::NAN,
            track_rms_y: f64::NAN,
            photometric_cv: f64::NAN,
            cosmic_fraction: f64::NAN,
            velocity_index: -1,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DetectionOutput {
    pub candidates: Vec<Candidate>,
    pub ref_frame: usize,
    pub runtime_s: f64,
    pub n_velocities: usize,
    pub backend_used: String,
}
