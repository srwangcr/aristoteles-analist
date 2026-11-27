//! Motor de apilamiento sintético por hipótesis de velocidad.

use rayon::prelude::*;
use rustfft::num_complex::Complex;

use crate::backend::{choose_backend, BackendDecision};
use crate::config::ComputeBackend;
use crate::fft::{signed_frequency, C, Fft2};
use crate::gpu::GpuStacker;

pub struct ShiftStackEngine {
    residuals: Vec<f64>,
    n_frames: usize,
    height: usize,
    width: usize,
    ref_frame: usize,
    spectra: Vec<Vec<C>>,
    fft: Fft2,
    pub decision: BackendDecision,
    gpu: Option<GpuStacker>,
}

impl ShiftStackEngine {
    pub fn new(
        residuals: Vec<f64>,
        n_frames: usize,
        height: usize,
        width: usize,
        ref_frame: Option<usize>,
        requested: ComputeBackend,
        n_velocities: usize,
    ) -> Result<Self, String> {
        let ref_frame = ref_frame.unwrap_or((n_frames - 1) / 2);
        let size = height * width;
        if residuals.len() != n_frames * size {
            return Err("tamaño del cubo inconsistente".to_string());
        }
        let gpu_available = crate::gpu::gpu_available();
        let decision = choose_backend(
            requested,
            n_frames,
            height,
            width,
            n_velocities,
            gpu_available,
        );
        let fft = Fft2::new(height, width);
        let mut spectra = Vec::with_capacity(n_frames);
        for frame in residuals.chunks_exact(size) {
            let mut spec: Vec<C> = frame.iter().map(|&v| C::new(v, 0.0)).collect();
            fft.process(&mut spec, false);
            spectra.push(spec);
        }
        let gpu = if decision.backend == ComputeBackend::Gpu {
            let mut stacker = GpuStacker::new(n_frames, height, width)?;
            stacker.upload_and_forward(&residuals)?;
            Some(stacker)
        } else {
            None
        };
        Ok(Self {
            residuals,
            n_frames,
            height,
            width,
            ref_frame,
            spectra,
            fft,
            decision,
            gpu,
        })
    }

    pub fn backend(&self) -> ComputeBackend {
        self.decision.backend
    }

    pub fn ref_frame(&self) -> usize {
        self.ref_frame
    }

    pub fn n_frames(&self) -> usize {
        self.n_frames
    }

    pub fn residuals(&self) -> &[f64] {
        &self.residuals
    }

    /// Réplica exacta de `np.fft.rfft2` / `irfft2` con la fase de desplazamiento
    /// de la línea base Python. El espectro completo se construye de modo que
    /// una sola IFFT 2D por frame produzca el mismo resultado que:
    ///   FFT_x → FFT_y → fase → IFFT_y → extensión hermítica → IFFT_x.
    /// Paralelizado con Rayon por frame.
    pub fn stack_cpu(&self, vx: f64, vy: f64) -> Vec<f64> {
        let (height, width) = (self.height, self.width);
        let size = height * width;
        let half = width / 2 + 1;
        let two_pi = 2.0 * std::f64::consts::PI;
        let mut parts: Vec<Vec<f64>> = (0..self.n_frames).map(|_| vec![0.0; size]).collect();
        parts.par_iter_mut().enumerate().for_each(|(t, output)| {
            let shift_x = -vx * (t as f64 - self.ref_frame as f64);
            let shift_y = -vy * (t as f64 - self.ref_frame as f64);
            let mut spec = self.spectra[t].clone();
            if shift_x.abs() < 1e-12 && shift_y.abs() < 1e-12 {
                self.fft.process(&mut spec, true);
                for (dst, value) in output.iter_mut().zip(spec.iter()) {
                    *dst = value.re;
                }
                return;
            }
            let kx_phase: Vec<C> = (0..half)
                .map(|kx| Complex::from_polar(1.0, -two_pi * kx as f64 * shift_x / width as f64))
                .collect();
            let ky_phase: Vec<C> = (0..height)
                .map(|y| {
                    Complex::from_polar(
                        1.0,
                        -two_pi * signed_frequency(y, height) * shift_y / height as f64,
                    )
                })
                .collect();
            // Mitad no negativa del espectro: fase de NumPy (kx sin signo).
            for kx in 0..half {
                let factor = kx_phase[kx];
                for y in 0..height {
                    let index = y * width + kx;
                    spec[index] *= factor * ky_phase[y];
                }
            }
            // Frecuencias negativas inducidas por `irfft`: conjugadas con
            // inversión del índice y, no simples conjugadas de la misma fila.
            for kx in half..width {
                let m = width - kx;
                for y in 0..height {
                    let mirrored = if y == 0 { 0 } else { height - y };
                    spec[y * width + kx] = spec[mirrored * width + m].conj();
                }
            }
            self.fft.process(&mut spec, true);
            for (dst, value) in output.iter_mut().zip(spec.iter()) {
                *dst = value.re;
            }
        });
        let mut stacked = vec![0.0_f64; size];
        for part in parts {
            for (dst, value) in stacked.iter_mut().zip(part.iter()) {
                *dst += value;
            }
        }
        stacked
    }

    /// Despacha al backend seleccionado. Si GPU falla, degrada a CPU.
    pub fn stack(&mut self, vx: f64, vy: f64) -> Result<Vec<f64>, String> {
        if self.decision.backend == ComputeBackend::Gpu {
            if let Some(gpu) = &self.gpu {
                match gpu.stack(vx, vy, self.ref_frame) {
                    Ok(result) => return Ok(result),
                    Err(error) => {
                        eprintln!("GPU failed, falling back to CPU: {error}");
                    }
                }
            }
        }
        Ok(self.stack_cpu(vx, vy))
    }
}
