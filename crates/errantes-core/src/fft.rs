//! FFT 2D sobre vectores row-major de complexos, equivalente a NumPy.

use std::sync::Arc;

use rustfft::{num_complex::Complex, Fft, FftPlanner};

pub type C = Complex<f64>;

#[derive(Clone)]
pub struct Fft2 {
    height: usize,
    width: usize,
    plan_rows: Arc<dyn Fft<f64>>,
    plan_cols: Arc<dyn Fft<f64>>,
    plan_rows_inv: Arc<dyn Fft<f64>>,
    plan_cols_inv: Arc<dyn Fft<f64>>,
}

impl Fft2 {
    pub fn new(height: usize, width: usize) -> Self {
        let mut planner = FftPlanner::new();
        Self {
            height,
            width,
            plan_rows: planner.plan_fft_forward(width),
            plan_cols: planner.plan_fft_forward(height),
            plan_rows_inv: planner.plan_fft_inverse(width),
            plan_cols_inv: planner.plan_fft_inverse(height),
        }
    }

    /// Transformada 2D en sitio. `inverse = true` incluye la normalización 1/(H*W).
    pub fn process(&self, data: &mut [C], inverse: bool) {
        if data.len() != self.height * self.width {
            panic!("FFT2: tamaño de datos incorrecto");
        }
        let (row_plan, col_plan) = if inverse {
            (&self.plan_rows_inv, &self.plan_cols_inv)
        } else {
            (&self.plan_rows, &self.plan_cols)
        };
        let mut row = vec![C::new(0.0, 0.0); self.width];
        for y in 0..self.height {
            for x in 0..self.width {
                row[x] = data[y * self.width + x];
            }
            row_plan.process(&mut row);
            for x in 0..self.width {
                data[y * self.width + x] = row[x];
            }
        }
        let mut col = vec![C::new(0.0, 0.0); self.height];
        for x in 0..self.width {
            for y in 0..self.height {
                col[y] = data[y * self.width + x];
            }
            col_plan.process(&mut col);
            for y in 0..self.height {
                data[y * self.width + x] = col[y];
            }
        }
        if inverse {
            let scale = 1.0 / ((self.height * self.width) as f64);
            for value in data.iter_mut() {
                *value *= scale;
            }
        }
    }
}

/// Frecuencia firmada en unidades de índice, igual que `np.fft.fftfreq(n) * n`.
pub fn signed_frequency(index: usize, n: usize) -> f64 {
    if index < n / 2 {
        index as f64
    } else {
        index as f64 - n as f64
    }
}

/// FFT 1D con planes reutilizables. La inversa incluye la normalización 1/n,
/// igual que `np.fft.ifft`.
pub struct Fft1 {
    n: usize,
    plan_fwd: Arc<dyn Fft<f64>>,
    plan_inv: Arc<dyn Fft<f64>>,
}

impl Fft1 {
    pub fn new(n: usize) -> Self {
        let mut planner = FftPlanner::new();
        Self {
            n,
            plan_fwd: planner.plan_fft_forward(n),
            plan_inv: planner.plan_fft_inverse(n),
        }
    }

    pub fn process(&self, data: &mut [C], inverse: bool) {
        assert_eq!(data.len(), self.n);
        if inverse {
            self.plan_inv.process(data);
            let scale = 1.0 / self.n as f64;
            for value in data.iter_mut() {
                *value *= scale;
            }
        } else {
            self.plan_fwd.process(data);
        }
    }
}
