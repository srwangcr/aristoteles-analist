//! Bindings PyO3 del motor Rust. La API espeja `errantes.detection.ShiftStackEngine`
//! para que el wrapper Python pueda delegar el hot path sin cambiar su interfaz.

use std::sync::Mutex;

use errantes_core::backend::{choose_backend, detected_vram_bytes, BackendDecision};
use errantes_core::config::{ComputeBackend, PipelineConfig};
use errantes_core::detection::track_metrics;
use errantes_core::preprocess::temporal_median_subtract;
use errantes_core::types::Candidate;
use errantes_core::ShiftStackEngine as CoreEngine;
use numpy::ndarray::{Array1, Array2, Array3};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray3};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn parse_backend(value: &str) -> PyResult<ComputeBackend> {
    match value.to_ascii_lowercase().as_str() {
        "auto" => Ok(ComputeBackend::Auto),
        "cpu" => Ok(ComputeBackend::Cpu),
        "gpu" => Ok(ComputeBackend::Gpu),
        other => Err(PyValueError::new_err(format!(
            "backend desconocido '{other}'; use auto, cpu o gpu"
        ))),
    }
}

fn decision_to_dict<'py>(
    py: Python<'py>,
    d: &BackendDecision,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("backend", format!("{:?}", d.backend).to_ascii_lowercase())?;
    dict.set_item("reason", &d.reason)?;
    dict.set_item("estimated_cpu_s", d.estimated_cpu_s)?;
    dict.set_item("estimated_gpu_s", d.estimated_gpu_s)?;
    dict.set_item("required_vram_bytes", d.required_vram_bytes)?;
    dict.set_item("available_vram_bytes", d.available_vram_bytes)?;
    Ok(dict)
}

#[pyclass]
struct ShiftStackEngine {
    inner: Mutex<CoreEngine>,
    shape: (usize, usize),
}

#[pymethods]
impl ShiftStackEngine {
    #[new]
    #[pyo3(signature = (residuals, ref_frame=None, backend="auto", n_velocities=169))]
    fn new(
        residuals: PyReadonlyArray3<'_, f64>,
        ref_frame: Option<usize>,
        backend: &str,
        n_velocities: usize,
    ) -> PyResult<Self> {
        let view = residuals.as_array();
        let shape3 = view.shape();
        let (n_frames, height, width) = (shape3[0], shape3[1], shape3[2]);
        if n_frames == 0 || height == 0 || width == 0 {
            return Err(PyValueError::new_err("el cubo no puede estar vacío"));
        }
        let data: Vec<f64> = view.iter().copied().collect();
        let engine = CoreEngine::new(
            data,
            n_frames,
            height,
            width,
            ref_frame,
            parse_backend(backend)?,
            n_velocities,
        )
        .map_err(PyRuntimeError::new_err)?;
        Ok(Self {
            inner: Mutex::new(engine),
            shape: (height, width),
        })
    }

    /// Suma los frames desplazados. Devuelve un array (H, W) float64.
    fn stack<'py>(&self, py: Python<'py>, vx: f64, vy: f64) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let mut engine = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("mutex del motor envenenado"))?;
        let stacked = engine.stack(vx, vy).map_err(PyRuntimeError::new_err)?;
        let (height, width) = self.shape;
        let array = numpy::ndarray::Array2::from_shape_vec((height, width), stacked)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(array.into_pyarray(py))
    }

    #[getter]
    fn ref_frame(&self) -> usize {
        self.inner.lock().map(|e| e.ref_frame()).unwrap_or(0)
    }

    #[getter]
    fn n_frames(&self) -> usize {
        self.inner.lock().map(|e| e.n_frames()).unwrap_or(0)
    }

    #[getter]
    fn backend(&self) -> String {
        let guard = self.inner.lock();
        format!("{:?}", guard.map(|e| e.backend()).unwrap_or(ComputeBackend::Cpu))
            .to_ascii_lowercase()
    }

    fn decision<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let guard = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("mutex del motor envenenado"))?;
        decision_to_dict(py, &guard.decision)
    }
}

#[pyfunction]
#[pyo3(signature = (backend="auto", n_frames=12, height=96, width=96, n_velocities=169))]
fn backend_decision<'py>(
    py: Python<'py>,
    backend: &str,
    n_frames: usize,
    height: usize,
    width: usize,
    n_velocities: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let decision = choose_backend(
        parse_backend(backend)?,
        n_frames,
        height,
        width,
        n_velocities,
        errantes_core::gpu::gpu_available(),
    );
    decision_to_dict(py, &decision)
}

#[pyfunction]
fn gpu_available() -> bool {
    errantes_core::gpu::gpu_available()
}

#[pyfunction]
fn vram_bytes() -> u64 {
    detected_vram_bytes()
}

fn finite(value: f64) -> f64 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

fn candidate_to_dict<'py>(py: Python<'py>, candidate: &Candidate) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("x", candidate.x)?;
    dict.set_item("y", candidate.y)?;
    dict.set_item("vx", candidate.vx)?;
    dict.set_item("vy", candidate.vy)?;
    dict.set_item("snr", finite(candidate.snr))?;
    dict.set_item("stacked_flux", finite(candidate.stacked_flux))?;
    dict.set_item("peak", finite(candidate.peak))?;
    dict.set_item("n_pixels", candidate.n_pixels)?;
    dict.set_item("ref_frame", candidate.ref_frame)?;
    dict.set_item("velocity_index", candidate.velocity_index)?;
    dict.set_item("track_fluxes", candidate.track_fluxes.clone())?;
    let rms = finite(candidate.track_residual_rms);
    dict.set_item("track_rms", rms)?;
    dict.set_item("track_residual_rms", rms)?;
    dict.set_item("track_rms_x", finite(candidate.track_rms_x))?;
    dict.set_item("track_rms_y", finite(candidate.track_rms_y))?;
    dict.set_item("photometric_cv", finite(candidate.photometric_cv))?;
    dict.set_item("cosmic_fraction", finite(candidate.cosmic_fraction))?;
    Ok(dict)
}

/// Sustracción de mediana temporal por píxel y máscara dilatada de estrellas.
#[pyfunction]
#[pyo3(signature = (residuals, star_sigma=5.0, mask_radius=3, core_sigma=100.0))]
fn subtract_temporal_median<'py>(
    py: Python<'py>,
    residuals: PyReadonlyArray3<'_, f64>,
    star_sigma: f64,
    mask_radius: usize,
    core_sigma: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let view = residuals.as_array();
    let shape = view.shape();
    let (n_frames, height, width) = (shape[0], shape[1], shape[2]);
    let frames: Vec<f64> = view.iter().copied().collect();
    let clean = temporal_median_subtract(
        &frames, n_frames, height, width, star_sigma, mask_radius, core_sigma,
    )
    .map_err(PyValueError::new_err)?;
    let dict = PyDict::new(py);
    dict.set_item(
        "cleaned",
        Array3::from_shape_vec((n_frames, height, width), clean.cleaned)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .into_pyarray(py),
    )?;
    dict.set_item(
        "median_frame",
        Array2::from_shape_vec((height, width), clean.median_frame)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .into_pyarray(py),
    )?;
    dict.set_item(
        "star_mask",
        Array2::from_shape_vec((height, width), clean.star_mask)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .into_pyarray(py),
    )?;
    dict.set_item("level", clean.level)?;
    dict.set_item("sigma", clean.sigma)?;
    dict.set_item("n_masked", clean.n_masked)?;
    Ok(dict)
}

/// Detección completa en Rust: mediana temporal, v_min, NMS 4D y diagnósticos.
#[pyfunction]
#[pyo3(signature = (
    residuals,
    v_min=0.08,
    r_nms=5.0,
    detection_sigma=5.0,
    velocity_max=1.5,
    velocity_step=0.25,
    match_filter_sigma=0.8,
    min_component_pixels=2,
    star_sigma=5.0,
    star_mask_radius=3,
    star_core_sigma=100.0,
    temporal_median=true,
    diagnostic_radius=2.5,
    ref_frame=None,
    backend="auto",
    max_candidates=200,
))]
#[allow(clippy::too_many_arguments)]
fn detect_candidates<'py>(
    py: Python<'py>,
    residuals: PyReadonlyArray3<'_, f64>,
    v_min: f64,
    r_nms: f64,
    detection_sigma: f64,
    velocity_max: f64,
    velocity_step: f64,
    match_filter_sigma: f64,
    min_component_pixels: usize,
    star_sigma: f64,
    star_mask_radius: usize,
    star_core_sigma: f64,
    temporal_median: bool,
    diagnostic_radius: f64,
    ref_frame: Option<usize>,
    backend: &str,
    max_candidates: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let view = residuals.as_array();
    let shape = view.shape();
    let (n_frames, height, width) = (shape[0], shape[1], shape[2]);
    if n_frames == 0 || height == 0 || width == 0 {
        return Err(PyValueError::new_err("el cubo no puede estar vacío"));
    }
    let frames: Vec<f64> = view.iter().copied().collect();
    let mut config = PipelineConfig::default();
    config.v_min = v_min.max(0.0);
    config.r_nms = r_nms.max(0.0);
    config.detection_sigma = detection_sigma;
    config.velocity_max = velocity_max;
    config.velocity_step = if velocity_step > 0.0 { velocity_step } else { 0.25 };
    config.match_filter_sigma = match_filter_sigma.max(0.0);
    config.min_component_pixels = min_component_pixels.max(1);
    config.star_sigma = star_sigma.max(0.0);
    config.star_mask_radius = star_mask_radius;
    config.star_core_sigma = star_core_sigma.max(0.0);
    config.temporal_median = temporal_median;
    config.diagnostic_radius = diagnostic_radius;
    config.max_candidates = max_candidates;
    config.backend = parse_backend(backend)?;
    let result = errantes_core::detect_candidates(
        &frames, n_frames, height, width, &config, ref_frame,
    )
    .map_err(PyRuntimeError::new_err)?;

    let dict = PyDict::new(py);
    let mut rows: Vec<Bound<'py, PyDict>> = Vec::new();
    for candidate in &result.candidates {
        rows.push(candidate_to_dict(py, candidate)?);
    }
    dict.set_item("candidates", rows)?;
    // El mejor mapa apilado se reconstruye con el tamaño del cubo.
    let size = height * width;
    let mut best = result.best_stack.clone();
    best.resize(size, 0.0);
    dict.set_item(
        "best_stack",
        Array2::from_shape_vec((height, width), best)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .into_pyarray(py),
    )?;
    dict.set_item(
        "snr_by_velocity",
        Array1::from_vec(result.snr_by_velocity.clone()).into_pyarray(py),
    )?;
    dict.set_item(
        "median_frame",
        Array2::from_shape_vec((height, width), result.preprocess.median_frame.clone())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .into_pyarray(py),
    )?;
    dict.set_item(
        "star_mask",
        Array2::from_shape_vec((height, width), result.preprocess.star_mask.clone())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .into_pyarray(py),
    )?;
    dict.set_item("ref_frame", result.ref_frame)?;
    dict.set_item("n_velocities", result.n_velocities)?;
    dict.set_item("n_masked", result.preprocess.n_masked)?;
    dict.set_item("runtime_s", result.runtime_s)?;
    dict.set_item("backend", result.backend_used.clone())?;
    Ok(dict)
}

/// Diagnósticos de trayectoria sobre un cubo ya saneado.
#[pyfunction]
#[pyo3(signature = (residuals, x, y, vx, vy, ref_frame=0, radius=2.5))]
fn track_diagnostics<'py>(
    py: Python<'py>,
    residuals: PyReadonlyArray3<'_, f64>,
    x: f64,
    y: f64,
    vx: f64,
    vy: f64,
    ref_frame: usize,
    radius: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let view = residuals.as_array();
    let shape = view.shape();
    let (n_frames, height, width) = (shape[0], shape[1], shape[2]);
    let frames: Vec<f64> = view.iter().copied().collect();
    let candidate = Candidate {
        x, y, vx, vy, ref_frame, ..Candidate::default()
    };
    let metrics = track_metrics(
        &frames, n_frames, height, width, &candidate, ref_frame, radius,
    );
    let dict = PyDict::new(py);
    dict.set_item("track_fluxes", metrics.fluxes.clone())?;
    dict.set_item("track_rms", finite(metrics.track_rms))?;
    dict.set_item("track_rms_x", finite(metrics.rms_x))?;
    dict.set_item("track_rms_y", finite(metrics.rms_y))?;
    dict.set_item("photometric_cv", finite(metrics.photometric_cv))?;
    dict.set_item("cosmic_fraction", finite(metrics.cosmic_fraction))?;
    Ok(dict)
}

#[pymodule]
fn errantes_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ShiftStackEngine>()?;
    m.add_function(wrap_pyfunction!(backend_decision, m)?)?;
    m.add_function(wrap_pyfunction!(gpu_available, m)?)?;
    m.add_function(wrap_pyfunction!(vram_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(subtract_temporal_median, m)?)?;
    m.add_function(wrap_pyfunction!(detect_candidates, m)?)?;
    m.add_function(wrap_pyfunction!(track_diagnostics, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
