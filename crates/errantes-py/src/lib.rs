//! Bindings PyO3 del motor Rust. La API espeja `errantes.detection.ShiftStackEngine`
//! para que el wrapper Python pueda delegar el hot path sin cambiar su interfaz.

use std::sync::Mutex;

use errantes_core::backend::{choose_backend, detected_vram_bytes, BackendDecision};
use errantes_core::config::ComputeBackend;
use errantes_core::ShiftStackEngine as CoreEngine;
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

#[pymodule]
fn errantes_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ShiftStackEngine>()?;
    m.add_function(wrap_pyfunction!(backend_decision, m)?)?;
    m.add_function(wrap_pyfunction!(gpu_available, m)?)?;
    m.add_function(wrap_pyfunction!(vram_bytes, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
