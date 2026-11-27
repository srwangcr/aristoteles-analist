//! Despachador automático CPU/GPU basado en coste, VRAM y transferencia PCIe.

use std::fs;
use std::path::Path;

use crate::config::ComputeBackend;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct BackendDecision {
    pub backend: ComputeBackend,
    pub reason: String,
    pub estimated_cpu_s: f64,
    pub estimated_gpu_s: f64,
    pub required_vram_bytes: u64,
    pub available_vram_bytes: u64,
}

/// Tamaño acolchado a potencia de dos usado por los shaders WGSL.
pub fn padded_size(height: usize, width: usize) -> usize {
    let n = height.max(width);
    n.next_power_of_two()
}

/// VRAM total de GPUs AMD visibles en sysfs; 0 si no se puede consultar.
pub fn detected_vram_bytes() -> u64 {
    let Ok(entries) = fs::read_dir("/sys/class/drm") else {
        return 0;
    };
    let mut best = 0_u64;
    for entry in entries.flatten() {
        let path = entry.path().join("device").join("mem_info_vram_total");
        if let Ok(text) = fs::read_to_string(&path) {
            if let Ok(value) = text.trim().parse::<u64>() {
                best = best.max(value);
            }
        }
    }
    // NVIDIA no expone mem_info_vram_total; nvidia-smi sí.
    if best == 0 && Path::new("/usr/bin/nvidia-smi").exists() {
        if let Ok(output) = std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
            .output()
        {
            if let Ok(text) = String::from_utf8(output.stdout) {
                best = text
                    .lines()
                    .filter_map(|line| line.trim().parse::<u64>().ok())
                    .map(|mb| mb * 1024 * 1024)
                    .max()
                    .unwrap_or(0);
            }
        }
    }
    best
}

/// Decide dónde ejecutar el barrido de velocidades.
pub fn choose_backend(
    requested: ComputeBackend,
    n_frames: usize,
    height: usize,
    width: usize,
    n_velocities: usize,
    gpu_available: bool,
) -> BackendDecision {
    let work = n_velocities * n_frames * height * width;
    let plane = padded_size(height, width);
    let required = (n_frames * plane * plane * 8 * 2
        + n_frames * plane * plane * 4
        + n_velocities * height * width * 4) as u64;
    let available = detected_vram_bytes();
    // Throughputs conservadores medidos en este workspace (CPU Rayon) y
    // estimados para un GPU Vulkan integrado/discreto de gama media.
    let estimated_cpu_s = (n_velocities * n_frames) as f64 / 2200.0;
    let bytes_gpu = (n_frames * plane * plane * 4
        + n_velocities * height * width * 4) as f64;
    let estimated_gpu_s = 0.02 + (n_velocities * n_frames) as f64 / 12000.0
        + bytes_gpu / 12.0e9;

    let backend = match requested {
        ComputeBackend::Cpu => ComputeBackend::Cpu,
        ComputeBackend::Gpu if !gpu_available => ComputeBackend::Cpu,
        ComputeBackend::Gpu => ComputeBackend::Gpu,
        ComputeBackend::Auto => {
            // El kernel GPU Vulkan está integrado pero su FFT 2D sigue en
            // validación numérica; Auto usa CPU Rayon para garantizar paridad.
            ComputeBackend::Cpu
        }
    };
    let reason = match (requested, backend) {
        (_, ComputeBackend::Cpu) if !gpu_available && requested != ComputeBackend::Cpu => {
            "sin adaptador Vulkan; CPU Rayon".to_string()
        }
        (_, ComputeBackend::Cpu) if work < 20_000 => "cubo pequeño; el PCIe domina".to_string(),
        (_, ComputeBackend::Cpu) if available > 0 && required > available * 8 / 10 => {
            "VRAM insuficiente".to_string()
        }
        (ComputeBackend::Auto, ComputeBackend::Cpu) => format!(
            "Auto: CPU Rayon por paridad exacta (shader FFT GPU en validación; est. GPU {:.3}s vs CPU {:.3}s, {} MB VRAM)",
            estimated_gpu_s, estimated_cpu_s, required / (1024 * 1024)
        ),
        (_, ComputeBackend::Cpu) => format!(
            "CPU seleccionada ({:.3}s est. CPU vs {:.3}s est. GPU)",
            estimated_cpu_s, estimated_gpu_s
        ),
        (_, ComputeBackend::Gpu) => format!(
            "GPU seleccionada ({:.3}s vs {:.3}s CPU, {} MB requeridos; shader en validación)",
            estimated_gpu_s, estimated_cpu_s, required / (1024 * 1024)
        ),
        (_, ComputeBackend::Auto) => unreachable!(),
    };
    BackendDecision {
        backend,
        reason,
        estimated_cpu_s,
        estimated_gpu_s,
        required_vram_bytes: required,
        available_vram_bytes: available,
    }
}
