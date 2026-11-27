//! Núcleo científico Rust del pipeline `errantes`.

pub mod backend;
pub mod config;
pub mod detection;
pub mod fft;
pub mod gpu;
pub mod physics;
pub mod preprocess;
pub mod shift_stack;
pub mod types;

pub use config::{ComputeBackend, PipelineConfig, PsfConfig, SimulatorConfig, WandererSpec};
pub use detection::{detect_candidates, detect_candidates_with_progress, DetectionResult};
pub use preprocess::{temporal_median_subtract, TemporalClean};
pub use shift_stack::ShiftStackEngine;
pub use types::{Candidate, Cube, DetectionOutput, GroundTruth, Wanderer};
