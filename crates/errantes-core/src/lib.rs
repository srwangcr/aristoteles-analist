//! Núcleo científico Rust del pipeline `errantes`.

pub mod backend;
pub mod config;
pub mod fft;
pub mod gpu;
pub mod physics;
pub mod shift_stack;
pub mod types;

pub use config::{ComputeBackend, PipelineConfig, PsfConfig, SimulatorConfig, WandererSpec};
pub use shift_stack::ShiftStackEngine;
pub use types::{Candidate, Cube, DetectionOutput, GroundTruth, Wanderer};
