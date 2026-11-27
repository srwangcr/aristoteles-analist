//! Backend GPU Vulkan/WGSL: FFT 2D por etapas radix-2 y acumulación de trazas.

use std::borrow::Cow;

use bytemuck::{Pod, Zeroable};
use futures::executor::block_on;
use wgpu::util::DeviceExt;

use crate::backend::padded_size;

const FFT_WGSL: &str = r#"
struct Params {
  plane_size: u32,
  len: u32,
  p: u32,
  axis: u32,
  direction: u32,
  n_planes: u32,
  pad0: u32,
  pad1: u32,
};
@group(0) @binding(0) var<storage, read_write> data: array<vec2<f32>>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.plane_size * params.n_planes) { return; }
  let plane = idx / params.plane_size;
  let local = idx % params.plane_size;
  var i0: u32;
  var j: u32;
  if (params.axis == 0u) {
    let row = local / params.p;
    let col = local % params.p;
    j = col % params.len;
    if (j >= params.len / 2u) { return; }
    // `local` ya es el primer elemento de la mariposa; el segundo está
    // desplazado len/2 dentro de la misma fila.
    i0 = idx;
  } else {
    let col = local % params.p;
    let row = local / params.p;
    j = row % params.len;
    if (j >= params.len / 2u) { return; }
    i0 = idx;
  }
  let i1 = select(i0 + params.len / 2u, i0 + (params.len / 2u) * params.p, params.axis == 1u);
  let sign = select(-1.0, 1.0, params.direction == 1u);
  let angle = sign * 6.283185307179586 * f32(j) / f32(params.len);
  let tw = vec2<f32>(cos(angle), sin(angle));
  let b = data[i1];
  let shifted = vec2<f32>(b.x * tw.x - b.y * tw.y, b.x * tw.y + b.y * tw.x);
  let a = data[i0];
  data[i0] = a + shifted;
  data[i1] = a - shifted;
}
"#;

const PACK_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> real: array<f32>;
@group(0) @binding(1) var<storage, read_write> complex: array<vec2<f32>>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&real)) { return; }
  complex[i] = vec2<f32>(real[i], 0.0);
}
"#;

const PHASE_WGSL: &str = r#"
struct Params {
  plane_size: u32,
  p: u32,
  n_planes: u32,
  pad0: u32,
  sx: f32,
  sy: f32,
  pad1: f32,
  pad2: f32,
};
@group(0) @binding(0) var<storage, read> spectrum: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> work: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.plane_size * params.n_planes) { return; }
  let local = idx % params.plane_size;
  let x = local % params.p;
  let y = local / params.p;
  let kx = select(f32(x), f32(x) - f32(params.p), x >= params.p / 2u);
  let ky = select(f32(y), f32(y) - f32(params.p), y >= params.p / 2u);
  let angle = -6.283185307179586 * (kx * params.sx + ky * params.sy) / f32(params.p);
  let ph = vec2<f32>(cos(angle), sin(angle));
  let s = spectrum[idx];
  work[idx] = vec2<f32>(s.x * ph.x - s.y * ph.y, s.x * ph.y + s.y * ph.x);
}
"#;

const ACCUM_WGSL: &str = r#"
struct Params {
  plane_size: u32,
  p: u32,
  width: u32,
  height: u32,
  n_planes: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
};
@group(0) @binding(0) var<storage, read> work: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }
  let x = idx % params.width;
  let y = idx / params.width;
  var sum = 0.0;
  for (var plane = 0u; plane < params.n_planes; plane = plane + 1u) {
    let offset = plane * params.plane_size + y * params.p + x;
    sum = sum + work[offset].x;
  }
  output[idx] = sum / f32(params.p * params.p);
}
"#;

const PERMUTE_WGSL: &str = r#"
struct Params {
  plane_size: u32,
  p: u32,
  bits: u32,
  n_planes: u32,
};
@group(0) @binding(0) var<storage, read> src: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

fn reverse_bits(input: u32) -> u32 {
  var value = input;
  var result = 0u;
  for (var i = 0u; i < params.bits; i = i + 1u) {
    result = (result << 1u) | (value & 1u);
    value = value >> 1u;
  }
  return result;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.plane_size * params.n_planes) { return; }
  let plane = idx / params.plane_size;
  let local = idx % params.plane_size;
  dst[plane * params.plane_size + reverse_bits(local)] = src[idx];
}
"#;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct PermParams {
    plane_size: u32,
    p: u32,
    bits: u32,
    n_planes: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct FftParams {
    plane_size: u32,
    len: u32,
    p: u32,
    axis: u32,
    direction: u32,
    n_planes: u32,
    pad0: u32,
    pad1: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct PhaseParams {
    plane_size: u32,
    p: u32,
    n_planes: u32,
    pad0: u32,
    sx: f32,
    sy: f32,
    pad1: f32,
    pad2: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct AccumParams {
    plane_size: u32,
    p: u32,
    width: u32,
    height: u32,
    n_planes: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

pub fn gpu_available() -> bool {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..Default::default()
    });
    block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .is_some()
}

/// Estado GPU reutilizable para un cubo de dimensiones fijas.
pub struct GpuStacker {
    device: wgpu::Device,
    queue: wgpu::Queue,
    fft_pipeline: wgpu::ComputePipeline,
    pack_pipeline: wgpu::ComputePipeline,
    phase_pipeline: wgpu::ComputePipeline,
    accum_pipeline: wgpu::ComputePipeline,
    permute_pipeline: wgpu::ComputePipeline,
    packed: wgpu::Buffer,
    spectrum: wgpu::Buffer,
    work: wgpu::Buffer,
    scratch: wgpu::Buffer,
    output: wgpu::Buffer,
    readback: wgpu::Buffer,
    phase_params: wgpu::Buffer,
    accum_params: wgpu::Buffer,
    perm_params: wgpu::Buffer,
    perm_params_single: wgpu::Buffer,
    fft_params_fwd: Vec<wgpu::Buffer>,
    fft_params_inv: Vec<wgpu::Buffer>,
    n_planes: usize,
    plane: usize,
    p: usize,
    height: usize,
    width: usize,
}

impl GpuStacker {
    pub fn new(n_frames: usize, height: usize, width: usize) -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok_or_else(|| "no se encontró adaptador Vulkan".to_string())?;
        let (device, queue) = block_on(
            adapter.request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("errantes-gpu"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            ),
        )
        .map_err(|e| format!("request_device falló: {e}"))?;

        let make_pipeline = |label: &str, src: &str| -> wgpu::ComputePipeline {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(src)),
            });
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[&layout],
                push_constant_ranges: &[],
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: "main",
                compilation_options: Default::default(),
            })
        };

        let fft_pipeline = make_pipeline("fft-radix2", FFT_WGSL);
        let _pipeline_layout = fft_pipeline.get_bind_group_layout(0);

        let make_three_buffer_pipeline = |label: &str, src: &str| -> wgpu::ComputePipeline {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(src)),
            });
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[&layout],
                push_constant_ranges: &[],
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: "main",
                compilation_options: Default::default(),
            })
        };
        let pack_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pack"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(PACK_WGSL)),
        });
        let pack_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pack"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pack_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pack"),
            bind_group_layouts: &[&pack_layout],
            push_constant_ranges: &[],
        });
        let pack_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pack"),
            layout: Some(&pack_pipeline_layout),
            module: &pack_module,
            entry_point: "main",
            compilation_options: Default::default(),
        });
        let phase_pipeline = make_three_buffer_pipeline("phase", PHASE_WGSL);
        let accum_pipeline = make_three_buffer_pipeline("accumulate", ACCUM_WGSL);
        let permute_pipeline = make_three_buffer_pipeline("permute", PERMUTE_WGSL);

        let p = padded_size(height, width);
        let plane = p * p;
        let n_planes = n_frames;
        let complex_bytes = (n_planes * plane * 8) as u64;
        let real_bytes = (n_planes * plane * 4) as u64;
        let create_buffer = |label: &str, size: u64, usage: wgpu::BufferUsages| -> wgpu::Buffer {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let packed = create_buffer(
            "padded-cube",
            real_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let spectrum = create_buffer(
            "spectrum",
            complex_bytes,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        );
        let work = create_buffer(
            "work",
            complex_bytes,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        );
        let scratch = create_buffer(
            "fft-scratch",
            complex_bytes,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        );
        let output = create_buffer(
            "stack-output",
            (plane * 4) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let readback = create_buffer(
            "stack-readback",
            (plane * 4) as u64,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let phase_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("phase-params"),
            contents: bytemuck::bytes_of(&PhaseParams::zeroed()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let accum_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("accum-params"),
            contents: bytemuck::bytes_of(&AccumParams {
                plane_size: plane as u32,
                p: p as u32,
                width: width as u32,
                height: height as u32,
                // El acumulador nunca ve más de un plano válido en `scratch`.
                n_planes: 1,
                pad0: 0,
                pad1: 0,
                pad2: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let perm_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("perm-params"),
            contents: bytemuck::bytes_of(&PermParams {
                plane_size: plane as u32,
                p: p as u32,
                bits: p.trailing_zeros(),
                n_planes: n_planes as u32,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let perm_params_single = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("perm-params-single"),
            contents: bytemuck::bytes_of(&PermParams {
                plane_size: plane as u32,
                p: p as u32,
                bits: p.trailing_zeros(),
                n_planes: 1,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let mut fft_params_fwd = Vec::new();
        let mut fft_params_inv = Vec::new();
        for direction in [0u32, 1u32] {
            for axis in [0u32, 1u32] {
                let mut len = 2;
                while len <= p {
                    let params = FftParams {
                        plane_size: plane as u32,
                        len: len as u32,
                        p: p as u32,
                        axis,
                        direction,
                        n_planes: n_planes as u32,
                        pad0: 0,
                        pad1: 0,
                    };
                    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("fft-stage"),
                        contents: bytemuck::bytes_of(&params),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });
                    if direction == 0 {
                        fft_params_fwd.push(buffer);
                    } else {
                        fft_params_inv.push(buffer);
                    }
                    len *= 2;
                }
            }
        }

        Ok(Self {
            device,
            queue,
            fft_pipeline,
            pack_pipeline,
            phase_pipeline,
            accum_pipeline,
            permute_pipeline,
            packed,
            spectrum,
            work,
            scratch,
            output,
            readback,
            phase_params,
            accum_params,
            perm_params,
            perm_params_single,
            fft_params_fwd,
            fft_params_inv,
            n_planes,
            plane,
            p,
            height,
            width,
        })
    }

    /// Crea los bind groups para cada etapa del FFT. En modo `single` solo se
    /// procesa el primer plano del buffer.
    fn fft_bind_group(
        &self,
        buffer: &wgpu::Buffer,
        params: &[wgpu::Buffer],
        _single: bool,
    ) -> Vec<wgpu::BindGroup> {
        let layout = self.fft_pipeline.get_bind_group_layout(0);
        params
            .iter()
            .map(|param| {
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("fft-bind"),
                    layout: &layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: param.as_entire_binding(),
                        },
                    ],
                })
            })
            .collect()
    }

    fn encode_permute(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        src: &wgpu::Buffer,
        dst: &wgpu::Buffer,
        single: bool,
    ) {
        let layout = self.permute_pipeline.get_bind_group_layout(0);
        let params = if single {
            &self.perm_params_single
        } else {
            &self.perm_params
        };
        let n_items = if single { self.plane } else { self.plane * self.n_planes };
        let size = std::num::NonZeroU64::new(((if single { self.plane } else { self.plane * self.n_planes }) * 8) as u64);
        let src_res = wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: src, offset: 0, size });
        let dst_res = wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: dst, offset: 0, size });
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("perm-bind"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: src_res },
                wgpu::BindGroupEntry { binding: 1, resource: dst_res },
                wgpu::BindGroupEntry { binding: 2, resource: params.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("permute-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.permute_pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(((n_items + 255) / 256) as u32, 1, 1);
    }

    fn encode_fft<'a>(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        buffer: &wgpu::Buffer,
        inverse: bool,
        single: bool,
    ) {
        let (params, scratch) = if inverse {
            (&self.fft_params_inv, &self.scratch)
        } else {
            (&self.fft_params_fwd, &self.scratch)
        };
        self.encode_permute(encoder, buffer, scratch, single);
        let copy_bytes = if single {
            (self.plane * 8) as u64
        } else {
            (self.plane * self.n_planes * 8) as u64
        };
        encoder.copy_buffer_to_buffer(scratch, 0, buffer, 0, copy_bytes);
        let groups = self.fft_bind_group(buffer, params, single);
        let n_items = if single { self.plane } else { self.plane * self.n_planes };
        for group in &groups {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fft-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fft_pipeline);
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(((n_items + 255) / 256) as u32, 1, 1);
        }
    }

    /// Sube el cubo y calcula los espectros forward una sola vez.
    pub fn upload_and_forward(&mut self, residuals: &[f64]) -> Result<(), String> {
        if residuals.len() != self.n_planes * self.height * self.width {
            return Err("tamaño del cubo incorrecto".to_string());
        }
        let mut padded = vec![0.0_f32; self.n_planes * self.plane];
        for t in 0..self.n_planes {
            for y in 0..self.height {
                for x in 0..self.width {
                    padded[t * self.plane + y * self.p + x] =
                        residuals[t * self.height * self.width + y * self.width + x] as f32;
                }
            }
        }
        self.queue.write_buffer(&self.packed, 0, bytemuck::cast_slice(&padded));

        let pack_layout = self.pack_pipeline.get_bind_group_layout(0);
        let pack_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pack-bind"),
            layout: &pack_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.packed.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.spectrum.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("pack-encode"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pack-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pack_pipeline);
            pass.set_bind_group(0, &pack_group, &[]);
            pass.dispatch_workgroups(((self.n_planes * self.plane + 255) / 256) as u32, 1, 1);
        }
        self.encode_fft(&mut encoder, &self.spectrum, false, false);
        self.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    /// El camino GPU mantiene la infraestructura wgpu/Vulkan pero delega el
    /// cálculo al kernel CPU de paridad exacta hasta que el shader FFT 2D esté
    /// validado en todas las GPUs objetivo. El dispatch, VRAM check y API
    /// permanecen listos para el swap.
    pub fn stack(&self, _vx: f64, _vy: f64, _ref_frame: usize) -> Result<Vec<f64>, String> {
        Err("GPU FFT shader en validación; usar backend CPU/Rayon para paridad exacta".to_string())
    }


}
