use metal::MTLGPUFamily;

use super::{DEFAULT_RESULTS_PER_SIMDGROUP, GemvTile};
use crate::backends::metal::context::LARGE_MIN_GPU_CORES;

const QUANT_N_BUCKET_MAXES: [u32; 6] = [512, 2048, 4096, 8192, 16384, 32768];
const QUANT_K_BUCKET_MAXES: [u32; 3] = [512, 2048, 8192];

/// Lane-sliced tiles split each quantization group across reduction lanes.
/// A lane stages 8 bytes: 16 W4 values or 8 W8 values.
const fn lane_tile(
    num_simdgroups: u32,
    results_per_simdgroup: u32,
    bits: u32,
    group: u32,
) -> GemvTile {
    let pack_factor = match bits {
        4 => 8,
        5 | 6 | 8 => 4,
        _ => 1,
    };
    GemvTile::quantized(num_simdgroups, results_per_simdgroup, 1, 32, group / (2 * pack_factor))
}

const fn lane_default(
    bits: u32,
    group: u32,
) -> GemvTile {
    lane_tile(8, DEFAULT_RESULTS_PER_SIMDGROUP, bits, group)
}

pub fn gathered_tile(
    bits: u32,
    group: u32,
    m: u32,
    n: u32,
) -> Option<GemvTile> {
    if matches!(bits, 5 | 6) {
        let tile = lane_default(bits, group);
        return (group == 64 && m == 1 && n >= tile.rows_per_lane()).then_some(tile);
    }
    if !matches!(bits, 4 | 8) || !matches!(group, 16 | 32 | 64 | 128) || !(1..=8).contains(&m) {
        return None;
    }
    let tile = lane_default(bits, group);
    (n >= tile.rows_per_lane()).then_some(tile)
}

fn table_bucket_index(
    value: u32,
    bucket_maxes: &[u32],
) -> usize {
    bucket_maxes.partition_point(|&max| value > max)
}

fn lane_policy(
    gpu_core_count: u32,
    apple_gpu_family: MTLGPUFamily,
    m: u32,
    n: u32,
    k: u32,
    bits: u32,
    group: u32,
) -> GemvTile {
    let is_large_gpu = gpu_core_count >= LARGE_MIN_GPU_CORES;
    if m != 1 || bits != 4 {
        return lane_default(bits, group);
    }

    let k_bucket = table_bucket_index(k, &QUANT_K_BUCKET_MAXES);
    let n_bucket = table_bucket_index(n, &QUANT_N_BUCKET_MAXES);
    let (num_simdgroups, results_per_simdgroup) = match (is_large_gpu, apple_gpu_family, k_bucket, n_bucket) {
        (true, _, 0, 1) => (4, 2),
        (true, _, 1, 0) => (2, 1),
        (true, _, 1, 1..=3) => (2, 2),
        (true, _, 1, 4) => (2, 1),
        (true, _, 1, 5) => (2, 4),
        (true, _, 2, 1) => (4, 2),
        (true, _, 3, 1) => (2, 2),
        (false, MTLGPUFamily::Apple9, 0, 1) => (4, 4),
        (false, MTLGPUFamily::Apple9, 1, 0) => (4, 2),
        (false, MTLGPUFamily::Apple9, 1, 1) => (2, 2),
        (false, MTLGPUFamily::Apple9, 1, 2) => (4, 2),
        (false, MTLGPUFamily::Apple9, 1, 4) => (2, 2),
        (false, MTLGPUFamily::Apple9, 1, 5) => (4, 2),
        (false, MTLGPUFamily::Apple9, 2, 1) => (4, 2),
        (false, MTLGPUFamily::Apple9, 3, 1) => (2, 2),
        (false, MTLGPUFamily::Apple8, 0, 1) => (4, 4),
        (false, MTLGPUFamily::Apple8, 1, _) | (false, MTLGPUFamily::Apple8, 2, 1) => (8, 2),
        (false, family, 0, 1) if family < MTLGPUFamily::Apple8 => (4, 8),
        (false, family, 1, 0..=3) if family < MTLGPUFamily::Apple8 => (8, 2),
        _ => (8, 4),
    };
    let selected = lane_tile(num_simdgroups, results_per_simdgroup, bits, group);
    if n < selected.results_per_simdgroup {
        lane_default(bits, group)
    } else {
        selected
    }
}

pub fn select(
    gpu_core_count: u32,
    apple_gpu_family: MTLGPUFamily,
    bits: u32,
    group: u32,
    m: u32,
    n: u32,
    k: u32,
    bf16_io: bool,
    qwen35_affine_layout: bool,
) -> Option<GemvTile> {
    if matches!(bits, 5 | 6) {
        if !qwen35_affine_layout || !bf16_io || group != 64 || !(1..=4).contains(&m) || !k.is_multiple_of(256) {
            return None;
        }
        let mut tile = lane_default(bits, group);
        tile.input_row_tile = if gpu_core_count >= LARGE_MIN_GPU_CORES && apple_gpu_family == MTLGPUFamily::Apple10 {
            m
        } else {
            1
        };
        return (n >= tile.rows_per_lane()).then_some(tile);
    }
    if !matches!(bits, 4 | 8) || !matches!(group, 16 | 32 | 64 | 128) || !(1..=8).contains(&m) {
        return None;
    }

    if !bf16_io {
        let tile = lane_default(bits, group);
        return (m <= 4 && n >= tile.rows_per_lane()).then_some(tile);
    }

    // Dense Qwen27 verification reuses each affine weight load across its
    // two to four input rows. Reuse the full verification row tile while
    // preserving scalar output geometry, K lanes and group arithmetic.
    if qwen35_affine_layout
        && gpu_core_count >= LARGE_MIN_GPU_CORES
        && apple_gpu_family == MTLGPUFamily::Apple10
        && group == 64
        && (2..=4).contains(&m)
        && matches!(
            (n, k),
            (5120, 17408)
                | (34816, 5120)
                | (16480, 5120)
                | (5120, 6144)
                | (14336, 5120)
                | (248320, 5120)
                | (5120, 10240)
        )
    {
        let mut tile = lane_default(bits, group);
        // A controlled cold-weight M5 comparison keeps the W8 down projection
        // faster with two input rows per tile at m=4; K reduction stays fixed.
        tile.input_row_tile = if bits == 8 && m == 4 && n == 5120 && k == 17408 {
            2
        } else {
            m
        };
        return Some(tile);
    }

    // M5 geometry for the fused Qwen3.5-2B decode projections.
    // Keep each row's K lanes and quantization-group partition unchanged;
    // increasing rows per SIMD group reuses its activation loads.
    if qwen35_affine_layout
        && gpu_core_count >= LARGE_MIN_GPU_CORES
        && apple_gpu_family == MTLGPUFamily::Apple10
        && bits == 4
        && group == 64
        && m == 1
        && k == 2048
        && matches!(n, 8224 | 12288)
    {
        return Some(lane_tile(2, 4, bits, group));
    }

    let tile = match m {
        1..=4 => lane_policy(gpu_core_count, apple_gpu_family, m, n, k, bits, group),
        5..=8 if n <= 64 => lane_default(bits, group),
        _ => return None,
    };
    // Input rows are tiled independently; partial tiles clamp loads and stores.
    if n < tile.rows_per_lane() {
        return None;
    }
    Some(tile)
}
