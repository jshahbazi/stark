use metal::MTLGPUFamily;
use stark_engine_macros::stark_test;

use super::*;
use crate::backends::{
    common::{gpu_types::gemm::GemmDTransform, kernel::matmul::QuantParamsLayout},
    metal::kernel::matmul::gemm::selection::GemmProblem,
};

fn shape(
    m: u32,
    n: u32,
    k: u32,
) -> MatmulShape {
    MatmulShape {
        m,
        n,
        k,
        b_transpose: true,
        b_leading_dimension: None,
        b_prologue: GemmBPrologueKind::ScaleBiasDequant,
        b_bits: Some(4),
        b_group_size: Some(64),
        signed_codes: false,
        a_full_precision: true,
        gathered: false,
        params_layout: Some(QuantParamsLayout::GroupOutput),
        d_transform: GemmDTransform::empty(),
    }
}

fn plan(shape: MatmulShape) -> GemmPlan {
    GemmProblem::new(shape, DataType::BF16, DataType::BF16, true, MTLGPUFamily::Apple10).select_plan()
}

fn bytes(
    shape: MatmulShape,
    plan: GemmPlan,
) -> Option<usize> {
    materialized_affine_bytes(shape, plan, DataType::BF16, DataType::BF16, DataType::BF16)
}

#[stark_test]
fn materialization_bounds_and_original_partitions() {
    for (m, n, k) in [(513, 129, 256), (1024, 6144, 2560), (1024, 2560, 4096), (1024, 19456, 2560), (1024, 2560, 9728)]
    {
        let shape = shape(m, n, k);
        let plan = plan(shape);
        assert_eq!(bytes(shape, plan), Some(n as usize * k as usize * 2));
        assert!((k / plan.split_k).is_multiple_of(256));
        assert_eq!(plan.tiling, GemmTiling::Tile128x128x256_Simdgroups4x4);
    }
    let at_bound = shape(1024, 16384, 4096);
    assert_eq!(bytes(at_bound, plan(at_bound)), Some(MAX_MATERIALIZED_AFFINE_BYTES));
    let beyond_bound = shape(1024, 16385, 4096);
    assert_eq!(bytes(beyond_bound, plan(beyond_bound)), None);
    let odd_partition = shape(512, 1024, 9728);
    let odd_partition_width = odd_partition.k / plan(odd_partition).split_k;
    assert!(odd_partition_width.is_multiple_of(64));
    assert!(!odd_partition_width.is_multiple_of(256));
    assert_eq!(bytes(odd_partition, plan(odd_partition)), None);
    let short_partition = shape(1024, 512, 1024);
    assert!(short_partition.k / plan(short_partition).split_k < 256);
    assert_eq!(bytes(short_partition, plan(short_partition)), None);
    let address_overflow = shape(65537, 128, 65536);
    assert_eq!(bytes(address_overflow, plan(address_overflow)), None);
    let near_address_limit = shape(65535, 128, 65536);
    assert_eq!(bytes(near_address_limit, plan(near_address_limit)), Some(16 * 1024 * 1024));
}

#[stark_test]
fn materialization_leaves_other_routes_unchanged() {
    let target = shape(1024, 2560, 9728);
    let selected = plan(target);
    for case in [
        MatmulShape {
            m: 511,
            ..target
        },
        MatmulShape {
            gathered: true,
            ..target
        },
        MatmulShape {
            a_full_precision: false,
            ..target
        },
        MatmulShape {
            b_transpose: false,
            ..target
        },
        MatmulShape {
            b_leading_dimension: Some(target.k + 1),
            ..target
        },
        MatmulShape {
            b_prologue: GemmBPrologueKind::FullPrecision,
            ..target
        },
        MatmulShape {
            b_prologue: GemmBPrologueKind::ScaleSymmetricDequant,
            ..target
        },
        MatmulShape {
            b_prologue: GemmBPrologueKind::ScaleZeroPointDequant,
            ..target
        },
        MatmulShape {
            b_bits: Some(8),
            ..target
        },
        MatmulShape {
            b_group_size: Some(32),
            ..target
        },
        MatmulShape {
            b_group_size: Some(128),
            ..target
        },
        MatmulShape {
            signed_codes: true,
            ..target
        },
        MatmulShape {
            d_transform: GemmDTransform::RHT,
            ..target
        },
    ] {
        assert_eq!(bytes(case, selected), None);
    }
    for (weights, input, output) in [
        (DataType::F32, DataType::BF16, DataType::BF16),
        (DataType::BF16, DataType::F32, DataType::BF16),
        (DataType::BF16, DataType::BF16, DataType::F32),
    ] {
        assert_eq!(materialized_affine_bytes(target, selected, weights, input, output), None);
    }
    assert_eq!(
        bytes(
            target,
            GemmPlan {
                engine: GemmEngine::Simdgroup,
                ..selected
            }
        ),
        None
    );
    assert_eq!(
        bytes(
            target,
            GemmPlan {
                split_k: 0,
                ..selected
            }
        ),
        None
    );
    assert_eq!(
        bytes(
            target,
            GemmPlan {
                split_k: 3,
                ..selected
            }
        ),
        None
    );
    assert_eq!(
        bytes(
            target,
            GemmPlan {
                tiling: GemmTiling::Tile64x64x256_Simdgroups2x2,
                ..selected
            }
        ),
        None
    );
    assert_eq!(
        bytes(
            shape(1024, u32::MAX, u32::MAX),
            GemmPlan {
                split_k: 1,
                ..selected
            }
        ),
        None
    );
}

#[stark_test]
fn one_materialization_workspace_page_is_reused_across_request_batches() {
    use std::cell::Cell;

    use crate::backends::common::allocator::pool::PoolAllocator;

    let pool = PoolAllocator::new();
    let allocation_count = Cell::new(0);
    let mut submitted_pools = Vec::new();
    for _chunk in 0..40 {
        let batch_pool = pool.clone();
        for _layer in 0..36 {
            for _matrix in 0..4 {
                let workspace = batch_pool
                    .allocate(MAX_MATERIALIZED_AFFINE_BYTES, |allocated_size| {
                        allocation_count.set(allocation_count.get() + 1);
                        assert_eq!(allocated_size, MAX_MATERIALIZED_AFFINE_BYTES);
                        Ok::<_, ()>(allocation_count.get())
                    })
                    .unwrap();
                assert_eq!(*workspace.page(), 1);
                // This mirrors the encode lifetime: no model or kernel owns
                // the page after its materialized-weight dispatches are queued.
                drop(workspace);
            }
        }
        // Pending command buffers retain the same request pool while all
        // batches are submitted ahead of the first completion wait.
        submitted_pools.push(batch_pool);
    }
    assert_eq!(allocation_count.get(), 1);
    assert_eq!(submitted_pools.len(), 40);
}
