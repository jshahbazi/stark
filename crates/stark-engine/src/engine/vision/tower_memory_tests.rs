//! Production tower allocation controls; synthetic weights remain private.

use std::{
    fs::OpenOptions,
    io::{Seek, Write},
};

use stark_engine_macros::stark_test;

use super::*;
use crate::{
    backends::{
        common::{Buffer, CommandBufferExecutable, CommandBufferPending, Context},
        metal::Metal,
    },
    parameters::ParameterLoader,
};

#[stark_test]
fn tower_constant_budget_matches_bump_pages_across_scalar_page_boundaries() {
    let context = <Metal as Backend>::Context::new().unwrap();
    let mut command = context.create_command_buffer(None, None).unwrap();
    let uploads = [1_179_648u32, 4_718_592, 1_179_648];
    let scalar_count = 256 * 1024 / 2 + 1;
    for size in uploads {
        drop(command.allocate_constant(size as usize).unwrap());
    }
    for _ in 0..scalar_count {
        drop(command.allocate_constant(2).unwrap());
    }
    let actual = command.validation_payload_reserved_bytes().0;
    let bound = super::super::memory::constant_budget_for_requests(&uploads, scalar_count).unwrap();
    assert_eq!(actual, bound, "command-lifetime scalar uploads exceed the page inventory");
    command.end_encoding().submit().wait_until_completed().unwrap();
}

#[stark_test]
fn vision_tower_workspace_bounds_pool_reuse_and_constants_metal() {
    let context = <Metal as Backend>::Context::new().unwrap();
    let mut observations = Vec::new();
    for depth in [1u32, 3] {
        let config = VisionTowerConfig {
            depth,
            hidden_size: 64,
            intermediate_size: 96,
            num_heads: 1,
            num_position_embeddings: 4,
            out_hidden_size: 80,
            patch_size: 2,
            spatial_merge_size: 2,
            temporal_patch_size: 2,
            in_channels: 3,
            layer_norm_eps: 1e-6,
        };
        let directory = super::super::tests::PrivateDirectory::new();
        let mut file = OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .open(directory.path().join("synthetic-tower.safetensors"))
            .unwrap();
        let mut tensors = vec![
            ("patch_embed.proj.weight".to_owned(), vec![64, 3, 2, 2, 2]),
            ("patch_embed.proj.bias".to_owned(), vec![64]),
            ("pos_embed.weight".to_owned(), vec![4, 64]),
        ];
        for layer in 0..depth {
            for norm in ["norm1", "norm2"] {
                for suffix in ["weight", "bias"] {
                    tensors.push((format!("blocks.{layer}.{norm}.{suffix}"), vec![64]));
                }
            }
            for (name, input, output) in
                [("attn.qkv", 64, 192), ("attn.proj", 64, 64), ("mlp.linear_fc1", 64, 96), ("mlp.linear_fc2", 96, 64)]
            {
                tensors.push((format!("blocks.{layer}.{name}.weight"), vec![output, input]));
                tensors.push((format!("blocks.{layer}.{name}.bias"), vec![output]));
            }
        }
        for suffix in ["weight", "bias"] {
            tensors.push((format!("merger.norm.{suffix}"), vec![64]));
        }
        for (name, input, output) in [("linear_fc1", 256, 256), ("linear_fc2", 256, 80)] {
            tensors.push((format!("merger.{name}.weight"), vec![output, input]));
            tensors.push((format!("merger.{name}.bias"), vec![output]));
        }
        let mut payload_bytes = 0usize;
        let mut header = serde_json::Map::new();
        for (name, shape) in tensors {
            let length = shape.iter().map(|n| *n as usize).product::<usize>() * 2;
            header.insert(
                name,
                serde_json::json!({
                    "dtype":"BF16", "shape":shape, "data_offsets":[payload_bytes,payload_bytes+length],
                }),
            );
            payload_bytes += length;
        }
        let mut header = serde_json::to_vec(&header).unwrap();
        header.resize(header.len().next_multiple_of(8), b' ');
        file.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
        file.write_all(&header).unwrap();
        file.write_all(&vec![0; payload_bytes]).unwrap();
        file.rewind().unwrap();
        let loader = ParameterLoader::<Metal>::new(&file, &context).unwrap();
        let tower = VisionTower::<Metal>::new(&*context, &config, &loader.tree()).unwrap();
        loader.tree().assert_all_tensors_validated().unwrap();
        // Rectangular non-power-of-two row/width sizes exercise page rounding.
        let grid = [1, 4, 6];
        let rows = 24;
        let patches = vec![bf16::ZERO; rows as usize * 24];
        let pool = context.create_allocation_pool();
        let mut command = context.create_command_buffer(None, Some(pool.clone())).unwrap();
        let output = tower.encode(&patches, grid, &mut command).unwrap();
        let (constants, scratch) = command.validation_payload_reserved_bytes();
        let (constant_bound, scratch_bound) =
            super::super::memory::qwen_device_allocation_bounds(rows, &config).unwrap();
        assert!(constants <= constant_bound, "tower constants exceed the admitted bump-page inventory");
        assert!(scratch <= scratch_bound, "tower scratch exceeds the rounded one-block inventory");
        let total = tower.workspace_bytes(grid).unwrap();
        assert!(constants + scratch + patches.len() * 2 <= total, "tower workspace exceeds admission");
        assert_eq!(output.size(), 6 * 80 * 2);
        println!(
            "{}",
            serde_json::json!({
                "scope":"synthetic-production-tower-allocation", "depth":depth,
                "constant_reserved_bytes":constants, "scratch_reserved_bytes":scratch,
                "constant_bound_bytes":constant_bound, "scratch_bound_bytes":scratch_bound,
                "total_admission_bytes":total,
            })
        );
        drop(output);
        command.end_encoding().submit().wait_until_completed().unwrap();
        observations.push((constants, scratch));
    }
    // This observes the real encode path and allocator, rather than replaying
    // the estimator. Repeating identical blocks must reuse its size classes.
    assert_eq!(observations[0].1, observations[1].1, "tower scratch grew with repeated identical blocks");
    assert!(observations[1].0 >= observations[0].0, "command constants must remain accounted separately");
}
