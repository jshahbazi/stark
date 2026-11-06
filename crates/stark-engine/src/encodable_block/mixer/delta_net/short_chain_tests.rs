//! Synthetic projected inputs with private checkpoint parameters. No payloads
//! or tensor values are printed; each accepted prefix must match serial T=1.

use std::{fs::File, path::PathBuf};

use half::bf16;
use serde_json::Value;
use stark_engine_macros::stark_test;

use super::*;
use crate::{
    backends::{
        common::{CommandBufferExecutable, CommandBufferPending, gpu_types::trie::TrieNode},
        metal::Metal,
    },
    parameters::ParameterLoader,
};

#[stark_test]
fn short_chain_fused_raw_matches_serial_t1() {
    const KEY_DIM: usize = 16 * 128;
    const VALUE_DIM: usize = 48 * 128;
    let context = <Metal as Backend>::Context::new().unwrap();
    if !context.supports_mxu {
        return;
    }
    let verify = <<Metal as Backend>::Kernels as Kernels>::DeltaNetTreeVerify::new(
        &context,
        &TreeVerifyNewArguments {
            data_type: DataType::BF16,
            num_k_heads: 16,
            num_v_heads: 48,
            head_k_dim: 128,
            head_v_dim: 128,
        },
    )
    .unwrap();
    let advance =
        <<Metal as Backend>::Kernels as Kernels>::StateAdvanceKernel::new(&context, DataType::BF16, 128, 48, 16)
            .unwrap();
    for seed in 0..3usize {
        for count in 2..=4usize {
            let q = (0..count * KEY_DIM)
                .map(|i| bf16::from_f32(((i * 17 + seed * 31) % 103) as f32 * 0.002 - 0.102))
                .collect::<Vec<_>>();
            let k = (0..count * KEY_DIM)
                .map(|i| bf16::from_f32(((i * 23 + seed * 19) % 107) as f32 * 0.002 - 0.106))
                .collect::<Vec<_>>();
            let v = (0..count * VALUE_DIM)
                .map(|i| bf16::from_f32(((i * 11 + seed * 37) % 109) as f32 * 0.01 - 0.54))
                .collect::<Vec<_>>();
            let beta = (0..count * 48).map(|i| 0.05 + (i % 17) as f32 * 0.05).collect::<Vec<_>>();
            let decay = (0..count * 48).map(|i| -0.01 - (i % 29) as f32 * 0.15).collect::<Vec<_>>();
            let initial = (0..48 * 128 * 128)
                .map(|i| {
                    if seed == 0 {
                        0.0
                    } else {
                        ((i * 13 + seed) % 113) as f32 * 0.0002 - 0.0112
                    }
                })
                .collect::<Vec<_>>();
            let mut h0 = context.create_buffer(initial.len() * 4).unwrap();
            h0.copyin(&initial);
            let mut command = context.create_command_buffer(None, None).unwrap();
            let mut qs = command.allocate_scratch(q.len() * 2).unwrap();
            let mut ks = command.allocate_scratch(k.len() * 2).unwrap();
            let mut vs = command.allocate_scratch(v.len() * 2).unwrap();
            let mut bs = command.allocate_scratch(beta.len() * 4).unwrap();
            let mut ds = command.allocate_scratch(decay.len() * 4).unwrap();
            qs.copyin(&q);
            ks.copyin(&k);
            vs.copyin(&v);
            bs.copyin(&beta);
            ds.copyin(&decay);
            let trie = command
                .allocate_constant_from_slice(&[TrieNode {
                    trie_start: 0,
                    trie_end: 0,
                    height: 0,
                }])
                .unwrap();
            let actual = verify
                .encode_short_chain(
                    TreeVerifyEncodeArguments {
                        q: qs.subrange(..),
                        k: ks.subrange(..),
                        v: vs.subrange(..),
                        trie: &trie,
                        log_decay: ds.subrange(..),
                        beta: bs.subrange(..),
                        h0: &h0,
                        tree_size: count as u32,
                    },
                    &mut command,
                )
                .unwrap()
                .expect("qualified synthetic geometry must use fused kernel");
            let mut working = command.allocate_scratch(h0.size()).unwrap();
            command.encode_copy(&h0, &mut working);
            let accepted = command.allocate_constant_from_slice(&[0u32]).unwrap();
            let mut reference = command.allocate_scratch(count * VALUE_DIM * 2).unwrap();
            for row in 0..count {
                let q = qs.subrange(row * KEY_DIM * 2..(row + 1) * KEY_DIM * 2);
                let k = ks.subrange(row * KEY_DIM * 2..(row + 1) * KEY_DIM * 2);
                let v = vs.subrange(row * VALUE_DIM * 2..(row + 1) * VALUE_DIM * 2);
                let beta = bs.subrange(row * 48 * 4..(row + 1) * 48 * 4);
                let log_decay = ds.subrange(row * 48 * 4..(row + 1) * 48 * 4);
                let output = verify
                    .encode(
                        TreeVerifyEncodeArguments {
                            q,
                            k,
                            v,
                            trie: &trie,
                            log_decay,
                            beta,
                            h0: &working,
                            tree_size: 1,
                        },
                        &mut command,
                    )
                    .unwrap();
                command.encode_copy(&output, reference.subrange_mut(row * VALUE_DIM * 2..(row + 1) * VALUE_DIM * 2));
                if row + 1 < count {
                    advance.encode(k, v, log_decay, beta, &accepted, &mut working, 1, &mut command);
                }
            }
            let mut actual_readback = context.create_buffer(actual.size()).unwrap();
            let mut reference_readback = context.create_buffer(reference.size()).unwrap();
            command.encode_copy(&actual, &mut actual_readback);
            command.encode_copy(&reference, &mut reference_readback);
            drop(trie);
            drop(accepted);
            command.end_encoding().submit().wait_until_completed().unwrap();
            assert!(
                actual_readback.copyout::<u16>() == reference_readback.copyout::<u16>(),
                "raw fused mismatch T={count} seed={seed}"
            );
            assert!(h0.copyout::<f32>() == initial, "fused verification modified committed state");
        }
    }
}

fn seeded_state(
    delta: &DeltaNet<Metal>,
    context: &<Metal as Backend>::Context,
) -> DeltaNetState<Metal> {
    let conv_len = (delta.conv_dim * (delta.kernel_size - 1)) as usize;
    let ssm_len = (delta.num_heads * delta.value_head_dim * delta.head_dim) as usize;
    let mut conv_state = context.create_buffer(conv_len * 4).unwrap();
    let mut ssm_state = context.create_buffer(ssm_len * 4).unwrap();
    conv_state.copyin(&(0..conv_len).map(|i| ((i % 53) as f32 - 26.0) * 0.003).collect::<Vec<_>>());
    ssm_state.copyin(&(0..ssm_len).map(|i| ((i % 97) as f32 - 48.0) * 0.0001).collect::<Vec<_>>());
    DeltaNetState {
        conv_state,
        ssm_state,
        suffix_status: None,
        #[cfg(feature = "deferred-state-init")]
        needs_zero_initialization: false,
        state_advance: <<Metal as Backend>::Kernels as Kernels>::StateAdvanceKernel::new(
            context,
            DataType::BF16,
            delta.head_dim,
            delta.num_heads,
            delta.num_groups,
        )
        .unwrap(),
    }
}

// Requires the exact private native artifact via STARK_TEST_MODEL_PATH.
#[ignore]
#[stark_test]
fn short_chain_batched_prep_matches_serial_t1() {
    let artifact = PathBuf::from(std::env::var_os("STARK_TEST_MODEL_PATH").expect("set private native artifact"));
    let config: Value = serde_json::from_slice(&std::fs::read(artifact.join("config.json")).unwrap()).unwrap();
    let transformer = &config["decoder_config"]["transformer_config"];
    let delta_config: DeltaNetConfig =
        serde_json::from_value(transformer["layer_configs"][0]["mixer_config"].clone()).unwrap();
    let context = <Metal as Backend>::Context::new().unwrap();
    let file = File::open(artifact.join("model.safetensors")).unwrap();
    let loader = ParameterLoader::<Metal>::new(&file, &context).unwrap();
    let tree = loader.tree().subtree("decoder.transformer.layers.0.mixer");
    let (delta, _retained) = DeltaNet::<Metal>::new(
        transformer["model_dim"].as_u64().unwrap() as u32,
        DataType::BF16,
        &delta_config,
        &tree,
        &context,
    )
    .unwrap();
    assert!(delta.tree_verify.is_some());
    assert!(context.supports_mxu);
    assert_eq!(delta.num_groups, 16);
    assert_eq!(delta.num_heads, 48);
    for count in 2..=4u32 {
        let projected_values = (0..count as usize * delta.total_proj_dim as usize)
            .map(|i| bf16::from_f32(((i * 17 % 101) as f32 - 50.0) * 0.01))
            .collect::<Vec<_>>();
        let mut projected = context.create_buffer(projected_values.len() * 2).unwrap();
        projected.copyin(&projected_values);
        let nodes = (0..count)
            .map(|height| TrieNode {
                trie_start: height,
                trie_end: count - 1,
                height,
            })
            .collect::<Vec<_>>();
        let batch = BatchTopology::new(&nodes, false);
        let root_node = [TrieNode {
            trie_start: 0,
            trie_end: 0,
            height: 0,
        }];
        let one = BatchTopology::new(&root_node, false);
        let mut serial = seeded_state(&delta, &context);
        let mut command = context.create_command_buffer(None, None).unwrap();
        let output_bytes = delta.value_dim as usize * 2;
        let projection_bytes = delta.total_proj_dim as usize * 2;
        let mut serial_outputs = command.allocate_scratch(count as usize * output_bytes).unwrap();
        let mut serial_states = Vec::new();
        for row in 0..count as usize {
            let output = delta
                .encode_tree_verify(
                    projected.subrange(row * projection_bytes..(row + 1) * projection_bytes),
                    &one,
                    &mut serial,
                    false,
                    &mut command,
                )
                .unwrap();
            command.encode_copy(&output, serial_outputs.subrange_mut(row * output_bytes..(row + 1) * output_bytes));
            serial.encode_accept(&[0], &mut command).unwrap();
            let mut conv = context.create_buffer(serial.conv_state.size()).unwrap();
            let mut ssm = context.create_buffer(serial.ssm_state.size()).unwrap();
            command.encode_copy(&serial.conv_state, &mut conv);
            command.encode_copy(&serial.ssm_state, &mut ssm);
            serial_states.push((conv, ssm));
        }
        let expected_output = delta.out_projection.encode(serial_outputs, count, &mut command).unwrap();
        let mut expected_readback = context.create_buffer(expected_output.size()).unwrap();
        command.encode_copy(&expected_output, &mut expected_readback);
        command.end_encoding().submit().wait_until_completed().unwrap();
        for accepted_len in 1..=count as usize {
            for use_fused in [false, true] {
                let mut candidate = seeded_state(&delta, &context);
                let initial_conv = candidate.conv_state.copyout::<u32>();
                let initial_ssm = candidate.ssm_state.copyout::<u32>();
                let mut command = context.create_command_buffer(None, None).unwrap();
                let actual = delta
                    .encode_short_chain_verify_with_mode(&projected, &batch, &mut candidate, use_fused, &mut command)
                    .unwrap();
                let mut actual_readback = context.create_buffer(actual.size()).unwrap();
                command.encode_copy(&actual, &mut actual_readback);
                command.end_encoding().submit().wait_until_completed().unwrap();
                assert!(
                    actual_readback.copyout::<u16>() == expected_readback.copyout::<u16>(),
                    "T={count} output differs"
                );
                assert!(
                    candidate.conv_state.copyout::<u32>() == initial_conv,
                    "verification changed committed convolution"
                );
                assert!(candidate.ssm_state.copyout::<u32>() == initial_ssm, "verification changed committed SSM");
                let mut command = context.create_command_buffer(None, None).unwrap();
                candidate.encode_accept(&(0..accepted_len as u32).collect::<Vec<_>>(), &mut command).unwrap();
                command.end_encoding().submit().wait_until_completed().unwrap();
                let (conv, ssm) = &serial_states[accepted_len - 1];
                assert!(
                    candidate.conv_state.copyout::<u32>() == conv.copyout::<u32>(),
                    "T={count} accepted convolution differs"
                );
                assert!(candidate.ssm_state.copyout::<u32>() == ssm.copyout::<u32>(), "T={count} accepted SSM differs");
            }
        }
    }
}
