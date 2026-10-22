use std::{iter::repeat_with, mem::MaybeUninit};

use num_traits::{Float, NumCast};
use proptest::prelude::*;
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use rand_distr::Normal;
use stark_engine_macros::stark_test;

use crate::{
    array::ArrayElement,
    backends::common::{
        Backend, BufferRef, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context,
        gpu_types::trie::TrieNode,
    },
    data_type::DataType,
    dispatch_dtype,
    encodable_block::{
        batch_topology::BatchTopology,
        sampling::{Sampling, SamplingMethod},
    },
    tests::proptest::{ComparableTestResults, TestContextes, for_each_context, kernel_data_type},
};

struct SamplingTestResults(Vec<u32>);

impl ComparableTestResults for SamplingTestResults {
    fn compare(
        backend: &str,
        actual: &SamplingTestResults,
        reference: &SamplingTestResults,
    ) -> Result<(), TestCaseError> {
        prop_assert_eq!(&actual.0, &reference.0, "{} doesn't match cpu", backend);

        Ok(())
    }
}

fn do_sampling_backend<B: Backend, T: ArrayElement + Float>(
    context: &B::Context,
    logits: &[T],
    seeds: Option<&[u64]>,
    bitmask: Option<&[u32]>,
    vocab_size: usize,
    method: &SamplingMethod,
    batch_size: u32,
) -> Result<SamplingTestResults, TestCaseError> {
    let sampling = Sampling::<B>::new(T::data_type(), vocab_size as u32);

    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    let logits_buffer = command_buffer.allocate_constant_from_slice(logits).unwrap();
    let seeds_buffer = seeds.map(|seeds| command_buffer.allocate_constant_from_slice(seeds).unwrap());
    let bitmask_buffer = bitmask.map(|bitmask| command_buffer.allocate_constant_from_slice(bitmask).unwrap());

    let nodes = (0..batch_size)
        .map(|index| TrieNode {
            trie_start: index,
            trie_end: batch_size - 1,
            height: index,
        })
        .collect::<Box<[_]>>();
    let batch_topology = BatchTopology::new(&nodes, true);

    let sampled_buffer = sampling
        .encode(
            &logits_buffer,
            seeds_buffer.as_ref(),
            bitmask_buffer.as_ref(),
            None::<&B::GlobalBuffer>,   // TODO
            None::<&B::ConstantBuffer>, // TODO
            method,
            &batch_topology,
            (0..batch_size).into(),
            &mut command_buffer,
        )
        .unwrap();
    drop(logits_buffer);
    drop(seeds_buffer);
    drop(bitmask_buffer);
    command_buffer.end_encoding().submit().wait_until_completed().unwrap();

    let sampled = sampled_buffer.copyout::<u32>();

    Ok(SamplingTestResults(sampled))
}

fn get_data<T: ArrayElement + Float>(
    seed: u64,
    batch_size: usize,
    vocab_size: usize,
    bitmask: bool,
    stochastic: bool,
) -> (Box<[T]>, Option<Box<[u64]>>, Option<Box<[u32]>>) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mean = rng.random_range(-100.0..100.0);
    let std_dev = rng.random_range(24.0..64.0);

    let distribution: Normal<f32> = Normal::new(mean, std_dev).unwrap();
    let mut logits: Box<[MaybeUninit<T>]> = Box::new_uninit_slice(batch_size * vocab_size);
    for logit in logits.iter_mut() {
        logit.write(<T as NumCast>::from(rng.sample(distribution)).unwrap());
    }

    let seeds = stochastic.then(|| repeat_with(|| rng.random::<u64>()).take(batch_size).collect());
    let bitmask = bitmask.then(|| {
        let bitmask_size = vocab_size.div_ceil(u32::BITS as usize);
        let mut bitmask: Box<[MaybeUninit<u32>]> = Box::new_uninit_slice(batch_size * bitmask_size);
        for entry in bitmask.iter_mut() {
            entry.write(rng.random::<u32>());
        }
        unsafe { bitmask.assume_init() }
    });

    (unsafe { logits.assume_init() }, seeds, bitmask)
}

#[derive(Debug, Clone)]
struct SamplingCase {
    data_type: DataType,
    seed: u64,
    batch_size: usize,
    vocab_size: usize,
    bitmask: bool,
    method: SamplingMethod,
}

fn sampling_case() -> impl Strategy<Value = SamplingCase> {
    (kernel_data_type(), any::<u64>().no_shrink(), 1usize..=7, 1usize..=123456, any::<bool>()).prop_flat_map(
        |(data_type, seed, batch_size, vocab_size, bitmask)| {
            let temperature = prop_oneof![Just(None), (0.05f32..=4.0).prop_map(Some),];

            let top_k = prop_oneof![Just(None), (1u32..=vocab_size as u32).prop_map(Some),];

            let top_p = prop_oneof![Just(None), (0.01f32..=1.0).prop_map(Some),];

            let min_p = prop_oneof![Just(None), (0.0f32..=1.0).prop_map(Some),];

            prop_oneof![
                Just(SamplingMethod::Greedy),
                (temperature, top_k, top_p, min_p).prop_map(|(temperature, top_k, top_p, min_p)| {
                    SamplingMethod::Stochastic {
                        temperature,
                        top_k,
                        top_p,
                        min_p,
                        repetition_penalty: None,       // TODO
                        suffix_repetition_length: None, // TODO
                    }
                },),
            ]
            .prop_map(move |method| SamplingCase {
                data_type,
                seed,
                batch_size,
                vocab_size,
                bitmask,
                method,
            })
        },
    )
}

#[stark_test]
fn test_sampling_prop() {
    let contextes = TestContextes::new();

    proptest!(|(sampling_case in sampling_case())| {
        dispatch_dtype!(|(T: sampling_case.data_type)| {
            let (logits, seeds, bitmask) = get_data(sampling_case.seed, sampling_case.batch_size, sampling_case.vocab_size, sampling_case.bitmask, matches!(sampling_case.method, SamplingMethod::Stochastic { .. }));

            for_each_context!(contextes, |context: C| do_sampling_backend::<<C as Context>::Backend, T>(
                context,
                logits.as_ref(),
                seeds.as_ref().map(Box::as_ref),
                bitmask.as_ref().map(Box::as_ref),
                sampling_case.vocab_size,
                &sampling_case.method,
                sampling_case.batch_size as u32,
            ))
            .compare_results()?
        });
    });
}

#[stark_test]
fn native_sampling_options_reject_invalid_effective_values() {
    let method = |temperature, top_k, top_p, min_p| SamplingMethod::Stochastic {
        temperature,
        top_k,
        top_p,
        min_p,
        repetition_penalty: None,
        suffix_repetition_length: None,
    };
    for temperature in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::from_bits(1)] {
        assert!(method(Some(temperature), None, None, None).validate(63).is_err());
    }
    for top_k in [0, 64, u32::MAX] {
        assert!(method(None, Some(top_k), None, None).validate(63).is_err());
    }
    for top_p in [0.0, -1.0, 1.1, f32::NAN, f32::INFINITY] {
        assert!(method(None, None, Some(top_p), None).validate(63).is_err());
    }
    for min_p in [-1.0, 1.1, f32::NAN, f32::INFINITY] {
        assert!(method(None, None, None, Some(min_p)).validate(63).is_err());
    }
    assert!(SamplingMethod::Greedy.validate(0).is_err());
    assert!(SamplingMethod::Greedy.validate(63).is_ok());
    assert!(method(None, None, None, None).validate(63).is_ok());
    assert!(method(Some(0.7), Some(63), Some(1.0), Some(0.0)).validate(63).is_ok());
    assert!(method(Some(0.7), Some(1), Some(0.9), Some(1.0)).validate(63).is_ok());
}

#[stark_test]
fn native_sampling_keeps_nonfinite_candidates_within_vocabulary() {
    let contexts = TestContextes::new();
    let check = || -> Result<(), TestCaseError> {
        for logits in [[f32::NAN; 3], [f32::NEG_INFINITY; 3], [1.0, 3.0, 2.0]] {
            for_each_context!(contexts, |context: C| do_sampling_backend::<<C as Context>::Backend, f32>(
                context,
                &logits,
                None,
                None,
                3,
                &SamplingMethod::Greedy,
                1,
            ))
            .compare_results()?;
        }
        Ok(())
    };
    check().unwrap();
}

fn repetition_penalty_backend<B: Backend>(context: &B::Context) {
    use crate::backends::common::{BufferMut, Kernels, kernel::RepetitionPenaltyKernel};

    let kernel = <B::Kernels as Kernels>::RepetitionPenaltyKernel::new(context, DataType::F32).unwrap();
    for (ring, suffix, token, expected) in [
        (vec![0u32, 0], 1, 2, [2.0f32, 4.0]),
        (vec![0, 0], 1, u32::MAX, [2.0, 4.0]),
        (vec![0, 0], 1, 1, [2.0, 2.0]),
        (vec![0, 1, u32::MAX, 0], 2, 0, [1.0, 4.0]),
    ] {
        let mut commands = context.create_command_buffer(None, None).unwrap();
        let logits = commands.allocate_constant_from_slice(&[2.0f32, 4.0]).unwrap();
        let ring = commands.allocate_constant_from_slice(&ring).unwrap();
        let token = commands.allocate_constant_from_slice(&[token]).unwrap();
        let mut output = context.create_buffer(8).unwrap();
        output.copyin(&[2.0f32, 4.0]);
        kernel.encode(&logits, &mut output, &ring, &token, 2.0, suffix, 2, 0, 1, &mut commands);
        drop(logits);
        drop(ring);
        drop(token);
        commands.end_encoding().submit().wait_until_completed().unwrap();
        assert_eq!(output.copyout::<f32>(), expected);
    }
}

#[stark_test]
fn native_repetition_penalty_handles_invalid_input_and_retained_ring_tokens() {
    let contexts = TestContextes::new();
    repetition_penalty_backend::<crate::backends::cpu::Cpu>(&contexts.cpu);
    #[cfg(backend = "metal")]
    repetition_penalty_backend::<crate::backends::metal::Metal>(&contexts.metal);
}
