use std::fmt::{Debug, Display};

use half::{bf16, f16};
use num_traits::Float;
use stark_engine_macros::stark_test;

use crate::{
    array::ArrayElement,
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
            gpu_types::ActivationType, kernel::ActivationKernel,
        },
        cpu::Cpu,
    },
    data_type::DataType,
    tests::{
        assert::assert_eq_float,
        helpers::{buffer_to_vec, create_buffer, create_buffer_with_data, for_each_non_cpu_backend},
    },
};

struct Input<T: ArrayElement + Float> {
    data: Box<[T]>,
    act_type: ActivationType,
    in_place: bool,
}

fn make_data<T: ArrayElement + Float>(n: usize) -> Vec<T> {
    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        // Use a range of values including negative to exercise both branches of activations
        data.push(T::from((i as f32 * 0.3) - 2.0).unwrap());
    }
    data
}

fn get_output<T: ArrayElement + Float, B: Backend>(input: &Input<T>) -> Vec<T> {
    let context = B::Context::new().expect("Failed to create Context");
    let kernel = <<B as Backend>::Kernels as Kernels>::ActivationKernel::new(&context, T::data_type(), input.in_place)
        .expect("Failed to create ActivationKernel");

    let n = input.data.len();

    if input.in_place {
        let mut output_buffer = create_buffer_with_data::<B, T>(&context, &input.data);
        let mut command_buffer = context.create_command_buffer(None, None).expect("Failed to get command buffer");
        kernel.encode(None::<&B::GlobalBuffer>, &mut output_buffer, n as u32, input.act_type, &mut command_buffer);
        command_buffer.end_encoding().submit().wait_until_completed().unwrap();
        buffer_to_vec::<B, T>(&output_buffer)
    } else {
        let input_buffer = create_buffer_with_data::<B, T>(&context, &input.data);
        let mut output_buffer = create_buffer::<B, T>(&context, n);
        let mut command_buffer = context.create_command_buffer(None, None).expect("Failed to get command buffer");
        kernel.encode(Some(&input_buffer), &mut output_buffer, n as u32, input.act_type, &mut command_buffer);
        command_buffer.end_encoding().submit().wait_until_completed().unwrap();
        buffer_to_vec::<B, T>(&output_buffer)
    }
}

fn get_test_data<T: ArrayElement + Float>(
    act_type: ActivationType,
    in_place: bool,
) -> (Input<T>, Vec<T>) {
    let n = 128;
    let data = make_data::<T>(n);

    let input = Input {
        data: data.into_boxed_slice(),
        act_type,
        in_place,
    };

    let expected = get_output::<T, Cpu>(&input);
    (input, expected)
}

/// Large input to exercise more threads on GPU.
fn get_test_data_large<T: ArrayElement + Float>(
    act_type: ActivationType,
    in_place: bool,
) -> (Input<T>, Vec<T>) {
    let n = 4096;
    let data = make_data::<T>(n);

    let input = Input {
        data: data.into_boxed_slice(),
        act_type,
        in_place,
    };

    let expected = get_output::<T, Cpu>(&input);
    (input, expected)
}

fn test_internal<T: ArrayElement + Float + Debug + Display>(
    input: &Input<T>,
    expected: &[T],
    test_name: &str,
) {
    let eps = if matches!(T::data_type(), DataType::F16 | DataType::BF16) {
        0.02f32
    } else {
        1e-5
    };

    for_each_non_cpu_backend!(|B| {
        let actual = get_output::<T, B>(input);
        let msg = format!("Activation {} failed with backend={}", test_name, std::any::type_name::<B>(),);
        assert_eq_float::<T>(expected, &actual, eps, &msg);
    });
}

fn test_activation<T: ArrayElement + Float + Debug + Display>(
    act_type: ActivationType,
    in_place: bool,
) {
    let label = format!(
        "{:?}_{}",
        act_type,
        if in_place {
            "in_place"
        } else {
            "out_of_place"
        }
    );
    let (input, expected) = get_test_data::<T>(act_type, in_place);
    test_internal::<T>(&input, &expected, &label);
}

fn test_activation_large<T: ArrayElement + Float + Debug + Display>(
    act_type: ActivationType,
    in_place: bool,
) {
    let label = format!(
        "{:?}_{}_large",
        act_type,
        if in_place {
            "in_place"
        } else {
            "out_of_place"
        }
    );
    let (input, expected) = get_test_data_large::<T>(act_type, in_place);
    test_internal::<T>(&input, &expected, &label);
}

// SILU out-of-place tests
#[stark_test]
fn test_silu_f32() {
    test_activation::<f32>(ActivationType::SILU, false);
}

#[stark_test]
fn test_silu_f16() {
    test_activation::<f16>(ActivationType::SILU, false);
}

#[stark_test]
fn test_silu_bf16() {
    test_activation::<bf16>(ActivationType::SILU, false);
}

// SILU in-place tests
#[stark_test]
fn test_silu_in_place_f32() {
    test_activation::<f32>(ActivationType::SILU, true);
}

#[stark_test]
fn test_silu_in_place_f16() {
    test_activation::<f16>(ActivationType::SILU, true);
}

#[stark_test]
fn test_silu_in_place_bf16() {
    test_activation::<bf16>(ActivationType::SILU, true);
}

// GELU out-of-place tests
#[stark_test]
fn test_gelu_f32() {
    test_activation::<f32>(ActivationType::GELUApprox, false);
}

#[stark_test]
fn test_gelu_f16() {
    test_activation::<f16>(ActivationType::GELUApprox, false);
}

#[stark_test]
fn test_gelu_bf16() {
    test_activation::<bf16>(ActivationType::GELUApprox, false);
}

#[stark_test]
fn materialized_gelu_preserves_bf16_polynomial_primitives() {
    let data: Vec<bf16> = (-128..=128).map(|i| bf16::from_f32(i as f32 / 32.0)).collect();
    let round = |x: f64| bf16::from_f64(x).to_f64();
    let expected: Vec<bf16> = data
        .iter()
        .map(|x| {
            let x = x.to_f64();
            let cubic = round(x * x * x);
            let term = round(round(0.044715) * cubic);
            let inner = round(x + term);
            let argument = round(round((2.0 / std::f64::consts::PI).sqrt()) * inner);
            let activated = round(argument.tanh());
            bf16::from_f64(round(0.5 * x) * round(1.0 + activated))
        })
        .collect();
    let input = Input {
        data: data.into_boxed_slice(),
        act_type: ActivationType::GELUMaterializedPolynomial,
        in_place: false,
    };
    let canonical: Vec<bf16> = input.data.iter().map(|x| ActivationType::GELUApprox.activate(*x)).collect();
    assert_ne!(canonical, expected, "fixture must distinguish the source polynomial profile");
    assert_eq!(get_output::<bf16, Cpu>(&input), expected);
    for_each_non_cpu_backend!(|B| {
        assert_eq!(get_output::<bf16, B>(&input), expected);
    });
}

#[stark_test]
fn gelu_source_profile_preserves_legacy_config_defaults() {
    use crate::config::activation::AnyActivation;
    let canonical: AnyActivation = serde_json::from_str(r#"{"type":"GELU","approximate":true}"#).unwrap();
    let source: AnyActivation =
        serde_json::from_str(r#"{"type":"GELU","approximate":true,"materialize_polynomial":true}"#).unwrap();
    assert_eq!(canonical.act_type(), ActivationType::GELUApprox);
    assert_eq!(source.act_type(), ActivationType::GELUMaterializedPolynomial);
}

#[stark_test]
#[ignore]
fn materialized_gelu_matches_source_quantized_first_layer_inputs() {
    use std::{fs, path::PathBuf};

    use sha2::{Digest, Sha256};
    let root = PathBuf::from(
        std::env::var_os("STARK_GEMMA4_PRIMITIVE_FIXTURE_PATH").expect("missing private primitive fixture"),
    );
    let receipt: serde_json::Value = serde_json::from_slice(&fs::read(root.join("receipt.json")).unwrap()).unwrap();
    assert_eq!(receipt["device"], "gpu");
    assert_eq!(receipt["identity"]["mlx"], "0.32.3");
    assert_eq!(receipt["identity"]["mlx_lm_revision"], "1f7057ca530f8a594d4aab50bac41a2ed5e5c5c7");
    let read = |name: &str| -> Vec<bf16> {
        let bytes = fs::read(root.join(format!("{name}.f32"))).unwrap();
        let row = receipt["operations"].as_array().unwrap().iter().find(|row| row["operation"] == name).unwrap();
        assert_eq!(format!("{:x}", Sha256::digest(&bytes)), row["sha256"].as_str().unwrap());
        assert_eq!(bytes.len() % 4, 0);
        bytes.as_chunks::<4>().0.iter().map(|value| bf16::from_f32(f32::from_le_bytes(*value))).collect()
    };
    for prefix in ["actual_mlp", "actual_ple"] {
        let input = Input {
            data: read(&format!("{prefix}_gate_input")).into_boxed_slice(),
            act_type: ActivationType::GELUMaterializedPolynomial,
            in_place: false,
        };
        let expected = read(&format!("{prefix}_gelu"));
        let check = |actual: Vec<bf16>| {
            assert_eq!(actual.len(), expected.len());
            let different = actual.iter().zip(&expected).filter(|(a, b)| a != b).count();
            let maximum =
                actual.iter().zip(&expected).map(|(a, b)| (a.to_f32() - b.to_f32()).abs()).fold(0.0f32, f32::max);
            assert!(different == 0, "source GELU mismatch: count={different}/{}, max_abs={maximum}", expected.len());
        };
        check(get_output::<bf16, Cpu>(&input));
        for_each_non_cpu_backend!(|B| {
            check(get_output::<bf16, B>(&input));
        });
    }
}

// Exact GELU out-of-place tests
#[stark_test]
fn test_gelu_exact_f32() {
    test_activation::<f32>(ActivationType::GELUExact, false);
}

#[stark_test]
fn test_gelu_exact_f16() {
    test_activation::<f16>(ActivationType::GELUExact, false);
}

#[stark_test]
fn test_gelu_exact_bf16() {
    test_activation::<bf16>(ActivationType::GELUExact, false);
}

// GELU in-place tests
#[stark_test]
fn test_gelu_in_place_f32() {
    test_activation::<f32>(ActivationType::GELUApprox, true);
}

#[stark_test]
fn test_gelu_in_place_f16() {
    test_activation::<f16>(ActivationType::GELUApprox, true);
}

#[stark_test]
fn test_gelu_in_place_bf16() {
    test_activation::<bf16>(ActivationType::GELUApprox, true);
}

// Large SILU tests
#[stark_test]
fn test_silu_large_f32() {
    test_activation_large::<f32>(ActivationType::SILU, false);
}

#[stark_test]
fn test_silu_large_f16() {
    test_activation_large::<f16>(ActivationType::SILU, false);
}

#[stark_test]
fn test_silu_large_bf16() {
    test_activation_large::<bf16>(ActivationType::SILU, false);
}

// Large GELU tests
#[stark_test]
fn test_gelu_large_f32() {
    test_activation_large::<f32>(ActivationType::GELUApprox, false);
}

#[stark_test]
fn test_gelu_large_f16() {
    test_activation_large::<f16>(ActivationType::GELUApprox, false);
}

#[stark_test]
fn test_gelu_large_bf16() {
    test_activation_large::<bf16>(ActivationType::GELUApprox, false);
}

// Large in-place tests
#[stark_test]
fn test_silu_in_place_large_f32() {
    test_activation_large::<f32>(ActivationType::SILU, true);
}

#[stark_test]
fn test_gelu_in_place_large_f32() {
    test_activation_large::<f32>(ActivationType::GELUApprox, true);
}
