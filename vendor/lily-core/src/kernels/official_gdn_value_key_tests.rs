// SPDX-License-Identifier: Apache-2.0
// Synthetic aggregate/hash controls. No model data or arrays are logged.
use super::*;
use crate::sha256::{hex, sha256};

const HK: usize = 16;
const HV: usize = 48;
const D: usize = 128;
const C: usize = 10240;
const CASES: [u16; 33] = [
    0xc0db, 0x0000, 0x8000, 0x3ebf, 0xbebf, 0x3ec0, 0xbec0, 0x3ec1, 0xbec1, 0x3f9f,
    0xbf9f, 0x3fa0, 0xbfa0, 0x3fff, 0xbfff, 0x4000, 0xc000, 0x4001, 0xc001, 0x401f,
    0xc01f, 0x4020, 0xc020, 0x4021, 0xc021, 0x40ff, 0xc0ff, 0x4100, 0xc100, 0x4180,
    0xc180, 0x42a0, 0xc2a0,
];

fn bf(x: f32) -> f32 {
    half::bf16::from_f32(x).to_f32()
}

fn bf_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|x| half::bf16::from_f32(*x).to_bits().to_le_bytes())
        .collect()
}

fn hash_is(bytes: &[u8], expected: &str, category: &str) -> Result<()> {
    let actual = hex(&sha256(bytes));
    ensure!(
        actual == expected,
        "{category}: actual_sha256={actual}, expected_sha256={expected}"
    );
    Ok(())
}

fn same_f32_bits(a: &[f32], b: &[f32], category: &str) -> Result<()> {
    ensure!(a.len() == b.len(), "{category}: length");
    let count = a.iter().zip(b).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    ensure!(count == 0, "{category}: mismatch_count={count}");
    Ok(())
}

fn transpose_states(values: &[f32]) -> Vec<f32> {
    let count = values.len() / (HV * D * D);
    let mut out = vec![0.0; values.len()];
    for t in 0..count {
        for h in 0..HV {
            for v in 0..D {
                for k in 0..D {
                    out[((t * HV + h) * D + k) * D + v] =
                        values[((t * HV + h) * D + v) * D + k];
                }
            }
        }
    }
    out
}

struct Fixture {
    history: Vec<f32>,
    // Stock K1 stores channel-major taps; native conv stores tap-major channels.
    weights: Vec<f32>,
    state: Vec<f32>,
    a_log: Vec<f32>,
    dt: Vec<f32>,
}

fn fixture(dense: bool) -> Fixture {
    Fixture {
        history: (0..3 * C)
            .map(|i| bf((((i * 7 + 3) % 59) as i32 - 29) as f32 / 47.0))
            .collect(),
        weights: (0..C)
            .flat_map(|ch| {
                (0..4).map(move |tap| {
                    bf((((ch * 3 + tap * 11) % 31) as i32 - 15) as f32 / 37.0)
                })
            })
            .collect(),
        state: (0..HV * D * D)
            .map(|i| {
                if dense {
                    let first = (((i * 17 + 3) % 257) as i32 - 128) as f32 / 997.0;
                    let second = (i as i32 % 11 - 5) as f32 / 65537.0;
                    first + second
                } else {
                    0.0
                }
            })
            .collect(),
        a_log: (0..HV).map(|h| bf((h as i32 % 9 - 4) as f32 / 13.0)).collect(),
        dt: (0..HV).map(|h| bf((h as i32 % 5 - 2) as f32 / 17.0)).collect(),
    }
}

fn token(step: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    (
        (0..C)
            .map(|i| bf((((i * 13 + step * 7) % 113) as i32 - 56) as f32 / 53.0))
            .collect(),
        (0..HV)
            .map(|h| bf((h as i32 % 7 - 3) as f32 / 5.0 + step as f32 * 0.125))
            .collect(),
        (0..HV).map(|h| bf((h as i32 % 17 - 8) as f32 / 7.0)).collect(),
    )
}

fn native_history(stock: &[f32]) -> Vec<f32> {
    let mut result = vec![0.0; 3 * C];
    for ch in 0..C {
        for tap in 0..3 {
            result[ch * 3 + tap] = stock[tap * C + ch];
        }
    }
    result
}

fn stock_history(native: &Tensor) -> Result<Vec<u8>> {
    let values = native.to_f32()?;
    let mut result = vec![0.0; 3 * C];
    for ch in 0..C {
        for tap in 0..3 {
            result[tap * C + ch] = values[ch * 3 + tap];
        }
    }
    Ok(bf_bytes(&result))
}

#[test]
fn official_value_key_production_safe_policy_keeps_custom_all_finite_hashes()
-> Result<()> {
    // Repeat the actual all-finite control with the production SOURCE and
    // scoped Safe policy, separately from the preserved default-policy run.
    let bits: Vec<u16> = (0..=u16::MAX).filter(|v| v & 0x7f80 != 0x7f80).collect();
    let bytes: Vec<u8> = bits.iter().flat_map(|v| v.to_le_bytes()).collect();
    hash_is(
        &bytes,
        "81100c8586b90228fcdfc535d5f2dc1222be8d798bad124b03afd8df05858fc2",
        "production finite BF16 input",
    )?;
    let ctx = MetalContext::new()?;
    let input = Tensor::from_bytes(&ctx, &bytes, &[bits.len()], DType::BF16)?;
    let outputs = (0..6)
        .map(|_| Tensor::zeros(&ctx, &[bits.len()], DType::BF16))
        .collect::<Result<Vec<_>>>()?;
    let pipeline = ctx.pipeline_with_math_mode(
        "official_gdn_k1_helper_exhaustive",
        SOURCE,
        MslVersion::V3_1,
        MslMathMode::Safe,
    )?;
    let pass = ctx.begin_concurrent()?;
    pass.dispatch_at(
        &pipeline,
        &[
            input.binding(),
            outputs[0].binding(),
            outputs[1].binding(),
            outputs[2].binding(),
            outputs[3].binding(),
            outputs[4].binding(),
            outputs[5].binding(),
        ],
        &[],
        Grid::Threads { grid: (bits.len(), 1, 1), threadgroup: (256, 1, 1) },
    )?;
    pass.commit()?.wait()?;
    hash_is(
        outputs[0].raw_bytes(),
        "6d8c7069994a4298cd6aaf5e4ab8ef5464176df027e072082314de5ec6e7bd6b",
        "production-safe custom sigmoid",
    )?;
    hash_is(
        outputs[1].raw_bytes(),
        "b35ebfe3e2574fa39079c24c08ea3d6139682976980da7c1c19759dbc36391bf",
        "production-safe custom SiLU",
    )?;
    Ok(())
}

#[test]
fn official_value_key_custom_conv_beta_and_normgate_match_actual_stock_adversarial_hashes()
-> Result<()> {
    // Actual stock CAPI report465134d6b5a84cc48cf41523bad26f593635e78e0c55da9cfe51bbe9a063a9cc.
    let ctx = MetalContext::new()?;
    let case = fixture(true);
    let values: Vec<f32> = (0..C)
        .map(|i| half::bf16::from_bits(CASES[i % CASES.len()]).to_f32())
        .collect();
    let betas: Vec<f32> = values[..HV].to_vec();
    let (_, av, _) = token(0);
    let stock_weights: Vec<f32> = (0..C).flat_map(|_| [0.0, 0.0, 0.0, 1.0]).collect();
    hash_is(
        &bf_bytes(&values),
        "53c37f5c5d2e8c0ddcfdb216bb89d6c1b60e479f9aa3db7272905bf6797238f8",
        "adversarial QKV fixture",
    )?;
    hash_is(
        &bf_bytes(&betas),
        "800900c6e6575b0af03708a1208915c3a5bb57bbdacce93b4ca4a0fcd83dc9c2",
        "adversarial beta fixture",
    )?;
    hash_is(
        &bf_bytes(&stock_weights),
        "6615c78982759da158eb961646dabc4b562dfd889760b5c7d5640d5c2e0d9913",
        "adversarial conv weights",
    )?;
    let window =
        Tensor::from_f32_as_bf16(&ctx, &native_history(&case.history), &[C, 3])?;
    let weights: Vec<f32> =
        (0..4 * C).map(|i| if i / C == 3 { 1.0 } else { 0.0 }).collect();
    let w = Tensor::from_f32_as_bf16(&ctx, &weights, &[4, C])?;
    let x = Tensor::from_f32_as_bf16(&ctx, &values, &[C])?;
    let a = Tensor::from_f32_as_bf16(&ctx, &av, &[HV])?;
    let b = Tensor::from_f32_as_bf16(&ctx, &betas, &[HV])?;
    let a_log = Tensor::from_f32(&ctx, &case.a_log, &[HV])?;
    let dt = Tensor::from_f32_as_bf16(&ctx, &case.dt, &[HV])?;
    let state = Tensor::from_f32(&ctx, &case.state, &[HV, D, D])?;
    let conv = Tensor::zeros(&ctx, &[C], DType::BF16)?;
    let raw = Tensor::zeros(&ctx, &[HV, D], DType::BF16)?;
    let pass = ctx.begin_concurrent()?;
    conv1d_step_with_layout(
        &ctx,
        &pass,
        &window,
        &x,
        &w,
        &conv,
        true,
        GdnStateLayout::ValueKey,
    )?;
    pass.level_barrier(&[&window, &conv])?;
    gdn_step_raw_value_key_mlx(
        &ctx,
        &pass,
        &conv,
        &a,
        &b,
        &a_log,
        &dt,
        &state,
        &raw,
        1.0 / 128.0f32.sqrt(),
        HK,
    )?;
    pass.commit()?.wait()?;
    hash_is(
        &stock_history(&window)?,
        "40b47e28865890b8990be06bf00cd44fd32b2bf7ceb8991408a53bbf47862d18",
        "adversarial stock conv state",
    )?;
    hash_is(
        raw.raw_bytes(),
        "d68893def5fc93fa8b9b28b79713be0502f9c493814edfbf2afb8482ce290947",
        "adversarial stock K1 raw Y",
    )?;
    hash_is(
        state.raw_bytes(),
        "a3cf14286768d88496b4454ff7c0bfe3be13bf09d833430a662d65aaac843acd",
        "adversarial stock K1 F32 state",
    )?;

    let yv: Vec<f32> = (0..HV * D)
        .map(|i| bf((((i * 7 + 3) % 53) as i32 - 26) as f32 / 19.0))
        .collect();
    let zv: Vec<f32> = (0..HV * D)
        .map(|i| half::bf16::from_bits(CASES[i % CASES.len()]).to_f32())
        .collect();
    let wv: Vec<f32> =
        (0..D).map(|i| bf((i as i32 % 7 - 3) as f32 / 17.0 + 1.0)).collect();
    hash_is(
        &bf_bytes(&yv),
        "31c42e80b6f53638a1843a0c3fdf5479b30ad24389f6ca896ed3519a81e36361",
        "stock norm Y input",
    )?;
    hash_is(
        &bf_bytes(&zv),
        "e78f29a2cb9c7d39996a52bf7d0430cb541105a7486f9e8f9fb38b8f4879f957",
        "stock norm Z input",
    )?;
    hash_is(
        &bf_bytes(&wv),
        "071b0d878adbdd692d486b893894aaa2cab2d288246b29116d9612a10b71b557",
        "stock norm BF16 gain input",
    )?;
    let y = Tensor::from_f32_as_bf16(&ctx, &yv, &[HV, D])?;
    let z = Tensor::from_f32_as_bf16(&ctx, &zv, &[HV, D])?;
    let gain = Tensor::from_f32(&ctx, &wv, &[D])?;
    let out = Tensor::zeros(&ctx, &[HV, D], DType::BF16)?;
    let pass = ctx.begin_concurrent()?;
    gated_rmsnorm_value_key_decode(&ctx, &pass, &y, &z, &gain, &out, 1e-6)?;
    pass.commit()?.wait()?;
    hash_is(
        out.raw_bytes(),
        "644b03418feaf9111d22e66433bd5c339936a5e29c4056d1d9c0cbf276f0183f",
        "actual stock S1 TAB0 norm gate",
    )?;
    Ok(())
}

#[test]
fn official_value_key_policy_is_strict_and_legacy_remains_key_value() {
    assert_eq!(
        GdnStateLayout::for_model(true, 16, 48, 128, 128),
        GdnStateLayout::ValueKey
    );
    for args in [
        (false, 16, 48, 128, 128),
        (true, 1, 3, 128, 128),
        (true, 16, 48, 64, 128),
        (true, 16, 48, 128, 64),
        (true, 16, 64, 128, 128),
    ] {
        assert_eq!(
            GdnStateLayout::for_model(args.0, args.1, args.2, args.3, args.4),
            GdnStateLayout::KeyValue
        );
    }
}

#[test]
fn official_value_key_full_conv_chain_matches_actual_stock_k1_hashes() -> Result<()> {
    // Stock run: matched-mlx-gdn-k1-oracle-report-v1.json
    // SHA-256 518efebf309c03499ca727689aabf9fbb5a908800e303f15fc7793ac5d13dd4c.
    // Controller fb5b3d66487ce94a9a6110ef1e199c536a1c6d0bd2b8be681da90fc25226553f,
    // source fc49d3147ca3a99d616556464b42d1adada426540a7545985dd7c67baa4bf16c.
    // Input hashes below prevent accepting a fixture-generation mismatch.
    let ctx = MetalContext::new()?;
    let expected_state = [
        ["f1ef2a2a680e6e0d05ef44cd64c1c51d4ac9aae827fc79f4b21d3d5d16ed0493", "", ""],
        [
            "b2bc7c3a1e93ed11c3fbfb3f67dc90426d5238687795b6af2f590af07a81fcb5",
            "647d5990d0e401d17d7486975d5425c95d469fd78fccccb312ee67f9a16744d1",
            "a375743eb8bc537be6460b8c544481f35e2198ccfe97d0effedf5f21c47d93b1",
        ],
    ];
    let expected_y = [
        ["bc858f8d2438e977460043ccb18209a515e270c702ab3005dd5df73eb4aee3fe", "", ""],
        [
            "600561ebe1c3d24a8c0a226eda8ac396565fcb68973581a15ac4f8bd38ccd834",
            "657f0f3c437857652ed2af851bba2056378ca480d006878c30bcf6d3094bafcd",
            "92563851c292992b459cbcb20f51908e46d537974b21332250bfe5a2687fd8f9",
        ],
    ];
    let expected_conv = [
        "9619ce9c108c8c29518d414799dc2b995e4df41293a87c33308445fc5eab30f7",
        "61bf35a522b9477054c5b2ebccfa978871f031dff84a0d9e9b5e18107cd657bf",
        "7b9c047714208e6568c1b19594077b0a231c63062ede0b82d1420ccd5d69cdda",
    ];
    let input_qkv = [
        "01f84dc4a3574ec8382ea494d5e5b73c88c9e846913cd6f94101dc6f5bba30ad",
        "594eb3a7e51a2529f72b7eaa6b980bf869a1c3854fe832e33d47ddc52a7a0db6",
        "e18d4dc0716a3137f178490f48c6afeda1461d378223bd00f8cbfbb2b8e59eff",
    ];
    let input_a = [
        "f5094b27b00d4a811a42eacb7611f72b078ce22445eeb1690ccaedb4bfe9e6e6",
        "9b17274ac7aba9f9956f07ea0ad19c1399c7c8de30a9cc85f84a603458502108",
        "466b49390b78c5aa29e184416a08c931184f25675d5657c3cb9fbdcb2ca46d28",
    ];
    for (case_index, dense) in [false, true].into_iter().enumerate() {
        let case = fixture(dense);
        hash_is(
            &bf_bytes(&case.history),
            "9ead6777082e7e39afc463adfa33836a3f466bd256fa941e9ca0014ecddfa2f3",
            "stock initial conv input",
        )?;
        hash_is(
            &bf_bytes(&case.weights),
            "377b6b26c283cb0fcaf229795e72236040e77faea181c0a57888ffa43bd03d3e",
            "stock conv weight input",
        )?;
        hash_is(
            bytemuck::cast_slice(&case.state),
            if dense {
                "def64ab0bb4e680fa4942788b9ac5fd30752055f04e215c2c7a1e3e82cc5afb9"
            } else {
                "bbd05cf6097ac9b1f89ea29d2542c1b7b67ee46848393895f5a9e43fa1f621e5"
            },
            "stock initial F32 VK input",
        )?;
        hash_is(
            &bf_bytes(&case.a_log),
            "46ca3b0728af18f888a10316b956a2bfe0c57b7fd52cc0d5b93b7b3a3135229a",
            "stock A_log input",
        )?;
        hash_is(
            &bf_bytes(&case.dt),
            "4411f9ed91c1af08d9dd618eb3b6f71734686160d154983931e76e51f9fac33d",
            "stock dt input",
        )?;
        let mut native_history = vec![0.0; 3 * C];
        for ch in 0..C {
            for tap in 0..3 {
                native_history[ch * 3 + tap] = case.history[tap * C + ch];
            }
        }
        let window = Tensor::from_f32_as_bf16(&ctx, &native_history, &[C, 3])?;
        let mut native_weights = vec![0.0; 4 * C];
        for ch in 0..C {
            for tap in 0..4 {
                native_weights[tap * C + ch] = case.weights[ch * 4 + tap];
            }
        }
        let weights = Tensor::from_f32_as_bf16(&ctx, &native_weights, &[4, C])?;
        let state = Tensor::from_f32(&ctx, &case.state, &[HV, D, D])?;
        let a_log = Tensor::from_f32(&ctx, &case.a_log, &[HV])?;
        let dt = Tensor::from_f32_as_bf16(&ctx, &case.dt, &[HV])?;
        let conv = Tensor::zeros(&ctx, &[C], DType::BF16)?;
        let raw = Tensor::zeros(&ctx, &[HV, D], DType::BF16)?;
        for step in 0..if dense { 3 } else { 1 } {
            let (qkv, a, b) = token(step);
            hash_is(&bf_bytes(&qkv), input_qkv[step], "stock token QKV input")?;
            hash_is(&bf_bytes(&a), input_a[step], "stock token A input")?;
            hash_is(
                &bf_bytes(&b),
                "533c60568f837234f9edc1a17f2aa4380be2e7563f382f1c731dbed9d27e5bb6",
                "stock token B input",
            )?;
            let x = Tensor::from_f32_as_bf16(&ctx, &qkv, &[C])?;
            let a = Tensor::from_f32_as_bf16(&ctx, &a, &[HV])?;
            let b = Tensor::from_f32_as_bf16(&ctx, &b, &[HV])?;
            let pass = ctx.begin_concurrent()?;
            conv1d_step_with_layout(
                &ctx,
                &pass,
                &window,
                &x,
                &weights,
                &conv,
                true,
                GdnStateLayout::ValueKey,
            )?;
            pass.level_barrier(&[&conv, &window])?;
            gdn_step_raw_value_key_mlx(
                &ctx,
                &pass,
                &conv,
                &a,
                &b,
                &a_log,
                &dt,
                &state,
                &raw,
                1.0 / 128.0f32.sqrt(),
                HK,
            )?;
            pass.commit()?.wait()?;
            let native_window = window.to_f32()?;
            let mut stock_window = vec![0.0; 3 * C];
            for ch in 0..C {
                for tap in 0..3 {
                    stock_window[tap * C + ch] = native_window[ch * 3 + tap];
                }
            }
            hash_is(
                &bf_bytes(&stock_window),
                expected_conv[step],
                "native conv history vs stock K1",
            )?;
            hash_is(
                raw.raw_bytes(),
                expected_y[case_index][step],
                "native raw Y vs stock K1",
            )?;
            hash_is(
                state.raw_bytes(),
                expected_state[case_index][step],
                "native F32 VK state vs stock K1",
            )?;
        }
    }
    Ok(())
}

#[test]
fn official_value_key_prefill_preserves_key_value_scan_bits_at_17_and_129() -> Result<()>
{
    let ctx = MetalContext::new()?;
    let case = fixture(true);
    let key_value = transpose_states(&case.state);
    let scale = 1.0 / 128.0f32.sqrt();
    for m in [17, 129] {
        let mut qkv = Vec::new();
        let mut a = Vec::new();
        let mut b = Vec::new();
        for t in 0..m {
            let (qt, at, bt) = token(t % 3);
            qkv.extend(qt);
            a.extend(at);
            b.extend(bt);
        }
        let qkv = Tensor::from_f32_as_bf16(&ctx, &qkv, &[m, C])?;
        let a = Tensor::from_f32_as_bf16(&ctx, &a, &[m, HV])?;
        let b = Tensor::from_f32_as_bf16(&ctx, &b, &[m, HV])?;
        let a_log = Tensor::from_f32(&ctx, &case.a_log, &[HV])?;
        let dt = Tensor::from_f32_as_bf16(&ctx, &case.dt, &[HV])?;
        let kv = Tensor::from_f32(&ctx, &key_value, &[HV, D, D])?;
        let vk = Tensor::from_f32(&ctx, &case.state, &[HV, D, D])?;
        let qk = Tensor::zeros(&ctx, &[m, 2 * HK * D], DType::BF16)?;
        let decay = Tensor::zeros(&ctx, &[m, HV], DType::F32)?;
        let beta = Tensor::zeros(&ctx, &[m, HV], DType::F32)?;
        let staging =
            GdnRegscanStaging { qk_norm: &qk, decay: &decay, beta: &beta, chunk: None };
        let kv_raw = Tensor::zeros(&ctx, &[m, HV, D], DType::BF16)?;
        let vk_raw = Tensor::zeros(&ctx, &[m, HV, D], DType::BF16)?;
        let kv_mid = Tensor::zeros(&ctx, &[3, HV, D, D], DType::F32)?;
        let vk_mid = Tensor::zeros(&ctx, &[3, HV, D, D], DType::F32)?;
        let pass = ctx.begin_concurrent()?;
        gdn_prefill_mid_with_rounding(
            &ctx,
            &pass,
            &qkv,
            &a,
            &b,
            &a_log,
            &dt,
            &staging,
            &kv,
            &kv_raw,
            scale,
            HK,
            Some(&kv_mid),
            true,
        )?;
        pass.level_barrier(&[&qk, &decay, &beta, &kv_raw, &kv, &kv_mid])?;
        gdn_prefill_mid_with_layout(
            &ctx,
            &pass,
            &qkv,
            &a,
            &b,
            &a_log,
            &dt,
            &staging,
            &vk,
            &vk_raw,
            scale,
            HK,
            Some(&vk_mid),
            true,
            GdnStateLayout::ValueKey,
        )?;
        pass.commit()?.wait()?;
        ensure!(
            kv_raw.raw_bytes() == vk_raw.raw_bytes(),
            "VK prefill readout changed arithmetic"
        );
        same_f32_bits(
            &kv.to_f32()?,
            &transpose_states(&vk.to_f32()?),
            "VK prefill final state transpose",
        )?;
        same_f32_bits(
            &kv_mid.to_f32()?,
            &transpose_states(&vk_mid.to_f32()?),
            "VK prefill mid state transpose",
        )?;
        // Continue the live VK prefill result through an actual new decode
        // step, independently seeded from the retained KV scan's transpose.
        let from_retained =
            Tensor::from_f32(&ctx, &transpose_states(&kv.to_f32()?), &[HV, D, D])?;
        let (qv, av, bv) = token(1);
        let step_q = Tensor::from_f32_as_bf16(&ctx, &qv, &[C])?;
        let step_a = Tensor::from_f32_as_bf16(&ctx, &av, &[HV])?;
        let step_b = Tensor::from_f32_as_bf16(&ctx, &bv, &[HV])?;
        let next_raw = Tensor::zeros(&ctx, &[HV, D], DType::BF16)?;
        let control_raw = Tensor::zeros(&ctx, &[HV, D], DType::BF16)?;
        let pass = ctx.begin_concurrent()?;
        for (state, out) in [(&vk, &next_raw), (&from_retained, &control_raw)] {
            gdn_step_raw_value_key_mlx(
                &ctx, &pass, &step_q, &step_a, &step_b, &a_log, &dt, state, out, scale,
                HK,
            )?;
        }
        pass.commit()?.wait()?;
        ensure!(
            next_raw.raw_bytes() == control_raw.raw_bytes()
                && vk.raw_bytes() == from_retained.raw_bytes(),
            "prefill-to-VK-decode continuation changed physical state"
        );
    }
    Ok(())
}

#[test]
fn official_value_key_mixed_policy_rows_keep_states_and_output_views_isolated()
-> Result<()> {
    let ctx = MetalContext::new()?;
    let case = fixture(true);
    let layouts = [GdnStateLayout::KeyValue, GdnStateLayout::ValueKey];
    let mut states = Vec::new();
    let mut controls = Vec::new();
    let mut windows = Vec::new();
    let mut control_windows = Vec::new();
    let mut qv = Vec::new();
    let mut av = Vec::new();
    let mut bv = Vec::new();
    for (r, layout) in layouts.into_iter().enumerate() {
        let initial = if layout == GdnStateLayout::ValueKey {
            case.state.clone()
        } else {
            transpose_states(&case.state)
        };
        states.push(Tensor::from_f32(&ctx, &initial, &[HV, D, D])?);
        controls.push(Tensor::from_f32(&ctx, &initial, &[HV, D, D])?);
        let history: Vec<f32> = native_history(&case.history)
            .iter()
            .map(|x| bf(*x + r as f32 / 8.0))
            .collect();
        windows.push(Tensor::from_f32_as_bf16(&ctx, &history, &[C, 3])?);
        control_windows.push(Tensor::from_f32_as_bf16(&ctx, &history, &[C, 3])?);
        let (q, a, _) = token(r);
        qv.extend(q);
        av.extend(a);
        bv.extend(vec![-6.84375; HV]);
    }
    let q = Tensor::from_f32_as_bf16(&ctx, &qv, &[2, C])?;
    let a = Tensor::from_f32_as_bf16(&ctx, &av, &[2, HV])?;
    let b = Tensor::from_f32_as_bf16(&ctx, &bv, &[2, HV])?;
    let a_log = Tensor::from_f32(&ctx, &case.a_log, &[HV])?;
    let dt = Tensor::from_f32_as_bf16(&ctx, &case.dt, &[HV])?;
    let z = Tensor::from_f32_as_bf16(&ctx, &vec![-6.84375; 2 * HV * D], &[2, HV * D])?;
    let gain = Tensor::from_f32(&ctx, &vec![1.25; D], &[D])?;
    let weights = Tensor::from_f32_as_bf16(
        &ctx,
        &(0..4 * C).map(|i| [0.25, -0.5, 0.375, 1.0][i / C]).collect::<Vec<_>>(),
        &[4, C],
    )?;
    let conv = Tensor::zeros(&ctx, &[2, C], DType::BF16)?;
    let control_conv = Tensor::zeros(&ctx, &[2, C], DType::BF16)?;
    let out_storage = Tensor::from_f32_as_bf16(
        &ctx,
        &vec![17.0; 4 + 2 * HV * D + 4],
        &[4 + 2 * HV * D + 4],
    )?;
    let raw_storage = Tensor::from_f32_as_bf16(
        &ctx,
        &vec![17.0; 4 + 2 * HV * D + 4],
        &[4 + 2 * HV * D + 4],
    )?;
    let controls_out = Tensor::zeros(&ctx, &[2, HV * D], DType::BF16)?;
    let control_raw = Tensor::zeros(&ctx, &[2, HV * D], DType::BF16)?;
    let pass = ctx.begin_concurrent()?;
    for r in 0..2 {
        conv1d_step_with_layout(
            &ctx,
            &pass,
            &windows[r],
            &q.view(r * C, &[C])?,
            &weights,
            &conv.view(r * C, &[C])?,
            true,
            layouts[r],
        )?;
    }
    pass.level_barrier(&[&conv])?;
    for r in 0..2 {
        gdn_step_gated_fused_with_layout(
            &ctx,
            &pass,
            &conv.view(r * C, &[C])?,
            &a.view(r * HV, &[HV])?,
            &b.view(r * HV, &[HV])?,
            &a_log,
            &dt,
            &states[r],
            &z.view(r * HV * D, &[HV, D])?,
            &gain,
            &out_storage.view(4 + r * HV * D, &[HV, D])?,
            1.0 / 128.0f32.sqrt(),
            HK,
            1e-6,
            GdnGate::Sigmoid,
            true,
            layouts[r],
            &raw_storage.view(4 + r * HV * D, &[HV, D])?,
        )?;
        // Independent policy references call the old KeyValue entry directly,
        // or the raw VK step + custom norm gate, rather than the dispatcher.
        if layouts[r] == GdnStateLayout::KeyValue {
            conv1d_step_with_rounding(
                &ctx,
                &pass,
                &control_windows[r],
                &q.view(r * C, &[C])?,
                &weights,
                &control_conv.view(r * C, &[C])?,
                true,
            )?;
        } else {
            conv1d_step_with_layout(
                &ctx,
                &pass,
                &control_windows[r],
                &q.view(r * C, &[C])?,
                &weights,
                &control_conv.view(r * C, &[C])?,
                true,
                GdnStateLayout::ValueKey,
            )?;
        }
        pass.level_barrier(&[&control_conv])?;
        if layouts[r] == GdnStateLayout::KeyValue {
            gdn_step_gated_fused_with_rounding(
                &ctx,
                &pass,
                &control_conv.view(r * C, &[C])?,
                &a.view(r * HV, &[HV])?,
                &b.view(r * HV, &[HV])?,
                &a_log,
                &dt,
                &controls[r],
                &z.view(r * HV * D, &[HV, D])?,
                &gain,
                &controls_out.view(r * HV * D, &[HV, D])?,
                1.0 / 128.0f32.sqrt(),
                HK,
                1e-6,
                GdnGate::Sigmoid,
                true,
            )?;
        } else {
            gdn_step_raw_value_key_mlx(
                &ctx,
                &pass,
                &control_conv.view(r * C, &[C])?,
                &a.view(r * HV, &[HV])?,
                &b.view(r * HV, &[HV])?,
                &a_log,
                &dt,
                &controls[r],
                &control_raw.view(r * HV * D, &[HV, D])?,
                1.0 / 128.0f32.sqrt(),
                HK,
            )?;
            pass.level_barrier(&[&control_raw, &controls[r]])?;
            gated_rmsnorm_value_key_decode(
                &ctx,
                &pass,
                &control_raw.view(r * HV * D, &[HV, D])?,
                &z.view(r * HV * D, &[HV, D])?,
                &gain,
                &controls_out.view(r * HV * D, &[HV, D])?,
                1e-6,
            )?;
        }
    }
    pass.commit()?.wait()?;
    for r in 0..2 {
        ensure!(
            states[r].raw_bytes() == controls[r].raw_bytes()
                && windows[r].raw_bytes() == control_windows[r].raw_bytes(),
            "mixed-policy row state/window mismatch"
        );
        ensure!(
            out_storage.view(4 + r * HV * D, &[HV, D])?.raw_bytes()
                == controls_out.view(r * HV * D, &[HV, D])?.raw_bytes(),
            "mixed-policy row output mismatch"
        );
    }
    let canary = bf_bytes(&[17.0; 4]);
    for t in [&out_storage, &raw_storage] {
        ensure!(
            t.raw_bytes()[..8] == canary && t.raw_bytes()[t.byte_len() - 8..] == canary,
            "mixed-policy row output canary overwritten"
        );
    }
    Ok(())
}
