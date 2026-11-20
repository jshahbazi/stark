// Modified by Stark: in-process core extraction; see UPSTREAM.md.
//! GPU token sampling: penalties, temperature, top-k / top-p / min-p, and the
//! draw itself, so pipelined decode never moves logits to the host.

use anyhow::{Result, ensure};

use crate::kernels::elementwise::{ARGMAX_GROUPS, argmax_f32};
use crate::kernels::{f32_bytes, u32_bytes};
use crate::metal::{ComputePass, Grid, MetalContext, MslVersion};
use crate::tensor::{DType, Tensor};

const SOURCE: &str = include_str!("metal/sample.metal");
const FINITE_SOURCE: &str = include_str!("metal/stark_finite_readout.metal");
const TG: usize = 1024;
const PREP_TG: usize = 256;
const PREP_GROUPS: usize = 64;
/// Largest candidate set the kernel considers; larger `top_k` values (and
/// "no top-k") are capped here.
pub const SAMPLE_K_CAP: usize = 1024;

/// One request's sampling configuration. `temperature == 0` means greedy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplingParams {
    pub temperature: f32,
    /// `0` disables top-k (subject to [`SAMPLE_K_CAP`]).
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub repetition_penalty: f32,
    pub seed: u64,
}

impl SamplingParams {
    pub const fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: 1,
            top_p: 1.0,
            min_p: 0.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repetition_penalty: 1.0,
            seed: 0,
        }
    }

    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0 || self.top_k == 1
    }

    pub fn uses_penalties(&self) -> bool {
        self.presence_penalty != 0.0
            || self.frequency_penalty != 0.0
            || self.repetition_penalty != 1.0
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.temperature.is_finite() && self.temperature >= 0.0,
            "temperature must be a finite number >= 0"
        );
        ensure!(
            self.top_p.is_finite() && self.top_p > 0.0 && self.top_p <= 1.0,
            "top_p must be in (0, 1]"
        );
        ensure!(
            self.min_p.is_finite() && (0.0..=1.0).contains(&self.min_p),
            "min_p must be in [0, 1]"
        );
        ensure!(
            self.repetition_penalty.is_finite() && self.repetition_penalty > 0.0,
            "repetition_penalty must be > 0"
        );
        ensure!(
            self.presence_penalty.is_finite() && self.frequency_penalty.is_finite(),
            "penalties must be finite"
        );
        Ok(())
    }

    /// The parameters as the kernel consumes them: greedy becomes an exact
    /// top-1 draw (penalties still apply), and the candidate cap is enforced.
    fn effective(&self) -> (f32, usize) {
        if self.is_greedy() {
            (1.0, 1)
        } else {
            let k = if self.top_k == 0 {
                SAMPLE_K_CAP
            } else {
                self.top_k.min(SAMPLE_K_CAP)
            };
            (self.temperature, k)
        }
    }
}

/// Threadgroups of the two-phase selection's first phase
/// (`LILY_SAMPLE_LOCAL_GROUPS` overrides it, for measurement) and the
/// largest candidate count it serves (the shader's SAMPLE_LOCAL_K_MAX).
const LOCAL_GROUPS: usize = 64;
const LOCAL_GROUPS_MAX: usize = 256;
const LOCAL_K_MAX: usize = 64;

fn local_groups() -> usize {
    LOCAL_GROUPS
}

/// Per-engine sampler buffers, sized for one vocabulary.
pub struct SamplerScratch {
    /// F32 `[V]`: penalised, temperature-scaled logits.
    adjusted: Tensor,
    /// U32 `[V]`: how often each id was emitted in the current request.
    counts: Tensor,
    /// F32 `[PREP_GROUPS]`: per-threadgroup maxima of the adjusted logits.
    maxima: Tensor,
    /// Partials for the exact argmax the greedy path uses.
    argmax_partials: Tensor,
    /// U32 `[LOCAL_GROUPS_MAX * LOCAL_K_MAX]`: the slices' top-k ids of the
    /// two-phase selection.
    local_ids: Tensor,
}

impl SamplerScratch {
    pub fn new(ctx: &MetalContext, vocab: usize) -> Result<Self> {
        Ok(Self {
            adjusted: Tensor::zeros(ctx, &[vocab], DType::F32)?,
            counts: Tensor::zeros(ctx, &[vocab], DType::U32)?,
            maxima: Tensor::zeros(ctx, &[PREP_GROUPS], DType::F32)?,
            argmax_partials: Tensor::zeros(ctx, &[2 * ARGMAX_GROUPS], DType::U32)?,
            local_ids: Tensor::zeros(
                ctx,
                &[LOCAL_GROUPS_MAX * LOCAL_K_MAX],
                DType::U32,
            )?,
        })
    }

    /// Undoes one count of `token` (a draw that was discarded). No committed
    /// pass may touch the counts (greedy draft passes never bind them).
    pub fn uncount(&self, token: u32) {
        let i = token as usize;
        if i >= self.counts.numel() {
            return;
        }
        // SAFETY: shared-storage buffer, GPU idle by contract, index in range.
        unsafe {
            let p = self.counts.contents_ptr().cast::<u32>().add(i);
            *p = (*p).saturating_sub(1);
        }
    }

    /// Clears the emission histogram at the start of a request. The GPU must
    /// be idle on this scratch.
    pub fn reset_counts(&self) {
        self.counts.zero_fill();
    }

    /// Takes over `other`'s emission histogram (a request moving between
    /// the engine's sampler and a batch slot's). The GPU must be idle on
    /// both scratches.
    pub fn copy_counts_from(&self, other: &SamplerScratch) -> Result<()> {
        ensure!(
            other.counts.numel() == self.counts.numel(),
            "sampler scratches for different vocabularies"
        );
        self.counts.write_bytes(other.counts.contents())
    }

    pub fn vocab(&self) -> usize {
        self.adjusted.numel()
    }
}

/// The distributions the draft head drew its proposals from, one slot per
/// draft: the kept candidate ids, their probabilities normalised over the
/// kept mass, and the kept count. Written by [`sample_draft_f32`] on the
/// GPU and read by [`sample_spec_f32`] in the verify pass that checks the
/// proposal, so nothing about them crosses to the host.
pub struct DraftDists {
    ids: Tensor,
    probs: Tensor,
    counts: Tensor,
}

/// One slot of a [`DraftDists`].
pub struct DraftDist {
    ids: Tensor,
    probs: Tensor,
    count: Tensor,
}

impl DraftDists {
    pub fn new(ctx: &MetalContext, slots: usize) -> Result<Self> {
        Ok(Self {
            ids: Tensor::zeros(ctx, &[slots * SAMPLE_K_CAP], DType::U32)?,
            probs: Tensor::zeros(ctx, &[slots * SAMPLE_K_CAP], DType::F32)?,
            counts: Tensor::zeros(ctx, &[slots], DType::U32)?,
        })
    }

    pub fn slot(&self, i: usize) -> Result<DraftDist> {
        Ok(DraftDist {
            ids: self.ids.view(i * SAMPLE_K_CAP, &[SAMPLE_K_CAP])?,
            probs: self.probs.view(i * SAMPLE_K_CAP, &[SAMPLE_K_CAP])?,
            count: self.counts.view(i, &[1])?,
        })
    }
}

/// The RNG index of the draft head's draw for the proposal that the
/// request's draw `step` will check: its own domain (high bit set), so it
/// never shares a uniform with the trunk's draws or the residual draws.
pub fn draft_step(step: usize) -> usize {
    step | 0x4000_0000
}

/// The sampler the draft head draws with: the request's temperature and
/// truncation, without the penalties (the proposals are not emissions and
/// the draft pass never binds the histogram).
pub fn draft_params(params: &SamplingParams) -> SamplingParams {
    SamplingParams {
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        repetition_penalty: 1.0,
        ..*params
    }
}

/// The sampler runs when the draw is not an exact argmax: sampling, or
/// greedy with penalties (a top-1 draw over the penalised logits).
fn samples(params: &SamplingParams) -> bool {
    !params.is_greedy() || params.uses_penalties()
}

/// What a draw needs beyond the plain kernel.
enum Draw<'a> {
    Plain,
    /// Export the kept distribution to the slot.
    Draft(&'a DraftDist),
    /// Speculative draw against the slot's distribution and its proposal.
    Spec(&'a DraftDist, &'a Tensor),
}

#[allow(clippy::too_many_arguments)]
fn encode_draw(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    logits: &Tensor,
    scratch: &SamplerScratch,
    params: &SamplingParams,
    step: usize,
    out: &Tensor,
    draw: Draw<'_>,
) -> Result<()> {
    let v = logits.numel();
    ensure!(v > 0 && logits.dtype() == DType::F32, "logits must be F32 [V]");
    ensure!(scratch.vocab() == v, "sampler scratch sized for a different vocabulary");
    ensure!(out.numel() == 1 && out.dtype() == DType::U32, "out must be U32 [1]");
    params.validate()?;
    let (temperature, top_k) = params.effective();
    let penalties = u32_bytes(usize::from(params.uses_penalties()));
    let prepare = ctx.pipeline("sample_prepare_f32", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &prepare,
        &[
            logits.binding(),
            scratch.adjusted.binding(),
            scratch.counts.binding(),
            scratch.maxima.binding(),
        ],
        &[
            &f32_bytes(temperature),
            &f32_bytes(params.presence_penalty),
            &f32_bytes(params.frequency_penalty),
            &f32_bytes(params.repetition_penalty),
            &u32_bytes(v),
            &penalties,
        ],
        Grid::Threadgroups {
            groups: (PREP_GROUPS, 1, 1),
            threadgroup: (PREP_TG, 1, 1),
        },
    )?;
    pass.level_barrier(&[&scratch.adjusted, &scratch.maxima])?;
    // Two phases for small k (the server's top-k 20): 64 threadgroups each
    // select their slice's top-k, and the draw selects over the union.
    let groups = local_groups();
    let two_phase = top_k <= LOCAL_K_MAX && v > 4 * groups * top_k && two_phase_on();
    let (select_v, mapped) = if two_phase {
        let local = ctx.pipeline("sample_local_topk_f32", SOURCE, MslVersion::V3_1)?;
        pass.dispatch_at(
            &local,
            &[
                scratch.adjusted.binding(),
                scratch.maxima.binding(),
                scratch.local_ids.binding(),
            ],
            &[&u32_bytes(top_k), &u32_bytes(v), &u32_bytes(groups)],
            Grid::Threadgroups { groups: (groups, 1, 1), threadgroup: (TG, 1, 1) },
        )?;
        pass.level_barrier(&[&scratch.local_ids])?;
        (groups * top_k, 1usize)
    } else {
        (v, 0usize)
    };
    let (name, mut buffers) = match &draw {
        Draw::Plain => ("sample_f32", Vec::new()),
        Draw::Draft(dist) => (
            "sample_draft_f32",
            vec![dist.ids.binding(), dist.probs.binding(), dist.count.binding()],
        ),
        Draw::Spec(dist, draft) => {
            ensure!(
                draft.numel() == 1 && draft.dtype() == DType::U32,
                "draft must be U32 [1]"
            );
            (
                "sample_spec_f32",
                vec![
                    dist.ids.binding(),
                    dist.probs.binding(),
                    dist.count.binding(),
                    draft.binding(),
                ],
            )
        }
    };
    let mut bindings = vec![
        scratch.adjusted.binding(),
        scratch.maxima.binding(),
        scratch.counts.binding(),
        out.binding(),
    ];
    bindings.append(&mut buffers);
    // The candidate map binds after the draw's own buffers and before the
    // constants (unused unless `mapped`).
    bindings.push(scratch.local_ids.binding());
    let select = ctx.pipeline(name, SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &select,
        &bindings,
        &[
            &f32_bytes(params.top_p),
            &f32_bytes(params.min_p),
            &u32_bytes(top_k),
            &u32_bytes(select_v),
            &u32_bytes((params.seed & 0xffff_ffff) as usize),
            &u32_bytes((params.seed >> 32) as usize),
            &u32_bytes(step),
            &penalties,
            &u32_bytes(mapped),
        ],
        Grid::Threadgroups { groups: (1, 1, 1), threadgroup: (TG, 1, 1) },
    )
}

/// `LILY_SAMPLE_TWO_PHASE=0` keeps every draw on the single-threadgroup
/// selection (for measurement).
fn two_phase_on() -> bool {
    true
}

/// Encodes one draw from `logits` (F32 `[V]`) into `out` (U32 `[1]`);
/// `step` indexes the request's draws for the RNG.
pub fn sample_f32(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    logits: &Tensor,
    scratch: &SamplerScratch,
    params: &SamplingParams,
    step: usize,
    out: &Tensor,
) -> Result<()> {
    if !samples(params) {
        // Exact and cheap: the multi-threadgroup argmax.
        argmax_f32(ctx, pass, logits, &scratch.argmax_partials, out)?;
    } else {
        encode_draw(ctx, pass, logits, scratch, params, step, out, Draw::Plain)?;
    }
    reject_nonfinite_readout_f32(ctx, pass, logits, out)
}

/// Keeps a draw private and invalidates its ID if any raw F32 logit is not
/// finite. Stark's caller rejects the sentinel before accepting a decision.
fn reject_nonfinite_readout_f32(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    logits: &Tensor,
    out: &Tensor,
) -> Result<()> {
    let count = logits.numel();
    ensure!(
        logits.dtype() == DType::F32 && count > 0 && count <= u32::MAX as usize,
        "finite readout validation requires non-empty F32 logits"
    );
    ensure!(
        out.dtype() == DType::U32 && out.numel() == 1,
        "finite readout validation requires one U32 output"
    );
    // The ordinary draw must finish before any validator can replace its ID.
    pass.level_barrier(&[logits, out])?;
    let kernel = ctx.pipeline(
        "stark_reject_nonfinite_readout_f32",
        FINITE_SOURCE,
        MslVersion::V3_1,
    )?;
    pass.dispatch_at(
        &kernel,
        &[logits.binding(), out.binding()],
        &[&u32_bytes(count)],
        Grid::Threads { grid: (count, 1, 1), threadgroup: (256.min(count), 1, 1) },
    )?;
    pass.level_barrier(&[out])
}

/// Status bit [`argmax_chain_f32`] sets when its selection must be redone on
/// the host: the selected logit is a zero or subnormal.
pub const CHAIN_ARGMAX_TINY: u32 = 1;

/// Stark-local greedy draw for a device draft chain: the first-max argmax of
/// `logits` into `out`, which a later dispatch may consume as an input id,
/// and a status word `flag` the host clears before commit. `out` always
/// stays in range, so a nonfinite readout cannot put a sentinel into an
/// embedding gather. Afterwards `flag` is 0 when `out` is exactly the host
/// rule's choice (the first ID no logit exceeds under IEEE `>`, signed zeros
/// equal), `u32::MAX` when the readout has a nonfinite value, and otherwise
/// [`CHAIN_ARGMAX_TINY`]: the selected logit is a zero or subnormal, where
/// device comparisons may flush subnormals and differ from the host.
pub fn argmax_chain_f32(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    logits: &Tensor,
    scratch: &SamplerScratch,
    out: &Tensor,
    flag: &Tensor,
) -> Result<()> {
    let count = logits.numel();
    ensure!(
        out.dtype() == DType::U32
            && out.numel() == 1
            && flag.dtype() == DType::U32
            && flag.numel() == 1,
        "chained argmax requires one U32 output and status word"
    );
    argmax_f32(ctx, pass, logits, &scratch.argmax_partials, out)?;
    pass.level_barrier(&[out])?;
    let kernel =
        ctx.pipeline("stark_flag_tiny_selection_f32", FINITE_SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &kernel,
        &[logits.binding(), out.binding(), flag.binding()],
        &[&u32_bytes(count)],
        Grid::Threads { grid: (1, 1, 1), threadgroup: (1, 1, 1) },
    )?;
    // Orders the status update before the nonfinite store, which wins.
    pass.level_barrier(&[flag])?;
    reject_nonfinite_readout_f32(ctx, pass, logits, flag)
}

/// The draft head's draw of one proposal: as [`sample_f32`], and under a
/// sampler that samples the kept distribution goes to `dist` for the
/// verify pass. Greedy proposals (an argmax) export nothing: the verify
/// pass compares them exactly.
#[allow(clippy::too_many_arguments)]
pub fn sample_draft_f32(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    logits: &Tensor,
    scratch: &SamplerScratch,
    params: &SamplingParams,
    step: usize,
    out: &Tensor,
    dist: &DraftDist,
) -> Result<()> {
    if !samples(params) {
        return argmax_f32(ctx, pass, logits, &scratch.argmax_partials, out);
    }
    encode_draw(ctx, pass, logits, scratch, params, step, out, Draw::Draft(dist))
}

/// The verify pass's draw for a row whose proposal `draft` (U32 `[1]`) the
/// draft head drew from `dist`: speculative sampling, which returns the
/// proposal exactly when it accepts it and otherwise a token from the
/// residual, distributed overall exactly as [`sample_f32`] would draw.
/// Under a greedy sampler this is the plain argmax (the proposal was one
/// too, and equality is the acceptance test).
#[allow(clippy::too_many_arguments)]
pub fn sample_spec_f32(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    logits: &Tensor,
    scratch: &SamplerScratch,
    params: &SamplingParams,
    step: usize,
    dist: &DraftDist,
    draft: &Tensor,
    out: &Tensor,
) -> Result<()> {
    if !samples(params) {
        return argmax_f32(ctx, pass, logits, &scratch.argmax_partials, out);
    }
    encode_draw(ctx, pass, logits, scratch, params, step, out, Draw::Spec(dist, draft))
}

/// The kernel's uniform draw for `(seed, step)`, for tests and replay.
pub fn uniform_for(seed: u64, step: u32) -> f32 {
    let mut z = seed.wrapping_add((step as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    ((z >> 40) as u32) as f32 * (1.0 / 16_777_216.0)
}

#[cfg(test)]
mod stark_finite_tests {
    use super::*;

    /// The host greedy rule: the first ID that no later logit exceeds under
    /// IEEE `>` (signed zeros equal), defined only for finite readouts.
    fn host_first_max(values: &[f32]) -> u32 {
        let mut best = 0;
        for index in 1..values.len() {
            if values[index] > values[best] {
                best = index;
            }
        }
        best as u32
    }

    #[test]
    #[ignore = "requires a physical Apple Silicon Metal device"]
    fn chained_argmax_matches_host_first_max_and_flags_nonfinite_in_range() -> Result<()>
    {
        let ctx = MetalContext::new()?;
        let mut word = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = move || {
            word ^= word << 13;
            word ^= word >> 7;
            word ^= word << 17;
            word
        };
        let mut checked = 0usize;
        let mut tiny_cases = 0usize;
        for count in [1usize, 2, 255, 256, 257, 4_097, 131_073, 248_320] {
            let scratch = SamplerScratch::new(&ctx, count)?;
            let logits = Tensor::zeros(&ctx, &[count], DType::F32)?;
            let words = Tensor::zeros(&ctx, &[2], DType::U32)?;
            let out = words.view(0, &[1])?;
            let flag = words.view(1, &[1])?;
            let mut run = |values: &[f32]| -> Result<(u32, u32)> {
                logits.write_bytes(bytemuck::cast_slice(values))?;
                words.write_bytes(bytemuck::cast_slice(&[0x1234_5678u32, 0]))?;
                let pass = ctx.begin_concurrent()?;
                argmax_chain_f32(&ctx, &pass, &logits, &scratch, &out, &flag)?;
                pass.commit_wait()?;
                let observed = words.to_u32()?;
                Ok((observed[0], observed[1]))
            };
            let mut cases: Vec<Vec<f32>> = Vec::new();
            // Positive, negative and tiny maxima, each with many exact ties,
            // signed zeros and subnormals of both signs among the logits.
            for offset in [1.0f32, -3.0, -1.0e6] {
                for levels in [1u64, 2, 3, 1 << 20] {
                    cases.push(
                        (0..count)
                            .map(|_| match next() % 9 {
                                0 => -0.0,
                                1 if offset < 0.0 => -f32::from_bits(1),
                                1 => 0.0,
                                2 if offset < -1.0 => -f32::MIN_POSITIVE,
                                2 => f32::from_bits(1),
                                _ => (next() % levels) as f32 / 7.0 + offset,
                            })
                            .collect(),
                    );
                }
            }
            // A subnormal maximum among positive and negative zeros.
            let mut tiny = vec![-0.0f32; count];
            tiny[count / 2] = f32::from_bits(3);
            tiny[count - 1] = f32::from_bits(3);
            cases.push(tiny);
            // Signed-zero maxima, with the negative zero first and last.
            let mut zeros = vec![-1.0f32; count];
            zeros[count - 1] = 0.0;
            zeros[count / 3] = -0.0;
            cases.push(zeros);
            cases.push(vec![-f32::MAX; count]);
            let mut late = vec![-f32::MAX; count];
            late[count - 1] = f32::MAX;
            late[count / 2] = f32::MAX;
            cases.push(late);
            for values in &cases {
                let (id, status) = run(values)?;
                let host = host_first_max(values);
                let tiny = values[host as usize].to_bits() & 0x7f80_0000 == 0;
                ensure!((id as usize) < count, "chained argmax output range");
                // Exact whenever the status is clear; flagged iff the host
                // maximum is a zero or subnormal.
                ensure!(
                    if tiny {
                        status == CHAIN_ARGMAX_TINY
                    } else {
                        status == 0 && id == host
                    },
                    "chained argmax tie-break"
                );
                tiny_cases += usize::from(tiny);
                checked += 1;
            }
            for bits in [0x7f80_0000u32, 0xff80_0000, 0x7fc0_0001, 0xffc0_0002] {
                for position in [0, count / 2, count - 1] {
                    let mut values = cases[0].clone();
                    values[position] = f32::from_bits(bits);
                    let (id, failed) = run(&values)?;
                    ensure!(
                        failed == u32::MAX && (id as usize) < count,
                        "chained argmax nonfinite guard"
                    );
                    checked += 1;
                }
            }
        }
        ensure!(checked > tiny_cases && tiny_cases > 0, "chained argmax coverage");
        Ok(())
    }

    #[test]
    #[ignore = "requires a physical Apple Silicon Metal device"]
    fn greedy_readout_rejects_nonfinite_values_in_full_vector_and_tails() -> Result<()>
    {
        let ctx = MetalContext::new()?;
        let nonfinite = [
            0x7f80_0000,
            0xff80_0000,
            0x7fc0_0001,
            0xffc0_0002,
            0x7f80_0001,
            0xff80_0001,
        ];
        for count in [1usize, 255, 256, 257, 131_073] {
            let scratch = SamplerScratch::new(&ctx, count)?;
            // Out-of-view nonfinite values must not poison a finite readout.
            let mut values = vec![f32::from_bits(0x7fc0_0001); count + 2];
            values[1..=count].fill(-f32::MAX);
            if count > 1 {
                values[1] = f32::from_bits(1); // Finite subnormal.
            }
            values[count] = f32::MAX;
            let storage = Tensor::from_f32(&ctx, &values, &[count + 2])?;
            let logits = storage.view(1, &[count])?;
            let tokens = Tensor::zeros(&ctx, &[3], DType::U32)?;
            let out = tokens.view(1, &[1])?;
            let canary = 0x1234_5678u32;
            let initial = [canary; 3];
            let run = |expected: u32| -> Result<()> {
                tokens.write_bytes(bytemuck::cast_slice(&initial))?;
                let pass = ctx.begin_concurrent()?;
                sample_f32(
                    &ctx,
                    &pass,
                    &logits,
                    &scratch,
                    &SamplingParams::greedy(),
                    0,
                    &out,
                )?;
                pass.commit_wait()?;
                let observed = tokens.to_u32()?;
                ensure!(
                    observed[0] == canary
                        && observed[1] == expected
                        && observed[2] == canary,
                    "finite readout guard result or output bounds mismatch"
                );
                Ok(())
            };
            run((count - 1) as u32)?;
            let mut positions = vec![0, count / 2, count - 1];
            positions.sort_unstable();
            positions.dedup();
            for position in positions {
                let saved = values[position + 1];
                for bits in nonfinite {
                    values[position + 1] = f32::from_bits(bits);
                    storage.write_bytes(bytemuck::cast_slice(&values))?;
                    run(u32::MAX)?;
                }
                values[position + 1] = saved;
                storage.write_bytes(bytemuck::cast_slice(&values))?;
                run((count - 1) as u32)?;
            }
            // Every finite IEEE exponent is accepted, including signed zeros
            // and subnormals that the model's floating kernels may flush.
            let finite: [u32; 6] =
                [0, 0x8000_0000, 1, 0x8000_0001, 0x7f7f_ffff, 0xff7f_ffff];
            for bits in finite {
                logits.write_bytes(bytemuck::cast_slice(&vec![bits; count]))?;
                out.write_bytes(bytemuck::cast_slice(&[canary]))?;
                let pass = ctx.begin_concurrent()?;
                reject_nonfinite_readout_f32(&ctx, &pass, &logits, &out)?;
                pass.commit_wait()?;
                ensure!(out.to_u32()?[0] == canary, "finite exponent was rejected");
            }
        }
        Ok(())
    }
}
