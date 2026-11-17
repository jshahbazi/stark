use sha2::{Digest, Sha256};
use stark_engine_macros::stark_test;

use super::*;

fn digest_logits(logits: &[f32]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for logit in logits {
        hasher.update(logit.to_bits().to_le_bytes());
    }
    hasher.finalize().into()
}

fn argmax(logits: &[f32]) -> u32 {
    logits.iter().enumerate().max_by(|(_, a), (_, b)| a.total_cmp(b)).unwrap().0 as u32
}

#[derive(Default, PartialEq)]
struct SamplerTrace {
    draft: Vec<[u8; 32]>,
    target: Vec<[u8; 32]>,
}

impl MtpSampler for SamplerTrace {
    fn begin_round(&mut self) {}

    fn draft(&mut self, logits: &[f32]) -> Result<u32> {
        self.draft.push(digest_logits(logits));
        Ok(argmax(logits))
    }

    fn target(&mut self, logits: &[f32], _proposal: Option<u32>) -> Result<u32> {
        self.target.push(digest_logits(logits));
        Ok(argmax(logits))
    }
}

struct Observation {
    prompt_logits: Vec<f32>,
    sampler: SamplerTrace,
    tokens: Vec<u32>,
    work: [usize; 4],
}

fn run(
    model: &mut LilyModel,
    input: &[u32],
    midpoint: bool,
    restore: Option<&PrefixSnapshot>,
    capture: Option<usize>,
) -> (Observation, Option<PrefixSnapshot>) {
    model.reset();
    model.prefix_midpoint = midpoint;
    model.set_mtp_enabled(true).unwrap();
    let start = restore.map_or(0, |snapshot| {
        model.restore_prefix(snapshot).unwrap();
        snapshot.position()
    });
    if let Some(position) = capture {
        model.capture_prefix_at(position).unwrap();
    }
    for &token in &input[start..input.len() - 1] {
        model.ingest(token).unwrap();
    }
    let prompt_logits = model.step(*input.last().unwrap()).unwrap();
    let mut sampler = SamplerTrace::default();
    let tokens = model.mtp_decode(argmax(&prompt_logits), 24, 2, &mut sampler).unwrap()
        .collect::<Result<Vec<_>>>().unwrap();
    let stats = model.mtp_stats();
    let work = [stats.rounds, stats.proposed, stats.accepted, stats.verified_rows];
    assert!(work[1] > 0 && work[3] == work[0] + work[1]);
    (Observation { prompt_logits, sampler, tokens, work }, model.take_prefix_snapshot())
}

fn compare(label: &str, input: usize, position: usize, a: &Observation, b: &Observation) -> bool {
    assert_eq!(a.prompt_logits.len(), b.prompt_logits.len());
    let unequal_logits = a.prompt_logits.iter().zip(&b.prompt_logits)
        .filter(|(x, y)| x.to_bits() != y.to_bits()).count();
    let drafts_match = a.sampler.draft == b.sampler.draft;
    let targets_match = a.sampler.target == b.sampler.target;
    let tokens_match = a.tokens == b.tokens;
    let work_matches = a.work == b.work;
    println!("{}", serde_json::json!({
        "scope": label, "input_tokens": input, "checkpoint": position,
        "vocab_logits": a.prompt_logits.len(), "unequal_prompt_logits": unequal_logits,
        "draft_readouts": a.sampler.draft.len(), "target_readouts": a.sampler.target.len(),
        "draft_logits_match": drafts_match, "target_logits_match": targets_match,
        "emitted_tokens_match": tokens_match, "mtp_work_matches": work_matches,
        "reference_mtp_work": a.work, "candidate_mtp_work": b.work,
    }));
    unequal_logits == 0 && drafts_match && targets_match && tokens_match && work_matches
}

// The fixture contains public-source coding tokens staged in an ignored
// private file. Print only result counts and equality verdicts.
#[stark_test]
#[ignore]
fn private_lily_midpoint_full_logits_and_mtp() {
    let path = std::env::var_os("STARK_LILY_TEST_MODEL_PATH").expect("private model fixture path");
    let fixture = std::env::var_os("STARK_LILY_MIDPOINT_FIXTURE").expect("private input fixture path");
    let input: Vec<Vec<u32>> = serde_json::from_reader(std::fs::File::open(fixture).unwrap()).unwrap();
    assert_eq!(input.iter().map(Vec::len).collect::<Vec<_>>(), [8192, 12288, 32768]);
    let mut model = LilyModel::load_with_options(Path::new(&path), true, true).unwrap();
    assert_eq!(model.prefill_chunk_tokens(), 4096);
    assert!(model.supports_images());
    model.enable_prefix_midpoint();
    assert!(!model.prefix_suffix_compatible(2048, 127));
    assert!(model.prefix_suffix_compatible(2048, 128));
    assert!(model.prefix_suffix_compatible(4096, 2));

    let mut exact = true;
    for (source, followed, checkpoint) in [
        (&input[0][..2205], &input[0][..2176], 2048),
        (&input[0][..2205], &input[0][..2341], 2048),
        (&input[0][..2205], &input[0][..4517], 2048),
        (input[0].as_slice(), input[0].as_slice(), 6144),
        (input[1].as_slice(), input[1].as_slice(), 10240),
        (input[2].as_slice(), input[2].as_slice(), 30720),
    ] {
        let (plain, _) = run(&mut model, source, false, None, None);
        // Fresh reference first, so no unrelated forward can overwrite the
        // captured prefix's live KV between capture and restore.
        let extended = (source.len() != followed.len())
            .then(|| run(&mut model, followed, false, None, None).0);
        let (captured, snapshot) = run(&mut model, source, true, None, Some(checkpoint));
        let snapshot = snapshot.expect("midpoint snapshot was not captured");
        assert_eq!(snapshot.position(), checkpoint);
        assert!(snapshot.mtp_enabled());
        exact &= compare("capture", source.len(), checkpoint, &plain, &captured);
        let (resumed, _) = run(&mut model, followed, true, Some(&snapshot), None);
        exact &= compare("resume", followed.len(), checkpoint, extended.as_ref().unwrap_or(&plain), &resumed);
    }
    assert!(exact, "midpoint failed exact full-model capture or resume");
}
