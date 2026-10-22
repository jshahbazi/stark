use std::{
    collections::HashSet,
    io::{BufRead, Read, Write},
    time::{Duration, Instant},
};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

pub(crate) const MAX_CONTEXT_TOKENS: u32 = 16_384;
pub(crate) const MAX_NEW_TOKENS: u32 = 4_096;
const MAX_REQUEST_BYTES: usize = 1_048_576;
const MAX_EXTRA_STOPS: usize = 256;

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum Sampling {
    Greedy {},
    Stochastic {
        temperature: f32,
        #[serde(default)]
        top_k: Option<u32>,
        #[serde(default)]
        top_p: Option<f32>,
        #[serde(default)]
        min_p: Option<f32>,
    },
}

impl Default for Sampling {
    fn default() -> Self {
        Self::Greedy {}
    }
}

impl Sampling {
    fn validate(
        &self,
        vocab_size: usize,
    ) -> Result<(), &'static str> {
        if let Self::Stochastic {
            temperature,
            top_k,
            top_p,
            min_p,
        } = self
            && (!temperature.is_finite()
                || *temperature <= 0.0
                || *temperature > 100.0
                || top_k.is_some_and(|k| k == 0 || k as usize > vocab_size)
                || top_p.is_some_and(|p| !p.is_finite() || p <= 0.0 || p > 1.0)
                || min_p.is_some_and(|p| !p.is_finite() || !(0.0..=1.0).contains(&p)))
        {
            return Err("invalid_sampling_request");
        }
        Ok(())
    }
}

#[derive(Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case", deny_unknown_fields)]
enum Request {
    Tokenize {
        text: String,
        #[serde(default)]
        add_special_tokens: bool,
    },
    Decode {
        token_ids: Vec<u32>,
        #[serde(default)]
        skip_special_tokens: bool,
    },
    Generate {
        #[serde(default)]
        rendered_prompt: Option<String>,
        #[serde(default)]
        input_token_ids: Option<Vec<u32>>,
        context_tokens: u32,
        max_new_tokens: u32,
        #[serde(default)]
        seed: u64,
        #[serde(default)]
        sampling: Sampling,
        #[serde(default)]
        stop_token_ids: Vec<u32>,
    },
}

pub(crate) struct GenerationOptions {
    pub context_tokens: u32,
    pub max_new_tokens: u32,
    pub seed: u64,
    pub sampling: Sampling,
    pub eos_token_ids: Vec<u32>,
    pub extra_stop_token_ids: Vec<u32>,
}

#[derive(Debug, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum FinishReason {
    Eos,
    StopToken,
    Length,
    Context,
}

#[derive(Debug, Serialize)]
pub(crate) struct GeneratedTokens {
    pub generated_token_ids: Vec<u32>,
    pub sampled_tokens: usize,
    pub finish_reason: FinishReason,
    pub stop_token_id: Option<u32>,
    pub prefill_seconds: Option<f64>,
    pub decode_seconds: Option<f64>,
}

fn tokens_per_second(
    tokens: usize,
    seconds: Option<f64>,
) -> Option<f64> {
    seconds.filter(|seconds| *seconds > 0.0 && seconds.is_finite()).and_then(|seconds| {
        let rate = tokens as f64 / seconds;
        (tokens > 0 && rate.is_finite()).then_some(rate)
    })
}

/// Own the stream so its Drop drains native work before the result is observable.
pub(crate) fn collect_tokens(
    stream: impl Iterator<Item = Result<u64, &'static str>>,
    input_tokens: usize,
    options: &GenerationOptions,
    vocab_size: usize,
    started: Instant,
) -> Result<GeneratedTokens, &'static str> {
    collect_tokens_with_clock(stream, input_tokens, options, vocab_size, || started.elapsed())
}

fn collect_tokens_with_clock(
    mut stream: impl Iterator<Item = Result<u64, &'static str>>,
    input_tokens: usize,
    options: &GenerationOptions,
    vocab_size: usize,
    mut elapsed: impl FnMut() -> Duration,
) -> Result<GeneratedTokens, &'static str> {
    let capacity = options.context_tokens as usize - input_tokens;
    let budget = capacity.min(options.max_new_tokens as usize);
    let mut result = GeneratedTokens {
        generated_token_ids: Vec::with_capacity(budget),
        sampled_tokens: 0,
        finish_reason: if capacity < options.max_new_tokens as usize {
            FinishReason::Context
        } else {
            FinishReason::Length
        },
        stop_token_id: None,
        prefill_seconds: None,
        decode_seconds: None,
    };
    let mut first_sample = None;
    let mut last_sample = None;
    for _ in 0..budget {
        let token = stream.next().ok_or("invalid_generation_output")??;
        let sample_time = elapsed();
        if token >= vocab_size as u64 {
            return Err("invalid_generation_output");
        }
        let token = token as u32;
        first_sample.get_or_insert(sample_time);
        last_sample = Some(sample_time);
        result.sampled_tokens += 1;
        if options.eos_token_ids.contains(&token) {
            result.finish_reason = FinishReason::Eos;
            result.stop_token_id = Some(token);
            break;
        }
        if options.extra_stop_token_ids.contains(&token) {
            result.finish_reason = FinishReason::StopToken;
            result.stop_token_id = Some(token);
            break;
        }
        result.generated_token_ids.push(token);
    }
    result.prefill_seconds = first_sample.map(|time| time.as_secs_f64());
    if result.sampled_tokens > 1 {
        result.decode_seconds =
            last_sample.zip(first_sample).map(|(last, first)| last.saturating_sub(first).as_secs_f64());
    }
    drop(stream);
    Ok(result)
}

pub(crate) trait GenerationBackend {
    fn vocab_size(&self) -> usize;
    fn max_context_length(&self) -> Option<u32>;
    fn eos_token_ids(&self) -> &[u32];
    fn tokenize(
        &mut self,
        text: &str,
        special: bool,
    ) -> Result<Vec<u32>, &'static str>;
    fn decode(
        &mut self,
        ids: &[u32],
        skip_special: bool,
    ) -> Result<String, &'static str>;
    fn generate(
        &mut self,
        input: &[u32],
        options: &GenerationOptions,
    ) -> Result<GeneratedTokens, &'static str>;
}

pub(crate) struct GenerationWorker<B> {
    backend: B,
}

fn valid_ids(
    ids: &[u32],
    vocab_size: usize,
) -> bool {
    ids.iter().all(|id| (*id as usize) < vocab_size)
}

fn write_reply(
    writer: &mut impl Write,
    value: &impl Serialize,
) -> Result<(), &'static str> {
    serde_json::to_writer(&mut *writer, value).map_err(|_| "protocol_write_failed")?;
    writer.write_all(b"\n").map_err(|_| "protocol_write_failed")?;
    writer.flush().map_err(|_| "protocol_write_failed")
}

impl<B: GenerationBackend> GenerationWorker<B> {
    pub(crate) fn new(backend: B) -> Result<Self, &'static str> {
        if backend.vocab_size() == 0
            || backend.vocab_size() > u32::MAX as usize
            || !valid_ids(backend.eos_token_ids(), backend.vocab_size())
        {
            return Err("invalid_model_config");
        }
        Ok(Self {
            backend,
        })
    }

    fn tokenize(
        &mut self,
        text: &str,
        special: bool,
    ) -> Result<Vec<u32>, &'static str> {
        let ids = self.backend.tokenize(text, special)?;
        if !valid_ids(&ids, self.backend.vocab_size()) {
            return Err("invalid_tokenization_output");
        }
        Ok(ids)
    }

    fn handle(
        &mut self,
        request: Request,
    ) -> Result<Value, &'static str> {
        match request {
            Request::Tokenize {
                text,
                add_special_tokens,
            } => {
                let ids = self.tokenize(&text, add_special_tokens)?;
                Ok(json!({"token_ids":ids,"token_count":ids.len()}))
            },
            Request::Decode {
                token_ids,
                skip_special_tokens,
            } => {
                if !valid_ids(&token_ids, self.backend.vocab_size()) {
                    return Err("invalid_decoding_request");
                }
                let text = self.backend.decode(&token_ids, skip_special_tokens)?;
                Ok(json!({"text":text,"token_count":token_ids.len()}))
            },
            Request::Generate {
                rendered_prompt,
                input_token_ids,
                context_tokens,
                max_new_tokens,
                seed,
                sampling,
                stop_token_ids,
            } => {
                if context_tokens == 0
                    || context_tokens > MAX_CONTEXT_TOKENS
                    || max_new_tokens == 0
                    || max_new_tokens > MAX_NEW_TOKENS
                    || stop_token_ids.len() > MAX_EXTRA_STOPS
                    || !valid_ids(&stop_token_ids, self.backend.vocab_size())
                    || stop_token_ids.iter().collect::<HashSet<_>>().len() != stop_token_ids.len()
                {
                    return Err("invalid_generation_request");
                }
                if self.backend.max_context_length().is_some_and(|max| context_tokens > max) {
                    return Err("context_overflow");
                }
                sampling.validate(self.backend.vocab_size())?;
                let input = match (rendered_prompt, input_token_ids) {
                    (Some(prompt), None) => self.tokenize(&prompt, false)?,
                    (None, Some(ids)) => ids,
                    _ => return Err("invalid_generation_request"),
                };
                if input.is_empty() || !valid_ids(&input, self.backend.vocab_size()) {
                    return Err("invalid_generation_request");
                }
                if input.len() >= context_tokens as usize {
                    return Err("context_overflow");
                }
                let options = GenerationOptions {
                    context_tokens,
                    max_new_tokens,
                    seed,
                    sampling,
                    eos_token_ids: self.backend.eos_token_ids().to_vec(),
                    extra_stop_token_ids: stop_token_ids,
                };
                let started = Instant::now();
                let generated = self.backend.generate(&input, &options)?;
                if !valid_ids(&generated.generated_token_ids, self.backend.vocab_size())
                    || generated.sampled_tokens > options.max_new_tokens as usize
                    || input.len() + generated.sampled_tokens > context_tokens as usize
                    || generated.sampled_tokens
                        != generated.generated_token_ids.len() + usize::from(generated.stop_token_id.is_some())
                {
                    return Err("invalid_generation_output");
                }
                // Decode the complete vector, preserving UTF-8 boundaries and specials.
                let text = self.backend.decode(&generated.generated_token_ids, false)?;
                Ok(json!({"generated_token_ids":generated.generated_token_ids,"text":text,
                    "input_tokens":input.len(),"output_tokens":generated.generated_token_ids.len(),
                    "sampled_tokens":generated.sampled_tokens,"finish_reason":generated.finish_reason,
                    "stop_token_id":generated.stop_token_id,"generation_seconds":started.elapsed().as_secs_f64(),
                    "prefill_seconds":generated.prefill_seconds,"decode_seconds":generated.decode_seconds,
                    "input_tokens_per_second":tokens_per_second(input.len(), generated.prefill_seconds),
                    "output_tokens_per_second":tokens_per_second(generated.sampled_tokens.saturating_sub(1), generated.decode_seconds),
                    "timing_method":"first-token-wall-clock-v1"}))
            },
        }
    }

    fn process_line(
        &mut self,
        bytes: &[u8],
        output: &mut impl Write,
    ) -> Result<(), &'static str> {
        let result = serde_json::from_slice::<Request>(bytes)
            .map_err(|_| "invalid_request")
            .and_then(|request| self.handle(request));
        match result {
            Ok(value) => write_reply(output, &value),
            Err(code) => {
                write_reply(output, &json!({"error":code}))?;
                if matches!(
                    code,
                    "native_generation_failed"
                        | "invalid_generation_output"
                        | "invalid_tokenization_output"
                        | "decoding_failed"
                ) {
                    Err("native_operation_failed")
                } else {
                    Ok(())
                }
            },
        }
    }
}

pub(crate) fn run_lines<B: GenerationBackend>(
    worker: &mut GenerationWorker<B>,
    input: &mut impl BufRead,
    output: &mut impl Write,
) -> Result<(), &'static str> {
    write_reply(
        output,
        &json!({"ready":true,"engine":"stark","version":env!("CARGO_PKG_VERSION"),
        "mode":"text-generation-v1","operations":["tokenize","decode","generate"],
        "state_policy":"fresh-per-request","vocab_size":worker.backend.vocab_size(),
        "max_context_tokens":MAX_CONTEXT_TOKENS.min(worker.backend.max_context_length().unwrap_or(MAX_CONTEXT_TOKENS)),
        "max_new_tokens":MAX_NEW_TOKENS,"eos_token_ids":worker.backend.eos_token_ids(),
        "sampling_modes":["greedy","stochastic"]}),
    )?;
    loop {
        let mut bytes = Vec::new();
        let count = Read::by_ref(input)
            .take(MAX_REQUEST_BYTES as u64 + 1)
            .read_until(b'\n', &mut bytes)
            .map_err(|_| "protocol_read_failed")?;
        if count == 0 {
            return Ok(());
        }
        if bytes.len() > MAX_REQUEST_BYTES {
            // Drain the rest of this frame so the next request remains independent.
            if bytes.last() != Some(&b'\n') {
                loop {
                    let chunk = input.fill_buf().map_err(|_| "protocol_read_failed")?;
                    if chunk.is_empty() {
                        break;
                    }
                    let newline = chunk.iter().position(|byte| *byte == b'\n');
                    let count = newline.map_or(chunk.len(), |position| position + 1);
                    input.consume(count);
                    if newline.is_some() {
                        break;
                    }
                }
            }
            write_reply(output, &json!({"error":"request_too_large"}))?;
            continue;
        }
        worker.process_line(&bytes, output)?;
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::VecDeque,
        io,
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        },
    };

    use super::*;

    struct TestStream {
        tokens: VecDeque<Result<u64, &'static str>>,
        drained: Arc<AtomicBool>,
    }
    impl Iterator for TestStream {
        type Item = Result<u64, &'static str>;
        fn next(&mut self) -> Option<Self::Item> {
            self.tokens.pop_front()
        }
    }
    impl Drop for TestStream {
        fn drop(&mut self) {
            self.drained.store(true, Ordering::SeqCst);
        }
    }

    struct FakeBackend {
        tokens: Vec<Result<u64, &'static str>>,
        eos: Vec<u32>,
        generation_calls: usize,
        inputs: Vec<Vec<u32>>,
        seeds: Vec<u64>,
        sampling: Vec<Sampling>,
        drained: Arc<AtomicBool>,
        invalid_tokenization: bool,
    }
    impl GenerationBackend for FakeBackend {
        fn vocab_size(&self) -> usize {
            8
        }
        fn max_context_length(&self) -> Option<u32> {
            Some(1024)
        }
        fn eos_token_ids(&self) -> &[u32] {
            &self.eos
        }
        fn tokenize(
            &mut self,
            _: &str,
            special: bool,
        ) -> Result<Vec<u32>, &'static str> {
            Ok(if self.invalid_tokenization {
                vec![8]
            } else if special {
                vec![0, 2]
            } else {
                vec![2]
            })
        }
        fn decode(
            &mut self,
            ids: &[u32],
            skip_special: bool,
        ) -> Result<String, &'static str> {
            assert!(self.drained.load(Ordering::SeqCst));
            Ok(ids
                .iter()
                .map(|id| match id {
                    3 => "ré",
                    4 => "sultat\n",
                    1 if !skip_special => "<eos>",
                    7 if !skip_special => "<tool_call>",
                    _ => "",
                })
                .collect())
        }
        fn generate(
            &mut self,
            input: &[u32],
            options: &GenerationOptions,
        ) -> Result<GeneratedTokens, &'static str> {
            self.generation_calls += 1;
            self.inputs.push(input.to_vec());
            self.seeds.push(options.seed);
            self.sampling.push(options.sampling.clone());
            self.drained.store(false, Ordering::SeqCst);
            collect_tokens(
                TestStream {
                    tokens: self.tokens.clone().into(),
                    drained: self.drained.clone(),
                },
                input.len(),
                options,
                self.vocab_size(),
                Instant::now(),
            )
        }
    }
    fn worker() -> GenerationWorker<FakeBackend> {
        GenerationWorker::new(FakeBackend {
            tokens: vec![Ok(3), Ok(4), Ok(1), Ok(7)],
            eos: vec![1],
            generation_calls: 0,
            inputs: Vec::new(),
            seeds: Vec::new(),
            sampling: Vec::new(),
            drained: Arc::new(AtomicBool::new(true)),
            invalid_tokenization: false,
        })
        .unwrap()
    }
    fn standard() -> Value {
        json!({"operation":"generate","input_token_ids":[2],"context_tokens":16,"max_new_tokens":8})
    }
    fn post(
        worker: &mut GenerationWorker<FakeBackend>,
        request: Value,
    ) -> Value {
        let mut output = Vec::new();
        worker.process_line(&serde_json::to_vec(&request).unwrap(), &mut output).unwrap();
        serde_json::from_slice(&output).unwrap()
    }

    #[test]
    fn stage_timing_separates_first_sample_from_subsequent_samples_and_cleanup() {
        let options = GenerationOptions {
            context_tokens: 16,
            max_new_tokens: 8,
            seed: 0,
            sampling: Sampling::Greedy {},
            eos_token_ids: vec![1],
            extra_stop_token_ids: Vec::new(),
        };
        let drained = Arc::new(AtomicBool::new(false));
        let mut clock = [Duration::from_secs(2), Duration::from_secs(3), Duration::from_secs(6)].into_iter();
        let result = collect_tokens_with_clock(
            TestStream {
                tokens: vec![Ok(3), Ok(4), Ok(1), Ok(7)].into(),
                drained: drained.clone(),
            },
            4,
            &options,
            8,
            || clock.next().unwrap(),
        )
        .unwrap();
        assert_eq!(result.prefill_seconds, Some(2.0));
        assert_eq!(result.decode_seconds, Some(4.0));
        assert_eq!(result.sampled_tokens, 3);
        assert_eq!(result.generated_token_ids.len(), 2);
        assert_eq!(tokens_per_second(4, result.prefill_seconds), Some(2.0));
        // The first sample belongs to prefill; terminal EOS is a decode sample.
        assert_eq!(tokens_per_second(result.sampled_tokens - 1, result.decode_seconds), Some(0.5));
        assert!(drained.load(Ordering::SeqCst));
        assert!(clock.next().is_none());
    }

    #[test]
    fn one_sample_and_unmeasurable_durations_do_not_invent_decode_throughput() {
        let mut worker = worker();
        worker.backend.tokens = vec![Ok(1)];
        let value = post(&mut worker, standard());
        assert_eq!(value["timing_method"], "first-token-wall-clock-v1");
        assert!(value["prefill_seconds"].is_number());
        assert!(value["decode_seconds"].is_null());
        assert!(value["output_tokens_per_second"].is_null());
        for seconds in [None, Some(0.0), Some(-1.0), Some(f64::NAN), Some(f64::INFINITY)] {
            assert_eq!(tokens_per_second(2, seconds), None);
        }
        assert_eq!(tokens_per_second(0, Some(1.0)), None);
    }

    #[test]
    fn stops_exclude_terminal_token_and_preserve_exact_text_after_cleanup() {
        let mut worker = worker();
        let value = post(&mut worker, standard());
        assert_eq!(value["generated_token_ids"], json!([3, 4]));
        assert_eq!(value["text"], "résultat\n");
        assert_eq!(value["finish_reason"], "eos");
        assert_eq!(value["stop_token_id"], 1);
        assert_eq!(value["output_tokens"], 2);
        assert_eq!(value["sampled_tokens"], 3);
        assert!(worker.backend.drained.load(Ordering::SeqCst));
        let mut request = standard();
        request["stop_token_ids"] = json!([4]);
        let value = post(&mut worker, request);
        assert_eq!(value["generated_token_ids"], json!([3]));
        assert_eq!(value["finish_reason"], "stop_token");
        assert_eq!(value["stop_token_id"], 4);
        assert_eq!(value["sampled_tokens"], 2);
    }

    #[test]
    fn model_eos_cannot_be_overridden_and_eos_at_first_token_is_empty_text() {
        let mut worker = worker();
        worker.backend.tokens = vec![Ok(1), Ok(3)];
        let mut request = standard();
        request["stop_token_ids"] = json!([1, 4]);
        let value = post(&mut worker, request);
        assert_eq!(value["generated_token_ids"], json!([]));
        assert_eq!(value["text"], "");
        assert_eq!(value["finish_reason"], "eos");
        assert_eq!(value["sampled_tokens"], 1);
    }

    #[test]
    fn length_and_context_limits_never_consume_an_extra_output_token() {
        let mut worker = worker();
        for (context, maximum, reason) in [(16, 1, "length"), (2, 8, "context"), (2, 1, "length")] {
            let mut request = standard();
            request["context_tokens"] = json!(context);
            request["max_new_tokens"] = json!(maximum);
            let value = post(&mut worker, request);
            assert_eq!(value["generated_token_ids"], json!([3]));
            assert_eq!(value["sampled_tokens"], 1);
            assert_eq!(value["finish_reason"], reason);
            assert!(worker.backend.drained.load(Ordering::SeqCst));
        }
    }

    #[test]
    fn each_generation_dispatches_fresh_input_seed_and_sampling() {
        let mut worker = worker();
        let mut request = standard();
        request["seed"] = json!(41);
        request["sampling"] = json!({"mode":"stochastic","temperature":0.6,"top_k":4,"top_p":0.95});
        let first = post(&mut worker, request.clone());
        let second = post(&mut worker, request);
        assert_eq!(first["generated_token_ids"], second["generated_token_ids"]);
        assert_eq!(worker.backend.generation_calls, 2);
        assert_eq!(worker.backend.inputs, vec![vec![2], vec![2]]);
        assert_eq!(worker.backend.seeds, vec![41, 41]);
        assert_eq!(worker.backend.sampling[0], worker.backend.sampling[1]);
        let prompt =
            json!({"operation":"generate","rendered_prompt":"rendered text", "context_tokens":16,"max_new_tokens":8});
        post(&mut worker, prompt);
        assert_eq!(worker.backend.inputs.last(), Some(&vec![2]));
        assert_eq!(worker.backend.seeds.last(), Some(&0));
        assert_eq!(worker.backend.sampling.last(), Some(&Sampling::Greedy {}));
    }

    #[test]
    fn bad_request_does_not_dispatch_native_generation_or_poison_following_request() {
        let mut worker = worker();
        let cases = [
            json!({"context_tokens":0}),
            json!({"context_tokens":16385}),
            json!({"context_tokens":1025}),
            json!({"max_new_tokens":0}),
            json!({"max_new_tokens":4097}),
            json!({"input_token_ids":[]}),
            json!({"input_token_ids":[8]}),
            json!({"input_token_ids":[2,3],"context_tokens":2}),
            json!({"stop_token_ids":[8]}),
            json!({"stop_token_ids":[1,1]}),
            json!({"rendered_prompt":"both sources"}),
            json!({"input_token_ids":null}),
            json!({"sampling":{"mode":"stochastic","temperature":0.0}}),
            json!({"sampling":{"mode":"stochastic","temperature":0.6,"top_k":0}}),
            json!({"sampling":{"mode":"stochastic","temperature":0.6,"top_k":9}}),
            json!({"sampling":{"mode":"stochastic","temperature":0.6,"top_p":0.0}}),
            json!({"sampling":{"mode":"stochastic","temperature":0.6,"top_p":1.1}}),
            json!({"sampling":{"mode":"stochastic","temperature":0.6,"min_p":-0.1}}),
            json!({"sampling":{"mode":"stochastic","temperature":0.6,"min_p":1.1}}),
        ];
        for case in cases {
            let mut request = standard();
            request.as_object_mut().unwrap().extend(case.as_object().unwrap().clone());
            assert!(post(&mut worker, request).get("error").is_some());
        }
        assert_eq!(worker.backend.generation_calls, 0);
        assert_eq!(post(&mut worker, standard())["finish_reason"], "eos");
        assert_eq!(worker.backend.generation_calls, 1);
    }

    #[test]
    fn unknown_fields_and_wrong_types_are_rejected_at_every_level() {
        for patch in [
            json!({"private":"content"}),
            json!({"input_token_ids":[-1]}),
            json!({"input_token_ids":[1.5]}),
            json!({"seed":-1}),
            json!({"sampling":{"mode":"greedy","temperature":0.6}}),
            json!({"sampling":{"mode":"stochastic","temperature":0.6,"repetition_penalty":1.1}}),
        ] {
            let mut value = standard();
            value.as_object_mut().unwrap().extend(patch.as_object().unwrap().clone());
            assert!(serde_json::from_value::<Request>(value).is_err());
        }
        let request: Request = serde_json::from_value(standard()).unwrap();
        assert!(matches!(
            request,
            Request::Generate {
                seed: 0,
                sampling: Sampling::Greedy {},
                ..
            }
        ));
        assert!(
            Sampling::Stochastic {
                temperature: f32::NAN,
                top_k: None,
                top_p: None,
                min_p: None
            }
            .validate(8)
            .is_err()
        );
        assert!(
            Sampling::Stochastic {
                temperature: 0.6,
                top_k: None,
                top_p: Some(f32::INFINITY),
                min_p: None
            }
            .validate(8)
            .is_err()
        );
    }

    #[test]
    fn native_errors_invalid_ids_and_early_exhaustion_drain_then_fence_worker() {
        for tokens in [vec![Err("native_generation_failed")], vec![Ok(8)], vec![]] {
            let mut worker = worker();
            worker.backend.tokens = tokens;
            let mut output = Vec::new();
            assert_eq!(
                worker.process_line(&serde_json::to_vec(&standard()).unwrap(), &mut output),
                Err("native_operation_failed")
            );
            assert!(serde_json::from_slice::<Value>(&output).unwrap().get("error").is_some());
            assert!(worker.backend.drained.load(Ordering::SeqCst));
        }
        let mut worker = worker();
        worker.backend.invalid_tokenization = true;
        assert_eq!(
            worker.process_line(br#"{"operation":"tokenize","text":"secret"}"#, &mut Vec::new()),
            Err("native_operation_failed")
        );
    }

    #[test]
    fn framed_protocol_recovers_from_malformed_and_oversized_lines() {
        let mut worker = worker();
        let mut bytes = b"{\n".to_vec();
        bytes.extend(vec![b'x'; MAX_REQUEST_BYTES + 30]);
        bytes.extend(b"\n");
        bytes.extend(serde_json::to_vec(&standard()).unwrap());
        bytes.extend(b"\n");
        let mut output = Vec::new();
        run_lines(&mut worker, &mut io::Cursor::new(bytes), &mut output).unwrap();
        let replies: Vec<Value> = output
            .split(|byte| *byte == b'\n')
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_slice(line).unwrap())
            .collect();
        assert_eq!(replies.len(), 4);
        assert_eq!(replies[0]["ready"], true);
        assert_eq!(replies[1], json!({"error":"invalid_request"}));
        assert_eq!(replies[2], json!({"error":"request_too_large"}));
        assert_eq!(replies[3]["finish_reason"], "eos");
        assert_eq!(worker.backend.generation_calls, 1);
    }

    #[test]
    fn tokenize_and_decode_helpers_preserve_specials_and_support_empty_vectors() {
        let mut worker = worker();
        assert_eq!(post(&mut worker, json!({"operation":"tokenize","text":""}))["token_ids"], json!([2]));
        assert_eq!(
            post(&mut worker, json!({"operation":"tokenize","text":"x","add_special_tokens":true}))["token_ids"],
            json!([0, 2])
        );
        assert_eq!(
            post(&mut worker, json!({"operation":"decode","token_ids":[7,3,4,1]}))["text"],
            "<tool_call>résultat\n<eos>"
        );
        assert_eq!(
            post(&mut worker, json!({"operation":"decode","token_ids":[7,3,4,1],"skip_special_tokens":true}))["text"],
            "résultat\n"
        );
        assert_eq!(post(&mut worker, json!({"operation":"decode","token_ids":[]}))["text"], "");
        assert_eq!(
            post(&mut worker, json!({"operation":"decode","token_ids":[8]})),
            json!({"error":"invalid_decoding_request"})
        );
    }

    struct FailingWriter {
        fail_flush: bool,
    }
    impl Write for FailingWriter {
        fn write(
            &mut self,
            bytes: &[u8],
        ) -> io::Result<usize> {
            if self.fail_flush {
                Ok(bytes.len())
            } else {
                Err(io::Error::other("synthetic failure"))
            }
        }
        fn flush(&mut self) -> io::Result<()> {
            Err(io::Error::other("synthetic failure"))
        }
    }
    #[test]
    fn output_write_and_flush_failures_are_fatal_after_stream_cleanup() {
        for fail_flush in [false, true] {
            let mut worker = worker();
            assert_eq!(
                worker.process_line(
                    &serde_json::to_vec(&standard()).unwrap(),
                    &mut FailingWriter {
                        fail_flush
                    }
                ),
                Err("protocol_write_failed")
            );
            assert!(worker.backend.drained.load(Ordering::SeqCst));
        }
    }
}
