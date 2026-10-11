use libtest_mimic::{Arguments, Trial};

const METAL_CAPTURE_ENABLED: &str = "METAL_CAPTURE_ENABLED";
const STARK_CAPTURE_BENCH: &str = "STARK_CAPTURE_BENCH";

pub enum StarkTest {
    Bench(&'static dyn Fn()),
    Test(&'static StarkTestCase),
}

pub struct StarkTestCase {
    pub name: &'static str,
    pub ignore: bool,
    pub run: fn(),
}

#[cfg(target_os = "ios")]
fn ios_set_current_dir() {
    use objc2_foundation::{NSSearchPathDirectory, NSSearchPathDomainMask, NSSearchPathForDirectoriesInDomains};

    let paths = NSSearchPathForDirectoriesInDomains(NSSearchPathDirectory(9), NSSearchPathDomainMask(1), true);
    if let Some(docs) = paths.firstObject() {
        let _ = std::env::set_current_dir(docs.to_string());
    }
}

pub fn stark_harness(tests: &[&StarkTest]) {
    let args = std::env::args().collect::<Vec<String>>();
    let benchmarks = args.contains(&"--bench".to_string());
    if benchmarks {
        #[cfg(target_os = "ios")]
        ios_set_current_dir();
        enable_benchmark_gpu_capture_if_requested();
        let bench_tests: Vec<&dyn Fn()> = tests
            .iter()
            .filter_map(|test| match test {
                StarkTest::Bench(test) => Some(*test),
                StarkTest::Test(_) => None,
            })
            .collect::<Vec<_>>();
        criterion::runner(bench_tests.as_slice());
    } else {
        let trials: Vec<Trial> = tests
            .iter()
            .filter_map(|test| match **test {
                StarkTest::Bench(_) => None,
                StarkTest::Test(case) => Some(
                    Trial::test(case.name, move || {
                        (case.run)();
                        Ok(())
                    })
                    .with_ignored_flag(case.ignore),
                ),
            })
            .collect();
        libtest_mimic::run(&Arguments::from_args(), trials).exit();
    }
}

fn enable_benchmark_gpu_capture_if_requested() {
    if enabled(STARK_CAPTURE_BENCH) {
        unsafe {
            std::env::set_var(METAL_CAPTURE_ENABLED, "1");
        }
    }
}

fn enabled(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("yes") || v.eq_ignore_ascii_case("true"))
}
