//! Automatic content and transport admission, independent of model token context.

pub const DEFAULT_MIB: usize = 16;
pub const MIN_MIB: usize = 1;
const MIB: usize = 1024 * 1024;
// Keep every derived byte allowance representable as an allocation size. The
// cumulative template allowance is the largest formula, 32 MiB + 32B. This is
// derived from the allocation address space.
pub const MAX_MIB: usize = (isize::MAX as usize - 32 * MIB) / (32 * MIB);
pub const ENVIRONMENT_VARIABLE: &str = "STARK_REQUEST_BUDGET_MIB";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Budget {
    pub mib: usize,
    pub input_bytes: usize,
    pub wire_bytes: usize,
    pub template_live_bytes: usize,
    pub template_cumulative_bytes: usize,
}

impl Budget {
    pub const DEFAULT: Self = Self::build(DEFAULT_MIB);

    const fn build(mib: usize) -> Self {
        let input_bytes = mib.checked_mul(MIB).expect("budget input overflow");
        Self {
            mib,
            input_bytes,
            // A JSON control byte can expand to six ASCII bytes. Reserve
            // additional space for template source, keys and protocol framing.
            wire_bytes: input_bytes
                .checked_mul(6)
                .expect("budget wire overflow")
                .checked_add(2 * MIB)
                .expect("budget wire overflow"),
            template_live_bytes: input_bytes
                .checked_mul(8)
                .expect("budget live overflow")
                .checked_add(128 * MIB)
                .expect("budget live overflow"),
            template_cumulative_bytes: input_bytes
                .checked_mul(32)
                .expect("budget cumulative overflow")
                .checked_add(32 * MIB)
                .expect("budget cumulative overflow"),
        }
    }

    pub fn for_mib(mib: usize) -> Result<Self, &'static str> {
        if !(MIN_MIB..=MAX_MIB).contains(&mib) {
            return Err("invalid_resource_budget");
        }
        Ok(Self::build(mib))
    }

    pub fn parse(value: &str) -> Result<Self, &'static str> {
        if value.is_empty()
            || value.len() > MAX_MIB.ilog10() as usize + 1
            || !value.bytes().all(|byte| byte.is_ascii_digit())
        {
            return Err("invalid_resource_budget");
        }
        Self::for_mib(value.parse().map_err(|_| "invalid_resource_budget")?)
    }

    pub fn automatic(available_bytes: usize) -> Self {
        Self::build((available_bytes / (128 * MIB)).clamp(DEFAULT_MIB, MAX_MIB))
    }

    /// Read once at process startup; callers must not mutate this environment
    /// setting while a loaded worker is running.
    pub fn from_environment() -> Result<Self, &'static str> {
        match std::env::var(ENVIRONMENT_VARIABLE) {
            Ok(value) => Self::parse(&value),
            Err(std::env::VarError::NotPresent) => Ok(Self::automatic(available_memory_bytes().unwrap_or(0))),
            Err(_) => Err("invalid_resource_budget"),
        }
    }
}

/// Bound JSON container expansion before Serde constructs its tagged request.
/// Numeric leaves have separate headroom for full-context token vectors. This
/// is an allocation admission scan; the request parser still validates syntax.
pub fn check_json_structure(
    data: &[u8],
    budget: Budget,
) -> Result<(), &'static str> {
    let mut offset = 0;
    let mut depth = 0usize;
    let mut nodes = 0usize;
    let mut numeric = 0usize;
    while let Some(&byte) = data.get(offset) {
        offset += 1;
        match byte {
            b' ' | b'\n' | b'\r' | b'\t' | b',' | b':' => continue,
            b'}' | b']' => {
                depth = depth.saturating_sub(1);
                continue;
            },
            b'{' | b'[' => {
                depth += 1;
                if depth > 64 {
                    return Err("resource_budget_exceeded");
                }
            },
            b'"' => {
                while let Some(&character) = data.get(offset) {
                    offset += 1;
                    match character {
                        b'\\' => offset = offset.saturating_add(1),
                        b'"' => break,
                        _ => {},
                    }
                }
            },
            _ => {
                if byte == b'-' || byte.is_ascii_digit() {
                    numeric += 1;
                    if numeric > budget.input_bytes / std::mem::size_of::<u32>() + 65_536 {
                        return Err("resource_budget_exceeded");
                    }
                } else {
                    nodes += 1;
                }
                while data.get(offset).is_some_and(|byte| {
                    !matches!(byte, b' ' | b'\n' | b'\r' | b'\t' | b',' | b':' | b'[' | b']' | b'{' | b'}' | b'"')
                }) {
                    offset += 1;
                }
                if nodes > 65_536 {
                    return Err("resource_budget_exceeded");
                }
                continue;
            },
        }
        nodes += 1;
        if nodes > 65_536 {
            return Err("resource_budget_exceeded");
        }
    }
    Ok(())
}

/// Available memory is admission evidence, never a reservation or RSS ceiling.
#[cfg(target_os = "macos")]
#[allow(deprecated)] // The locked libc exports the system Mach ABI used here.
pub fn available_memory_bytes() -> Option<usize> {
    unsafe extern "C" {
        fn mach_port_deallocate(
            task: libc::mach_port_t,
            name: libc::mach_port_t,
        ) -> libc::kern_return_t;
    }
    let mut stats = std::mem::MaybeUninit::<libc::vm_statistics64>::zeroed();
    let mut count = libc::HOST_VM_INFO64_COUNT;
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if page_size <= 0 {
        return None;
    }
    let host = unsafe { libc::mach_host_self() };
    let result = unsafe { libc::host_statistics64(host, libc::HOST_VM_INFO64, stats.as_mut_ptr().cast(), &mut count) };
    // One task-local send right was returned by mach_host_self.
    unsafe { mach_port_deallocate(libc::mach_task_self(), host) };
    if result != libc::KERN_SUCCESS || !valid_vm_statistics_count(count) {
        return None;
    }
    let stats = unsafe { stats.assume_init() };
    // Match the host's psutil policy: free and inactive pages are reclaimable.
    (stats.free_count as usize).checked_add(stats.inactive_count as usize)?.checked_mul(page_size as usize)
}

#[cfg(target_os = "macos")]
fn valid_vm_statistics_count(count: libc::mach_msg_type_number_t) -> bool {
    // New SDK/libc definitions can include trailing counters absent from an
    // older kernel revision. Only the free/active/inactive prefix is read,
    // and the entire destination was zero-initialized before the syscall.
    let prefix_bytes =
        std::mem::offset_of!(libc::vm_statistics64, inactive_count) + std::mem::size_of::<libc::natural_t>();
    let prefix_words = prefix_bytes.div_ceil(std::mem::size_of::<libc::integer_t>());
    (count as usize) >= prefix_words && count <= libc::HOST_VM_INFO64_COUNT
}

#[cfg(target_os = "linux")]
pub fn available_memory_bytes() -> Option<usize> {
    let data = std::fs::read_to_string("/proc/meminfo").ok()?;
    let line = data.lines().find_map(|line| line.strip_prefix("MemAvailable:"))?;
    line.split_whitespace().next()?.parse::<usize>().ok()?.checked_mul(1024)
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
pub fn available_memory_bytes() -> Option<usize> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_has_escaping_and_allocation_headroom() {
        for mib in [MIN_MIB, DEFAULT_MIB, MAX_MIB] {
            let budget = Budget::for_mib(mib).unwrap();
            assert_eq!(budget.input_bytes, mib * MIB);
            assert_eq!(budget.wire_bytes, 6 * budget.input_bytes + 2 * MIB);
            assert!(budget.template_live_bytes > budget.wire_bytes);
            assert!(budget.template_cumulative_bytes > budget.input_bytes);
            assert!(budget.template_cumulative_bytes <= isize::MAX as usize);
        }
        assert_eq!(Budget::for_mib(MAX_MIB + 1), Err("invalid_resource_budget"));
    }

    #[test]
    fn configuration_is_strict_and_errors_are_content_free() {
        for value in ["", "0", " 16", "16 ", "+16", "1.5", "١٦", "private"] {
            assert_eq!(Budget::parse(value), Err("invalid_resource_budget"));
        }
        assert_eq!(Budget::parse("16"), Ok(Budget::DEFAULT));
        assert_eq!(Budget::parse("01"), Budget::for_mib(1));
        assert_eq!(Budget::parse("128"), Budget::for_mib(128));
        assert_eq!(Budget::parse(&MAX_MIB.to_string()), Budget::for_mib(MAX_MIB));
        assert_eq!(Budget::parse(&(MAX_MIB + 1).to_string()), Err("invalid_resource_budget"));
        assert_eq!(Budget::parse(&usize::MAX.to_string()), Err("invalid_resource_budget"));
    }

    #[test]
    fn automatic_budget_tracks_headroom_without_a_fixed_content_ceiling() {
        for (available, mib) in [(0, 16), (2 << 30, 16), (4 << 30, 32), (8 << 30, 64), (16 << 30, 128)] {
            assert_eq!(Budget::automatic(available).mib, mib);
        }
        assert_eq!(Budget::automatic(usize::MAX).mib, usize::MAX / (128 * MIB));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn older_kernel_memory_statistics_prefix_remains_usable() {
        assert!(valid_vm_statistics_count(40));
        assert!(valid_vm_statistics_count(libc::HOST_VM_INFO64_COUNT));
        assert!(!valid_vm_statistics_count(2));
        assert!(!valid_vm_statistics_count(libc::HOST_VM_INFO64_COUNT + 1));
    }

    #[test]
    fn structural_scan_preserves_long_vectors_and_bounds_container_expansion() {
        let budget = Budget::for_mib(1).unwrap();
        let vector = format!("[{}0]", "0,".repeat(100_000));
        assert_eq!(check_json_structure(vector.as_bytes(), budget), Ok(()));
        let tree = format!("[{}{{}}]", "{},".repeat(65_536));
        assert_eq!(check_json_structure(tree.as_bytes(), budget), Err("resource_budget_exceeded"));
        assert_eq!(check_json_structure(&b"[".repeat(65), budget), Err("resource_budget_exceeded"));
        let scalar_bomb = format!("[{}0]", "0,".repeat(budget.input_bytes / 4 + 65_536));
        assert_eq!(check_json_structure(scalar_bomb.as_bytes(), budget), Err("resource_budget_exceeded"));
        assert_eq!(check_json_structure(br#"{"value":"\\\"[[[}}}"}"#, budget), Ok(()));
    }
}
