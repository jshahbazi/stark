//! Explicit stage policy, independent of native artifacts and environment I/O.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NativeProfile {
    HeadlessReference,
    PrivateValidation,
    Bf16NativeExperimental,
}

pub(crate) fn select(
    profile: Option<&str>,
    private: bool,
    validation: bool,
    api: u64,
    prefix: bool,
) -> Result<NativeProfile, &'static str> {
    match profile.unwrap_or("headless-reference") {
        "headless-reference" if private == validation && (!private || api == 1) => Ok(if private {
            NativeProfile::PrivateValidation
        } else {
            NativeProfile::HeadlessReference
        }),
        "bf16-native-experimental" if !private && !validation && api == 1 && prefix => {
            Ok(NativeProfile::Bf16NativeExperimental)
        },
        _ => Err("invalid GLM native profile, capability or validation combination"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_and_private_stages_retain_their_gates() {
        assert_eq!(select(None, false, false, 1, false), Ok(NativeProfile::HeadlessReference));
        assert_eq!(select(None, true, true, 1, true), Ok(NativeProfile::PrivateValidation));
        assert!(select(None, true, false, 1, true).is_err());
        assert!(select(None, false, true, 1, true).is_err());
        assert!(select(None, true, true, 0, true).is_err());
    }

    #[test]
    fn retained_profile_requires_its_complete_contract() {
        let name = Some("bf16-native-experimental");
        assert_eq!(select(name, false, false, 1, true), Ok(NativeProfile::Bf16NativeExperimental));
        for (private, validation, api, prefix) in
            [(true, false, 1, true), (false, true, 1, true), (false, false, 0, true), (false, false, 1, false)]
        {
            assert!(select(name, private, validation, api, prefix).is_err());
        }
        assert!(select(Some("unrecognized"), false, false, 1, true).is_err());
    }
}
