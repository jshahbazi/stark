// SPDX-License-Identifier: Apache-2.0
//! Explicit diagnostic-only MTP source plumbing. Dense tensors retain their
//! original encoding; the expert bank retains its four physical Q2 planes.
//! Admission covers a mapped graph experiment and does not qualify residency,
//! model capability, worker execution, prefix reuse, or production MTP.

use super::*;

const TARGET_BANK_BYTES: usize = 43_819_991_040;
const MIN_HEAD_DYNAMIC_BYTES: usize = 512 * 1024 * 1024;
const MTP_EXPERT_RECORD_BYTES: usize = 1_382_400;

fn admit_mtp_memory(
    target_admitted: bool,
    target_resident: bool,
    target_dynamic_bytes: usize,
    head_dynamic_bytes: usize,
    cap: usize,
) -> Result<()> {
    if !target_admitted {
        return Err(Error::Invalid("Edge0 MTP head requires admitted target"));
    }
    if target_resident {
        return Err(Error::Unsupported("Edge0 MTP resident head is unqualified"));
    }
    if head_dynamic_bytes < MIN_HEAD_DYNAMIC_BYTES
        || TARGET_BANK_BYTES
            .checked_add(MAX_STATIC_VIEW_BYTES)
            .and_then(|n| n.checked_add(target_dynamic_bytes))
            .and_then(|n| n.checked_add(head_dynamic_bytes))
            .is_none_or(|n| n > cap)
    {
        return Err(Error::Invalid("Edge0 MTP head memory admission"));
    }
    Ok(())
}

fn dense_tensor(
    file: usize,
    encoding: Edge0MtpEncoding,
    shape: [usize; 2],
    offset: usize,
    bytes: usize,
    file_bytes: usize,
) -> Result<Tensor> {
    let kind = match encoding {
        Edge0MtpEncoding::F32 => 0,
        Edge0MtpEncoding::Bf16 => 30,
        Edge0MtpEncoding::Q8_0 => 8,
    };
    if shape[0] == 0
        || shape[1] == 0
        || shape[0] > 12288
        || shape[1] > 10240
        || file_bytes != mtp_storage::DENSE_BYTES as usize
        || shape[0].checked_mul(encoded_size(kind, shape[1])?) != Some(bytes)
        || offset.checked_add(bytes).is_none_or(|end| end > file_bytes)
    {
        return Err(Error::Invalid("Edge0 MTP dense mapped geometry"));
    }
    Ok(Tensor {
        file,
        offset,
        size: bytes,
        shape: shape.to_vec(),
        kind,
        affine: None,
    })
}

fn final_gamma(
    bytes: &[u8],
    shape: [usize; 2],
    encoding: Edge0MtpEncoding,
) -> Result<Vec<f32>> {
    if encoding != Edge0MtpEncoding::F32
        || shape[0] != 1
        || shape[1] == 0
        || shape[1] > 10240
        || shape[1].checked_mul(4) != Some(bytes.len())
    {
        return Err(Error::Invalid("Edge0 MTP final F32 vector geometry"));
    }
    let values = bytes.as_chunks::<4>().0.iter().map(|word| f32::from_le_bytes(*word)).collect::<Vec<_>>();
    if values.iter().any(|value| !value.is_finite()) {
        return Err(Error::Invalid("Edge0 MTP nonfinite final gamma"));
    }
    // mtp_rt.py already adds one to centered norm weights during preparation.
    Ok(values)
}

impl Weights {
    /// Admit an explicit bounded diagnostic head after target Metal admission.
    /// Head weights consume the existing static-view reserve. Additional head
    /// state and graph scratch consume a separate allowance of at least 512 MiB.
    pub(in crate::engine::hybrid) fn prepare_edge0_mtp_head(
        &self,
        extra_dynamic_bytes: usize,
    ) -> Result<()> {
        let data = self.edge0.as_ref().ok_or(Error::Invalid("Edge0 MTP storage unavailable"))?;
        if data.mtp_admitted.load(Ordering::Acquire) {
            return Err(Error::Invalid("Edge0 MTP head already admitted"));
        }
        let target_admitted = data.metal_admitted.load(Ordering::Acquire);
        let cap = edge0_memory_cap(crate::engine::hybrid::weight_metal::recommended_working_set()?);
        admit_mtp_memory(
            target_admitted,
            data.metal_resident.load(Ordering::Acquire),
            data.metal_dynamic_allowance.load(Ordering::Acquire),
            extra_dynamic_bytes,
            cap,
        )?;
        // Admission creates no Metal wrappers and does not read weight pages.
        // Target's existing cold bank/static reservation remains unchanged.
        data.mtp_admitted
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| Error::Invalid("Edge0 MTP head already admitted"))?;
        Ok(())
    }

    fn admitted_edge0_mtp(&self) -> Result<&Edge0Data> {
        let data = self.edge0.as_ref().ok_or(Error::Invalid("Edge0 MTP storage unavailable"))?;
        if !data.mtp_admitted.load(Ordering::Acquire) {
            return Err(Error::Invalid("Edge0 MTP diagnostic head not admitted"));
        }
        Ok(data)
    }

    /// Read a final F32 normalization vector without applying another +1.
    pub(in crate::engine::hybrid) fn edge0_mtp_vector(
        &self,
        name: &str,
    ) -> Result<Vec<f32>> {
        let view = self.edge0_mtp_dense(name)?;
        final_gamma(view.bytes()?, view.shape(), view.encoding())
    }

    pub(in crate::engine::hybrid) fn encode_edge0_mtp_dense(
        &self,
        graph: &mut DecodeGraph,
        name: &str,
        input: &DeviceVector,
        tokens: usize,
    ) -> Result<DeviceVector> {
        let data = self.admitted_edge0_mtp()?;
        let view = data.mtp.dense(name)?;
        let tensor = dense_tensor(
            data.mtp_dense_file,
            view.encoding(),
            view.shape(),
            view.offset(),
            view.bytes()?.len(),
            self.files[data.mtp_dense_file].length,
        )?;
        let mapped = self.mapped_matrix(&tensor, 0, tensor.shape[0], tensor.shape[1], true)?;
        graph.edge0_packed_batch(&mapped, self.matrix_format(&tensor)?, tensor.shape[0], tensor.shape[1], input, tokens)
    }

    fn edge0_mtp_expert_matrix(
        &self,
        expert: usize,
    ) -> Result<MappedMatrix> {
        let data = self.admitted_edge0_mtp()?;
        let view = data.mtp.expert(expert)?;
        if view.gate_up_shape() != [640, 2560]
            || view.down_shape() != [2560, 640]
            || view.record_bytes() != MTP_EXPERT_RECORD_BYTES
            || expert.checked_mul(MTP_EXPERT_RECORD_BYTES) != Some(view.offset())
        {
            return Err(Error::Invalid("Edge0 MTP original expert geometry"));
        }
        // One lazy retained raw view of the complete unchanged expert file.
        // Cached here as static storage, never mislabeled as Packed(42).
        let bank = self.metal_view(data.mtp_expert_file, 0, mtp_storage::EXPERT_BYTES as usize, true)?;
        Ok(MappedMatrix {
            codes: bank.subslice(view.offset(), view.record_bytes())?,
            planes: None,
        })
    }

    pub(in crate::engine::hybrid) fn encode_edge0_mtp_gate_up(
        &self,
        graph: &mut DecodeGraph,
        expert: usize,
        input: &DeviceVector,
        tokens: usize,
    ) -> Result<DeviceVector> {
        let matrix = self.edge0_mtp_expert_matrix(expert)?;
        graph.edge0_mtp_q2_gate_up(&matrix, 2560, 640, input, tokens)
    }

    pub(in crate::engine::hybrid) fn encode_edge0_mtp_down(
        &self,
        graph: &mut DecodeGraph,
        expert: usize,
        input: &DeviceVector,
        tokens: usize,
    ) -> Result<DeviceVector> {
        let matrix = self.edge0_mtp_expert_matrix(expert)?;
        graph.edge0_mtp_q2_down(&matrix, 2560, 640, input, tokens)
    }
}

#[cfg(test)]
mod tests {
    use stark_engine_macros::stark_test;

    use super::*;

    #[stark_test]
    fn edge0_mtp_head_memory_admission_includes_target_and_extra_dynamic_reserves() {
        let target_dynamic = 1024 * 1024 * 1024;
        let exact = TARGET_BANK_BYTES + MAX_STATIC_VIEW_BYTES + target_dynamic + MIN_HEAD_DYNAMIC_BYTES;
        assert!(admit_mtp_memory(true, false, target_dynamic, MIN_HEAD_DYNAMIC_BYTES, exact).is_ok());
        assert!(admit_mtp_memory(true, false, target_dynamic, MIN_HEAD_DYNAMIC_BYTES, exact - 1).is_err());
        assert!(admit_mtp_memory(false, false, target_dynamic, MIN_HEAD_DYNAMIC_BYTES, usize::MAX).is_err());
        assert!(admit_mtp_memory(true, true, target_dynamic, MIN_HEAD_DYNAMIC_BYTES, usize::MAX).is_err());
        assert!(admit_mtp_memory(true, false, target_dynamic, MIN_HEAD_DYNAMIC_BYTES - 1, usize::MAX).is_err());
        assert!(admit_mtp_memory(true, false, usize::MAX, MIN_HEAD_DYNAMIC_BYTES, usize::MAX).is_err());
        assert!(admit_mtp_memory(true, false, 0, usize::MAX, usize::MAX).is_err());
        assert_eq!(edge0_memory_cap(64 * 1024 * 1024 * 1024), 48 * 1024 * 1024 * 1024);
        assert_eq!(edge0_memory_cap(128 * 1024 * 1024 * 1024), BANK_CAP);
        assert_eq!(edge0_memory_cap(usize::MAX), BANK_CAP);
    }

    #[stark_test]
    fn edge0_mtp_dense_transport_preserves_encoding_and_rejects_counterexamples() {
        let file_bytes = mtp_storage::DENSE_BYTES as usize;
        let tensor =
            dense_tensor(6, Edge0MtpEncoding::Q8_0, [12288, 2560], 81_231_872, 33_423_360, file_bytes).unwrap();
        assert!(tensor.file == 6 && tensor.kind == 8 && tensor.shape == [12288, 2560] && tensor.affine.is_none());
        assert!(dense_tensor(6, Edge0MtpEncoding::Q8_0, [12288, 2560], 81_231_872, 33_423_359, file_bytes).is_err());
        assert!(dense_tensor(6, Edge0MtpEncoding::Q8_0, [12288, 2559], 81_231_872, 33_423_360, file_bytes).is_err());
        assert!(dense_tensor(6, Edge0MtpEncoding::Bf16, [320, 10240], 13_967_360, 6_553_600, file_bytes).is_ok());
        assert!(dense_tensor(6, Edge0MtpEncoding::F32, [1, 10240], 13_926_400, 40_960, file_bytes).is_ok());
        assert!(dense_tensor(6, Edge0MtpEncoding::F32, [0, 10240], 0, 0, file_bytes).is_err());
        assert!(dense_tensor(6, Edge0MtpEncoding::F32, [1, 10240], usize::MAX, 40_960, file_bytes).is_err());
        assert!(dense_tensor(6, Edge0MtpEncoding::F32, [1, 10240], file_bytes - 1, 40_960, file_bytes).is_err());
        assert!(dense_tensor(6, Edge0MtpEncoding::F32, [1, 10240], 0, 40_960, file_bytes - 1).is_err());
    }

    #[stark_test]
    fn edge0_mtp_final_gamma_preserves_values_and_rejects_nonfinite_or_wrong_encoding() {
        let values = [0.0_f32, -1.0, 0.125, 2.0];
        let bytes = values.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>();
        assert!(final_gamma(&bytes, [1, 4], Edge0MtpEncoding::F32).unwrap() == values);
        assert!(final_gamma(&bytes, [1, 4], Edge0MtpEncoding::Bf16).is_err());
        assert!(final_gamma(&bytes, [1, 4], Edge0MtpEncoding::Q8_0).is_err());
        assert!(final_gamma(&bytes, [2, 2], Edge0MtpEncoding::F32).is_err());
        assert!(final_gamma(&bytes[..15], [1, 4], Edge0MtpEncoding::F32).is_err());
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(final_gamma(&value.to_le_bytes(), [1, 1], Edge0MtpEncoding::F32).is_err());
        }
    }
}
