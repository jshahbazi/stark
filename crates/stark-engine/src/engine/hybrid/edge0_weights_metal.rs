//! Checked, retained layer windows for Edge0's explicit Metal experiment.
//!
//! The same registry and residency manager used by the existing encoder path
//! own these original-byte windows. The n-gram table is never registered here.

use super::*;
use crate::engine::hybrid::weight_metal::{DecodeGraph, DeviceVector, MappedMatrix, MatrixFormat};

const BANK_CAP: usize = 56 * 1024 * 1024 * 1024;

#[derive(Clone)]
struct Bank {
    name: String,
    file: usize,
    offset: usize,
    bytes: usize,
}

impl Edge0Data {
    fn bank(
        &self,
        layer: usize,
        projection: bool,
    ) -> Result<Bank> {
        if layer >= 48 {
            return Err(Error::Invalid("Edge0 bank layer"));
        }
        if projection {
            let bytes = usize::try_from(self.projections.layer_bytes)
                .map_err(|_| Error::Invalid("Edge0 projection bank size"))?;
            Ok(Bank {
                name: format!("edge0.projections.{layer}"),
                file: self.projection_file,
                offset: layer.checked_mul(bytes).ok_or(Error::Invalid("Edge0 projection bank offset"))?,
                bytes,
            })
        } else {
            let span = self.experts.layers.get(layer).ok_or(Error::Invalid("Edge0 expert bank layer"))?;
            Ok(Bank {
                name: format!("edge0.experts.{layer}"),
                file: self.expert_file,
                offset: usize::try_from(span.offset).map_err(|_| Error::Invalid("Edge0 bank offset"))?,
                bytes: usize::try_from(span.bytes).map_err(|_| Error::Invalid("Edge0 bank size"))?,
            })
        }
    }

    fn banks(&self) -> Result<Vec<Bank>> {
        let mut banks = Vec::with_capacity(96);
        if self.experts.layers.len() != 48 {
            return Err(Error::Invalid("Edge0 complete expert inventory"));
        }
        for layer in 0..48 {
            banks.push(self.bank(layer, false)?);
            banks.push(self.bank(layer, true)?);
        }
        Ok(banks)
    }
}

impl Weights {
    pub(in crate::engine::hybrid) fn prepare_edge0_metal(
        &self,
        dynamic_allowance: usize,
    ) -> Result<()> {
        let data = self.edge0.as_ref().ok_or(Error::Invalid("Edge0 layout required"))?;
        if data.metal_admitted.load(Ordering::Acquire) {
            return Err(Error::Invalid("Edge0 Metal already admitted"));
        }
        let banks = data.banks()?;
        let cap = BANK_CAP.min(super::super::super::weight_metal::recommended_working_set()? / 4 * 3);
        let mut bank_bytes = 0usize;
        for bank in &banks {
            let bytes = self.files[bank.file].metal_span(bank.offset, bank.bytes)?.1;
            if bytes > super::super::super::weight_metal::max_buffer_length()?.min(u32::MAX as usize) {
                return Err(Error::Invalid("Edge0 layer bank exceeds buffer limit"));
            }
            bank_bytes = bank_bytes.checked_add(bytes).ok_or(Error::Invalid("Edge0 complete bank overflow"))?;
        }
        if banks.len() != 96
            || bank_bytes != 43_819_991_040usize
            || bank_bytes
                .checked_add(MAX_STATIC_VIEW_BYTES)
                .and_then(|n| n.checked_add(dynamic_allowance))
                .is_none_or(|n| n > cap)
        {
            return Err(Error::Invalid("Edge0 complete bank memory admission"));
        }
        let names: HashSet<_> = banks.iter().map(|bank| bank.name.clone()).collect();
        if names.len() != 96 {
            return Err(Error::Invalid("Edge0 duplicate bank window"));
        }
        let resident = std::env::var_os("STARK_EDGE0_RESIDENT").is_some_and(|v| v == "1");
        let cache = self.encoder_bank_views.lock().map_err(|_| Error::Invalid("Edge0 bank registry lock"))?;
        if !cache.matrices.is_empty() {
            return Err(Error::Invalid("Edge0 bank registry already populated"));
        }
        drop(cache);
        if resident {
            let statics: Vec<_> = self
                .static_views
                .lock()
                .map_err(|_| Error::Invalid("Edge0 static registry lock"))?
                .entries
                .values()
                .cloned()
                .collect();
            let mut configured = self.encoder_residency.lock().map_err(|_| Error::Invalid("Edge0 residency lock"))?;
            if configured.is_some() {
                return Err(Error::Invalid("Edge0 residency already configured"));
            }
            let mut state = EncoderResidency::new(names, dynamic_allowance, cap, MAX_STATIC_VIEW_BYTES)?;
            state.add(&statics.iter().collect::<Vec<_>>(), false)?;
            *configured = Some(state);
            self.encoder_residency_enabled.store(true, Ordering::Release);
        }
        // Admit the complete prospective inventory above before creating any
        // GPU wrapper. Actual registration remains lazy, one layer at a time.
        super::super::super::profile::record("native.edge0.bank.admitted_bytes", 0.0, bank_bytes);
        super::super::super::profile::record("native.edge0.bank.cap_bytes", 0.0, cap);
        data.metal_admitted.store(true, Ordering::Release);
        Ok(())
    }

    fn edge0_bank_matrix(
        &self,
        bank: &Bank,
    ) -> Result<MappedMatrix> {
        let mut cache = self.encoder_bank_views.lock().map_err(|_| Error::Invalid("Edge0 bank registry lock"))?;
        if let Some(matrix) = cache.matrices.get(&bank.name) {
            self.register_resident_bank(&bank.name, matrix)?;
            return Ok(matrix.clone());
        }
        let codes = self.files[bank.file].metal_slice(bank.offset, bank.bytes)?;
        let matrix = MappedMatrix {
            codes,
            planes: None,
        };
        self.register_resident_bank(&bank.name, &matrix)?;
        cache.virtual_bytes = cache
            .virtual_bytes
            .checked_add(matrix.codes.registered_bytes())
            .ok_or(Error::Invalid("Edge0 virtual bank overflow"))?;
        cache.matrices.insert(bank.name.clone(), matrix.clone());
        Ok(matrix)
    }

    fn edge0_graph_matrix(
        &self,
        matrix: Edge0Matrix,
    ) -> Result<(MappedMatrix, Tensor)> {
        let data = self.edge0.as_ref().ok_or(Error::Invalid("Edge0 layout required"))?;
        if !data.metal_admitted.load(Ordering::Acquire) {
            return Err(Error::Invalid("Edge0 Metal banks not admitted"));
        }
        let tensor = data.matrix(matrix)?;
        let (layer, projection) = match matrix {
            Edge0Matrix::ProjectionDown {
                layer,
                ..
            }
            | Edge0Matrix::ProjectionUp {
                layer,
                ..
            } => (layer, true),
            Edge0Matrix::ExpertGate {
                layer,
                ..
            }
            | Edge0Matrix::ExpertUp {
                layer,
                ..
            }
            | Edge0Matrix::ExpertDown {
                layer,
                ..
            } => (layer, false),
        };
        let bank = data.bank(layer, projection)?;
        let offset = tensor.offset.checked_sub(bank.offset).ok_or(Error::Invalid("Edge0 graph bank offset"))?;
        if tensor.file != bank.file || offset.checked_add(tensor.size).is_none_or(|end| end > bank.bytes) {
            return Err(Error::Invalid("Edge0 graph matrix outside layer bank"));
        }
        let mapped = self.edge0_bank_matrix(&bank)?;
        Ok((
            MappedMatrix {
                codes: mapped.codes.subslice(offset, tensor.size)?,
                planes: None,
            },
            tensor,
        ))
    }

    fn edge0_count_batch(
        &self,
        tokens: usize,
        kind: u32,
        expert: bool,
    ) {
        if tokens > 1
            && let Some(data) = &self.edge0
        {
            data.metal_counters[0].fetch_add(1, Ordering::Relaxed);
            if kind == 8 {
                data.metal_counters[1].fetch_add(1, Ordering::Relaxed);
            }
            if expert {
                data.metal_counters[2].fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    pub(in crate::engine::hybrid) fn encode_edge0_matvec_batch(
        &self,
        graph: &mut DecodeGraph,
        matrix: Edge0Matrix,
        input: &DeviceVector,
        tokens: usize,
    ) -> Result<DeviceVector> {
        let (mapped, tensor) = self.edge0_graph_matrix(matrix)?;
        let output = graph.edge0_packed_batch(
            &mapped,
            MatrixFormat::Packed(tensor.kind),
            tensor.shape[0],
            tensor.shape[1],
            input,
            tokens,
        )?;
        self.edge0_count_batch(
            tokens,
            tensor.kind,
            matches!(
                matrix,
                Edge0Matrix::ExpertGate { .. } | Edge0Matrix::ExpertUp { .. } | Edge0Matrix::ExpertDown { .. }
            ),
        );
        Ok(output)
    }

    pub(in crate::engine::hybrid) fn encode_edge0_gate_up_swiglu_batch(
        &self,
        graph: &mut DecodeGraph,
        layer: usize,
        expert: usize,
        input: &DeviceVector,
        tokens: usize,
    ) -> Result<DeviceVector> {
        let (gate, tensor) = self.edge0_graph_matrix(Edge0Matrix::ExpertGate {
            layer,
            expert,
        })?;
        let (up, right) = self.edge0_graph_matrix(Edge0Matrix::ExpertUp {
            layer,
            expert,
        })?;
        if tensor.kind != 11 || right.kind != 11 || tensor.shape != right.shape {
            return Err(Error::Invalid("Edge0 fused gate/up format"));
        }
        let output = graph.edge0_gate_up_swiglu_batch(&gate, &up, tensor.shape[0], tensor.shape[1], input, tokens)?;
        self.edge0_count_batch(tokens, 11, true);
        Ok(output)
    }

    pub(in crate::engine::hybrid) fn encode_edge0_named_batch(
        &self,
        graph: &mut DecodeGraph,
        name: &str,
        input: &DeviceVector,
        tokens: usize,
    ) -> Result<DeviceVector> {
        if !self.is_edge0() {
            return Err(Error::Invalid("Edge0 named batch requires Edge0 layout"));
        }
        let tensor = self.tensor(name)?;
        if tensor.shape.len() != 2 || tensor.affine.is_some() {
            return Err(Error::Invalid("Edge0 named batch shape"));
        }
        let mapped = self.mapped_matrix(tensor, 0, tensor.shape[0], tensor.shape[1], true)?;
        let output = graph.edge0_packed_batch(
            &mapped,
            self.matrix_format(tensor)?,
            tensor.shape[0],
            tensor.shape[1],
            input,
            tokens,
        )?;
        self.edge0_count_batch(tokens, tensor.kind, false);
        Ok(output)
    }
}
