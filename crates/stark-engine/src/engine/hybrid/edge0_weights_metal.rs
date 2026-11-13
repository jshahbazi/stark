//! Checked, retained weight windows for Edge0's explicit Metal experiment.
//!
//! The same registry and residency manager used by the existing encoder path
//! own these original-byte windows. The n-gram table is never registered here.

use super::*;
use crate::engine::hybrid::weight_metal::{DecodeGraph, DeviceVector, MappedMatrix, MatrixFormat};

const BANK_CAP: usize = 56 * 1024 * 1024 * 1024;
const EXPERT_RECORD_BYTES: usize = 1_630_720;
const EXPERT_LAYER_BYTES: usize = EXPERT_RECORD_BYTES * 512;
const EXPERT_FILE_BYTES: usize = EXPERT_LAYER_BYTES * 48;

#[path = "edge0_mtp_weights_metal.rs"]
mod mtp;

#[cfg(test)]
#[path = "edge0_window_transport_tests.rs"]
mod window_transport_tests;

fn edge0_memory_cap(recommended_working_set: usize) -> usize {
    BANK_CAP.min(recommended_working_set / 4 * 3)
}

fn admit_window_policy(
    expert_windows: bool,
    resident: bool,
) -> Result<()> {
    if expert_windows && resident {
        return Err(Error::Unsupported("Edge0 expert windows with residency are unqualified"));
    }
    Ok(())
}

#[derive(Clone)]
struct Bank {
    name: String,
    file: usize,
    offset: usize,
    bytes: usize,
}

fn expert_record_bank(
    experts: &edge0_source::NativeExperts,
    file: usize,
    layer: usize,
    expert: usize,
) -> Result<Bank> {
    if layer >= 48
        || expert >= 512
        || experts.layers.len() != 48
        || experts.gate_bytes != 492_800
        || experts.down_bytes != 645_120
        || experts.record_bytes != EXPERT_RECORD_BYTES as u64
        || experts.file_bytes != EXPERT_FILE_BYTES as u64
    {
        return Err(Error::Invalid("Edge0 expert window geometry"));
    }
    let span = experts.layers[layer];
    let layer_offset = layer * EXPERT_LAYER_BYTES;
    if span.offset != layer_offset as u64 || span.bytes != EXPERT_LAYER_BYTES as u64 {
        return Err(Error::Invalid("Edge0 expert window layer span"));
    }
    let record = experts.expert(layer, expert)?;
    let offset = layer_offset + expert * EXPERT_RECORD_BYTES;
    if record.gate.offset != offset as u64
        || record.gate.bytes != experts.gate_bytes
        || record.up.offset != record.gate.offset + record.gate.bytes
        || record.up.bytes != experts.gate_bytes
        || record.down.offset != record.up.offset + record.up.bytes
        || record.down.bytes != experts.down_bytes
        || record.down.offset + record.down.bytes != (offset + EXPERT_RECORD_BYTES) as u64
        || offset + EXPERT_RECORD_BYTES > layer_offset + EXPERT_LAYER_BYTES
    {
        return Err(Error::Invalid("Edge0 expert window record span"));
    }
    Ok(Bank {
        name: format!("edge0.experts.{layer}.{expert}"),
        file,
        offset,
        bytes: EXPERT_RECORD_BYTES,
    })
}

fn admit_bank_memory(
    count: usize,
    expert_windows: bool,
    bank_bytes: usize,
    dynamic_allowance: usize,
    cap: usize,
) -> Result<()> {
    let expected_count = if expert_windows {
        48 * 513
    } else {
        96
    };
    if count != expected_count
        || (!expert_windows && bank_bytes != 43_819_991_040usize)
        || (expert_windows && bank_bytes < 43_819_991_040usize)
        || bank_bytes
            .checked_add(MAX_STATIC_VIEW_BYTES)
            .and_then(|n| n.checked_add(dynamic_allowance))
            .is_none_or(|n| n > cap)
    {
        return Err(Error::Invalid("Edge0 complete bank memory admission"));
    }
    Ok(())
}

fn bank_matrix_offset(
    bank: &Bank,
    tensor: &Tensor,
) -> Result<usize> {
    let offset = tensor.offset.checked_sub(bank.offset).ok_or(Error::Invalid("Edge0 graph bank offset"))?;
    if tensor.file != bank.file || offset.checked_add(tensor.size).is_none_or(|end| end > bank.bytes) {
        return Err(Error::Invalid("Edge0 graph matrix outside bank window"));
    }
    Ok(offset)
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

    fn banks(
        &self,
        expert_windows: bool,
    ) -> Result<Vec<Bank>> {
        let mut banks = Vec::with_capacity(if expert_windows {
            48 * 513
        } else {
            96
        });
        if self.experts.layers.len() != 48 {
            return Err(Error::Invalid("Edge0 complete expert inventory"));
        }
        for layer in 0..48 {
            if expert_windows {
                for expert in 0..512 {
                    banks.push(expert_record_bank(&self.experts, self.expert_file, layer, expert)?);
                }
            } else {
                banks.push(self.bank(layer, false)?);
            }
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
        let expert_windows = std::env::var_os("STARK_EDGE0_EXPERT_WINDOWS").is_some_and(|v| v == "1");
        let resident = std::env::var_os("STARK_EDGE0_RESIDENT").is_some_and(|v| v == "1");
        admit_window_policy(expert_windows, resident)?;
        let banks = data.banks(expert_windows)?;
        let cap = edge0_memory_cap(super::super::super::weight_metal::recommended_working_set()?);
        let mut bank_bytes = 0usize;
        for bank in &banks {
            let bytes = self.files[bank.file].metal_span(bank.offset, bank.bytes)?.1;
            if bytes > super::super::super::weight_metal::max_buffer_length()?.min(u32::MAX as usize) {
                return Err(Error::Invalid("Edge0 layer bank exceeds buffer limit"));
            }
            bank_bytes = bank_bytes.checked_add(bytes).ok_or(Error::Invalid("Edge0 complete bank overflow"))?;
        }
        admit_bank_memory(banks.len(), expert_windows, bank_bytes, dynamic_allowance, cap)?;
        let names: HashSet<_> = banks.iter().map(|bank| bank.name.clone()).collect();
        if names.len() != banks.len() {
            return Err(Error::Invalid("Edge0 duplicate bank window"));
        }
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
        // GPU wrapper. Registration stays lazy under the frozen window policy.
        super::super::super::profile::record("native.edge0.bank.admitted_bytes", 0.0, bank_bytes);
        super::super::super::profile::record("native.edge0.bank.cap_bytes", 0.0, cap);
        data.metal_dynamic_allowance.store(dynamic_allowance, Ordering::Release);
        data.metal_resident.store(resident, Ordering::Release);
        data.metal_expert_windows.store(expert_windows, Ordering::Release);
        data.metal_bank_allowance.store(bank_bytes, Ordering::Release);
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
        let bank = match matrix {
            Edge0Matrix::ProjectionDown {
                layer,
                ..
            }
            | Edge0Matrix::ProjectionUp {
                layer,
                ..
            } => data.bank(layer, true)?,
            Edge0Matrix::ExpertGate {
                layer,
                expert,
            }
            | Edge0Matrix::ExpertUp {
                layer,
                expert,
            }
            | Edge0Matrix::ExpertDown {
                layer,
                expert,
            } => {
                if data.metal_expert_windows.load(Ordering::Acquire) {
                    expert_record_bank(&data.experts, data.expert_file, layer, expert)?
                } else {
                    data.bank(layer, false)?
                }
            },
        };
        let offset = bank_matrix_offset(&bank, &tensor)?;
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

#[cfg(test)]
mod tests {
    use stark_engine_macros::stark_test;

    use super::*;

    fn experts() -> edge0_source::NativeExperts {
        edge0_source::NativeExperts {
            layers: (0..48)
                .map(|layer| edge0_source::FileSpan {
                    offset: (layer * EXPERT_LAYER_BYTES) as u64,
                    bytes: EXPERT_LAYER_BYTES as u64,
                })
                .collect(),
            gate_bytes: 492_800,
            down_bytes: 645_120,
            record_bytes: EXPERT_RECORD_BYTES as u64,
            file_bytes: EXPERT_FILE_BYTES as u64,
        }
    }

    #[stark_test]
    fn edge0_expert_windows_cover_exact_records_and_all_selected_matrix_spans() {
        let experts = experts();
        let mut names = HashSet::new();
        let mut end = 0;
        for layer in 0..48 {
            for expert in 0..512 {
                let bank = expert_record_bank(&experts, 3, layer, expert).unwrap();
                assert!(names.insert(bank.name.clone()));
                assert_eq!(bank.offset, end);
                assert_eq!(bank.bytes, 1_630_720);
                let record = experts.expert(layer, expert).unwrap();
                for (span, expected_offset) in [(record.gate, 0), (record.up, 492_800), (record.down, 985_600)] {
                    let mut tensor = Tensor {
                        file: 3,
                        offset: span.offset as usize,
                        size: span.bytes as usize,
                        shape: vec![1, 1],
                        kind: 11,
                        affine: None,
                    };
                    assert_eq!(bank_matrix_offset(&bank, &tensor).unwrap(), expected_offset);
                    tensor.file = 4;
                    assert!(bank_matrix_offset(&bank, &tensor).is_err());
                    tensor.file = 3;
                    tensor.offset = bank.offset + bank.bytes;
                    assert!(bank_matrix_offset(&bank, &tensor).is_err());
                }
                end += bank.bytes;
            }
        }
        assert_eq!(names.len(), 24_576);
        assert_eq!(end, 40_076_574_720);
    }

    #[stark_test]
    fn edge0_expert_windows_reject_changed_geometry_and_out_of_bounds_selection() {
        let valid = experts();
        assert!(expert_record_bank(&valid, 0, 48, 0).is_err());
        assert!(expert_record_bank(&valid, 0, 0, 512).is_err());
        for field in 0..6 {
            let mut changed = valid.clone();
            match field {
                0 => changed.gate_bytes += 1,
                1 => changed.down_bytes += 1,
                2 => changed.record_bytes += 1,
                3 => changed.file_bytes += 1,
                4 => changed.layers[0].offset += 1,
                5 => changed.layers[0].bytes -= 1,
                _ => unreachable!(),
            }
            assert!(expert_record_bank(&changed, 0, 0, 0).is_err());
        }
        let mut changed = valid;
        changed.layers.pop();
        assert!(expert_record_bank(&changed, 0, 0, 0).is_err());
    }

    #[stark_test]
    fn edge0_expert_window_admission_requires_complete_inventory_and_exact_cap() {
        let rounded_inventory = 44_210_061_312usize;
        let dynamic = 128 * 1024 * 1024;
        let exact = rounded_inventory + MAX_STATIC_VIEW_BYTES + dynamic;
        assert!(admit_bank_memory(24_624, true, rounded_inventory, dynamic, exact).is_ok());
        assert!(admit_bank_memory(24_624, true, rounded_inventory, dynamic, exact - 1).is_err());
        assert!(admit_bank_memory(24_623, true, rounded_inventory, dynamic, exact).is_err());
        assert!(admit_bank_memory(24_624, true, usize::MAX, dynamic, usize::MAX).is_err());
        assert!(admit_bank_memory(96, false, 43_819_991_040, 0, BANK_CAP).is_ok());
        assert!(admit_bank_memory(96, false, 43_819_991_041, 0, BANK_CAP).is_err());
        assert_eq!(edge0_memory_cap(usize::MAX), BANK_CAP);
        assert_eq!(edge0_memory_cap(64 * 1024 * 1024 * 1024), 48 * 1024 * 1024 * 1024);
        assert!(admit_window_policy(false, false).is_ok());
        assert!(admit_window_policy(false, true).is_ok());
        assert!(admit_window_policy(true, false).is_ok());
        assert!(admit_window_policy(true, true).is_err());
    }
}
