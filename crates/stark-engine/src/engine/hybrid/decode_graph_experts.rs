// SPDX-License-Identifier: Apache-2.0
//! GPU route grouping and cross-token IQ expert projections. All arrays stay
//! graph-owned; routing IDs are raw u32 slots in DeviceVector storage.

use super::*;

#[derive(Clone)]
pub(crate) struct GroupedExpertRoutes {
    counts: DeviceVector,
    lists: DeviceVector,
    jobs: DeviceVector,
    experts: usize,
    tokens: usize,
    selected: usize,
}

impl GroupedExpertRoutes {
    pub(crate) fn route_count(&self) -> usize {
        self.tokens * self.selected
    }
}

struct ExpertPipelines {
    copy: PipelineState,
    lists: PipelineState,
    jobs: PipelineState,
    indirect: PipelineState,
    iq2: PipelineState,
    iq3: PipelineState,
}

static EXPERT_PIPELINES: OnceLock<Option<ExpertPipelines>> = OnceLock::new();

fn expert_pipelines() -> Option<&'static ExpertPipelines> {
    EXPERT_PIPELINES
        .get_or_init(|| {
            let executor = executor().ok()?;
            let mut source = String::from("#include <metal_stdlib>\nusing namespace metal;\n");
            for (name, table) in
                [("iq2xxs_grid", weights::IQ2XXS_GRID.as_slice()), ("iq2xs_grid", weights::IQ2XS_GRID.as_slice())]
            {
                source.push_str(&format!("constant ulong {name}[{}]={{", table.len()));
                for value in table {
                    source.push_str(&format!("0x{value:016x}ul,"));
                }
                source.push_str("};\n");
            }
            source.push_str("constant uchar ksigns[128]={");
            for value in weights::KSIGNS_IQ2XS {
                source.push_str(&format!("{value},"));
            }
            source.push_str("};\nconstant short iq4values[16]={");
            for value in weights::KVALUES_IQ4NL {
                source.push_str(&format!("{value},"));
            }
            source.push_str("};\nconstant uint iq3xxs_grid[256]={");
            for value in weights::IQ3XXS_GRID {
                source.push_str(&format!("0x{value:08x}u,"));
            }
            source.push_str("};\n");
            // Reuse source-bound decode helpers and the source vector8 risk
            // fallback. Only the cross-input-row loop is new.
            for text in [
                include_str!("packed_matvec.metal"),
                include_str!("packed_vector_matvec.metal"),
                include_str!("packed_iq_multirow_matvec.metal"),
                include_str!("packed_iq_staged_matvec.metal"),
                include_str!("packed_iq_grouped32_matvec.metal"),
                include_str!("grouped_expert_prefill.metal"),
            ] {
                source.push_str(text);
            }
            let options = metal::MTLCompileOptions::new();
            options.set_math_mode(metal::MTLMathMode::Safe);
            let library = executor.device.new_library_with_source(&source, Some(&options)).ok()?;
            let simple = |name| {
                let function = library.new_function_with_name(name)?;
                executor.device.new_compute_pipeline_state_with_function(&function).ok()
            };
            let matrix = |kind| {
                let values = MTLFunctionConstantValues::new();
                for (index, value) in [
                    (0, kind),
                    (4, 256),
                    (
                        5,
                        if kind == 17 {
                            74
                        } else {
                            98
                        },
                    ),
                    (71, 4),
                ] {
                    values.set_constant_value_type_at_index(NonNull::from(&value).cast(), MTLDataType::UInt, index);
                }
                let function =
                    library.new_function_with_name_constant_values("stark_expert_iq_grouped", &values).ok()?;
                executor.device.new_compute_pipeline_state_with_function(&function).ok()
            };
            let result = ExpertPipelines {
                copy: simple("stark_expert_copy_indices")?,
                lists: simple("stark_expert_lists")?,
                jobs: simple("stark_expert_jobs")?,
                indirect: simple("stark_expert_indirect")?,
                iq2: matrix(17)?,
                iq3: matrix(18)?,
            };
            [&result.copy, &result.lists, &result.jobs, &result.indirect, &result.iq2, &result.iq3]
                .iter()
                .all(|pipeline| {
                    pipeline.thread_execution_width() == 32 && pipeline.max_total_threads_per_threadgroup() >= 64
                })
                .then_some(result)
        })
        .as_ref()
}

fn route_geometry(
    experts: usize,
    tokens: usize,
    selected: usize,
) -> bool {
    (1..=2048).contains(&experts)
        && (1..=128).contains(&tokens)
        && (1..=64).contains(&selected)
        && selected <= experts
        && tokens.checked_mul(selected).is_some_and(|routes| routes <= 1024)
        && experts.checked_mul(tokens * selected).is_some_and(|n| n <= 8_000_000)
}

fn projection_geometry(
    matrix: &MappedMatrix,
    format: MatrixFormat,
    shape: [usize; 3],
    tokens: usize,
    selected: usize,
) -> Result<bool> {
    let [experts, rows, columns] = shape;
    if !matches!(format, MatrixFormat::Packed(17 | 18)) {
        return Ok(false);
    }
    // parameters_for_input checks the logical byte length. Check the actual
    // backing span as well, so even a forged low-level descriptor cannot bind
    // an out-of-bounds typed16 view before optional pipeline admission.
    if matrix.planes.is_some()
        || matrix.codes.length == 0
        || !matrix.codes.offset.is_multiple_of(2)
        || matrix.codes.offset > u32::MAX as usize
        || matrix.codes.offset.checked_add(matrix.codes.length).is_none_or(|end| end > matrix.codes.buffer.length())
    {
        return Err(Error::Invalid("grouped expert checked weight span"));
    }
    if !route_geometry(experts, tokens, selected)
        || !matches!(columns, 2048 | 4096)
        || rows == 0
        || rows.checked_mul(tokens * selected).is_none_or(|n| n > 8_000_000)
    {
        return Ok(false);
    }
    MappedOperation {
        matrices: vec![matrix.clone()],
        inputs: &[],
        rows: rows.checked_mul(experts).ok_or(Error::Invalid("grouped expert bank geometry"))?,
        columns,
        rows_per_batch: 0,
        format,
    }
    .parameters_for_input(columns)?;
    Ok(true)
}

impl DecodeGraph {
    /// Routing IDs are u32 bit patterns, including subnormal F32 encodings.
    /// Copy them without a floating-point load, finite scan or conversion.
    pub(crate) fn copy_expert_indices(
        &mut self,
        input: &DeviceVector,
        output: &DeviceVector,
    ) -> Result<()> {
        input.validate()?;
        output.validate()?;
        if input.len != output.len
            || input.len > 1024
            || self.projection_level
            || (std::ptr::eq(&*input.buffer, &*output.buffer)
                && input.offset < output.offset + output.len * 4
                && output.offset < input.offset + input.len * 4)
        {
            return Err(Error::Invalid("grouped expert index copy dimensions or alias"));
        }
        let pipelines = expert_pipelines().ok_or(Error::Unsupported("grouped expert pipeline unavailable"))?;
        self.prepare_external_write(output)?;
        self.expert_bind(&pipelines.copy, &[input, output], &[input.len as u32], input.len, 64);
        Ok(())
    }

    /// Call for all gate/up/down banks before grouping encodes anything. The
    /// existing selected signed-factor policy is required for exact admission.
    pub(crate) fn expert_grouped_admitted(
        matrix: &MappedMatrix,
        format: MatrixFormat,
        shape: [usize; 3],
        tokens: usize,
        selected: usize,
    ) -> Result<bool> {
        if !std::env::var_os("STARK_GLM53_EXPERT_PREFILL_BATCH").is_some_and(|value| value == "1")
            || !iq_signed_factor_schedule(format, shape[2], matrix.codes.offset)
            || !projection_geometry(matrix, format, shape, tokens, selected)?
        {
            return Ok(false);
        }
        Ok(expert_pipelines().is_some())
    }

    fn expert_bind(
        &mut self,
        pipeline: &PipelineState,
        vectors: &[&DeviceVector],
        p: &[u32],
        grid: usize,
        threads: usize,
    ) {
        self.order_dispatch();
        self.encoding.set_compute_pipeline_state(pipeline);
        for (index, vector) in vectors.iter().enumerate() {
            self.encoding.set_buffer(Some(&vector.buffer), vector.offset, index);
            self.resources.push(vector.buffer.clone());
        }
        self.encoding.set_bytes(
            NonNull::new(p.as_ptr().cast_mut()).unwrap().cast(),
            std::mem::size_of_val(p),
            vectors.len(),
        );
        self.encoding.set_buffer(Some(&self.failure.buffer), self.failure.offset, 30);
        self.encoding.dispatch_threads(
            MTLSize {
                width: grid,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threads,
                height: 1,
                depth: 1,
            },
        );
    }

    pub(crate) fn group_expert_routes(
        &mut self,
        indices: &DeviceVector,
        experts: usize,
        tokens: usize,
        selected: usize,
    ) -> Result<Option<GroupedExpertRoutes>> {
        if !std::env::var_os("STARK_GLM53_EXPERT_PREFILL_BATCH").is_some_and(|value| value == "1") {
            return Ok(None);
        }
        self.group_expert_routes_inner(indices, experts, tokens, selected)
    }

    fn group_expert_routes_inner(
        &mut self,
        indices: &DeviceVector,
        experts: usize,
        tokens: usize,
        selected: usize,
    ) -> Result<Option<GroupedExpertRoutes>> {
        if !route_geometry(experts, tokens, selected) {
            return Ok(None);
        }
        if indices.len != tokens * selected || self.projection_level {
            return Err(Error::Invalid("grouped expert route dimensions"));
        }
        indices.validate()?;
        let Some(pipelines) = expert_pipelines() else {
            return Ok(None);
        };
        let routes = tokens * selected;
        // Complete bounded allocation before the first dispatch. Each list
        // has route-count capacity, and the sum of jobs cannot exceed routes.
        let counts = self.scratch(experts)?;
        let lists = self.scratch(experts * routes)?;
        let jobs = self.scratch(1 + 5 * routes)?;
        let p = [experts as u32, routes as u32, selected as u32];
        self.expert_bind(&pipelines.lists, &[indices, &counts, &lists], &p, experts, 64);
        self.expert_bind(&pipelines.jobs, &[&counts, &lists, &jobs], &p, 1, 1);
        profile::record("native.glm.prefill.expert_group.routes", 0.0, routes);
        Ok(Some(GroupedExpertRoutes {
            counts,
            lists,
            jobs,
            experts,
            tokens,
            selected,
        }))
    }

    /// Original route order is restored in the output. For down projections,
    /// input_batched means one incoming row per original route slot.
    pub(crate) fn mapped_expert_grouped(
        &mut self,
        matrix: &MappedMatrix,
        format: MatrixFormat,
        shape: [usize; 3],
        routes: &GroupedExpertRoutes,
        input: &DeviceVector,
        input_batched: bool,
    ) -> Result<Option<DeviceVector>> {
        if !Self::expert_grouped_admitted(matrix, format, shape, routes.tokens, routes.selected)? {
            return Ok(None);
        }
        self.mapped_expert_grouped_inner(matrix, format, shape, routes, input, input_batched).map(Some)
    }

    fn mapped_expert_grouped_inner(
        &mut self,
        matrix: &MappedMatrix,
        format: MatrixFormat,
        shape: [usize; 3],
        routes: &GroupedExpertRoutes,
        input: &DeviceVector,
        input_batched: bool,
    ) -> Result<DeviceVector> {
        let [experts, rows, columns] = shape;
        if !projection_geometry(matrix, format, shape, routes.tokens, routes.selected)?
            || experts != routes.experts
            || self.projection_level
            || input.len
                != columns
                    * if input_batched {
                        routes.route_count()
                    } else {
                        routes.tokens
                    }
        {
            return Err(Error::Invalid("grouped expert projection dimensions"));
        }
        input.validate()?;
        let pipelines = expert_pipelines().ok_or(Error::Unsupported("grouped expert pipeline unavailable"))?;
        let output = self.scratch(rows * routes.route_count())?;
        let args = self.scratch(3)?;
        let bounds = self.iq_input_bounds(
            input,
            columns,
            if input_batched {
                routes.route_count()
            } else {
                routes.tokens
            },
        )?;
        self.expert_bind(
            &pipelines.indirect,
            &[&routes.jobs, &args],
            &[rows as u32, routes.route_count() as u32],
            1,
            1,
        );
        self.prepare_external_write(&output)?;
        self.order_dispatch();
        let pipeline = if format == MatrixFormat::Packed(17) {
            &pipelines.iq2
        } else {
            &pipelines.iq3
        };
        self.encoding.set_compute_pipeline_state(pipeline);
        self.encoding.set_buffer(Some(&matrix.codes.buffer), matrix.codes.offset, 0);
        for (index, vector) in [(1, input), (2, &output), (3, &routes.jobs), (4, &bounds)] {
            self.encoding.set_buffer(Some(&vector.buffer), vector.offset, index);
            self.resources.push(vector.buffer.clone());
        }
        let p = [
            rows as u32,
            columns as u32,
            experts as u32,
            u32::from(input_batched),
            routes.selected as u32,
            routes.route_count() as u32,
        ];
        self.encoding.set_bytes(NonNull::from(&p).cast(), std::mem::size_of_val(&p), 5);
        self.encoding.set_buffer(Some(&self.failure.buffer), self.failure.offset, 30);
        self.encoding.dispatch_threadgroups_indirect(
            &args.buffer,
            args.offset,
            MTLSize {
                width: 64,
                height: 1,
                depth: 1,
            },
        );
        self.resources.extend([
            matrix.codes.buffer.clone(),
            args.buffer.clone(),
            routes.counts.buffer.clone(),
            routes.lists.buffer.clone(),
        ]);
        self.check_finite(&output)?;
        profile::record("native.glm.prefill.expert_group.projections", 0.0, routes.route_count());
        Ok(output)
    }
}

#[cfg(test)]
#[path = "grouped_expert_prefill_tests.rs"]
mod tests;
