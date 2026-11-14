//! Allocator-rounded request-owned image workspace inventories.
//!
//! Repeated tower blocks have identical allocation shapes. PoolAllocation pages
//! return to their size class at scope drop, and Metal inserts dispatch barriers
//! before reusing their storage. Counting every allocation in one block, plus
//! every edge-stage allocation, bounds the pool's retained size-class high-water
//! without multiplying scratch by depth. Command constants are counted separately.

use super::tower::VisionTowerConfig;

const MIN_BUFFER: usize = 16 * 1024;
const CONSTANT_PAGE: usize = 256 * 1024;
const MAX_SCRATCH: usize = 2 * 1024 * 1024 * 1024;
const MARGIN: usize = 32 * 1024 * 1024;

#[derive(Default)]
struct Inventory {
    scratch: usize,
    constants: usize,
    constant_page: usize,
    constant_offset: usize,
    host: usize,
}

fn bytes(
    rows: u32,
    width: u32,
    element: usize,
) -> Option<usize> {
    (rows as usize).checked_mul(width as usize)?.checked_mul(element)
}

impl Inventory {
    fn scratch(
        &mut self,
        rows: u32,
        width: u32,
        element: usize,
    ) -> Option<()> {
        let requested = bytes(rows, width, element)?;
        if requested == 0 || requested > MAX_SCRATCH {
            return None;
        }
        // PoolAllocator's power-of-two classes, then Metal's 16 KiB minimum.
        self.scratch = self.scratch.checked_add(requested.checked_next_power_of_two()?.max(MIN_BUFFER))?;
        Some(())
    }

    fn constant(
        &mut self,
        rows: u32,
        width: u32,
        element: usize,
    ) -> Option<()> {
        let requested = bytes(rows, width, element)?;
        if requested == 0 {
            return None;
        }
        // BumpAllocator's actual page/alignment contract: small scalar constants
        // share a page and persist until their command completes.
        let alignment = requested.min(64).checked_next_power_of_two()?;
        let aligned = self.constant_offset.checked_add(alignment - 1)? & !(alignment - 1);
        let end = aligned.checked_add(requested)?;
        if end <= self.constant_page {
            self.constant_offset = end;
        } else {
            self.constant_page = requested.max(CONSTANT_PAGE);
            self.constant_offset = requested;
            self.constants = self.constants.checked_add(self.constant_page)?;
        }
        Some(())
    }

    fn host(
        &mut self,
        rows: u32,
        width: u32,
        element: usize,
    ) -> Option<()> {
        self.host = self.host.checked_add(bytes(rows, width, element)?)?;
        Some(())
    }

    fn linear(
        &mut self,
        rows: u32,
        input: u32,
        output: u32,
        clipped: bool,
    ) -> Option<()> {
        if clipped {
            self.scratch(rows, input, 2)?;
        }
        self.scratch(rows, output, 4)?;
        self.scratch(rows, output, 2)?;
        if clipped {
            self.scratch(rows, output, 2)?;
        }
        Some(())
    }

    fn eager_attention(
        &mut self,
        rows: u32,
        width: u32,
        head: u32,
    ) -> Option<()> {
        self.scratch(rows, width, 2)?;
        self.scratch(rows, rows, 4)?;
        self.scratch(rows, rows, 2)?;
        self.scratch(rows, head, 4)?;
        self.scratch(rows, head, 2)
    }

    fn total(
        &self,
        depth: u32,
    ) -> Option<usize> {
        // Payload accounting, not an RSS guarantee. Command/argument metadata
        // receives explicit fixed and depth-dependent allocation margin.
        self.scratch
            .checked_add(self.constants)?
            .checked_add(self.host)?
            .checked_add(MARGIN)?
            .checked_add((depth as usize).checked_mul(1024 * 1024)?)
    }
}

pub(super) fn qwen_workspace_bytes(
    rows: u32,
    config: &VisionTowerConfig,
) -> Option<usize> {
    qwen_inventory(rows, config)?.total(config.depth)
}

#[cfg(all(test, backend = "metal"))]
pub(super) fn qwen_device_allocation_bounds(
    rows: u32,
    config: &VisionTowerConfig,
) -> Option<(usize, usize)> {
    let inventory = qwen_inventory(rows, config)?;
    Some((inventory.constants, inventory.scratch))
}

#[cfg(all(test, backend = "metal"))]
pub(super) fn constant_budget_for_requests(
    uploads: &[u32],
    scalar_count: usize,
) -> Option<usize> {
    let mut inventory = Inventory::default();
    for &size in uploads {
        inventory.constant(1, size, 1)?;
    }
    for _ in 0..scalar_count {
        inventory.constant(1, 1, 2)?;
    }
    Some(inventory.constants)
}

fn qwen_inventory(
    rows: u32,
    config: &VisionTowerConfig,
) -> Option<Inventory> {
    if rows == 0
        || rows > 4096
        || config.depth == 0
        || config.num_heads == 0
        || !config.hidden_size.is_multiple_of(config.num_heads)
    {
        return None;
    }
    let width = config.hidden_size;
    let inner = config.intermediate_size;
    let head = width / config.num_heads;
    let merge = config.spatial_merge_size.checked_mul(config.spatial_merge_size)?;
    if merge == 0 || !rows.is_multiple_of(merge) {
        return None;
    }
    let merged_rows = rows / merge;
    let merged_width = width.checked_mul(merge)?;
    let patch = config
        .in_channels
        .checked_mul(config.temporal_patch_size)?
        .checked_mul(config.patch_size)?
        .checked_mul(config.patch_size)?;
    let mut inventory = Inventory::default();
    // prepare_hidden: constant uploads, CPU staging and initial projection.
    inventory.constant(rows, patch, 2)?;
    inventory.constant(rows, width, 2)?;
    inventory.constant(rows, head, 4)?;
    inventory.constant(rows, head, 4)?;
    inventory.host(rows, patch, 2)?;
    inventory.host(rows, width, 2)?;
    inventory.host(rows, head, 8)?;
    inventory.host(rows, 2, 4)?;
    inventory.linear(rows, patch, width, false)?;
    inventory.scratch(rows, width, 2)?;
    // One complete encode_block: all RawLinear F32/BF16 boundaries included.
    inventory.scratch(rows, width, 2)?;
    inventory.linear(rows, width, width.checked_mul(3)?, false)?;
    for _ in 0..3 {
        inventory.scratch(rows, width, 2)?;
    }
    inventory.eager_attention(rows, width, head)?;
    inventory.linear(rows, width, width, false)?;
    inventory.scratch(rows, width, 2)?;
    inventory.scratch(rows, width, 2)?;
    inventory.linear(rows, width, inner, false)?;
    inventory.linear(rows, inner, width, false)?;
    inventory.scratch(rows, width, 2)?;
    for _ in 0..config.depth {
        inventory.constant(1, 1, 2)?;
    }
    // Merger and old zero features / device readback / new CPU features.
    inventory.scratch(rows, width, 2)?;
    inventory.linear(merged_rows, merged_width, merged_width, false)?;
    inventory.linear(merged_rows, merged_width, config.out_hidden_size, false)?;
    inventory.host(merged_rows, config.out_hidden_size, 6)?;
    Some(inventory)
}

pub(super) fn gemma3_workspace_bytes(output: u32) -> Option<usize> {
    if !matches!(output, 2560 | 3840 | 5376) {
        return None;
    }
    let (rows, width, inner, head, tokens, patch) = (4096, 1152, 4304, 72, 256, 588);
    let mut inventory = Inventory::default();
    inventory.host(rows, patch, 2)?;
    inventory.host(rows, width, 2)?; // persistent hidden global buffer
    // Separate initial/block/final commands have independent pools. Summing
    // their inventories also covers upstream pages cached between submissions.
    inventory.constant(rows, patch, 2)?;
    inventory.linear(rows, patch, width, false)?;
    inventory.scratch(rows, width, 2)?;
    inventory.scratch(rows, width, 2)?; // block input copy
    inventory.scratch(rows, width, 2)?;
    inventory.linear(rows, width, 3 * width, false)?;
    for _ in 0..3 {
        inventory.scratch(rows, width, 2)?;
    }
    inventory.constant(rows, head, 4)?;
    inventory.constant(rows, head, 4)?;
    inventory.host(rows, head, 8)?;
    inventory.eager_attention(rows, width, head)?;
    inventory.constant(1, 1, 2)?;
    inventory.linear(rows, width, width, false)?;
    inventory.scratch(rows, width, 2)?;
    inventory.scratch(rows, width, 2)?;
    inventory.linear(rows, width, inner, false)?;
    inventory.linear(rows, inner, width, false)?;
    inventory.scratch(rows, width, 2)?;
    inventory.scratch(rows, width, 2)?; // finish input copy
    inventory.scratch(rows, width, 2)?;
    inventory.scratch(tokens, width, 2)?;
    inventory.scratch(tokens, width, 2)?;
    inventory.linear(tokens, width, output, false)?;
    inventory.host(tokens, output, 6)?;
    inventory.total(1)
}

pub(super) fn gemma4_workspace_bytes(
    rows: u32,
    output: u32,
) -> Option<usize> {
    if rows == 0 || rows > 2520 || !rows.is_multiple_of(9) {
        return None;
    }
    // The two source-admitted tower layouts selected by Gemma4VisionTower::new.
    let (width, inner, heads, head, padded, depth, clipped) = match output {
        1536 | 2560 => (768u32, 3072, 12u32, 64, 64, 16, true),
        2816 | 5376 => (1152, 4304, 16, 72, 128, 27, false),
        _ => return None,
    };
    let tokens = rows / 9;
    let mut inventory = Inventory::default();
    inventory.host(rows, 768, 2)?;
    inventory.host(rows, width, 2)?;
    inventory.host(rows, head, 4)?;
    inventory.constant(rows, 768, 2)?;
    inventory.constant(rows, width, 2)?;
    inventory.constant(rows, head, 2)?;
    inventory.constant(rows, head, 2)?;
    inventory.constant(1, width, 2)?;
    inventory.scratch(rows, 768, 2)?;
    inventory.linear(rows, 768, width, false)?;
    inventory.scratch(rows, width, 2)?;
    for _ in 0..3 {
        inventory.linear(rows, width, width, clipped)?;
    }
    for _ in 0..3 {
        inventory.scratch(rows, width, 2)?;
    }
    for _ in 0..3 {
        inventory.scratch(rows, heads.checked_mul(padded)?, 2)?;
    }
    if clipped {
        inventory.scratch(rows, width, 2)?;
    } else {
        // Four fixed FlashAttention buffers are reused across key tiles.
        inventory.scratch(rows, heads.checked_mul(512)?, 4)?;
        inventory.scratch(rows, heads.checked_mul(512)?, 2)?;
        inventory.scratch(rows, heads.checked_mul(3)?, 4)?;
        inventory.scratch(rows, heads.checked_mul(72)?, 4)?;
        inventory.scratch(rows, width, 2)?;
    }
    inventory.linear(rows, width, width, clipped)?;
    inventory.scratch(rows, width, 2)?;
    inventory.scratch(rows, width, 2)?;
    inventory.linear(rows, width, inner, clipped)?;
    inventory.linear(rows, width, inner, clipped)?;
    inventory.scratch(rows, inner, 2)?;
    inventory.linear(rows, inner, width, clipped)?;
    inventory.scratch(rows, width, 2)?;
    inventory.scratch(tokens, width, 2)?;
    inventory.scratch(tokens, width, 2)?;
    inventory.linear(tokens, width, output, false)?;
    inventory.host(tokens, output, 6)?;
    inventory.total(depth)
}

pub(super) fn gemma4_unified_workspace_bytes(rows: u32) -> Option<usize> {
    if rows == 0 || rows > 280 {
        return None;
    }
    let (patch, width) = (6912, 3840);
    let mut inventory = Inventory::default();
    inventory.host(rows, patch, 2)?;
    inventory.constant(rows, patch, 2)?;
    inventory.scratch(rows, patch, 2)?;
    inventory.scratch(rows, patch, 2)?;
    inventory.linear(rows, patch, width, false)?;
    for _ in 0..4 {
        inventory.scratch(rows, width, 2)?;
    }
    inventory.linear(rows, width, width, false)?;
    inventory.host(rows, width, 6)?;
    inventory.total(0)
}

#[cfg(test)]
mod tests {
    use stark_engine_macros::stark_test;

    use super::*;

    #[stark_test]
    fn tower_workspace_inventory_bounds_admitted_profiles_and_rejects_invalid_geometry() {
        let small = gemma4_workspace_bytes(2304, 1536).unwrap();
        assert!(small < 1024 * 1024 * 1024, "small tower admission must not sum all layers");
        assert!(gemma4_workspace_bytes(2520, 5376).unwrap() > small);
        assert!(gemma3_workspace_bytes(2560).unwrap() > small);
        assert!(gemma4_unified_workspace_bytes(280).unwrap() < small);
        assert!(gemma4_workspace_bytes(2521, 1536).is_none());
        assert!(gemma4_workspace_bytes(2304, 1).is_none());
        assert!(gemma3_workspace_bytes(1).is_none());
        assert!(gemma4_unified_workspace_bytes(0).is_none());
        let mut inventory = Inventory::default();
        assert!(inventory.scratch(u32::MAX, u32::MAX, 4).is_none());
        println!(
            "{}",
            serde_json::json!({
                "scope":"vision-workspace-inventory",
                "gemma4_e2b_square_bytes":small,
                "gemma4_larger_max_bytes":gemma4_workspace_bytes(2520,5376).unwrap(),
                "gemma3_4b_bytes":gemma3_workspace_bytes(2560).unwrap(),
                "gemma4_unified_max_bytes":gemma4_unified_workspace_bytes(280).unwrap(),
            })
        );
    }
}
