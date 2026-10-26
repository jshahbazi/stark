use crate::backends::common::gpu_types::trie::TrieNode;

pub struct BatchTopology<'a> {
    nodes: &'a [TrieNode],
    parents: Box<[i32]>,
    is_flat: bool,
    full_accept: bool,
    rotary_positions: Option<&'a [[u32; 3]]>,
}

impl<'a> BatchTopology<'a> {
    pub fn new(
        nodes: &'a [TrieNode],
        full_accept: bool,
    ) -> Self {
        let mut stack: Vec<usize> = Vec::new();

        let mut parents = Box::new_uninit_slice(nodes.len());
        let mut is_flat = true;

        for (node, index) in nodes.iter().zip(0u32..) {
            stack.truncate(node.height as usize);
            parents[index as usize].write(stack.last().map(|i| *i as i32).unwrap_or(-1));
            stack.push(index as usize);

            is_flat &= node.height == index;
        }
        let parents = unsafe { parents.assume_init() };

        assert!(is_flat || !full_accept, "full_accept requires trie to be flat");

        Self {
            nodes,
            parents,
            is_flat,
            full_accept,
            rotary_positions: None,
        }
    }

    /// Spatial rotary coordinates never replace trie heights or cache offsets.
    pub(crate) fn with_rotary_positions(
        nodes: &'a [TrieNode],
        full_accept: bool,
        positions: &'a [[u32; 3]],
    ) -> Result<Self, crate::backends::common::InvalidInput> {
        if nodes.len() != positions.len()
            || !full_accept
            || nodes.iter().enumerate().any(|(i, node)| node.height as usize != i)
        {
            return Err(crate::backends::common::InvalidInput(
                "multimodal positions require an equally sized flat accepted batch",
            ));
        }
        let mut batch = Self::new(nodes, full_accept);
        batch.rotary_positions = Some(positions);
        Ok(batch)
    }

    pub(crate) fn rotary_positions(&self) -> Option<&[[u32; 3]]> {
        self.rotary_positions
    }

    pub fn nodes(&self) -> &'a [TrieNode] {
        self.nodes
    }

    pub fn size(&self) -> u32 {
        self.nodes.len() as u32
    }

    pub fn is_flat(&self) -> bool {
        self.is_flat
    }

    pub fn full_accept(&self) -> bool {
        self.full_accept
    }

    pub fn heights(&self) -> impl Iterator<Item = u32> + 'a {
        self.nodes.iter().map(|trie| trie.height)
    }

    pub fn parents(&self) -> &[i32] {
        &self.parents
    }
}
