//! Essence and EssenceKey types for grouping identical sequences.

use crate::config::CANONICAL_SAMPLES;
use crate::types::{MutBag, MutList, Mutation};
use rustc_hash::FxHashMap;
use std::hash::{Hash, Hasher};

/// Unique identifier for an antibody sequence type.
///
/// For unpaired sequences, only heavy chain fields are used.
/// For paired sequences, light chain fields contribute to identity and scoring.
///
/// Two sequences with identical (junction, v_gene, j_gene, light_v_gene, light_j_gene)
/// are considered the same "type" and will be grouped into a single Essence.
#[derive(Clone, Debug)]
pub struct EssenceKey {
    /// CDR3/junction amino acid sequence (heavy chain)
    pub junction: String,
    /// V gene identifier - heavy chain (interned)
    pub v_gene: u8,
    /// J gene identifier - heavy chain (interned)
    pub j_gene: u8,
    /// V gene identifier - light chain (interned), None for unpaired
    pub light_v_gene: Option<u8>,
    /// J gene identifier - light chain (interned), None for unpaired
    pub light_j_gene: Option<u8>,
}

impl EssenceKey {
    /// Create a new essence key for unpaired sequences.
    pub fn new(junction: String, v_gene: u8, j_gene: u8) -> Self {
        Self {
            junction,
            v_gene,
            j_gene,
            light_v_gene: None,
            light_j_gene: None,
        }
    }

    /// Create a new essence key for paired sequences.
    pub fn new_paired(
        junction: String,
        v_gene: u8,
        j_gene: u8,
        light_v_gene: u8,
        light_j_gene: u8,
    ) -> Self {
        Self {
            junction,
            v_gene,
            j_gene,
            light_v_gene: Some(light_v_gene),
            light_j_gene: Some(light_j_gene),
        }
    }

    /// Check if this is a paired sequence.
    pub fn is_paired(&self) -> bool {
        self.light_v_gene.is_some()
    }
}

impl PartialEq for EssenceKey {
    fn eq(&self, other: &Self) -> bool {
        self.v_gene == other.v_gene
            && self.j_gene == other.j_gene
            && self.junction == other.junction
            && self.light_v_gene == other.light_v_gene
            && self.light_j_gene == other.light_j_gene
    }
}

impl Eq for EssenceKey {}

impl Hash for EssenceKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.junction.hash(state);
        self.v_gene.hash(state);
        self.j_gene.hash(state);
        self.light_v_gene.hash(state);
        self.light_j_gene.hash(state);
    }
}

impl PartialOrd for EssenceKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for EssenceKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.v_gene
            .cmp(&other.v_gene)
            .then_with(|| self.j_gene.cmp(&other.j_gene))
            .then_with(|| self.light_v_gene.cmp(&other.light_v_gene))
            .then_with(|| self.light_j_gene.cmp(&other.light_j_gene))
            .then_with(|| self.junction.cmp(&other.junction))
    }
}

/// Hash function for mutation lists (for deduplication).
fn hash_mutations(mutations: &[Mutation]) -> u64 {
    let mut h: u64 = 1;
    for &m in mutations {
        h = h.wrapping_mul(547129405631827).wrapping_add(m as u64);
    }
    h
}

/// Represents one or more sequences with identical (V, J, CDR3).
///
/// An Essence groups all sequences that share the same EssenceKey,
/// tracking the different mutation variants observed.
#[derive(Clone, Debug)]
pub struct Essence {
    /// Unique identifier for this essence
    pub key: EssenceKey,

    /// Canonical mutation list (used for fast dissimilarity)
    pub canonical_mutlist: MutList,

    /// All distinct mutation lists observed (hash -> MutList)
    mutlists: FxHashMap<u64, MutList>,

    /// Total number of sequences in this essence
    weight: u32,

    /// Accumulator for canonical mutation computation
    mutsum: MutBag,

    /// Assigned cluster ID (after clustering)
    pub cluster_id: Option<u32>,
}

impl Essence {
    /// Create a new essence with the given key.
    pub fn new(key: EssenceKey) -> Self {
        Self {
            key,
            canonical_mutlist: MutList::new(),
            mutlists: FxHashMap::default(),
            weight: 0,
            mutsum: MutBag::new(),
            cluster_id: None,
        }
    }

    /// Add a sequence with the given mutations.
    pub fn push_mutlist(&mut self, mutations: Vec<Mutation>) {
        let hash = hash_mutations(&mutations);

        if let Some(existing) = self.mutlists.get_mut(&hash) {
            existing.increment_weight();
        } else {
            let mutlist = MutList::from_vec(mutations);
            self.mutlists.insert(hash, mutlist);
        }

        // Update canonical mutation accumulator
        self.weight += 1;
        if self.weight > 1 && self.weight as usize <= CANONICAL_SAMPLES {
            if self.weight == 2 {
                // Add all existing mutation lists
                for ml in self.mutlists.values() {
                    self.mutsum.add(ml);
                }
            } else if let Some(ml) = self.mutlists.get(&hash) {
                self.mutsum.add(ml);
            }
        }
    }

    /// Get the total weight (number of sequences).
    pub fn weight(&self) -> u32 {
        self.weight
    }

    /// Get an iterator over all mutation list variants.
    pub fn mutlists(&self) -> impl Iterator<Item = &MutList> {
        self.mutlists.values()
    }

    /// Finalize parsing and compute canonical mutation list.
    ///
    /// Returns true if this essence qualifies as a cluster center
    /// (weight >= min_center_size).
    pub fn finalize(&mut self, min_center_size: usize) -> bool {
        let n = self.weight as usize;

        if n == 1 {
            // Single variant: use its mutation list directly
            if let Some(ml) = self.mutlists.values().next() {
                self.canonical_mutlist = ml.clone();
            }
        } else {
            // Multiple variants: use majority vote
            let p = n.min(CANONICAL_SAMPLES);
            let threshold = (p / 2) as u32;
            self.canonical_mutlist = self.mutsum.quantize(threshold);
        }

        n >= min_center_size
    }
}

/// Interning table for gene names.
///
/// Maps gene name strings to compact u8 identifiers.
#[derive(Default)]
pub struct GeneIntern {
    mapping: FxHashMap<String, u8>,
    values: Vec<String>,
}

impl GeneIntern {
    /// Create a new gene interning table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Intern a gene name, returning its compact ID.
    ///
    /// If the gene name has not been seen before, assigns a new ID.
    pub fn intern(&mut self, name: &str) -> u8 {
        if let Some(&id) = self.mapping.get(name) {
            return id;
        }

        let id = self.values.len() as u8;
        self.values.push(name.to_string());
        self.mapping.insert(name.to_string(), id);
        id
    }

    /// Look up the gene name for an ID.
    pub fn lookup(&self, id: u8) -> Option<&str> {
        self.values.get(id as usize).map(|s| s.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_essence_key_equality() {
        let k1 = EssenceKey::new("CARFDY".to_string(), 1, 2);
        let k2 = EssenceKey::new("CARFDY".to_string(), 1, 2);
        let k3 = EssenceKey::new("CARFDY".to_string(), 1, 3);

        assert_eq!(k1, k2);
        assert_ne!(k1, k3);
    }

    #[test]
    fn test_essence_key_ordering() {
        let k1 = EssenceKey::new("AAA".to_string(), 1, 1);
        let k2 = EssenceKey::new("AAA".to_string(), 1, 2);
        let k3 = EssenceKey::new("AAA".to_string(), 2, 1);

        assert!(k1 < k2); // Same v_gene, different j_gene
        assert!(k1 < k3); // Different v_gene
        assert!(k2 < k3); // v_gene takes precedence
    }

    #[test]
    fn test_essence_weight() {
        let key = EssenceKey::new("CARFDY".to_string(), 1, 2);
        let mut ess = Essence::new(key);

        assert_eq!(ess.weight(), 0);

        ess.push_mutlist(vec![1, 2, 3]);
        assert_eq!(ess.weight(), 1);

        ess.push_mutlist(vec![1, 2, 4]);
        assert_eq!(ess.weight(), 2);
    }

    #[test]
    fn test_essence_finalize_single() {
        let key = EssenceKey::new("CARFDY".to_string(), 1, 2);
        let mut ess = Essence::new(key);

        ess.push_mutlist(vec![1, 2, 3]);
        ess.finalize(1);

        assert_eq!(ess.canonical_mutlist.len(), 3);
    }

    #[test]
    fn test_gene_intern() {
        let mut intern = GeneIntern::new();

        let id1 = intern.intern("IGHV3-20");
        let id2 = intern.intern("IGHV1-2");
        let id3 = intern.intern("IGHV3-20"); // Same as id1

        assert_eq!(id1, id3);
        assert_ne!(id1, id2);
        assert_eq!(intern.lookup(id1), Some("IGHV3-20"));
        assert_eq!(intern.lookup(id2), Some("IGHV1-2"));
    }

    #[test]
    fn test_essence_key_paired() {
        let k1 = EssenceKey::new_paired("CARFDY".to_string(), 1, 2, 3, 4);
        let k2 = EssenceKey::new_paired("CARFDY".to_string(), 1, 2, 3, 4);
        let k3 = EssenceKey::new_paired("CARFDY".to_string(), 1, 2, 5, 4); // Different light V

        assert!(k1.is_paired());
        assert_eq!(k1, k2);
        assert_ne!(k1, k3);
    }

    #[test]
    fn test_essence_key_unpaired_not_equal_paired() {
        let unpaired = EssenceKey::new("CARFDY".to_string(), 1, 2);
        let paired = EssenceKey::new_paired("CARFDY".to_string(), 1, 2, 3, 4);

        assert!(!unpaired.is_paired());
        assert!(paired.is_paired());
        assert_ne!(unpaired, paired);
    }
}
