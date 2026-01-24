//! Core data types for mutation representation.

use std::cmp::Ordering;

/// A mutation encoded as a 16-bit value.
///
/// Format: `(position << 4) | mutation_code`
/// - Position: top 12 bits (0-4095)
/// - Mutation code: bottom 4 bits (encodes ref/alt bases)
///
/// The mutation code uses a simple encoding:
/// - `m1 = (ref_base / 2) % 4`
/// - `m2 = (alt_base / 2) % 4`
/// - `code = ((m1 - m2) % 4) * 4 + m2`
pub type Mutation = u16;

/// Bits used for mutation type encoding
pub const MUTTYPE_BITS: u32 = 4;

/// Encode a mutation from position and ref/alt bases.
///
/// # Arguments
/// * `position` - Position in the sequence (0-4095)
/// * `ref_base` - Reference nucleotide character
/// * `alt_base` - Alternate nucleotide character
///
/// # Returns
/// Encoded mutation value
pub fn encode_mutation(position: u16, ref_base: char, alt_base: char) -> Mutation {
    let pos = position % 4096; // MAX_MUTATION_LOC

    // Encode bases using ASCII value division (matches C++)
    let m1 = (ref_base as u8 / 2) % 4;
    let m2 = (alt_base as u8 / 2) % 4;

    // Handle deletion (ref = '-')
    let m1 = if ref_base == '-' { 0 } else { m1.wrapping_sub(m2) % 4 };

    let mut_code = (m1 % 4) * 4 + m2;
    (pos << MUTTYPE_BITS) | (mut_code as u16)
}

/// Extract the position from an encoded mutation.
pub fn mutation_position(mutation: Mutation) -> u16 {
    mutation >> MUTTYPE_BITS
}

/// Extract the mutation code from an encoded mutation.
pub fn mutation_code(mutation: Mutation) -> u8 {
    (mutation & 0xF) as u8
}

/// A sorted list of mutations associated with a sequence variant.
#[derive(Clone, Debug, Default)]
pub struct MutList {
    /// Sorted vector of mutations
    pub data: Vec<Mutation>,
    /// Number of sequences sharing this exact mutation list
    pub weight: u32,
}

impl MutList {
    /// Create a new empty mutation list.
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            weight: 0,
        }
    }

    /// Create a mutation list from a vector of mutations.
    ///
    /// The mutations will be sorted.
    pub fn from_vec(mut mutations: Vec<Mutation>) -> Self {
        mutations.sort_unstable();
        Self {
            data: mutations,
            weight: 1,
        }
    }

    /// Check if the mutation list is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get the number of mutations.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Get the weight (number of sequences with this exact mutation list).
    pub fn weight(&self) -> u32 {
        self.weight
    }

    /// Increment the weight.
    pub fn increment_weight(&mut self) {
        self.weight += 1;
    }
}

/// Count the number of shared mutations between two sorted mutation lists.
///
/// Uses a two-pointer merge algorithm for O(n + m) complexity.
pub fn num_shared_mutations(m1: &MutList, m2: &MutList) -> usize {
    let mut count = 0;
    let mut p1 = 0;
    let mut p2 = 0;

    while p1 < m1.data.len() && p2 < m2.data.len() {
        match m1.data[p1].cmp(&m2.data[p2]) {
            Ordering::Less => p1 += 1,
            Ordering::Greater => p2 += 1,
            Ordering::Equal => {
                count += 1;
                p1 += 1;
                p2 += 1;
            }
        }
    }

    count
}

/// Accumulator for mutations with occurrence counts.
///
/// Used to compute the canonical mutation list from multiple variants.
#[derive(Clone, Debug, Default)]
pub struct MutBag {
    /// Sorted vector of (mutation, count) pairs
    count: Vec<(Mutation, u32)>,
}

impl MutBag {
    /// Create a new empty mutation bag.
    pub fn new() -> Self {
        Self { count: Vec::new() }
    }

    /// Add a mutation list to the bag, incrementing counts.
    pub fn add(&mut self, mutlist: &MutList) {
        let initial_size = self.count.len();
        let mut p1 = 0;
        let mut p2 = 0;

        // Merge existing counts with new mutations
        while p1 < initial_size && p2 < mutlist.data.len() {
            match self.count[p1].0.cmp(&mutlist.data[p2]) {
                Ordering::Less => p1 += 1,
                Ordering::Greater => {
                    self.count.push((mutlist.data[p2], 1));
                    p2 += 1;
                }
                Ordering::Equal => {
                    self.count[p1].1 += 1;
                    p1 += 1;
                    p2 += 1;
                }
            }
        }

        // Add remaining new mutations
        while p2 < mutlist.data.len() {
            self.count.push((mutlist.data[p2], 1));
            p2 += 1;
        }

        // Re-sort to maintain sorted order
        self.count[initial_size..].sort_unstable_by_key(|&(m, _)| m);
        if initial_size > 0 && self.count.len() > initial_size {
            // Merge the two sorted halves
            let mut merged = Vec::with_capacity(self.count.len());
            let (left, right) = self.count.split_at(initial_size);
            let mut li = 0;
            let mut ri = 0;

            while li < left.len() && ri < right.len() {
                if left[li].0 <= right[ri].0 {
                    merged.push(left[li]);
                    li += 1;
                } else {
                    merged.push(right[ri]);
                    ri += 1;
                }
            }
            merged.extend_from_slice(&left[li..]);
            merged.extend_from_slice(&right[ri..]);
            self.count = merged;
        }
    }

    /// Extract mutations that appear at least `threshold` times.
    pub fn quantize(&self, threshold: u32) -> MutList {
        let data: Vec<Mutation> = self
            .count
            .iter()
            .filter(|&&(_, c)| c >= threshold)
            .map(|&(m, _)| m)
            .collect();

        MutList { data, weight: 0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_mutation() {
        // Test basic encoding
        let m = encode_mutation(100, 'G', 'T');
        assert_eq!(mutation_position(m), 100);

        // Test position wrapping
        let m2 = encode_mutation(5000, 'A', 'C');
        assert_eq!(mutation_position(m2), 5000 % 4096);
    }

    #[test]
    fn test_mutlist_from_vec() {
        let mutations = vec![300u16 << 4, 100u16 << 4, 200u16 << 4];
        let mutlist = MutList::from_vec(mutations);

        // Should be sorted
        assert_eq!(mutlist.len(), 3);
        assert!(mutlist.data[0] < mutlist.data[1]);
        assert!(mutlist.data[1] < mutlist.data[2]);
    }

    #[test]
    fn test_num_shared_mutations() {
        let m1 = MutList::from_vec(vec![1, 2, 3, 5, 7]);
        let m2 = MutList::from_vec(vec![2, 3, 4, 7, 8]);

        assert_eq!(num_shared_mutations(&m1, &m2), 3); // 2, 3, 7
    }

    #[test]
    fn test_num_shared_mutations_empty() {
        let m1 = MutList::new();
        let m2 = MutList::from_vec(vec![1, 2, 3]);

        assert_eq!(num_shared_mutations(&m1, &m2), 0);
    }

    #[test]
    fn test_mutbag_quantize() {
        let mut bag = MutBag::new();

        // Add same mutation list 3 times
        let m1 = MutList::from_vec(vec![1, 2, 3]);
        bag.add(&m1);
        bag.add(&m1);
        bag.add(&m1);

        // Add different mutations once
        let m2 = MutList::from_vec(vec![4, 5]);
        bag.add(&m2);

        // Quantize with threshold 2 should only include 1, 2, 3
        let result = bag.quantize(2);
        assert_eq!(result.len(), 3);
        assert!(result.data.contains(&1));
        assert!(result.data.contains(&2));
        assert!(result.data.contains(&3));
    }
}
