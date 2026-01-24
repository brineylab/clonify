//! Distance and dissimilarity metrics.

use crate::config::ClusterParams;
use crate::essence::Essence;
use crate::types::num_shared_mutations;

/// Compute Hamming distance between two equal-length strings.
///
/// Counts the number of positions where the characters differ.
pub fn hamming_distance(s1: &str, s2: &str) -> usize {
    debug_assert_eq!(s1.len(), s2.len());
    s1.bytes().zip(s2.bytes()).filter(|(a, b)| a != b).count()
}

/// Compute Levenshtein (edit) distance between two strings.
///
/// Uses dynamic programming with O(n*m) time and O(min(n,m)) space.
pub fn levenshtein_distance(s1: &str, s2: &str) -> usize {
    let s1_bytes = s1.as_bytes();
    let s2_bytes = s2.as_bytes();
    let n = s1_bytes.len();
    let m = s2_bytes.len();

    // Optimize: make s1 the shorter string for space efficiency
    if n > m {
        return levenshtein_distance(s2, s1);
    }

    // Use two rows instead of full matrix
    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr: Vec<usize> = vec![0; n + 1];

    for j in 1..=m {
        curr[0] = j;
        for i in 1..=n {
            let cost = if s1_bytes[i - 1] == s2_bytes[j - 1] {
                0
            } else {
                1
            };
            curr[i] = (prev[i] + 1) // deletion
                .min(curr[i - 1] + 1) // insertion
                .min(prev[i - 1] + cost); // substitution
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[n]
}

/// Compute the edit distance between two CDR3 sequences.
///
/// Uses Hamming distance for equal-length sequences, Levenshtein otherwise.
pub fn get_edit_distance(s1: &str, s2: &str) -> usize {
    if s1.len() == s2.len() {
        hamming_distance(s1, s2)
    } else {
        levenshtein_distance(s1, s2)
    }
}

/// Compare V genes, returning penalty for mismatch.
///
/// Returns 8 if genes differ, 0 if same.
pub fn v_compare(e1: &Essence, e2: &Essence) -> i32 {
    if e1.key.v_gene != e2.key.v_gene {
        8
    } else {
        0
    }
}

/// Compare J genes, returning penalty for mismatch.
///
/// Returns 8 if genes differ, 0 if same.
pub fn j_compare(e1: &Essence, e2: &Essence) -> i32 {
    if e1.key.j_gene != e2.key.j_gene {
        8
    } else {
        0
    }
}

/// Compare light chain V genes, returning penalty for mismatch.
///
/// Returns 8 if genes differ, 0 if same or if either is None (unpaired).
pub fn light_v_compare(e1: &Essence, e2: &Essence) -> i32 {
    match (e1.key.light_v_gene, e2.key.light_v_gene) {
        (Some(v1), Some(v2)) if v1 != v2 => 8,
        _ => 0,
    }
}

/// Compare light chain J genes, returning penalty for mismatch.
///
/// Returns 8 if genes differ, 0 if same or if either is None (unpaired).
pub fn light_j_compare(e1: &Essence, e2: &Essence) -> i32 {
    match (e1.key.light_j_gene, e2.key.light_j_gene) {
        (Some(j1), Some(j2)) if j1 != j2 => 8,
        _ => 0,
    }
}

/// Compute fast dissimilarity using canonical mutation lists.
///
/// This is used for megacluster center matching where speed is critical.
///
/// Formula:
/// ```text
/// dissimilarity = (LD + v_penalty + j_penalty + light_v_penalty + light_j_penalty - mut_bonus + len_penalty) / edit_length
/// ```
pub fn fast_dissimilarity(e1: &Essence, e2: &Essence, ld: usize, params: &ClusterParams) -> f64 {
    let v_penalty = v_compare(e1, e2);
    let j_penalty = j_compare(e1, e2);
    let light_v_penalty = light_v_compare(e1, e2);
    let light_j_penalty = light_j_compare(e1, e2);

    let mut_bonus =
        params.mut_value * num_shared_mutations(&e1.canonical_mutlist, &e2.canonical_mutlist) as f64;

    let basic = ld as f64
        + v_penalty as f64
        + j_penalty as f64
        + light_v_penalty as f64
        + light_j_penalty as f64;
    let with_bonus = (basic - mut_bonus).max(params.epsilon);

    let len_penalty = (e1.key.junction.len() as i32 - e2.key.junction.len() as i32).unsigned_abs()
        as f64
        * params.len_penalty as f64;

    let edit_length = e1.key.junction.len().min(e2.key.junction.len()) as f64;

    (with_bonus + len_penalty) / edit_length
}

/// Compute mutation bonus considering all mutation variants.
///
/// This is the weighted average of shared mutations across all variant pairs.
fn mut_bonus(e1: &Essence, e2: &Essence, ceiling: f64, params: &ClusterParams) -> f64 {
    let n1 = e1.weight() as f64;
    let n2 = e2.weight() as f64;
    let total_weight = n1 * n2;

    let mut sum = 0.0;

    for ml1 in e1.mutlists() {
        let w1 = ml1.weight() as f64;
        for ml2 in e2.mutlists() {
            let w2 = ml2.weight() as f64;
            let shared = num_shared_mutations(ml1, ml2) as f64;
            sum += shared.min(ceiling) * w1 * w2;
        }
    }

    params.mut_value * (sum / total_weight)
}

/// Compute full dissimilarity considering all mutation variants.
///
/// This is used in hierarchical clustering within megaclusters
/// where accuracy is more important than speed.
pub fn full_dissimilarity(e1: &Essence, e2: &Essence, ld: usize, params: &ClusterParams) -> f64 {
    let v_penalty = v_compare(e1, e2);
    let j_penalty = j_compare(e1, e2);
    let light_v_penalty = light_v_compare(e1, e2);
    let light_j_penalty = light_j_compare(e1, e2);

    let basic = ld as i32 + v_penalty + j_penalty + light_v_penalty + light_j_penalty;

    // Ceiling prevents over-rewarding high mutation counts
    let ceiling = (basic as f64) / params.mut_value - params.epsilon / params.mut_value;
    let mutation_bonus = mut_bonus(e1, e2, ceiling, params);

    let with_bonus = basic as f64 - mutation_bonus;

    let len_penalty = (e1.key.junction.len() as i32 - e2.key.junction.len() as i32).unsigned_abs()
        as f64
        * params.len_penalty as f64;

    let edit_length = e1.key.junction.len().min(e2.key.junction.len()) as f64;

    (with_bonus + len_penalty) / edit_length
}

/// Compute dissimilarity for megacluster center merging.
///
/// Returns the cutoff early if length penalty alone exceeds threshold.
pub fn merging_dissimilarity(e1: &Essence, e2: &Essence, cutoff: f64, params: &ClusterParams) -> f64 {
    let len_penalty = (e1.key.junction.len() as i32 - e2.key.junction.len() as i32).unsigned_abs()
        as f64
        * params.len_penalty as f64;

    let edit_length = e1.key.junction.len().min(e2.key.junction.len()) as f64;

    // Early exit if length penalty alone exceeds threshold
    if len_penalty >= edit_length * cutoff {
        return cutoff;
    }

    let ld = get_edit_distance(&e1.key.junction, &e2.key.junction);
    fast_dissimilarity(e1, e2, ld, params)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::essence::EssenceKey;

    #[test]
    fn test_hamming_distance() {
        assert_eq!(hamming_distance("CARFDY", "CARFDY"), 0);
        assert_eq!(hamming_distance("CARFDY", "CARFDA"), 1);
        assert_eq!(hamming_distance("CARFDY", "XARFDA"), 2);
        assert_eq!(hamming_distance("ABCDEF", "FEDCBA"), 6);
    }

    #[test]
    fn test_levenshtein_distance() {
        assert_eq!(levenshtein_distance("", ""), 0);
        assert_eq!(levenshtein_distance("abc", ""), 3);
        assert_eq!(levenshtein_distance("", "abc"), 3);
        assert_eq!(levenshtein_distance("abc", "abc"), 0);
        assert_eq!(levenshtein_distance("abc", "abcd"), 1); // insertion
        assert_eq!(levenshtein_distance("abcd", "abc"), 1); // deletion
        assert_eq!(levenshtein_distance("abc", "adc"), 1); // substitution
        assert_eq!(levenshtein_distance("kitten", "sitting"), 3);
    }

    #[test]
    fn test_get_edit_distance() {
        // Same length: uses Hamming
        assert_eq!(get_edit_distance("CARFDY", "CARFDA"), 1);

        // Different length: uses Levenshtein
        assert_eq!(get_edit_distance("CARFDY", "CARFDYW"), 1);
    }

    #[test]
    fn test_v_j_compare() {
        let key1 = EssenceKey::new("CARFDY".to_string(), 1, 2);
        let key2 = EssenceKey::new("CARFDY".to_string(), 1, 2);
        let key3 = EssenceKey::new("CARFDY".to_string(), 2, 2);
        let key4 = EssenceKey::new("CARFDY".to_string(), 1, 3);

        let e1 = Essence::new(key1);
        let e2 = Essence::new(key2);
        let e3 = Essence::new(key3);
        let e4 = Essence::new(key4);

        assert_eq!(v_compare(&e1, &e2), 0);
        assert_eq!(v_compare(&e1, &e3), 8);
        assert_eq!(j_compare(&e1, &e2), 0);
        assert_eq!(j_compare(&e1, &e4), 8);
    }

    #[test]
    fn test_fast_dissimilarity_identical() {
        let params = ClusterParams::default();

        let key1 = EssenceKey::new("CARFDY".to_string(), 1, 2);
        let key2 = EssenceKey::new("CARFDY".to_string(), 1, 2);

        let mut e1 = Essence::new(key1);
        let mut e2 = Essence::new(key2);

        e1.push_mutlist(vec![1, 2, 3]);
        e2.push_mutlist(vec![1, 2, 3]);
        e1.finalize(1);
        e2.finalize(1);

        let dist = fast_dissimilarity(&e1, &e2, 0, &params);
        // Should be close to epsilon (minimum) due to identical mutations
        assert!(dist < 0.1);
    }

    #[test]
    fn test_fast_dissimilarity_different_genes() {
        let params = ClusterParams::default();

        let key1 = EssenceKey::new("CARFDY".to_string(), 1, 2);
        let key2 = EssenceKey::new("CARFDY".to_string(), 2, 3); // Different V and J

        let mut e1 = Essence::new(key1);
        let mut e2 = Essence::new(key2);

        e1.push_mutlist(vec![]);
        e2.push_mutlist(vec![]);
        e1.finalize(1);
        e2.finalize(1);

        let dist = fast_dissimilarity(&e1, &e2, 0, &params);
        // Should be high due to V and J gene penalties (8 + 8 = 16)
        // Dissimilarity = (0 + 8 + 8) / 6 = 2.67
        assert!(dist > 2.0);
    }

    #[test]
    fn test_light_chain_compare() {
        // Unpaired sequences - no light chain penalty
        let key1 = EssenceKey::new("CARFDY".to_string(), 1, 2);
        let key2 = EssenceKey::new("CARFDY".to_string(), 1, 2);
        let e1 = Essence::new(key1);
        let e2 = Essence::new(key2);

        assert_eq!(light_v_compare(&e1, &e2), 0);
        assert_eq!(light_j_compare(&e1, &e2), 0);

        // Paired sequences - same light chain
        let key3 = EssenceKey::new_paired("CARFDY".to_string(), 1, 2, 3, 4);
        let key4 = EssenceKey::new_paired("CARFDY".to_string(), 1, 2, 3, 4);
        let e3 = Essence::new(key3);
        let e4 = Essence::new(key4);

        assert_eq!(light_v_compare(&e3, &e4), 0);
        assert_eq!(light_j_compare(&e3, &e4), 0);

        // Paired sequences - different light V gene
        let key5 = EssenceKey::new_paired("CARFDY".to_string(), 1, 2, 5, 4);
        let e5 = Essence::new(key5);

        assert_eq!(light_v_compare(&e3, &e5), 8);
        assert_eq!(light_j_compare(&e3, &e5), 0);

        // Paired sequences - different light J gene
        let key6 = EssenceKey::new_paired("CARFDY".to_string(), 1, 2, 3, 6);
        let e6 = Essence::new(key6);

        assert_eq!(light_v_compare(&e3, &e6), 0);
        assert_eq!(light_j_compare(&e3, &e6), 8);

        // Mixed paired/unpaired - no penalty
        assert_eq!(light_v_compare(&e1, &e3), 0);
        assert_eq!(light_j_compare(&e1, &e3), 0);
    }

    #[test]
    fn test_fast_dissimilarity_paired_different_light() {
        let params = ClusterParams::default();

        // Same heavy chain, different light chain
        let key1 = EssenceKey::new_paired("CARFDY".to_string(), 1, 2, 3, 4);
        let key2 = EssenceKey::new_paired("CARFDY".to_string(), 1, 2, 5, 6); // Different light V and J

        let mut e1 = Essence::new(key1);
        let mut e2 = Essence::new(key2);

        e1.push_mutlist(vec![]);
        e2.push_mutlist(vec![]);
        e1.finalize(1);
        e2.finalize(1);

        let dist = fast_dissimilarity(&e1, &e2, 0, &params);
        // Should have light chain penalties (8 + 8 = 16)
        // Dissimilarity = (0 + 0 + 0 + 8 + 8) / 6 = 2.67
        assert!(dist > 2.0);
    }
}
