//! Flat clustering from hierarchical dendrogram.
//!
//! Cuts the dendrogram at a specified distance threshold to form flat clusters.

/// Extract flat clusters from a dendrogram using a distance threshold.
///
/// # Arguments
/// * `dendrogram` - Linkage matrix as [node1, node2, distance, size] arrays
/// * `cutoff` - Distance threshold for cutting the dendrogram
/// * `n` - Number of original elements
///
/// # Returns
/// Vector of cluster IDs (1-indexed) for each original element
pub fn form_flat_clusters_from_dist(dendrogram: &[[f64; 4]], cutoff: f64, n: usize) -> Vec<u32> {
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![1];
    }

    // First, compute max distance for each internal node
    let mut max_dists = vec![0.0f64; n - 1];
    compute_max_distances(dendrogram, &mut max_dists, n);

    // Then form clusters using the max distances
    form_clusters_from_criterion(dendrogram, &max_dists, cutoff, n)
}

/// Compute the maximum merge distance in each subtree.
///
/// Uses iterative tree traversal (post-order).
fn compute_max_distances(dendrogram: &[[f64; 4]], max_dists: &mut [f64], n: usize) {
    if dendrogram.is_empty() {
        return;
    }

    let bits_per_byte = 8usize;
    let flag_size = (n + bits_per_byte - 1) / bits_per_byte;

    let mut cur_node = vec![0i32; n];
    let mut lvisited = vec![0u8; flag_size];
    let mut rvisited = vec![0u8; flag_size];

    let get_bit = |arr: &[u8], i: usize| -> bool {
        let byte_idx = i / bits_per_byte;
        let bit_idx = (bits_per_byte - 1) - (i % bits_per_byte);
        (arr[byte_idx] >> bit_idx) & 1 != 0
    };

    let set_bit = |arr: &mut [u8], i: usize| {
        let byte_idx = i / bits_per_byte;
        let bit_idx = (bits_per_byte - 1) - (i % bits_per_byte);
        arr[byte_idx] |= 1 << bit_idx;
    };

    let mut k = 0i32;
    cur_node[0] = (2 * n - 2) as i32; // Root node

    while k >= 0 {
        let ndid = cur_node[k as usize] as usize;
        let row = &dendrogram[ndid - n];
        let lid = row[0] as usize;
        let rid = row[1] as usize;

        // Visit left child if internal and not visited
        if lid >= n && !get_bit(&lvisited, ndid - n) {
            set_bit(&mut lvisited, ndid - n);
            cur_node[(k + 1) as usize] = lid as i32;
            k += 1;
            continue;
        }

        // Visit right child if internal and not visited
        if rid >= n && !get_bit(&rvisited, ndid - n) {
            set_bit(&mut rvisited, ndid - n);
            cur_node[(k + 1) as usize] = rid as i32;
            k += 1;
            continue;
        }

        // Both children visited, compute max distance
        let mut max_dist = row[2]; // This node's merge distance
        if lid >= n {
            max_dist = max_dist.max(max_dists[lid - n]);
        }
        if rid >= n {
            max_dist = max_dist.max(max_dists[rid - n]);
        }
        max_dists[ndid - n] = max_dist;
        k -= 1;
    }
}

/// Form flat clusters using precomputed max distances.
fn form_clusters_from_criterion(
    dendrogram: &[[f64; 4]],
    max_dists: &[f64],
    cutoff: f64,
    n: usize,
) -> Vec<u32> {
    let mut result = vec![0u32; n];

    if dendrogram.is_empty() {
        if n == 1 {
            result[0] = 1;
        }
        return result;
    }

    let bits_per_byte = 8usize;
    let flag_size = (n + bits_per_byte - 1) / bits_per_byte;

    let mut cur_node = vec![0i32; n];
    let mut lvisited = vec![0u8; flag_size];
    let mut rvisited = vec![0u8; flag_size];

    let get_bit = |arr: &[u8], i: usize| -> bool {
        let byte_idx = i / bits_per_byte;
        let bit_idx = (bits_per_byte - 1) - (i % bits_per_byte);
        (arr[byte_idx] >> bit_idx) & 1 != 0
    };

    let set_bit = |arr: &mut [u8], i: usize| {
        let byte_idx = i / bits_per_byte;
        let bit_idx = (bits_per_byte - 1) - (i % bits_per_byte);
        arr[byte_idx] |= 1 << bit_idx;
    };

    let mut nc = 0u32; // Number of clusters
    let mut ms = -1i32; // Marker for subtree below cutoff

    let mut k = 0i32;
    cur_node[0] = (2 * n - 2) as i32;

    while k >= 0 {
        let ndid = cur_node[k as usize] as usize;
        let row = &dendrogram[ndid - n];
        let lid = row[0] as usize;
        let rid = row[1] as usize;
        let max_crit = max_dists[ndid - n];

        // Check if we've entered a subtree below cutoff
        if ms == -1 && max_crit <= cutoff {
            ms = k;
            nc += 1;
        }

        // Visit left child
        if lid >= n && !get_bit(&lvisited, ndid - n) {
            set_bit(&mut lvisited, ndid - n);
            cur_node[(k + 1) as usize] = lid as i32;
            k += 1;
            continue;
        }

        // Visit right child
        if rid >= n && !get_bit(&rvisited, ndid - n) {
            set_bit(&mut rvisited, ndid - n);
            cur_node[(k + 1) as usize] = rid as i32;
            k += 1;
            continue;
        }

        // Process this node
        if ndid >= n {
            // Assign cluster IDs to leaf children
            if lid < n {
                if ms == -1 {
                    nc += 1;
                    result[lid] = nc;
                } else {
                    result[lid] = nc;
                }
            }
            if rid < n {
                if ms == -1 {
                    nc += 1;
                    result[rid] = nc;
                } else {
                    result[rid] = nc;
                }
            }

            // Check if we're leaving the subtree below cutoff
            if ms == k {
                ms = -1;
            }
        }
        k -= 1;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_form_flat_clusters_single() {
        let clusters = form_flat_clusters_from_dist(&[], 0.5, 1);
        assert_eq!(clusters, vec![1]);
    }

    #[test]
    fn test_form_flat_clusters_simple() {
        // 3 elements: 0 and 1 merged at distance 0.1, then with 2 at distance 0.5
        let dendrogram = vec![
            [0.0, 1.0, 0.1, 2.0], // Node 3: merge 0,1
            [2.0, 3.0, 0.5, 3.0], // Node 4: merge 2, (0,1)
        ];

        // Cutoff 0.3: should separate 2 from (0,1)
        let clusters = form_flat_clusters_from_dist(&dendrogram, 0.3, 3);
        assert_eq!(clusters[0], clusters[1]); // 0 and 1 in same cluster
        assert_ne!(clusters[0], clusters[2]); // 2 in different cluster

        // Cutoff 0.6: all in one cluster
        let clusters = form_flat_clusters_from_dist(&dendrogram, 0.6, 3);
        assert_eq!(clusters[0], clusters[1]);
        assert_eq!(clusters[1], clusters[2]);
    }

    #[test]
    fn test_form_flat_clusters_low_cutoff() {
        // Same dendrogram
        let dendrogram = vec![
            [0.0, 1.0, 0.1, 2.0],
            [2.0, 3.0, 0.5, 3.0],
        ];

        // Very low cutoff: each element in its own cluster
        let clusters = form_flat_clusters_from_dist(&dendrogram, 0.05, 3);
        assert_ne!(clusters[0], clusters[1]);
        assert_ne!(clusters[1], clusters[2]);
        assert_ne!(clusters[0], clusters[2]);
    }
}
