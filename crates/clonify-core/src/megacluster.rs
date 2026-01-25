//! Megacluster formation and hierarchical clustering within megaclusters.

use std::sync::atomic::{AtomicU32, Ordering};

use rayon::prelude::*;

use crate::config::ClusterParams;
use crate::distance::{full_dissimilarity, get_edit_distance, hamming_distance};
use crate::essence::Essence;
use crate::flat_cluster::form_flat_clusters_from_dist;
use crate::linkage::{generate_dendrogram, nn_chain_core};

/// Minimum megacluster size to use parallel computation.
/// Below this threshold, the overhead of parallelism exceeds benefits.
const PARALLEL_THRESHOLD: usize = 10;

/// A megacluster contains essences that will be hierarchically clustered together.
#[derive(Default)]
pub struct Megacluster {
    /// Indices into the essence pool
    pub essence_indices: Vec<usize>,
}

impl Megacluster {
    /// Create a new empty megacluster.
    pub fn new() -> Self {
        Self {
            essence_indices: Vec::new(),
        }
    }

    /// Add an essence index to this megacluster.
    pub fn add(&mut self, index: usize) {
        self.essence_indices.push(index);
    }

    /// Get the number of essences in this megacluster.
    pub fn len(&self) -> usize {
        self.essence_indices.len()
    }

    /// Check if the megacluster is empty.
    pub fn is_empty(&self) -> bool {
        self.essence_indices.is_empty()
    }

    /// Perform hierarchical clustering on the essences.
    ///
    /// # Arguments
    /// * `essences` - The essence pool (only cluster_id field is mutated)
    /// * `params` - Clustering parameters
    /// * `next_cluster` - Atomic counter for cluster IDs
    ///
    /// # Safety
    /// This method only writes to `cluster_id` fields of essences at indices
    /// in `self.essence_indices`. When called from parallel code, callers must
    /// ensure no two megaclusters share essence indices.
    pub fn cluster(
        &self,
        essences: &[Essence],
        params: &ClusterParams,
        next_cluster: &AtomicU32,
    ) {
        let n = self.essence_indices.len();
        if n == 0 {
            return;
        }

        if n == 1 {
            // Single essence: assign its own cluster
            let cluster_id = next_cluster.fetch_add(1, Ordering::Relaxed);
            essences[self.essence_indices[0]].set_cluster_id(cluster_id);
            return;
        }

        // Determine if we should use parallel computation
        let use_parallel = params.is_parallel() && n > PARALLEL_THRESHOLD;

        // Build base distance matrix (Hamming/Levenshtein)
        let base_matrix = self.build_base_matrix(essences, use_parallel);

        // Build full dissimilarity matrix
        let mut dist_matrix =
            self.build_dissimilarity_matrix(essences, &base_matrix, params, use_parallel);

        // Build member weights
        let mut members: Vec<i32> = self
            .essence_indices
            .iter()
            .map(|&i| essences[i].weight() as i32)
            .collect();

        // Perform hierarchical clustering
        let result = nn_chain_core(n, &mut dist_matrix, &mut members);
        let dendrogram = generate_dendrogram(&result, n);

        // Extract flat clusters
        let flat_clusters = form_flat_clusters_from_dist(&dendrogram, params.cutoff, n);

        // Assign cluster IDs
        // Reserve a block of cluster IDs atomically
        let n_clusters = *flat_clusters.iter().max().unwrap_or(&0);
        let base_cluster_id = next_cluster.fetch_add(n_clusters, Ordering::Relaxed);

        for (i, &cluster_num) in flat_clusters.iter().enumerate() {
            let cluster_id = base_cluster_id + cluster_num - 1;
            essences[self.essence_indices[i]].set_cluster_id(cluster_id);
        }
    }

    /// Build base distance matrix (Hamming/Levenshtein distances).
    ///
    /// Uses parallel computation when `parallel` is true and matrix is large enough.
    fn build_base_matrix(&self, essences: &[Essence], parallel: bool) -> Vec<u8> {
        let n = self.essence_indices.len();
        if n < 2 {
            return Vec::new();
        }

        // Pre-extract junction strings and lengths for better cache locality
        let junctions: Vec<(&str, usize)> = self
            .essence_indices
            .iter()
            .map(|&idx| {
                let junc = &essences[idx].key.junction;
                (junc.as_str(), junc.len())
            })
            .collect();

        if parallel {
            // Parallel: compute row by row, then flatten
            let rows: Vec<Vec<u8>> = (0..n)
                .into_par_iter()
                .map(|i| {
                    let (ji, ni) = junctions[i];
                    ((i + 1)..n)
                        .map(|j| {
                            let (jj, nj) = junctions[j];
                            let dist = if ni == nj {
                                hamming_distance(ji, jj)
                            } else {
                                get_edit_distance(ji, jj)
                            };
                            dist.min(255) as u8
                        })
                        .collect()
                })
                .collect();

            rows.into_iter().flatten().collect()
        } else {
            // Sequential: original algorithm
            let mut matrix = Vec::with_capacity(n * (n - 1) / 2);
            for i in 0..n {
                let (ji, ni) = junctions[i];
                for j in (i + 1)..n {
                    let (jj, nj) = junctions[j];
                    let dist = if ni == nj {
                        hamming_distance(ji, jj)
                    } else {
                        get_edit_distance(ji, jj)
                    };
                    matrix.push(dist.min(255) as u8);
                }
            }
            matrix
        }
    }

    /// Build full dissimilarity matrix from base distances.
    ///
    /// Uses parallel computation when `parallel` is true.
    fn build_dissimilarity_matrix(
        &self,
        essences: &[Essence],
        base_matrix: &[u8],
        params: &ClusterParams,
        parallel: bool,
    ) -> Vec<f64> {
        let n = self.essence_indices.len();
        if n < 2 {
            return Vec::new();
        }

        if parallel {
            // Parallel: compute row by row, then flatten
            let rows: Vec<Vec<f64>> = (0..n)
                .into_par_iter()
                .map(|i| {
                    ((i + 1)..n)
                        .map(|j| {
                            let idx = condensed_index(n, i, j);
                            let ei = &essences[self.essence_indices[i]];
                            let ej = &essences[self.essence_indices[j]];
                            full_dissimilarity(ei, ej, base_matrix[idx] as usize, params)
                        })
                        .collect()
                })
                .collect();

            rows.into_iter().flatten().collect()
        } else {
            // Sequential computation
            let mut matrix = vec![0.0f64; n * (n - 1) / 2];
            for i in 0..n {
                for j in (i + 1)..n {
                    let idx = condensed_index(n, i, j);
                    let ei = &essences[self.essence_indices[i]];
                    let ej = &essences[self.essence_indices[j]];
                    matrix[idx] = full_dissimilarity(ei, ej, base_matrix[idx] as usize, params);
                }
            }
            matrix
        }
    }
}

/// Index into condensed distance matrix.
#[inline]
fn condensed_index(n: usize, r: usize, c: usize) -> usize {
    debug_assert!(r < c);
    ((2 * n - 3 - r) * r) / 2 + c - 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::essence::EssenceKey;

    fn make_essence(junction: &str, v: u8, j: u8) -> Essence {
        let key = EssenceKey::new(junction.to_string(), v, j);
        let mut ess = Essence::new(key);
        ess.push_mutlist(vec![]);
        ess.finalize(1);
        ess
    }

    #[test]
    fn test_megacluster_single() {
        let essences = vec![make_essence("CARFDY", 1, 2)];
        let mut mc = Megacluster::new();
        mc.add(0);

        let params = ClusterParams::default();
        let next_cluster = AtomicU32::new(1);

        mc.cluster(&essences, &params, &next_cluster);

        assert_eq!(essences[0].cluster_id(), Some(1));
        assert_eq!(next_cluster.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn test_megacluster_identical() {
        let essences = vec![
            make_essence("CARFDY", 1, 2),
            make_essence("CARFDY", 1, 2),
        ];
        let mut mc = Megacluster::new();
        mc.add(0);
        mc.add(1);

        let params = ClusterParams::default();
        let next_cluster = AtomicU32::new(1);

        mc.cluster(&essences, &params, &next_cluster);

        // Identical essences should be in same cluster
        assert_eq!(essences[0].cluster_id(), essences[1].cluster_id());
    }

    #[test]
    fn test_megacluster_different() {
        let essences = vec![
            make_essence("CARFDY", 1, 2),
            make_essence("XYZABC", 3, 4), // Very different
        ];
        let mut mc = Megacluster::new();
        mc.add(0);
        mc.add(1);

        let params = ClusterParams::default();
        let next_cluster = AtomicU32::new(1);

        mc.cluster(&essences, &params, &next_cluster);

        // Very different essences should be in different clusters
        assert_ne!(essences[0].cluster_id(), essences[1].cluster_id());
    }
}
