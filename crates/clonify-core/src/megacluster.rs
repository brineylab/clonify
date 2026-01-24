//! Megacluster formation and hierarchical clustering within megaclusters.

use crate::config::ClusterParams;
use crate::distance::{full_dissimilarity, get_edit_distance, hamming_distance};
use crate::essence::Essence;
use crate::flat_cluster::form_flat_clusters_from_dist;
use crate::linkage::{generate_dendrogram, nn_chain_core};

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
    /// * `essences` - The essence pool
    /// * `params` - Clustering parameters
    /// * `next_cluster` - Counter for cluster IDs (mutated)
    pub fn cluster(
        &self,
        essences: &mut [Essence],
        params: &ClusterParams,
        next_cluster: &mut u32,
    ) {
        let n = self.essence_indices.len();
        if n == 0 {
            return;
        }

        if n == 1 {
            // Single essence: assign its own cluster
            essences[self.essence_indices[0]].cluster_id = Some(*next_cluster);
            *next_cluster += 1;
            return;
        }

        // Build base distance matrix (Hamming/Levenshtein)
        let base_matrix = self.build_base_matrix(essences);

        // Build full dissimilarity matrix
        let mut dist_matrix = vec![0.0f64; n * (n - 1) / 2];
        for i in 0..n {
            for j in (i + 1)..n {
                let idx = condensed_index(n, i, j);
                let ei = &essences[self.essence_indices[i]];
                let ej = &essences[self.essence_indices[j]];
                dist_matrix[idx] = full_dissimilarity(ei, ej, base_matrix[idx] as usize, params);
            }
        }

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
        let n_clusters = *flat_clusters.iter().max().unwrap_or(&0);
        for (i, &cluster_num) in flat_clusters.iter().enumerate() {
            let cluster_id = *next_cluster + cluster_num - 1;
            essences[self.essence_indices[i]].cluster_id = Some(cluster_id);
        }
        *next_cluster += n_clusters;
    }

    /// Build base distance matrix (Hamming/Levenshtein distances).
    fn build_base_matrix(&self, essences: &[Essence]) -> Vec<u8> {
        let n = self.essence_indices.len();
        let mut matrix = Vec::with_capacity(n * (n - 1) / 2);

        for i in 0..n {
            let ei = &essences[self.essence_indices[i]];
            let ni = ei.key.junction.len();

            for j in (i + 1)..n {
                let ej = &essences[self.essence_indices[j]];
                let nj = ej.key.junction.len();

                let dist = if ni == nj {
                    hamming_distance(&ei.key.junction, &ej.key.junction)
                } else {
                    get_edit_distance(&ei.key.junction, &ej.key.junction)
                };

                matrix.push(dist.min(255) as u8);
            }
        }

        matrix
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
        let mut essences = vec![make_essence("CARFDY", 1, 2)];
        let mut mc = Megacluster::new();
        mc.add(0);

        let params = ClusterParams::default();
        let mut next_cluster = 1u32;

        mc.cluster(&mut essences, &params, &mut next_cluster);

        assert_eq!(essences[0].cluster_id, Some(1));
        assert_eq!(next_cluster, 2);
    }

    #[test]
    fn test_megacluster_identical() {
        let mut essences = vec![
            make_essence("CARFDY", 1, 2),
            make_essence("CARFDY", 1, 2),
        ];
        let mut mc = Megacluster::new();
        mc.add(0);
        mc.add(1);

        let params = ClusterParams::default();
        let mut next_cluster = 1u32;

        mc.cluster(&mut essences, &params, &mut next_cluster);

        // Identical essences should be in same cluster
        assert_eq!(essences[0].cluster_id, essences[1].cluster_id);
    }

    #[test]
    fn test_megacluster_different() {
        let mut essences = vec![
            make_essence("CARFDY", 1, 2),
            make_essence("XYZABC", 3, 4), // Very different
        ];
        let mut mc = Megacluster::new();
        mc.add(0);
        mc.add(1);

        let params = ClusterParams::default();
        let mut next_cluster = 1u32;

        mc.cluster(&mut essences, &params, &mut next_cluster);

        // Very different essences should be in different clusters
        assert_ne!(essences[0].cluster_id, essences[1].cluster_id);
    }
}
