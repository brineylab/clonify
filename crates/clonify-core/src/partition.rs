//! Partitioning and dataset management for clustering.

use crate::config::{ClusterParams, MAX_AA_LENGTH, MAX_PARTITIONS, MIN_MEGACLUSTER_DISSIMILARITY};
use crate::distance::{fast_dissimilarity, get_edit_distance, merging_dissimilarity};
use crate::essence::{Essence, EssenceKey, GeneIntern};
use crate::megacluster::Megacluster;
use crate::types::Mutation;
use rustc_hash::FxHashMap;

/// A partition groups essences by V gene family for parallel processing.
pub struct Partition {
    /// All essences in this partition
    pub essences: Vec<Essence>,

    /// Map from EssenceKey to index in essences
    essence_map: FxHashMap<EssenceKey, usize>,

    /// Essences grouped by junction length
    essence_by_length: Vec<Vec<usize>>,

    /// Center candidates (high-weight essences)
    center_candidates: Vec<usize>,

    /// Centers for megacluster assignment, grouped by junction length
    centers: Vec<Vec<(usize, usize)>>, // (essence_idx, megacluster_idx)

    /// The megaclusters in this partition
    pub megaclusters: Vec<Megacluster>,
}

impl Default for Partition {
    fn default() -> Self {
        Self::new()
    }
}

impl Partition {
    /// Create a new empty partition.
    pub fn new() -> Self {
        Self {
            essences: Vec::new(),
            essence_map: FxHashMap::default(),
            essence_by_length: vec![Vec::new(); MAX_AA_LENGTH + 1],
            center_candidates: Vec::new(),
            centers: vec![Vec::new(); MAX_AA_LENGTH + 1],
            megaclusters: Vec::new(),
        }
    }

    /// Look up or create an essence by key.
    ///
    /// Returns the index of the essence.
    pub fn essence_lookup(&mut self, key: EssenceKey) -> usize {
        if let Some(&idx) = self.essence_map.get(&key) {
            return idx;
        }

        let idx = self.essences.len();
        let junc_len = key.junction.len().min(MAX_AA_LENGTH);

        self.essences.push(Essence::new(key.clone()));
        self.essence_map.insert(key, idx);
        self.essence_by_length[junc_len].push(idx);

        idx
    }

    /// Finalize parsing: compute canonical mutations and identify center candidates.
    pub fn finalize_parsing(&mut self, min_center_size: usize) {
        for (idx, essence) in self.essences.iter_mut().enumerate() {
            if essence.finalize(min_center_size) {
                self.center_candidates.push(idx);
            }
        }
    }

    /// Create megacluster centers from high-weight essences.
    pub fn create_centers(&mut self, params: &ClusterParams) {
        // Sort center candidates by weight (descending)
        self.center_candidates
            .sort_by(|&a, &b| self.essences[b].weight().cmp(&self.essences[a].weight()));

        let mut candidate_mega: Vec<usize> = Vec::with_capacity(self.center_candidates.len());

        // Collect merges to perform after the loop
        let mut pending_merges: Vec<(usize, usize)> = Vec::new();

        for i in 0..self.center_candidates.len() {
            let cand_idx = self.center_candidates[i];
            let mut mega_idx: Option<usize> = None;

            // Check distance to all previous candidates
            for j in 0..i {
                let prev_idx = self.center_candidates[j];
                let dist = merging_dissimilarity(
                    &self.essences[cand_idx],
                    &self.essences[prev_idx],
                    MIN_MEGACLUSTER_DISSIMILARITY,
                    params,
                );

                if dist < MIN_MEGACLUSTER_DISSIMILARITY {
                    let prev_mega = candidate_mega[j];
                    if let Some(curr_mega) = mega_idx {
                        if curr_mega != prev_mega {
                            // Schedule merge for later
                            pending_merges.push((curr_mega, prev_mega));
                        }
                    } else {
                        mega_idx = Some(prev_mega);
                    }
                }
            }

            // Create new megacluster if needed
            if mega_idx.is_none() {
                mega_idx = Some(self.megaclusters.len());
                self.megaclusters.push(Megacluster::new());
            }

            let mega_idx = mega_idx.unwrap();

            // Add as center
            let junc_len = self.essences[cand_idx].key.junction.len();
            self.centers[junc_len].push((cand_idx, mega_idx));

            candidate_mega.push(mega_idx);
        }

        // Process pending merges
        for (a, b) in pending_merges {
            let merged_to = self.merge_megaclusters(a, b);
            // Update candidate_mega entries
            for m in candidate_mega.iter_mut() {
                if *m == b {
                    *m = merged_to;
                }
            }
        }
    }

    /// Merge megacluster b into megacluster a.
    fn merge_megaclusters(&mut self, a: usize, b: usize) -> usize {
        if a == b {
            return a;
        }

        // Move essences from b to a
        let essences_b: Vec<usize> = std::mem::take(&mut self.megaclusters[b].essence_indices);
        self.megaclusters[a].essence_indices.extend(essences_b);

        // Update center references
        for length_centers in &mut self.centers {
            for (_, mega_idx) in length_centers.iter_mut() {
                if *mega_idx == b {
                    *mega_idx = a;
                }
            }
        }

        a
    }

    /// Assign all essences to megaclusters based on nearest center.
    pub fn assign_to_megaclusters(&mut self, params: &ClusterParams) {
        // Count centers
        let n_centers: usize = self.centers.iter().map(|v| v.len()).sum();

        if n_centers <= 1 {
            // All essences go to a single megacluster
            self.megaclusters.clear();
            self.megaclusters.push(Megacluster::new());

            for length_essences in &self.essence_by_length {
                for &ess_idx in length_essences {
                    self.megaclusters[0].add(ess_idx);
                }
            }
            return;
        }

        // Assign each essence to nearest center
        for (len, length_essences) in self.essence_by_length.iter().enumerate() {
            for &ess_idx in length_essences {
                let essence = &self.essences[ess_idx];
                let s0 = len;
                let mut best_dist = f64::INFINITY;
                let mut best_mega: Option<usize> = None;

                // Search centers at increasing length differences
                for ds in 0..=MAX_AA_LENGTH {
                    // Early exit if length penalty alone exceeds best distance
                    if (params.len_penalty as f64) * (ds as f64) >= best_dist * (s0 as f64) {
                        break;
                    }

                    // Check centers at length s0 + ds
                    if s0 + ds <= MAX_AA_LENGTH {
                        for &(center_idx, mega_idx) in &self.centers[s0 + ds] {
                            let center = &self.essences[center_idx];
                            let ld = get_edit_distance(&essence.key.junction, &center.key.junction);
                            let dist = fast_dissimilarity(essence, center, ld, params);
                            if dist < best_dist {
                                best_dist = dist;
                                best_mega = Some(mega_idx);
                            }
                        }
                    }

                    // Check centers at length s0 - ds
                    if ds > 0 && s0 >= ds {
                        let check_len = s0 - ds;
                        if (params.len_penalty as f64) * (ds as f64)
                            < best_dist * (check_len as f64)
                        {
                            for &(center_idx, mega_idx) in &self.centers[check_len] {
                                let center = &self.essences[center_idx];
                                let ld =
                                    get_edit_distance(&essence.key.junction, &center.key.junction);
                                let dist = fast_dissimilarity(essence, center, ld, params);
                                if dist < best_dist {
                                    best_dist = dist;
                                    best_mega = Some(mega_idx);
                                }
                            }
                        }
                    }
                }

                if let Some(mega_idx) = best_mega {
                    self.megaclusters[mega_idx].add(ess_idx);
                }
            }
        }
    }

    /// Perform hierarchical clustering within each megacluster.
    pub fn cluster(&mut self, params: &ClusterParams, next_cluster: &mut u32) {
        for mega in &self.megaclusters {
            mega.cluster(&mut self.essences, params, next_cluster);
        }
    }

    /// Process the partition: finalize, create centers, assign, and cluster.
    pub fn process(&mut self, params: &ClusterParams, next_cluster: &mut u32) {
        let min_center_size = params.auto_min_center_size(self.essences.len());
        self.finalize_parsing(min_center_size);
        self.create_centers(params);
        self.assign_to_megaclusters(params);
        self.cluster(params, next_cluster);
    }
}

/// Dataset containing all partitions and sequence tracking.
pub struct Dataset {
    /// Partitions by V gene family
    partitions: Vec<Partition>,

    /// Gene name interning
    pub v_gene_intern: GeneIntern,
    pub j_gene_intern: GeneIntern,

    /// Sequence tracking: (sequence_id, partition_idx, essence_idx)
    sequences: Vec<(String, usize, usize)>,

    /// Clustering parameters
    params: ClusterParams,
}

impl Dataset {
    /// Create a new dataset with the given parameters.
    pub fn new(params: ClusterParams) -> Self {
        Self {
            partitions: (0..MAX_PARTITIONS).map(|_| Partition::new()).collect(),
            v_gene_intern: GeneIntern::new(),
            j_gene_intern: GeneIntern::new(),
            sequences: Vec::new(),
            params,
        }
    }

    /// Extract V gene family number from gene name.
    ///
    /// Returns 0 (overflow partition) for unrecognized patterns.
    fn extract_v_family(v_gene: &str) -> usize {
        // Look for pattern like "IGHV3" or "IGKV1" or "IGLV2"
        if let Some(pos) = v_gene.find('V') {
            let rest = &v_gene[pos + 1..];
            let mut family = 0usize;
            for c in rest.chars() {
                if c.is_ascii_digit() {
                    family = family * 10 + (c as usize - '0' as usize);
                } else {
                    break;
                }
            }
            if (1..=7).contains(&family) {
                return family;
            }
        }
        0 // Overflow partition
    }

    /// Strip allele information from gene name.
    ///
    /// "IGHV3-20*01" -> "IGHV3-20"
    fn strip_allele(gene: &str) -> &str {
        gene.split('*').next().unwrap_or(gene)
    }

    /// Add a sequence to the dataset.
    pub fn add_sequence(
        &mut self,
        seq_id: &str,
        v_gene: &str,
        j_gene: &str,
        cdr3: &str,
        mutations: &[Mutation],
    ) {
        // Strip allele info
        let v_gene = Self::strip_allele(v_gene);
        let j_gene = Self::strip_allele(j_gene);

        // Get partition
        let partition_idx = Self::extract_v_family(v_gene);

        // Intern gene names
        let v_id = self.v_gene_intern.intern(v_gene);
        let j_id = self.j_gene_intern.intern(j_gene);

        // Truncate CDR3 if necessary
        let cdr3 = if cdr3.len() > MAX_AA_LENGTH {
            &cdr3[..MAX_AA_LENGTH]
        } else {
            cdr3
        };

        // Create essence key and look up/create essence
        let key = EssenceKey::new(cdr3.to_string(), v_id, j_id);
        let essence_idx = self.partitions[partition_idx].essence_lookup(key);

        // Add mutations to essence
        self.partitions[partition_idx].essences[essence_idx].push_mutlist(mutations.to_vec());

        // Track sequence
        self.sequences
            .push((seq_id.to_string(), partition_idx, essence_idx));
    }

    /// Process all partitions and return (sequence_id, cluster_id) pairs.
    pub fn process(&mut self) -> Vec<(String, u32)> {
        let mut next_cluster = 1u32;

        // Process each partition
        for partition in &mut self.partitions {
            partition.process(&self.params, &mut next_cluster);
        }

        // Collect results
        self.sequences
            .iter()
            .map(|(seq_id, part_idx, ess_idx)| {
                let cluster_id = self.partitions[*part_idx].essences[*ess_idx]
                    .cluster_id
                    .unwrap_or(0);
                (seq_id.clone(), cluster_id)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_v_family() {
        assert_eq!(Dataset::extract_v_family("IGHV3-20"), 3);
        assert_eq!(Dataset::extract_v_family("IGHV1-2"), 1);
        assert_eq!(Dataset::extract_v_family("IGKV4-1"), 4);
        assert_eq!(Dataset::extract_v_family("IGLV7-46"), 7);
        assert_eq!(Dataset::extract_v_family("unknown"), 0);
        assert_eq!(Dataset::extract_v_family("IGHV10-1"), 0); // >7 goes to overflow
    }

    #[test]
    fn test_strip_allele() {
        assert_eq!(Dataset::strip_allele("IGHV3-20*01"), "IGHV3-20");
        assert_eq!(Dataset::strip_allele("IGHV3-20"), "IGHV3-20");
        assert_eq!(Dataset::strip_allele("IGHJ4*02"), "IGHJ4");
    }

    #[test]
    fn test_dataset_basic() {
        let params = ClusterParams::default();
        let mut dataset = Dataset::new(params);

        dataset.add_sequence("seq1", "IGHV3-20*01", "IGHJ4*02", "CARFDY", &[]);
        dataset.add_sequence("seq2", "IGHV3-20*01", "IGHJ4*02", "CARFDY", &[]);
        dataset.add_sequence("seq3", "IGHV1-2*01", "IGHJ6*01", "CARDYF", &[]);

        let results = dataset.process();

        assert_eq!(results.len(), 3);

        // seq1 and seq2 should be in same cluster (identical)
        let seq1_cluster = results.iter().find(|(id, _)| id == "seq1").unwrap().1;
        let seq2_cluster = results.iter().find(|(id, _)| id == "seq2").unwrap().1;
        let seq3_cluster = results.iter().find(|(id, _)| id == "seq3").unwrap().1;

        assert_eq!(seq1_cluster, seq2_cluster);
        // seq3 is in different partition, likely different cluster
        // (but could be same if dissimilarity is low enough)
    }
}
