//! Clustering configuration parameters.

/// Partitioning strategy for clustering.
///
/// Controls how sequences are grouped into partitions for parallel processing.
/// Finer partitioning (VjGene) reduces partition sizes and speeds up pairwise
/// calculations, while coarser partitioning (VFamily) uses fewer partitions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PartitionLevel {
    /// Partition by V gene family (IGHV1-7 + overflow) - 8 partitions max
    VFamily,
    /// Partition by full V gene (~50-100 partitions)
    VGene,
    /// Partition by V+J gene combination (~300-600 partitions) - DEFAULT
    #[default]
    VjGene,
}

/// Default distance cutoff for flat clustering (0.35 = 35% dissimilarity)
pub const DEFAULT_CUTOFF: f64 = 0.35;

/// Default bonus for shared mutations (reduces dissimilarity)
pub const DEFAULT_MUT_VALUE: f64 = 0.35;

/// Default penalty per unit length difference in CDR3
pub const DEFAULT_LEN_PENALTY: i32 = 2;

/// Minimum dissimilarity value (prevents zero distances)
pub const DEFAULT_EPSILON: f64 = 0.001;

/// Number of samples used to compute canonical mutation list
pub const CANONICAL_SAMPLES: usize = 17;

/// Minimum megacluster dissimilarity for center merging
pub const MIN_MEGACLUSTER_DISSIMILARITY: f64 = 0.40;

/// Maximum CDR3 amino acid sequence length
pub const MAX_AA_LENGTH: usize = 64;

/// Maximum mutation position value (for encoding)
pub const MAX_MUTATION_LOC: u16 = 4096;

/// Number of V gene family partitions (1-7 + overflow)
pub const MAX_PARTITIONS: usize = 8;

/// Configuration parameters for clustering.
#[derive(Clone, Debug)]
pub struct ClusterParams {
    /// Distance threshold for flat clustering (default: 0.35)
    pub cutoff: f64,

    /// Bonus weight for shared mutations (default: 0.35)
    pub mut_value: f64,

    /// Penalty per unit length difference (default: 2)
    pub len_penalty: i32,

    /// Minimum dissimilarity value (default: 0.001)
    pub epsilon: f64,

    /// Minimum essence weight to be a cluster center (None = auto)
    pub min_center_size: Option<usize>,

    /// Number of threads for parallel processing (None = auto)
    pub n_threads: Option<usize>,

    /// Partitioning strategy (default: VjGene)
    pub partition_level: PartitionLevel,
}

impl Default for ClusterParams {
    fn default() -> Self {
        Self {
            cutoff: DEFAULT_CUTOFF,
            mut_value: DEFAULT_MUT_VALUE,
            len_penalty: DEFAULT_LEN_PENALTY,
            epsilon: DEFAULT_EPSILON,
            min_center_size: None,
            n_threads: None,
            partition_level: PartitionLevel::default(),
        }
    }
}

impl ClusterParams {
    /// Create new parameters with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the distance cutoff.
    pub fn with_cutoff(mut self, cutoff: f64) -> Self {
        self.cutoff = cutoff;
        self
    }

    /// Set the mutation bonus value.
    pub fn with_mut_value(mut self, mut_value: f64) -> Self {
        self.mut_value = mut_value;
        self
    }

    /// Set the length penalty.
    pub fn with_len_penalty(mut self, len_penalty: i32) -> Self {
        self.len_penalty = len_penalty;
        self
    }

    /// Set the minimum center size.
    pub fn with_min_center_size(mut self, size: usize) -> Self {
        self.min_center_size = Some(size);
        self
    }

    /// Set the partition level.
    pub fn with_partition_level(mut self, level: PartitionLevel) -> Self {
        self.partition_level = level;
        self
    }

    /// Compute the automatic minimum center size based on dataset size.
    pub fn auto_min_center_size(&self, n_sequences: usize) -> usize {
        if let Some(size) = self.min_center_size {
            size
        } else if n_sequences >= 30000 {
            9 // MIN_CENTER_SIZE_100K
        } else {
            3 // MIN_CENTER_SIZE_10K
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_params() {
        let params = ClusterParams::default();
        assert!((params.cutoff - 0.35).abs() < f64::EPSILON);
        assert!((params.mut_value - 0.35).abs() < f64::EPSILON);
        assert_eq!(params.len_penalty, 2);
    }

    #[test]
    fn test_auto_min_center_size() {
        let params = ClusterParams::default();
        assert_eq!(params.auto_min_center_size(1000), 3);
        assert_eq!(params.auto_min_center_size(50000), 9);
    }

    #[test]
    fn test_builder_pattern() {
        let params = ClusterParams::new()
            .with_cutoff(0.5)
            .with_mut_value(0.4)
            .with_len_penalty(3);

        assert!((params.cutoff - 0.5).abs() < f64::EPSILON);
        assert!((params.mut_value - 0.4).abs() < f64::EPSILON);
        assert_eq!(params.len_penalty, 3);
    }
}
