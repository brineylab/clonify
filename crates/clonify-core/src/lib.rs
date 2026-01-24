//! Clonify Core - Antibody Clonotype Clustering Library
//!
//! This library provides high-performance clustering of antibody sequences
//! into clonal lineages based on:
//! - CDR3/junction amino acid sequence similarity
//! - V and J gene usage
//! - Shared somatic mutations
//!
//! # Algorithm Overview
//!
//! The clustering algorithm uses a multi-stage approach:
//! 1. **Partitioning**: Sequences are partitioned by V gene family
//! 2. **Essence Grouping**: Identical (V, J, CDR3) tuples are grouped
//! 3. **Megaclustering**: High-weight essences become cluster centers
//! 4. **Hierarchical Clustering**: NN-chain UPGMA within megaclusters
//! 5. **Flat Clustering**: Distance threshold cuts the dendrogram

pub mod config;
pub mod distance;
pub mod essence;
pub mod flat_cluster;
pub mod linkage;
pub mod megacluster;
pub mod partition;
pub mod types;

// Re-export commonly used types
pub use config::ClusterParams;
pub use essence::{Essence, EssenceKey};
pub use types::{MutBag, MutList, Mutation};

/// Cluster a set of antibody sequences.
///
/// This is the main entry point for the clustering algorithm.
///
/// # Arguments
/// * `sequences` - Iterator of (id, v_gene, j_gene, cdr3, mutations)
/// * `params` - Clustering parameters
///
/// # Returns
/// A vector of (sequence_id, cluster_id) pairs
pub fn cluster<'a, I>(
    sequences: I,
    params: &ClusterParams,
) -> Vec<(String, u32)>
where
    I: IntoIterator<Item = (&'a str, &'a str, &'a str, &'a str, &'a [Mutation])>,
{
    let mut dataset = partition::Dataset::new(params.clone());

    for (id, v_gene, j_gene, cdr3, mutations) in sequences {
        dataset.add_sequence(id, v_gene, j_gene, cdr3, mutations);
    }

    dataset.process()
}
