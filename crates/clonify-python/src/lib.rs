//! Python bindings for the clonify clustering library.

use clonify_core::{ClusterParams, Mutation};
use pyo3::prelude::*;

/// Python-exposed clustering parameters.
#[pyclass(name = "ClusterParams")]
#[derive(Clone)]
pub struct PyClusterParams {
    inner: ClusterParams,
}

#[pymethods]
impl PyClusterParams {
    #[new]
    #[pyo3(signature = (
        cutoff = 0.35,
        mut_value = 0.35,
        len_penalty = 2,
        epsilon = 0.001,
        min_center_size = None
    ))]
    fn new(
        cutoff: f64,
        mut_value: f64,
        len_penalty: i32,
        epsilon: f64,
        min_center_size: Option<usize>,
    ) -> Self {
        Self {
            inner: ClusterParams {
                cutoff,
                mut_value,
                len_penalty,
                epsilon,
                min_center_size,
                n_threads: None,
            },
        }
    }

    #[getter]
    fn cutoff(&self) -> f64 {
        self.inner.cutoff
    }

    #[getter]
    fn mut_value(&self) -> f64 {
        self.inner.mut_value
    }

    #[getter]
    fn len_penalty(&self) -> i32 {
        self.inner.len_penalty
    }

    #[getter]
    fn epsilon(&self) -> f64 {
        self.inner.epsilon
    }
}

/// Encode a mutation from position and bases.
#[pyfunction]
fn encode_mutation(position: u16, ref_base: char, alt_base: char) -> u16 {
    clonify_core::types::encode_mutation(position, ref_base, alt_base)
}

/// Cluster antibody sequences.
///
/// # Arguments
/// * `sequence_ids` - List of sequence identifiers
/// * `v_genes` - List of V gene names
/// * `j_genes` - List of J gene names
/// * `cdr3s` - List of CDR3/junction amino acid sequences
/// * `mutations` - List of mutation lists (each as list of encoded mutations)
/// * `params` - Clustering parameters
///
/// # Returns
/// List of (sequence_id, cluster_id) tuples
#[pyfunction]
#[pyo3(signature = (sequence_ids, v_genes, j_genes, cdr3s, mutations, params = None))]
fn cluster(
    sequence_ids: Vec<String>,
    v_genes: Vec<String>,
    j_genes: Vec<String>,
    cdr3s: Vec<String>,
    mutations: Vec<Vec<u16>>,
    params: Option<PyClusterParams>,
) -> PyResult<Vec<(String, u32)>> {
    let params = params.map(|p| p.inner).unwrap_or_default();
    let mut dataset = clonify_core::partition::Dataset::new(params);

    for i in 0..sequence_ids.len() {
        let muts: Vec<Mutation> = mutations.get(i).map(|v| v.clone()).unwrap_or_default();
        dataset.add_sequence(
            &sequence_ids[i],
            &v_genes[i],
            &j_genes[i],
            &cdr3s[i],
            &muts,
        );
    }

    Ok(dataset.process())
}

/// Parse mutation string in format "pos:ref>alt|pos:ref>alt|..."
#[pyfunction]
fn parse_mutations(mutation_str: &str) -> Vec<u16> {
    if mutation_str.is_empty() {
        return Vec::new();
    }

    let mut mutations = Vec::new();

    for mut_part in mutation_str.split('|') {
        if let Some(colon_pos) = mut_part.find(':') {
            // Parse position
            let pos_str = &mut_part[..colon_pos];
            let pos: u16 = pos_str.parse().unwrap_or(0) % 4096;

            // Parse ref>alt
            if let Some(gt_pos) = mut_part.find('>') {
                if colon_pos + 1 < gt_pos && gt_pos + 1 < mut_part.len() {
                    let ref_base = mut_part.chars().nth(colon_pos + 1).unwrap_or('N');
                    let alt_base = mut_part.chars().nth(gt_pos + 1).unwrap_or('N');
                    mutations.push(clonify_core::types::encode_mutation(pos, ref_base, alt_base));
                }
            }
        }
    }

    mutations.sort_unstable();
    mutations
}

/// Compute Hamming distance between two equal-length strings.
#[pyfunction]
fn hamming_distance(s1: &str, s2: &str) -> usize {
    clonify_core::distance::hamming_distance(s1, s2)
}

/// Compute Levenshtein distance between two strings.
#[pyfunction]
fn levenshtein_distance(s1: &str, s2: &str) -> usize {
    clonify_core::distance::levenshtein_distance(s1, s2)
}

/// Python module definition.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyClusterParams>()?;
    m.add_function(wrap_pyfunction!(cluster, m)?)?;
    m.add_function(wrap_pyfunction!(encode_mutation, m)?)?;
    m.add_function(wrap_pyfunction!(parse_mutations, m)?)?;
    m.add_function(wrap_pyfunction!(hamming_distance, m)?)?;
    m.add_function(wrap_pyfunction!(levenshtein_distance, m)?)?;
    Ok(())
}
