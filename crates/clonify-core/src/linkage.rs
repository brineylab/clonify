//! Hierarchical clustering using the NN-chain algorithm with UPGMA linkage.
//!
//! This implements the nearest-neighbor chain algorithm described in:
//! Fionn Murtagh, Multidimensional Clustering Algorithms,
//! Vienna, Würzburg: Physica-Verlag, 1985.

/// A node in the dendrogram representing a merge.
#[derive(Clone, Debug)]
pub struct MergeNode {
    pub node1: i32,
    pub node2: i32,
    pub dist: f64,
}

/// Result of hierarchical clustering.
pub struct ClusterResult {
    nodes: Vec<MergeNode>,
}

impl ClusterResult {
    /// Get the merge nodes.
    pub fn nodes(&self) -> &[MergeNode] {
        &self.nodes
    }
}

/// Doubly linked list for tracking active nodes.
struct DoublyLinkedList {
    start: usize,
    succ: Vec<usize>,
    pred: Vec<usize>,
}

impl DoublyLinkedList {
    fn new(size: usize) -> Self {
        let mut succ = vec![0; size + 1];
        let mut pred = vec![0; size + 1];

        for i in 0..size {
            pred[i + 1] = i;
            succ[i] = i + 1;
        }

        Self {
            start: 0,
            succ,
            pred,
        }
    }

    fn remove(&mut self, idx: usize) {
        if idx == self.start {
            self.start = self.succ[idx];
        } else {
            self.succ[self.pred[idx]] = self.succ[idx];
            self.pred[self.succ[idx]] = self.pred[idx];
        }
        self.succ[idx] = 0; // Mark as inactive
    }

    #[allow(dead_code)]
    fn is_inactive(&self, idx: usize) -> bool {
        self.succ[idx] == 0
    }
}

/// Index into condensed distance matrix.
///
/// For a symmetric NxN matrix stored as a condensed upper triangular array,
/// the element (r, c) with r < c is at index:
/// `(2*N - 3 - r) * r / 2 + c - 1`
#[inline]
fn condensed_index(n: usize, r: usize, c: usize) -> usize {
    debug_assert!(r < c);
    ((2 * n - 3 - r) * r) / 2 + c - 1
}

/// Get distance from condensed matrix.
#[inline]
fn get_dist(dm: &[f64], n: usize, r: usize, c: usize) -> f64 {
    if r < c {
        dm[condensed_index(n, r, c)]
    } else {
        dm[condensed_index(n, c, r)]
    }
}

/// Set distance in condensed matrix.
#[inline]
fn set_dist(dm: &mut [f64], n: usize, r: usize, c: usize, val: f64) {
    if r < c {
        dm[condensed_index(n, r, c)] = val;
    } else {
        dm[condensed_index(n, c, r)] = val;
    }
}

/// Perform hierarchical clustering using NN-chain algorithm with UPGMA.
///
/// # Arguments
/// * `n` - Number of elements
/// * `distance_matrix` - Condensed distance matrix (upper triangular, length n*(n-1)/2)
/// * `members` - Initial member counts for each element
///
/// # Returns
/// ClusterResult containing the dendrogram
pub fn nn_chain_core(n: usize, distance_matrix: &mut [f64], members: &mut [i32]) -> ClusterResult {
    if n == 0 {
        return ClusterResult { nodes: vec![] };
    }
    if n == 1 {
        return ClusterResult { nodes: vec![] };
    }

    let mut result = ClusterResult {
        nodes: Vec::with_capacity(n - 1),
    };

    let mut nn_chain: Vec<usize> = vec![0; n];
    let mut nn_chain_tip: usize = 0;

    let mut active_nodes = DoublyLinkedList::new(n);

    for _ in 0..(n - 1) {
        let (mut idx1, mut idx2, mut min_dist);

        if nn_chain_tip <= 3 {
            // Start new chain from first active node
            idx1 = active_nodes.start;
            nn_chain[0] = idx1;
            nn_chain_tip = 1;

            // Find nearest neighbor
            idx2 = active_nodes.succ[idx1];
            min_dist = get_dist(distance_matrix, n, idx1, idx2);

            let mut i = active_nodes.succ[idx2];
            while i < n {
                let dist = get_dist(distance_matrix, n, idx1, i);
                if dist < min_dist {
                    min_dist = dist;
                    idx2 = i;
                }
                i = active_nodes.succ[i];
            }
        } else {
            // Continue from chain
            nn_chain_tip -= 3;
            idx1 = nn_chain[nn_chain_tip - 1];
            idx2 = nn_chain[nn_chain_tip];
            min_dist = get_dist(distance_matrix, n, idx1, idx2);
        }

        // Extend chain until we find mutual nearest neighbors
        loop {
            nn_chain[nn_chain_tip] = idx2;

            // Find nearest neighbor of idx2
            let mut i = active_nodes.start;
            while i < idx2 {
                let dist = get_dist(distance_matrix, n, i, idx2);
                if dist < min_dist {
                    min_dist = dist;
                    idx1 = i;
                }
                i = active_nodes.succ[i];
            }
            i = active_nodes.succ[idx2];
            while i < n {
                let dist = get_dist(distance_matrix, n, idx2, i);
                if dist < min_dist {
                    min_dist = dist;
                    idx1 = i;
                }
                i = active_nodes.succ[i];
            }

            idx2 = idx1;
            idx1 = nn_chain[nn_chain_tip];
            nn_chain_tip += 1;

            // Check for mutual nearest neighbors
            if nn_chain_tip >= 2 && idx2 == nn_chain[nn_chain_tip - 2] {
                break;
            }
        }

        // Record merge
        result.nodes.push(MergeNode {
            node1: idx1 as i32,
            node2: idx2 as i32,
            dist: min_dist,
        });

        // Ensure idx1 < idx2 for consistent updates
        if idx1 > idx2 {
            std::mem::swap(&mut idx1, &mut idx2);
        }

        // Update cluster sizes
        let size1 = members[idx1] as f64;
        let size2 = members[idx2] as f64;
        members[idx2] += members[idx1];

        // Remove smaller index from active nodes
        active_nodes.remove(idx1);

        // Update distances using UPGMA
        let s = size1 / (size1 + size2);
        let t = size2 / (size1 + size2);

        // Update distances for all remaining active nodes
        let mut i = active_nodes.start;
        while i < idx1 {
            let d_i_idx1 = get_dist(distance_matrix, n, i, idx1);
            let d_i_idx2 = get_dist(distance_matrix, n, i, idx2);
            let new_dist = s * d_i_idx1 + t * d_i_idx2;
            set_dist(distance_matrix, n, i, idx2, new_dist);
            i = active_nodes.succ[i];
        }
        // Skip idx1 (removed)
        i = active_nodes.succ[i];
        while i < idx2 {
            let d_idx1_i = get_dist(distance_matrix, n, idx1, i);
            let d_i_idx2 = get_dist(distance_matrix, n, i, idx2);
            let new_dist = s * d_idx1_i + t * d_i_idx2;
            set_dist(distance_matrix, n, i, idx2, new_dist);
            i = active_nodes.succ[i];
        }
        i = active_nodes.succ[idx2];
        while i < n {
            let d_idx1_i = get_dist(distance_matrix, n, idx1, i);
            let d_idx2_i = get_dist(distance_matrix, n, idx2, i);
            let new_dist = s * d_idx1_i + t * d_idx2_i;
            set_dist(distance_matrix, n, idx2, i, new_dist);
            i = active_nodes.succ[i];
        }
    }

    result
}

/// Union-Find data structure for cluster membership.
pub struct UnionFind {
    parent: Vec<i32>,
    next_parent: i32,
}

impl UnionFind {
    pub fn new(size: usize) -> Self {
        Self {
            parent: vec![0; 2 * size - 1],
            next_parent: size as i32,
        }
    }

    pub fn find(&mut self, mut idx: i32) -> i32 {
        if self.parent[idx as usize] != 0 {
            let mut p = idx;
            idx = self.parent[idx as usize];
            if self.parent[idx as usize] != 0 {
                // Path compression
                while self.parent[idx as usize] != 0 {
                    idx = self.parent[idx as usize];
                }
                while self.parent[p as usize] != idx {
                    let tmp = self.parent[p as usize];
                    self.parent[p as usize] = idx;
                    p = tmp;
                }
            }
        }
        idx
    }

    pub fn union(&mut self, node1: i32, node2: i32) {
        self.parent[node1 as usize] = self.next_parent;
        self.parent[node2 as usize] = self.next_parent;
        self.next_parent += 1;
    }
}

/// Generate SciPy-compatible dendrogram from clustering result.
///
/// # Arguments
/// * `result` - Clustering result from nn_chain_core
/// * `n` - Number of original elements
///
/// # Returns
/// Vector of [node1, node2, distance, size] arrays
pub fn generate_dendrogram(result: &ClusterResult, n: usize) -> Vec<[f64; 4]> {
    let mut nodes = result.nodes.clone();
    nodes.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());

    let mut uf = UnionFind::new(n);
    let mut output = Vec::with_capacity(n - 1);

    // Helper to get subtree size
    let get_size = |idx: i32, sizes: &[[f64; 4]]| -> f64 {
        if (idx as usize) < n {
            1.0
        } else {
            sizes[(idx as usize) - n][3]
        }
    };

    for node in &nodes {
        let node1 = uf.find(node.node1);
        let node2 = uf.find(node.node2);
        uf.union(node1, node2);

        let size = get_size(node1, &output) + get_size(node2, &output);

        let (n1, n2) = if node1 < node2 {
            (node1, node2)
        } else {
            (node2, node1)
        };

        output.push([n1 as f64, n2 as f64, node.dist, size]);
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_condensed_index() {
        // For n=4, the condensed matrix has 6 elements
        // Mapping: (0,1)->0, (0,2)->1, (0,3)->2, (1,2)->3, (1,3)->4, (2,3)->5
        assert_eq!(condensed_index(4, 0, 1), 0);
        assert_eq!(condensed_index(4, 0, 2), 1);
        assert_eq!(condensed_index(4, 0, 3), 2);
        assert_eq!(condensed_index(4, 1, 2), 3);
        assert_eq!(condensed_index(4, 1, 3), 4);
        assert_eq!(condensed_index(4, 2, 3), 5);
    }

    #[test]
    fn test_doubly_linked_list() {
        let mut list = DoublyLinkedList::new(5);
        assert_eq!(list.start, 0);
        assert_eq!(list.succ[0], 1);
        assert_eq!(list.succ[1], 2);

        list.remove(1);
        assert_eq!(list.succ[0], 2);
        assert!(list.is_inactive(1));

        list.remove(0);
        assert_eq!(list.start, 2);
    }

    #[test]
    fn test_nn_chain_simple() {
        // 3 elements with distances:
        // d(0,1) = 1.0, d(0,2) = 2.0, d(1,2) = 1.5
        let mut distances = vec![1.0, 2.0, 1.5];
        let mut members = vec![1, 1, 1];

        let result = nn_chain_core(3, &mut distances, &mut members);
        assert_eq!(result.nodes.len(), 2); // 3-1 = 2 merges
    }

    #[test]
    fn test_union_find() {
        let mut uf = UnionFind::new(5);

        // Initially, each element is its own root
        assert_eq!(uf.find(0), 0);
        assert_eq!(uf.find(1), 1);

        // Union 0 and 1
        uf.union(0, 1);
        assert_eq!(uf.find(0), uf.find(1));

        // Union 2 and 3
        uf.union(2, 3);
        assert_eq!(uf.find(2), uf.find(3));

        // 0,1 and 2,3 should be in different sets
        assert_ne!(uf.find(0), uf.find(2));
    }

    #[test]
    fn test_generate_dendrogram() {
        // Simple case: 3 elements
        let mut distances = vec![1.0, 3.0, 2.0]; // d(0,1)=1, d(0,2)=3, d(1,2)=2
        let mut members = vec![1, 1, 1];

        let result = nn_chain_core(3, &mut distances, &mut members);
        let dendrogram = generate_dendrogram(&result, 3);

        assert_eq!(dendrogram.len(), 2);
        // First merge should be 0 and 1 (distance 1.0)
        assert_eq!(dendrogram[0][2], 1.0);
    }
}
