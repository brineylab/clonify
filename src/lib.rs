use ahash::AHashMap;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(Clone, Copy, Debug)]
struct Params {
    shared_mutation_bonus: f64,
    length_penalty_multiplier: f64,
    v_penalty: f64,
    j_penalty: f64,
}

#[inline]
pub(crate) fn hamming(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

pub(crate) fn levenshtein(a: &[u8], b: &[u8]) -> usize {
    let n = a.len();
    let m = b.len();
    if n == 0 { return m; }
    if m == 0 { return n; }
    let mut prev: Vec<usize> = (0..=m).collect();
    let mut curr: Vec<usize> = vec![0; m + 1];
    for i in 1..=n {
        curr[0] = i;
        for j in 1..=m {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            let del = prev[j] + 1;
            let ins = curr[j - 1] + 1;
            let sub = prev[j - 1] + cost;
            curr[j] = del.min(ins).min(sub);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[m]
}

#[inline]
pub(crate) fn intersection_count(a: &[i32], b: &[i32]) -> usize {
    let mut i = 0usize;
    let mut j = 0usize;
    let mut count = 0usize;
    while i < a.len() && j < b.len() {
        if a[i] < b[j] {
            i += 1;
        } else if a[i] > b[j] {
            j += 1;
        } else {
            count += 1;
            i += 1;
            j += 1;
        }
    }
    count
}

#[pyclass]
struct NativeInputs {
    #[pyo3(get)]
    n: usize,
    cdr3: Vec<Vec<u8>>,
    v_ids: Vec<i32>,
    j_ids: Vec<i32>,
    mut_ids: Vec<i32>,
    mut_offsets: Vec<usize>,
    v_allelic: AHashMap<i32, Vec<i32>>,
}

#[pymethods]
impl NativeInputs {
    #[new]
    fn new(
        cdr3: Vec<&PyAny>,
        v_ids: Vec<i32>,
        j_ids: Vec<i32>,
        mut_ids: Vec<i32>,
        mut_offsets: Vec<usize>,
        v_allelic: Vec<(i32, Vec<i32>)>,
    ) -> PyResult<Self> {
        if v_ids.len() != cdr3.len() || j_ids.len() != cdr3.len() {
            return Err(PyValueError::new_err("Length mismatch"));
        }
        let n = cdr3.len();
        let mut cdr3_bytes: Vec<Vec<u8>> = Vec::with_capacity(n);
        for s in cdr3.into_iter() {
            let s_bytes: Vec<u8> = s.extract::<String>()?.into_bytes();
            cdr3_bytes.push(s_bytes);
        }
        if mut_offsets.len() != n + 1 {
            return Err(PyValueError::new_err("mut_offsets must be length n+1"));
        }
        let mut map: AHashMap<i32, Vec<i32>> = AHashMap::new();
        for (k, mut v) in v_allelic.into_iter() {
            v.sort_unstable();
            v.dedup();
            map.insert(k, v);
        }
        Ok(NativeInputs { n, cdr3: cdr3_bytes, v_ids, j_ids, mut_ids, mut_offsets, v_allelic: map })
    }
}

#[inline]
fn pair_distance(
    inp: &NativeInputs,
    i: usize,
    j: usize,
    params: &Params,
) -> f64 {
    let s1 = &inp.cdr3[i];
    let s2 = &inp.cdr3[j];
    let len1 = s1.len();
    let len2 = s2.len();
    let mut germline_penalty = 0.0f64;
    if inp.v_ids[i] != inp.v_ids[j] { germline_penalty += params.v_penalty; }
    if inp.j_ids[i] != inp.j_ids[j] { germline_penalty += params.j_penalty; }

    let length_penalty = (len1 as isize - len2 as isize).abs() as f64 * params.length_penalty_multiplier;
    let min_len = len1.min(len2) as f64;
    let dist = if len1 == len2 { hamming(s1, s2) } else { levenshtein(s1, s2) } as f64;

    let a0 = inp.mut_offsets[i];
    let a1 = inp.mut_offsets[i + 1];
    let b0 = inp.mut_offsets[j];
    let b1 = inp.mut_offsets[j + 1];
    let mut_a = &inp.mut_ids[a0..a1];
    let mut_b = &inp.mut_ids[b0..b1];
    let empty: Vec<i32> = Vec::new();
    let allelic_a = inp.v_allelic.get(&inp.v_ids[i]).unwrap_or(&empty);
    let allelic_b = inp.v_allelic.get(&inp.v_ids[j]).unwrap_or(&empty);
    let filtered_a: Vec<i32> = mut_a.iter().copied().filter(|x| allelic_a.binary_search(x).is_err()).collect();
    let filtered_b: Vec<i32> = mut_b.iter().copied().filter(|x| allelic_b.binary_search(x).is_err()).collect();
    let shared = intersection_count(&filtered_a, &filtered_b) as f64;
    let mutation_bonus = shared * params.shared_mutation_bonus;

    let score = germline_penalty + ((dist + length_penalty - mutation_bonus) / min_len);
    if score < 0.001 { 0.001 } else { score }
}

#[pyfunction]
fn average_linkage_cutoff(
    inp: &NativeInputs,
    shared_mutation_bonus: f64,
    length_penalty_multiplier: f64,
    v_penalty: f64,
    j_penalty: f64,
    distance_cutoff: f64,
    n_threads: Option<usize>,
) -> PyResult<Vec<i32>> {
    if let Some(t) = n_threads { rayon::ThreadPoolBuilder::new().num_threads(t).build_global().ok(); }
    let params = Params { shared_mutation_bonus, length_penalty_multiplier, v_penalty, j_penalty };
    let n = inp.n;
    if n == 0 { return Ok(vec![]); }
    if n == 1 { return Ok(vec![0]); }

    let mut active: Vec<bool> = vec![true; n];
    let mut labels: Vec<i32> = (0..n as i32).collect();
    let mut cluster_sizes: Vec<usize> = vec![1; n];
    let mut members: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();

    loop {
        let mut best_pair: Option<(usize, usize, f64)> = None;
        for i in 0..n {
            if !active[i] { continue; }
            for j in (i + 1)..n {
                if !active[j] { continue; }
                let m1 = &members[i];
                let m2 = &members[j];
                let total = (m1.len() * m2.len()) as f64;
                let sum: f64 = m1.par_iter().map(|&ii| {
                    m2.iter().map(|&jj| pair_distance(inp, ii, jj, &params)).sum::<f64>()
                }).sum();
                let d = sum / total;
                if d <= distance_cutoff {
                    match best_pair {
                        None => best_pair = Some((i, j, d)),
                        Some((_, _, bd)) if d < bd => best_pair = Some((i, j, d)),
                        _ => {}
                    }
                }
            }
        }
        match best_pair {
            None => break,
            Some((i, j, _d)) => {
                let mut b = Vec::new();
                b.append(&mut members[j]);
                members[i].append(&mut b);
                active[j] = false;
                cluster_sizes[i] += cluster_sizes[j];
                let new_label = labels[i];
                for &m in members[i].iter() {
                    labels[m] = new_label;
                }
            }
        }
    }

    let mut map: AHashMap<i32, i32> = AHashMap::new();
    let mut next = 0i32;
    for l in labels.iter_mut() {
        let entry = map.entry(*l).or_insert_with(|| { let v = next; next += 1; v });
        *l = *entry;
    }
    Ok(labels)
}

#[pymodule]
fn _native(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<NativeInputs>()?;
    m.add_function(wrap_pyfunction!(average_linkage_cutoff, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_hamming() {
        assert_eq!(hamming(b"ABCDEFG", b"ABCXEZG"), 2);
    }
    #[test]
    fn test_levenshtein_basic() {
        assert_eq!(levenshtein(b"", b"abc"), 3);
        assert_eq!(levenshtein(b"abc", b""), 3);
        assert_eq!(levenshtein(b"abc", b"abc"), 0);
        assert_eq!(levenshtein(b"kitten", b"sitting"), 3);
    }
    #[test]
    fn test_intersection_count() {
        let a = vec![1, 2, 3, 5, 7];
        let b = vec![2, 3, 4, 7, 9];
        assert_eq!(intersection_count(&a, &b), 3);
    }
}


