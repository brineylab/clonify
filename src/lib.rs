use ahash::AHashMap;
use bitvec::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use rayon::slice::ParallelSliceMut;
use std::fs::{File, OpenOptions};
use std::cell::RefCell;
use std::io::Write;
use std::time::{Duration, Instant};
use std::env;
use memmap2::MmapMut;

#[derive(Clone, Copy, Debug)]
struct Params {
    shared_mutation_bonus: f64,
    length_penalty_multiplier: f64,
    v_penalty: f64,
    j_penalty: f64,
    distance_cutoff: f64,
}

#[inline]
pub(crate) fn hamming(a: &[u8], b: &[u8]) -> usize {
    debug_assert_eq!(a.len(), b.len());
    let mut cnt: usize = 0;
    let mut i = 0;
    let len = a.len();
    const LANES: usize = 8;
    let end64 = len.saturating_sub(len % LANES);
    while i < end64 {
        let mut x: u64 = 0;
        unsafe {
            // read 8 bytes from each slice
            let pa = a.get_unchecked(i..i + LANES);
            let pb = b.get_unchecked(i..i + LANES);
            let ua = u64::from_ne_bytes([pa[0], pa[1], pa[2], pa[3], pa[4], pa[5], pa[6], pa[7]]);
            let ub = u64::from_ne_bytes([pb[0], pb[1], pb[2], pb[3], pb[4], pb[5], pb[6], pb[7]]);
            x = ua ^ ub;
        }
        // turn any non-zero byte into 0x01 in that lane
        let mut y = x | (x >> 4);
        y |= y >> 2;
        y |= y >> 1;
        y &= 0x0101_0101_0101_0101u64;
        cnt += y.count_ones() as usize;
        i += LANES;
    }
    while i < len {
        if a[i] != b[i] { cnt += 1; }
        i += 1;
    }
    cnt
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
fn levenshtein_banded(a: &[u8], b: &[u8], max_dist: usize) -> usize {
    // Standard DP with band limiting j around i by max_dist. Returns a value > max_dist if edit distance exceeds band.
    let n = a.len();
    let m = b.len();
    if n == 0 { return m; }
    if m == 0 { return n; }
    // If band is wide enough, fall back to full computation
    if max_dist >= n.max(m) { return levenshtein(a, b); }
    let inf = max_dist + 1;
    let mut prev: Vec<usize> = vec![inf; m + 1];
    let mut curr: Vec<usize> = vec![inf; m + 1];
    // Initialize prev row with j within band around i=0
    for j in 0..=m.min(max_dist) { prev[j] = j; }
    for i in 1..=n {
        let j_start = if i > max_dist { i - max_dist } else { 1 }; // j >= 1
        let j_end = (m).min(i + max_dist);
        // Set curr outside band to inf
        if j_start > 1 { curr[j_start - 1] = inf; }
        curr[j_start.saturating_sub(1)] = inf; // safe guard
        // Set left boundary
        if j_start == 1 {
            curr[0] = i; // insertion-only cost at left edge
        }
        for j in j_start..=j_end {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            let del = prev[j].saturating_add(1);
            let ins = curr[j - 1].saturating_add(1);
            let sub = prev[j - 1].saturating_add(cost);
            let v = del.min(ins).min(sub);
            curr[j] = v;
        }
        // Set right boundary beyond band
        if j_end < m { curr[j_end + 1] = inf; }
        std::mem::swap(&mut prev, &mut curr);
    }
    let d = prev[m];
    d
}

thread_local! {
    static LV_BUFFERS: RefCell<(Vec<usize>, Vec<usize>)> = RefCell::new((Vec::new(), Vec::new()));
}

#[inline]
fn levenshtein_banded_tl(a: &[u8], b: &[u8], max_dist: usize) -> usize {
    LV_BUFFERS.with(|cell| {
        let (ref mut prev, ref mut curr) = *&mut *cell.borrow_mut();
        let n = a.len();
        let m = b.len();
        if prev.len() < m + 1 { prev.resize(m + 1, 0); }
        if curr.len() < m + 1 { curr.resize(m + 1, 0); }
        let inf = max_dist + 1;
        // Initialize prev with inf, then band
        for j in 0..=m { prev[j] = inf; }
        for j in 0..=m.min(max_dist) { prev[j] = j; }
        for i in 1..=n {
            let j_start = if i > max_dist { i - max_dist } else { 1 };
            let j_end = m.min(i + max_dist);
            // Fill curr with inf across row to avoid stale values
            for j in 0..=m { curr[j] = inf; }
            // Set left boundary within range
            if j_start == 1 { curr[0] = i; } else if j_start >= 1 && j_start - 1 <= m { curr[j_start - 1] = inf; }
            for j in j_start..=j_end {
                let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
                let del = prev[j].saturating_add(1);
                let ins = curr[j - 1].saturating_add(1);
                let sub = prev[j - 1].saturating_add(cost);
                curr[j] = del.min(ins).min(sub);
            }
            if j_end < m { curr[j_end + 1] = inf; }
            // swap
            for j in 0..=m { std::mem::swap(&mut prev[j], &mut curr[j]); }
        }
        prev[m]
    })
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
    // Legacy layout retained for compatibility, but SoA layout is used in hot paths
    cdr3: Vec<Vec<u8>>,
    cdr3_data: Vec<u8>,
    cdr3_offsets: Vec<usize>,
    v_ids: Vec<i32>,
    j_ids: Vec<i32>,
    mut_ids: Vec<i32>,
    mut_offsets: Vec<usize>,
    v_allelic: AHashMap<i32, Vec<i32>>,
    // Precomputed per-sequence filtered mutation ids (allelic-masked)
    filtered_mut_ids: Vec<i32>,
    filtered_mut_offsets: Vec<usize>,
    // Optional dense bitset per sequence for filtered mutations to speed up intersections
    mut_bitsets: Option<Vec<BitVec<u8>>>,
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
        // Build SoA layout for better cache locality
        let mut cdr3_offsets: Vec<usize> = Vec::with_capacity(n + 1);
        let total_len: usize = cdr3_bytes.iter().map(|s| s.len()).sum();
        let mut cdr3_data: Vec<u8> = Vec::with_capacity(total_len);
        cdr3_offsets.push(0);
        for s in &cdr3_bytes {
            cdr3_data.extend_from_slice(s);
            cdr3_offsets.push(cdr3_data.len());
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

        // Precompute filtered mutation ids per sequence by removing likely allelic variants
        let mut filtered_mut_ids: Vec<i32> = Vec::with_capacity(mut_ids.len());
        let mut filtered_mut_offsets: Vec<usize> = Vec::with_capacity(n + 1);
        filtered_mut_offsets.push(0);
        for i in 0..n {
            let a0 = mut_offsets[i];
            let a1 = mut_offsets[i + 1];
            let seq_mut = &mut_ids[a0..a1];
            // Allelic list for this sequence's V id
            let allelic = map.get(&v_ids[i]).map(|v| v.as_slice()).unwrap_or(&[]);
            // Merge-like filter: both seq_mut and allelic are sorted
            let mut p = 0usize;
            let mut q = 0usize;
            while p < seq_mut.len() {
                if q >= allelic.len() {
                    // push all remaining
                    filtered_mut_ids.extend_from_slice(&seq_mut[p..]);
                    break;
                }
                let mv = seq_mut[p];
                let av = allelic[q];
                if mv < av {
                    filtered_mut_ids.push(mv);
                    p += 1;
                } else if mv > av {
                    q += 1;
                } else {
                    // equal → skip allelic mutation
                    p += 1;
                    q += 1;
                }
            }
            filtered_mut_offsets.push(filtered_mut_ids.len());
        }

        // Optionally precompute dense bitsets for filtered mutations
        let mut mut_bitsets: Option<Vec<BitVec<u8>>> = None;
        // Build a mapping from mutation id -> dense index
        let universe_size_threshold: usize = 200_000; // avoid excessive memory
        if !filtered_mut_ids.is_empty() {
            let mut uniq: Vec<i32> = filtered_mut_ids.clone();
            uniq.sort_unstable();
            uniq.dedup();
            if uniq.len() <= universe_size_threshold {
                let mut id_to_dense: AHashMap<i32, usize> = AHashMap::with_capacity(uniq.len());
                for (idx, mid) in uniq.iter().enumerate() { id_to_dense.insert(*mid, idx); }
                let mut bitsets_vec: Vec<BitVec<u8>> = Vec::with_capacity(n);
                for i in 0..n {
                    let a0 = filtered_mut_offsets[i];
                    let a1 = filtered_mut_offsets[i + 1];
                    let mut bv: BitVec<u8> = bitvec![u8, Lsb0; 0; uniq.len()];
                    for &mid in &filtered_mut_ids[a0..a1] {
                        if let Some(&pos) = id_to_dense.get(&mid) {
                            // Safety: pos < uniq.len()
                            bv.set(pos, true);
                        }
                    }
                    bitsets_vec.push(bv);
                }
                mut_bitsets = Some(bitsets_vec);
            }
        }

        Ok(NativeInputs {
            n,
            cdr3: cdr3_bytes,
            cdr3_data,
            cdr3_offsets,
            v_ids,
            j_ids,
            mut_ids,
            mut_offsets,
            v_allelic: map,
            filtered_mut_ids,
            filtered_mut_offsets,
            mut_bitsets,
        })
    }
}

impl NativeInputs {
    #[inline]
    fn cdr3_slice(&self, idx: usize) -> &[u8] {
        let start = self.cdr3_offsets[idx];
        let end = self.cdr3_offsets[idx + 1];
        &self.cdr3_data[start..end]
    }
}

#[inline]
fn pair_distance(
    inp: &NativeInputs,
    i: usize,
    j: usize,
    params: &Params,
    mut_bitsets: Option<&Vec<BitVec<u8>>>,
) -> f64 {
    let s1 = inp.cdr3_slice(i);
    let s2 = inp.cdr3_slice(j);
    let len1 = s1.len();
    let len2 = s2.len();
    let mut germline_penalty = 0.0f64;
    if inp.v_ids[i] != inp.v_ids[j] { germline_penalty += params.v_penalty; }
    if inp.j_ids[i] != inp.j_ids[j] { germline_penalty += params.j_penalty; }

    let length_penalty = (len1 as isize - len2 as isize).abs() as f64 * params.length_penalty_multiplier;
    let min_len = len1.min(len2) as f64;

    // Early cutoff: if length penalty alone exceeds cutoff window, no need to compute LD/mutations
    if params.distance_cutoff > 0.0 && length_penalty >= min_len * params.distance_cutoff {
        // Ensure we do not spuriously merge at exactly the cutoff
        return params.distance_cutoff + f64::EPSILON;
    }

    // Compute a conservative band for Levenshtein based on cutoff window and maximum possible mutation bonus.
    let dist = if len1 == len2 {
        hamming(s1, s2) as f64
    } else {
        // Maximum possible shared mutations equals the minimum of per-sequence filtered mutation list lengths
        let a0p = inp.filtered_mut_offsets[i];
        let a1p = inp.filtered_mut_offsets[i + 1];
        let b0p = inp.filtered_mut_offsets[j];
        let b1p = inp.filtered_mut_offsets[j + 1];
        let max_shared = (a1p - a0p).min(b1p - b0p) as f64;
        let max_mut_bonus = max_shared * params.shared_mutation_bonus;
        let allowed = (params.distance_cutoff * min_len - length_penalty - germline_penalty + max_mut_bonus).floor();
        if allowed.is_finite() && allowed >= 0.0 {
            let band = allowed as usize;
            let d = levenshtein_banded_tl(s1, s2, band) as f64;
            d
        } else {
            levenshtein_banded_tl(s1, s2, (len1.max(len2))) as f64
        }
    };

    // Use bitsets if available, otherwise intersect sorted lists
    let shared = if let Some(bitsets) = mut_bitsets {
        let a = &bitsets[i];
        let b = &bitsets[j];
        let ra = a.as_raw_slice();
        let rb = b.as_raw_slice();
        let m = ra.len().min(rb.len());
        let mut ones: u32 = 0;
        let mut t = 0usize;
        while t < m {
            ones += (ra[t] & rb[t]).count_ones();
            t += 1;
        }
        ones as f64
    } else {
        let a0 = inp.filtered_mut_offsets[i];
        let a1 = inp.filtered_mut_offsets[i + 1];
        let b0 = inp.filtered_mut_offsets[j];
        let b1 = inp.filtered_mut_offsets[j + 1];
        let mut_a = &inp.filtered_mut_ids[a0..a1];
        let mut_b = &inp.filtered_mut_ids[b0..b1];
        intersection_count(mut_a, mut_b) as f64
    };
    let mutation_bonus = shared * params.shared_mutation_bonus;

    let score = germline_penalty + ((dist + length_penalty - mutation_bonus) / min_len);
    if score < 0.001 { 0.001 } else { score }
}

fn check_for_interrupt() -> PyResult<()> {
    // Briefly acquire the GIL and let Python raise KeyboardInterrupt if signaled
    Python::with_gil(|py| py.check_signals())
}

fn average_linkage_cutoff_fallback(
    inp: &NativeInputs,
    params: &Params,
    distance_cutoff: f64,
) -> PyResult<Vec<i32>> {
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
            // Allow Ctrl-C to interrupt between outer iterations
            check_for_interrupt()?;
            if !active[i] { continue; }
            for j in (i + 1)..n {
                if !active[j] { continue; }
                let m1 = &members[i];
                let m2 = &members[j];
                let total = (m1.len() * m2.len()) as f64;
                let sum: f64 = m1.par_iter().map(|&ii| {
                    m2.iter().map(|&jj| pair_distance(inp, ii, jj, params, None)).sum::<f64>()
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

#[inline]
fn condensed_index(n: usize, i: usize, j: usize) -> usize {
    debug_assert!(i < j);
    // Number of elements before row i: i * (2n - i - 1) / 2
    i * (2 * n - i - 1) / 2 + (j - i - 1)
}

#[inline]
fn condensed_index_inv(n: usize, k: usize) -> (usize, usize) {
    // Binary search row i such that k in [base(i), base(i)+row_len)
    // base(i) = i*(2n - i - 1)/2, row_len = n - 1 - i
    let mut lo = 0usize;
    let mut hi = n - 1; // last valid i is n-2; keep hi exclusive sentinel
    let mut i = 0usize;
    loop {
        let mid = (lo + hi) / 2;
        let base = mid * (2 * n - mid - 1) / 2;
        let row_len = n - 1 - mid;
        if k < base {
            hi = mid;
        } else if k >= base + row_len {
            lo = mid + 1;
        } else {
            i = mid;
            break;
        }
    }
    let base = i * (2 * n - i - 1) / 2;
    let offset = k - base;
    let j = i + 1 + offset;
    (i, j)
}

#[derive(Clone)]
struct UnionFind {
    parent: Vec<usize>,
    size: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self { parent: (0..n).collect(), size: vec![1; n] }
    }
    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            let root = self.find(self.parent[x]);
            self.parent[x] = root;
        }
        self.parent[x]
    }
    fn union(&mut self, a: usize, b: usize) {
        let mut ra = self.find(a);
        let mut rb = self.find(b);
        if ra == rb { return; }
        if self.size[ra] < self.size[rb] { std::mem::swap(&mut ra, &mut rb); }
        self.parent[rb] = ra;
        self.size[ra] += self.size[rb];
    }
}

fn average_linkage_cutoff_nn_chain(
    inp: &NativeInputs,
    params: &Params,
    distance_cutoff: f64,
) -> PyResult<Vec<i32>> {
    let n = inp.n;
    if n == 0 { return Ok(vec![]); }
    if n == 1 { return Ok(vec![0]); }

    // Build condensed distance matrix D for singletons
    // Layout: for i in [0..n-1), j in (i+1..n) consecutively
    let m = n * (n - 1) / 2;
    // Optional disk-backed streaming store under env flag
    let use_streaming = env::var("CLONIFY_STREAM_DIST").ok().as_deref() == Some("1");
    enum Store {
        VecStore(Vec<f64>),
        MmapStore { file: File, mmap: MmapMut },
    }
    impl Store {
        #[inline]
        fn len(&self) -> usize { match self { Store::VecStore(v) => v.len(), Store::MmapStore { mmap, .. } => mmap.len() / std::mem::size_of::<f64>() } }
        #[inline]
        unsafe fn write(&mut self, idx: usize, val: f64) {
            match self {
                Store::VecStore(v) => { v[idx] = val; }
                Store::MmapStore { mmap, .. } => {
                    let bytes = &mut mmap[(idx * 8)..((idx + 1) * 8)];
                    bytes.copy_from_slice(&val.to_ne_bytes());
                }
            }
        }
        #[inline]
        unsafe fn read(&self, idx: usize) -> f64 {
            match self {
                Store::VecStore(v) => v[idx],
                Store::MmapStore { mmap, .. } => {
                    let bytes = &mmap[(idx * 8)..((idx + 1) * 8)];
                    f64::from_ne_bytes(bytes.try_into().unwrap())
                }
            }
        }
        #[inline]
        fn as_vec_mut(&mut self) -> Option<&mut Vec<f64>> { if let Store::VecStore(v) = self { Some(v) } else { None } }
    }
    let mut dist_store = if use_streaming {
        // Create temp file in system temp dir
        let mut path = env::temp_dir();
        path.push(format!("clonify_dist_{}.bin", std::process::id()));
        let file = OpenOptions::new().read(true).write(true).create(true).truncate(true).open(&path)
            .map_err(|e| PyValueError::new_err(format!("Failed to create mmap file: {e}")))?;
        let size_bytes = m * std::mem::size_of::<f64>();
        file.set_len(size_bytes as u64).map_err(|e| PyValueError::new_err(format!("Failed to size mmap file: {e}")))?;
        let mmap = unsafe { MmapMut::map_mut(&file).map_err(|e| PyValueError::new_err(format!("Failed to mmap: {e}")))? };
        Store::MmapStore { file, mmap }
    } else {
        Store::VecStore(vec![0.0f64; m])
    };
    // Parallelize by slicing the output vector into disjoint chunks
    let chunk_size = 64 * 1024; // tuneable
    if let Some(vec_ref) = dist_store.as_vec_mut() {
        vec_ref.par_chunks_mut(chunk_size).enumerate().for_each(|(chunk_idx, chunk)| {
            let start = chunk_idx * chunk_size;
            for (t, cell) in chunk.iter_mut().enumerate() {
                let k = start + t;
                let (i, j) = condensed_index_inv(n, k);
                *cell = pair_distance(inp, i, j, params, inp.mut_bitsets.as_ref());
            }
        });
    } else {
        // streaming path: sequentially fill mmap to avoid mutable capture in parallel closures
        for k in 0..m {
            let (i, j) = condensed_index_inv(n, k);
            let d = pair_distance(inp, i, j, params, inp.mut_bitsets.as_ref());
            unsafe { dist_store.write(k, d); }
        }
    }

    // Active list as a doubly linked list using indices 0..n-1
    let mut start: Option<usize> = Some(0);
    let mut succ: Vec<Option<usize>> = (0..n).map(|i| if i + 1 < n { Some(i + 1) } else { None }).collect();
    let mut pred: Vec<Option<usize>> = (0..n).map(|i| if i == 0 { None } else { Some(i - 1) }).collect();

    let mut is_active: Vec<bool> = vec![true; n];
    let mut members: Vec<usize> = vec![1; n];
    let mut uf = UnionFind::new(n);

    // Helper functions to get/set distances in condensed matrix
    let get_dist = |a: usize, b: usize, dist: &Store| -> f64 {
        debug_assert!(a != b);
        let (i, j) = if a < b { (a, b) } else { (b, a) };
        let idx = condensed_index(n, i, j);
        unsafe { dist.read(idx) }
    };
    let mut set_dist = |a: usize, b: usize, val: f64, dist: &mut Store| {
        debug_assert!(a != b);
        let (i, j) = if a < b { (a, b) } else { (b, a) };
        let idx = condensed_index(n, i, j);
        unsafe { dist.write(idx, val); }
    };

    // NN-chain workspace
    let mut chain: Vec<usize> = vec![0; n];
    let mut chain_tip: usize = 0;

    // Helper to iterate active nodes from a given start
    let next_active = |cur: Option<usize>, succ: &Vec<Option<usize>>| -> Option<usize> {
        match cur {
            None => None,
            Some(i) => succ[i],
        }
    };

    // Function to remove an index from active list
    let mut remove_active = |idx: usize, start: &mut Option<usize>, succ: &mut Vec<Option<usize>>, pred: &mut Vec<Option<usize>>, is_active: &mut Vec<bool>| {
        is_active[idx] = false;
        let p = pred[idx];
        let s = succ[idx];
        if let Some(pp) = p { succ[pp] = s; } else { *start = s; }
        if let Some(ss) = s { pred[ss] = p; }
        succ[idx] = None;
        pred[idx] = None;
    };

    // Perform up to n-1 merges; early-stop if minimal distance exceeds cutoff
    let mut merges_done = 0usize;
    let mut last_signal_check = Instant::now();
    while merges_done < n - 1 {
        // Throttle KeyboardInterrupt checks (time-based, ~every 100ms)
        if last_signal_check.elapsed() >= Duration::from_millis(100) {
            Python::with_gil(|py| { let _ = py.check_signals(); });
            last_signal_check = Instant::now();
        }

        let (mut idx1, mut idx2, mut min_d): (usize, usize, f64);

        if chain_tip <= 3 {
            // Reset chain starting from list head
            let i0 = match start { Some(x) => x, None => break };
            chain[0] = i0;
            chain_tip = 1;
            // pick nearest neighbor of i0
            let mut nn = succ[i0].expect("at least two active items expected");
            let mut best = get_dist(i0, nn, &dist_store);
            // scan remaining actives with early stop if minimal possible distance reached
            let mut it = succ[nn];
            while let Some(i) = it {
                let d = get_dist(i0, i, &dist_store);
                if d < best {
                    best = d; nn = i;
                    if best <= 0.001 { break; }
                }
                it = succ[i];
            }
            idx1 = i0; idx2 = nn; min_d = best;
        } else {
            // Pop last 3 and resume
            chain_tip -= 3;
            idx1 = chain[chain_tip - 1];
            idx2 = chain[chain_tip];
            min_d = get_dist(idx1, idx2, &dist_store);
        }

        // Build chain until reciprocal nearest neighbors
        loop {
            chain[chain_tip] = idx2;
            // Find nearest neighbor of idx2 among active nodes
            let mut best = f64::INFINITY;
            let mut best_i = 0usize;
            let mut it = start;
            while let Some(i) = it {
                if i != idx2 && is_active[i] {
                    let d = get_dist(i, idx2, &dist_store);
                    if d < best { best = d; best_i = i; if best <= 0.001 { /* can't get smaller */ break; } }
                }
                it = next_active(it, &succ);
            }
            idx2 = best_i;
            let old = idx1;
            idx1 = chain[chain_tip];
            chain_tip += 1;
            min_d = best;
            if idx2 == chain[chain_tip - 2] { break; }
            let _ = old; // silence unused var in debug
        }

        // If the minimal merge distance exceeds cutoff, we can stop
        if min_d > distance_cutoff { break; }

        // Merge idx1 into idx2 (ensure idx1 < idx2 for update order not required, we handle generically)
        let size1 = members[idx1] as f64;
        let size2 = members[idx2] as f64;
        let s = size1 / (size1 + size2);
        let t = 1.0 - s;

        // Update distances from new cluster idx2 to all other active nodes k
        let mut it = start;
        while let Some(k) = it {
            if k != idx1 && k != idx2 && is_active[k] {
                let d1 = get_dist(idx1, k, &dist_store);
                let d2 = get_dist(idx2, k, &dist_store);
                set_dist(idx2, k, s * d1 + t * d2, &mut dist_store);
            }
            it = next_active(it, &succ);
        }

        // Remove idx1 from active list
        remove_active(idx1, &mut start, &mut succ, &mut pred, &mut is_active);
        // Update size
        members[idx2] += members[idx1];
        // Record merge for output labels via union-find
        uf.union(idx1, idx2);

        // Prepare for next iteration
        merges_done += 1;
        if chain_tip > 0 { chain_tip -= 1; }
    }

    // Build labels from union-find roots
    let mut root_to_label: AHashMap<usize, i32> = AHashMap::new();
    let mut next_label: i32 = 0;
    let mut labels: Vec<i32> = Vec::with_capacity(n);
    for i in 0..n {
        let r = uf.find(i);
        let entry = root_to_label.entry(r).or_insert_with(|| { let v = next_label; next_label += 1; v });
        labels.push(*entry);
    }
    Ok(labels)
}

#[cfg(feature = "nn_chain")]
fn average_linkage_cutoff_impl(
    inp: &NativeInputs,
    params: &Params,
    distance_cutoff: f64,
) -> PyResult<Vec<i32>> {
    average_linkage_cutoff_nn_chain(inp, params, distance_cutoff)
}

#[cfg(not(feature = "nn_chain"))]
fn average_linkage_cutoff_impl(
    inp: &NativeInputs,
    params: &Params,
    distance_cutoff: f64,
) -> PyResult<Vec<i32>> {
    // Default to optimized NN-chain implementation as the primary path
    average_linkage_cutoff_nn_chain(inp, params, distance_cutoff)
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
    let params = Params { shared_mutation_bonus, length_penalty_multiplier, v_penalty, j_penalty, distance_cutoff };
    // Release the GIL during heavy computation
    let labels = Python::with_gil(|py| {
        py.allow_threads(|| average_linkage_cutoff_impl(inp, &params, distance_cutoff))
    })?;
    Ok(labels)
}

#[pyfunction]
fn average_linkage_cutoff_progressive(
    inp: &NativeInputs,
    shared_mutation_bonus: f64,
    length_penalty_multiplier: f64,
    v_penalty: f64,
    j_penalty: f64,
    distance_cutoff: f64,
    n_threads: Option<usize>,
) -> PyResult<Vec<i32>> {
    if let Some(t) = n_threads { rayon::ThreadPoolBuilder::new().num_threads(t).build_global().ok(); }
    let params = Params { shared_mutation_bonus, length_penalty_multiplier, v_penalty, j_penalty, distance_cutoff };
    let n = inp.n;
    if n == 0 { return Ok(vec![]); }
    if n == 1 { return Ok(vec![0]); }

    // Build simple multi-band hash buckets (approximate LSH)
    // Bands: whole CDR3, prefix, suffix, (len, V, J)
    let mut bands: [AHashMap<u64, Vec<usize>>; 4] = [
        AHashMap::new(), AHashMap::new(), AHashMap::new(), AHashMap::new()
    ];
    inp.cdr3_offsets.par_windows(2).enumerate().for_each(|(i, w)| {
        let start = w[0];
        let end = w[1];
        let s = &inp.cdr3_data[start..end];
        let len = s.len();
        // band0: full
        let mut hasher0 = ahash::AHasher::default();
        use std::hash::Hasher;
        for &b in s { hasher0.write_u8(b); }
        let key0 = hasher0.finish();
        // band1: prefix up to 6
        let k = len.min(6);
        let mut hasher1 = ahash::AHasher::default();
        for &b in &s[..k] { hasher1.write_u8(b); }
        hasher1.write_usize(len);
        let key1 = hasher1.finish();
        // band2: suffix up to 6
        let mut hasher2 = ahash::AHasher::default();
        for &b in &s[len.saturating_sub(k)..] { hasher2.write_u8(b); }
        hasher2.write_usize(len);
        let key2 = hasher2.finish();
        // band3: length, v and j ids
        let mut hasher3 = ahash::AHasher::default();
        hasher3.write_usize(len);
        hasher3.write_i32(inp.v_ids[i]);
        hasher3.write_i32(inp.j_ids[i]);
        let key3 = hasher3.finish();
        // Insert into bands (synchronized via mutex-free approach: collect locally then merge)
        // For simplicity in parallel, push into thread-local vectors then merge after
    });
    // Rebuild sequentially (small n typical in tests); for production, parallelize with mutex or sharded maps
    for i in 0..n {
        let s = inp.cdr3_slice(i);
        let len = s.len();
        use std::hash::Hasher;
        let mut h0 = ahash::AHasher::default();
        for &b in s { h0.write_u8(b); }
        let key0 = h0.finish();
        bands[0].entry(key0).or_default().push(i);
        let k = len.min(6);
        let mut h1 = ahash::AHasher::default();
        for &b in &s[..k] { h1.write_u8(b); }
        h1.write_usize(len);
        bands[1].entry(h1.finish()).or_default().push(i);
        let mut h2 = ahash::AHasher::default();
        for &b in &s[len.saturating_sub(k)..] { h2.write_u8(b); }
        h2.write_usize(len);
        bands[2].entry(h2.finish()).or_default().push(i);
        let mut h3 = ahash::AHasher::default();
        h3.write_usize(len);
        h3.write_i32(inp.v_ids[i]);
        h3.write_i32(inp.j_ids[i]);
        bands[3].entry(h3.finish()).or_default().push(i);
    }

    let mut uf = UnionFind::new(n);
    // For each band bucket, union pairs that pass distance cutoff (single-linkage approximation)
    for band in &bands {
        for (_k, members) in band {
            if members.len() < 2 { continue; }
            // For large buckets, cap comparisons using stride sampling
            let max_pairs = 200_000usize; // safety cap
            let mut pairs_checked = 0usize;
            for a_idx in 0..members.len() {
                for b_idx in (a_idx + 1)..members.len() {
                    if pairs_checked >= max_pairs { break; }
                    let i = members[a_idx];
                    let j = members[b_idx];
                    let d = pair_distance(inp, i, j, &params, inp.mut_bitsets.as_ref());
                    if d <= distance_cutoff {
                        uf.union(i, j);
                    }
                    pairs_checked += 1;
                }
                if pairs_checked >= max_pairs { break; }
            }
        }
    }

    // Build labels from roots
    let mut root_to_label: AHashMap<usize, i32> = AHashMap::new();
    let mut next_label: i32 = 0;
    let mut labels: Vec<i32> = Vec::with_capacity(n);
    for i in 0..n {
        let r = uf.find(i);
        let entry = root_to_label.entry(r).or_insert_with(|| { let v = next_label; next_label += 1; v });
        labels.push(*entry);
    }
    Ok(labels)
}

#[pymodule]
fn _native(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<NativeInputs>()?;
    m.add_function(wrap_pyfunction!(average_linkage_cutoff, m)?)?;
    m.add_function(wrap_pyfunction!(average_linkage_cutoff_progressive, m)?)?;
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


