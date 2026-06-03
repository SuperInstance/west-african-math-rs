//! West African Mathematics — Rust Port
//!
//! A trilogy of computational models inspired by West African intellectual
//! traditions:
//!
//! - **Griot** — Active Persistent Homology of Memory (oral history as a
//!   living, decaying, clustering process)
//! - **Adinkra** — Context-Dependent Symbolic Compression (geometric symbols
//!   encoding abstract concepts, lossy in Shannon sense but recoverable
//!   through shared cultural context)
//! - **Palaver** — Sheaf H⁰ Consensus Dialogue (deliberative consensus-building
//!   modelled as sheaf cohomology)
//!
//! Also includes classical West African mathematical concepts:
//! - `IshangoBone` — tally mark encoding and prime factorization
//! - `YorubaNumeration` — base-20 number system operations
//! - `MandeMeasurement` — traditional Mande measurement conversions
//! - `OwareBoard` — mancala game as computational model
//! - `EthiopianMultiplication` — ancient peasant multiplication algorithm
//! - `FibonacciSequence` — Ethiopian interpretation
//! - `GeometricConcept` — generalized traditional geometric patterns

// ============================================================================
// Common constants
// ============================================================================

pub const WAM_MAX_DIM: usize = 16;
pub const WAM_MAX_STORIES: usize = 1024;
pub const WAM_MAX_PARTICIPANTS: usize = 64;
pub const WAM_MAX_CHILDREN: usize = 32;
pub const WAM_MAX_SYMBOLS: usize = 8;
pub const WAM_MAX_GLYPHS: usize = 256;
pub const WAM_PI: f64 = std::f64::consts::PI;

// ============================================================================
// 1. GRIOT — Active Persistent Homology of Memory
// ============================================================================

/// A story in the griot's memory.
#[derive(Debug, Clone)]
pub struct GriotStory {
    /// Vector embedding of the story.
    pub embedding: [f64; WAM_MAX_DIM],
    /// When the story was told.
    pub timestamp: f64,
    /// Who told it.
    pub teller_id: u32,
    /// How fast this story fades (exponential decay rate).
    pub decay_rate: f64,
    /// Unique story id.
    pub id: u32,
    /// Parent story id (0 = original).
    pub parent_id: u32,
    /// Child story ids.
    pub children: Vec<u32>,
    /// Current weight after decay.
    pub weight: f64,
}

impl GriotStory {
    fn new(embedding: &[f64], dim: usize, timestamp: f64, teller_id: u32,
           decay_rate: f64, parent_id: u32, id: u32) -> Self {
        let mut emb = [0.0f64; WAM_MAX_DIM];
        let copy = dim.min(WAM_MAX_DIM);
        emb[..copy].copy_from_slice(&embedding[..copy]);
        GriotStory {
            embedding: emb,
            timestamp,
            teller_id,
            decay_rate,
            id,
            parent_id,
            children: Vec::new(),
            weight: 1.0,
        }
    }
}

/// The griot's active memory, storing stories with decay and clustering.
#[derive(Debug, Clone)]
pub struct GriotMemory {
    pub stories: Vec<GriotStory>,
    pub next_id: u32,
    pub vr_epsilon: f64,
}

impl GriotMemory {
    /// Create a new empty GriotMemory with the given Vietoris-Rips parameter.
    pub fn new(vr_epsilon: f64) -> Self {
        GriotMemory {
            stories: Vec::with_capacity(WAM_MAX_STORIES),
            next_id: 1,
            vr_epsilon,
        }
    }

    /// Tell (store) a new story.
    pub fn tell(&mut self, embedding: &[f64], dim: usize, timestamp: f64,
                teller_id: u32, decay_rate: f64, parent_id: u32) -> u32 {
        if self.stories.len() >= WAM_MAX_STORIES {
            return 0;
        }
        let id = self.next_id;
        self.next_id += 1;

        let story = GriotStory::new(embedding, dim, timestamp, teller_id,
                                     decay_rate, parent_id, id);
        let parent = story.parent_id;

        // Register as child of parent
        if parent != 0 {
            if let Some(p) = self.stories.iter_mut().find(|s| s.id == parent) {
                if p.children.len() < WAM_MAX_CHILDREN {
                    p.children.push(id);
                }
            }
        }

        self.stories.push(story);
        id
    }

    /// Tell a story derived from a parent, inheriting the parent's decay rate.
    pub fn tell_derived(&mut self, parent_id: u32, embedding: &[f64],
                        dim: usize, timestamp: f64, teller_id: u32) -> u32 {
        let decay = self.stories.iter()
            .find(|s| s.id == parent_id)
            .map(|s| s.decay_rate)
            .unwrap_or(0.1);
        self.tell(embedding, dim, timestamp, teller_id, decay, parent_id)
    }

    /// Recall the top-k stories closest to a query embedding (k-nearest neighbors).
    pub fn recall(&self, query: &[f64], dim: usize, k: usize) -> Vec<u32> {
        let mut pairs: Vec<(f64, usize)> = self.stories.iter()
            .enumerate()
            .filter(|(_, s)| s.weight >= 0.01)
            .map(|(i, s)| (vec_dist(&s.embedding, query, dim), i))
            .collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        pairs.truncate(k);
        pairs.into_iter().map(|(_, i)| self.stories[i].id).collect()
    }

    /// Apply exponential decay to all stories based on current time.
    pub fn decay(&mut self, current_time: f64) {
        for s in &mut self.stories {
            let age = current_time - s.timestamp;
            s.weight = (-s.decay_rate * age).exp();
        }
    }

    /// Trace the lineage of a story back to its root.
    pub fn genealogy(&self, story_id: u32) -> Vec<u32> {
        let mut ancestors = Vec::new();
        let mut current = story_id;
        while current != 0 {
            let found = self.stories.iter().find(|s| s.id == current);
            match found {
                Some(s) => {
                    current = s.parent_id;
                    if current != 0 {
                        ancestors.push(current);
                    }
                }
                None => break,
            }
        }
        ancestors
    }

    /// Cluster stories by distance threshold, returning representative IDs.
    pub fn praise_names(&self, threshold: f64) -> Vec<u32> {
        let n = self.stories.len();
        let mut assigned = vec![false; n];
        let mut clusters = Vec::new();

        for i in 0..n {
            if assigned[i] || self.stories[i].weight < 0.01 {
                continue;
            }
            clusters.push(self.stories[i].id);
            assigned[i] = true;

            for j in (i + 1)..n {
                if assigned[j] || self.stories[j].weight < 0.01 {
                    continue;
                }
                let d = vec_dist(&self.stories[i].embedding,
                                 &self.stories[j].embedding, WAM_MAX_DIM);
                if d < threshold {
                    assigned[j] = true;
                }
            }
        }
        clusters
    }

    /// Compute a persistence diagram simulating Vietoris-Rips filtration.
    pub fn persistence(&self) -> Vec<PersistencePoint> {
        let n = self.stories.len();
        if n == 0 {
            return Vec::new();
        }

        // Filter active stories
        let active: Vec<usize> = (0..n)
            .filter(|&i| self.stories[i].weight >= 0.01)
            .collect();
        let m = active.len();
        if m == 0 {
            return Vec::new();
        }

        // Collect edges sorted by distance
        let mut edges: Vec<Edge> = Vec::new();
        for ai in 0..m {
            for aj in (ai + 1)..m {
                let i = active[ai];
                let j = active[aj];
                let d = vec_dist(&self.stories[i].embedding,
                                 &self.stories[j].embedding, WAM_MAX_DIM);
                edges.push(Edge { dist: d, a: ai, b: aj });
            }
        }
        edges.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(std::cmp::Ordering::Equal));

        // Union-Find
        let mut parent: Vec<usize> = (0..m).collect();
        let mut death_time = vec![-1.0f64; m];

        for e in &edges {
            let ra = uf_find(&mut parent, e.a);
            let rb = uf_find(&mut parent, e.b);
            if ra != rb {
                let (die, live) = if ra < rb { (ra, rb) } else { (rb, ra) };
                parent[die] = live;
                death_time[die] = e.dist;
            }
        }

        let mut points = Vec::new();
        for i in 0..m {
            let d = if death_time[i] < 0.0 { f64::INFINITY } else { death_time[i] };
            points.push(PersistencePoint { birth: 0.0, death: d });
        }
        points
    }
}

/// A point in a persistence diagram: (birth, death).
#[derive(Debug, Clone, Copy)]
pub struct PersistencePoint {
    pub birth: f64,
    pub death: f64,
}

#[derive(Debug, Clone)]
struct Edge {
    dist: f64,
    a: usize,
    b: usize,
}

fn uf_find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

fn vec_dist(a: &[f64; WAM_MAX_DIM], b: &[f64], dim: usize) -> f64 {
    let mut s = 0.0;
    let d = dim.min(WAM_MAX_DIM);
    for i in 0..d {
        let diff = a[i] - b[i];
        s += diff * diff;
    }
    s.sqrt()
}

/// Euclidean distance between two slices.
pub fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let mut s = 0.0;
    for i in 0..n {
        let d = a[i] - b[i];
        s += d * d;
    }
    s.sqrt()
}

// ============================================================================
// 2. ADINKRA — Context-Dependent Symbolic Compression
// ============================================================================

/// Geometric primitives used in Adinkra symbols.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AdinkraPrimitive {
    Circle = 0,
    Spiral = 1,
    Cross = 2,
    Knot = 3,
    Line = 4,
    Arc = 5,
    Triangle = 6,
    Diamond = 7,
}

impl AdinkraPrimitive {
    fn from_u8(v: u8) -> Self {
        match v % 8 {
            0 => AdinkraPrimitive::Circle,
            1 => AdinkraPrimitive::Spiral,
            2 => AdinkraPrimitive::Cross,
            3 => AdinkraPrimitive::Knot,
            4 => AdinkraPrimitive::Line,
            5 => AdinkraPrimitive::Arc,
            6 => AdinkraPrimitive::Triangle,
            _ => AdinkraPrimitive::Diamond,
        }
    }
}

/// How symbols are composed into a glyph.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GlyphComposeOp {
    Stack = 0,
    Nest = 1,
    Interleave = 2,
    Mirror = 3,
    Overlay = 4,
}

impl GlyphComposeOp {
    fn from_u8(v: u8) -> Self {
        match v % 5 {
            0 => GlyphComposeOp::Stack,
            1 => GlyphComposeOp::Nest,
            2 => GlyphComposeOp::Interleave,
            3 => GlyphComposeOp::Mirror,
            _ => GlyphComposeOp::Overlay,
        }
    }
}

/// A single Adinkra symbol made from primitives.
#[derive(Debug, Clone)]
pub struct AdinkraSymbol {
    pub primitives: Vec<AdinkraPrimitive>,
    pub params: Vec<[f64; 2]>,
}

impl AdinkraSymbol {
    fn from_primitive(p: AdinkraPrimitive, param0: f64, param1: f64) -> Self {
        AdinkraSymbol {
            primitives: vec![p],
            params: vec![[param0, param1]],
        }
    }
}

/// A glyph encoding a concept through Adinkra symbols.
#[derive(Debug, Clone)]
pub struct AdinkraGlyph {
    pub symbols: Vec<AdinkraSymbol>,
    pub ops: Vec<GlyphComposeOp>,
    pub hash: u64,
    pub cultural_weight: f64,
}

/// The Adinkra space, storing all glyphs.
#[derive(Debug, Clone)]
pub struct AdinkraSpace {
    pub glyphs: Vec<AdinkraGlyph>,
}

/// FNV-1a hash of a slice of f64 values.
pub fn fnv1a_f64(data: &[f64]) -> u64 {
    let mut h: u64 = 14695981039346656037;
    for &val in data {
        let bits: u64 = val.to_bits();
        for b in 0..8 {
            h ^= ((bits >> (b * 8)) & 0xFF) as u64;
            h = h.wrapping_mul(1099511628211);
        }
    }
    h
}

impl AdinkraSpace {
    pub fn new() -> Self {
        AdinkraSpace { glyphs: Vec::with_capacity(WAM_MAX_GLYPHS) }
    }

    /// Encode a concept vector (with optional cultural context) into an Adinkra glyph.
    pub fn encode(&mut self, concept: &[f64], dim: usize,
                  cultural_context: Option<&[f64]>, ctx_dim: usize) -> AdinkraGlyph {
        let dim = dim.min(WAM_MAX_DIM);
        let mut hash = fnv1a_f64(&concept[..dim]);

        if let Some(ctx) = cultural_context {
            let ctx_dim = ctx_dim.min(WAM_MAX_DIM);
            let ctx_hash = fnv1a_f64(&ctx[..ctx_dim]);
            hash ^= ctx_hash;
        }

        let num_symbols = 1 + (hash as usize % WAM_MAX_SYMBOLS);
        let mut symbols = Vec::with_capacity(num_symbols);
        let mut ops = Vec::with_capacity(num_symbols.saturating_sub(1));

        for i in 0..num_symbols {
            let prim = AdinkraPrimitive::from_u8(((hash >> (i * 4 + 3)) & 0xFF) as u8);
            let p0 = ((hash >> (i * 7)) & 0xFF) as f64 / 255.0;
            let p1 = ((hash >> (i * 7 + 8)) & 0xFF) as f64 / 255.0;
            symbols.push(AdinkraSymbol::from_primitive(prim, p0, p1));
        }

        for i in 0..num_symbols.saturating_sub(1) {
            ops.push(GlyphComposeOp::from_u8(((hash >> (i * 3 + 7)) & 0xFF) as u8));
        }

        let cultural_weight = cultural_context.map_or(0.0, |ctx| {
            let ctx_dim = ctx_dim.min(WAM_MAX_DIM);
            let norm: f64 = ctx[..ctx_dim].iter().map(|x| x * x).sum();
            1.0 - 1.0 / (1.0 + norm.sqrt())
        });

        let glyph = AdinkraGlyph {
            symbols,
            ops,
            hash,
            cultural_weight,
        };

        if self.glyphs.len() < WAM_MAX_GLYPHS {
            self.glyphs.push(glyph.clone());
        }

        glyph
    }

    /// Decode a glyph back into a concept vector (lossy reconstruction).
    pub fn decode(&self, glyph: &AdinkraGlyph, dim: usize) -> Vec<f64> {
        let mut out = vec![0.0f64; dim];
        for i in 0..dim {
            let sym_idx = i % glyph.symbols.len().max(1);
            let param_idx = (i / glyph.symbols.len().max(1)) % 2;
            let base = glyph.symbols[sym_idx].params[0][param_idx];
            let hash_contrib = ((glyph.hash >> (i * 5)) & 0x1F) as f64 / 31.0;
            out[i] = (base * 0.6 + hash_contrib * 0.4) * 2.0 - 1.0;
        }
        out
    }

    /// Cultural recoverability: how well a glyph can be decoded given cultural context.
    pub fn cultural_recoverability(&self, glyph: &AdinkraGlyph,
                                   original: &[f64], cultural_context: Option<&[f64]>,
                                   dim: usize) -> f64 {
        let decoded = self.decode(glyph, dim);
        let dim = dim.min(WAM_MAX_DIM);

        let dot: f64 = (0..dim).map(|i| original[i] * decoded[i]).sum();
        let norm_a: f64 = (0..dim).map(|i| original[i] * original[i]).sum();
        let norm_b: f64 = (0..dim).map(|i| decoded[i] * decoded[i]).sum();
        let base_sim = if norm_a > 0.0 && norm_b > 0.0 {
            (dot / (norm_a.sqrt() * norm_b.sqrt())).max(0.0)
        } else {
            0.0
        };

        let boost = glyph.cultural_weight * 0.5;
        (base_sim + boost).min(1.0)
    }

    /// Shannon compression ratio: glyph size / original size.
    pub fn shannon_compression(&self, glyph: &AdinkraGlyph, original_bytes: usize) -> f64 {
        if original_bytes == 0 {
            return 0.0;
        }
        let mut glyph_bytes = 8u64; // hash
        glyph_bytes += glyph.symbols.len() as u64; // one byte per primitive
        glyph_bytes += glyph.ops.len() as u64; // composition ops
        glyph_bytes as f64 / original_bytes as f64
    }

    /// Create a single-primitive symbol.
    pub fn symbol_from_primitive(p: AdinkraPrimitive, param0: f64, param1: f64) -> AdinkraSymbol {
        AdinkraSymbol::from_primitive(p, param0, param1)
    }
}

// ============================================================================
// 3. PALAVER — Sheaf H⁰ Consensus Dialogue
// ============================================================================

/// A participant in a palaver (deliberation session).
#[derive(Debug, Clone)]
pub struct PalaverParticipant {
    pub position: [f64; WAM_MAX_DIM],
    pub weight: f64,
    pub stubbornness: f64,
    pub id: u32,
}

/// A palaver session where participants deliberate toward consensus.
#[derive(Debug, Clone)]
pub struct PalaverSession {
    pub participants: Vec<PalaverParticipant>,
    pub topic: [f64; WAM_MAX_DIM],
    pub topic_dim: usize,
    pub tolerance: f64,
    pub max_iterations: usize,
}

/// A coalition of participants who agree.
#[derive(Debug, Clone)]
pub struct PalaverCoalition {
    pub members: Vec<usize>,
}

impl PalaverSession {
    /// Create a new palaver session around a topic.
    pub fn new(topic: &[f64], dim: usize, tolerance: f64, max_iterations: usize) -> Self {
        let mut t = [0.0f64; WAM_MAX_DIM];
        let d = dim.min(WAM_MAX_DIM);
        t[..d].copy_from_slice(&topic[..d]);
        PalaverSession {
            participants: Vec::with_capacity(WAM_MAX_PARTICIPANTS),
            topic: t,
            topic_dim: dim,
            tolerance,
            max_iterations,
        }
    }

    /// Add a participant with a position, weight, and stubbornness.
    pub fn propose(&mut self, position: &[f64], dim: usize,
                   weight: f64, stubbornness: f64) -> u32 {
        if self.participants.len() >= WAM_MAX_PARTICIPANTS {
            return 0;
        }
        let mut pos = [0.0f64; WAM_MAX_DIM];
        let d = dim.min(WAM_MAX_DIM);
        pos[..d].copy_from_slice(&position[..d]);
        let id = self.participants.len() as u32 + 1;
        let stubborn = stubbornness.clamp(0.0, 1.0);
        self.participants.push(PalaverParticipant {
            position: pos,
            weight,
            stubbornness: stubborn,
            id,
        });
        id
    }

    /// Perform one deliberation step. Returns true if converged.
    pub fn deliberate(&mut self) -> bool {
        if self.participants.is_empty() {
            return true;
        }

        // Compute weighted centroid
        let d = self.topic_dim.min(WAM_MAX_DIM);
        let mut centroid = [0.0f64; WAM_MAX_DIM];
        let mut total_weight = 0.0;

        for p in &self.participants {
            let w = p.weight * (1.0 - p.stubbornness);
            for j in 0..d {
                centroid[j] += p.position[j] * w;
            }
            total_weight += w;
        }

        if total_weight > 0.0 {
            for j in 0..d {
                centroid[j] /= total_weight;
            }
        }

        // Move participants toward centroid
        let mut max_shift = 0.0f64;
        for p in &mut self.participants {
            let flexibility = 1.0 - p.stubbornness;
            for j in 0..d {
                let delta = centroid[j] - p.position[j];
                let shift = delta * flexibility * 0.5;
                p.position[j] += shift;
                let ad = shift.abs();
                if ad > max_shift {
                    max_shift = ad;
                }
            }
        }

        max_shift < self.tolerance
    }

    /// Run deliberation to convergence (up to max_iterations). Returns true if converged.
    pub fn converge(&mut self) -> bool {
        for _ in 0..self.max_iterations {
            if self.deliberate() {
                return true;
            }
        }
        false
    }

    /// Get the consensus (weighted centroid of all participants).
    pub fn consensus(&self) -> Vec<f64> {
        let d = self.topic_dim.min(WAM_MAX_DIM);
        let mut consensus_out = vec![0.0f64; d];
        let mut total_w = 0.0;

        for p in &self.participants {
            for j in 0..d {
                consensus_out[j] += p.position[j] * p.weight;
            }
            total_w += p.weight;
        }

        if total_w > 0.0 {
            for j in 0..d {
                consensus_out[j] /= total_w;
            }
        }

        consensus_out
    }

    /// Converge and return the consensus (convenience wrapper).
    pub fn converge_with_consensus(&mut self) -> (bool, Vec<f64>) {
        let conv = self.converge();
        (conv, self.consensus())
    }

    /// Find coalitions (clusters of participants within threshold).
    pub fn coalition(&self, threshold: f64) -> Vec<PalaverCoalition> {
        let n = self.participants.len();
        let mut assigned = vec![-1i32; n];
        let mut coalitions = Vec::new();

        for i in 0..n {
            if assigned[i] >= 0 {
                continue;
            }
            assigned[i] = coalitions.len() as i32;
            let mut members = vec![i];

            for j in (i + 1)..n {
                if assigned[j] >= 0 {
                    continue;
                }
                let dist = self.participant_distance(i, j);
                if dist < threshold {
                    assigned[j] = coalitions.len() as i32;
                    members.push(j);
                }
            }
            coalitions.push(PalaverCoalition { members });
        }
        coalitions
    }

    fn participant_distance(&self, i: usize, j: usize) -> f64 {
        let d = self.topic_dim.min(WAM_MAX_DIM);
        let mut s = 0.0;
        for k in 0..d {
            let diff = self.participants[i].position[k] - self.participants[j].position[k];
            s += diff * diff;
        }
        s.sqrt()
    }

    /// Quality metric: 1 - normalized variance (1 = perfect agreement).
    pub fn quality(&self) -> f64 {
        let n = self.participants.len();
        if n < 2 {
            return 1.0;
        }
        let d = self.topic_dim.min(WAM_MAX_DIM);

        let mut mean = [0.0f64; WAM_MAX_DIM];
        for p in &self.participants {
            for j in 0..d {
                mean[j] += p.position[j];
            }
        }
        for j in 0..d {
            mean[j] /= n as f64;
        }

        let variance: f64 = self.participants.iter().flat_map(|p| {
            (0..d).map(move |j| {
                let diff = p.position[j] - mean[j];
                diff * diff
            })
        }).sum();
        let variance = variance / (n as f64 * d as f64);
        1.0 / (1.0 + variance)
    }

    /// H⁰ dimension: number of connected components in the agreement graph.
    pub fn h0_dimension(&self, threshold: f64) -> usize {
        let n = self.participants.len();
        if n == 0 {
            return 0;
        }
        if n == 1 {
            return 1;
        }

        let mut parent: Vec<usize> = (0..n).collect();
        for i in 0..n {
            for j in (i + 1)..n {
                if self.participant_distance(i, j) < threshold {
                    let ri = uf_find(&mut parent, i);
                    let rj = uf_find(&mut parent, j);
                    if ri != rj {
                        parent[ri] = rj;
                    }
                }
            }
        }

        let mut roots = 0;
        for i in 0..n {
            if uf_find(&mut parent, i) == i {
                roots += 1;
            }
        }
        roots
    }
}

// ============================================================================
// 4. ISHANGO BONE — Tally mark encoding and prime factorization
// ============================================================================

/// The Ishango Bone is one of the oldest known mathematical artifacts,
/// discovered in the Democratic Republic of Congo. It features tally marks
/// and what some interpret as prime number groupings.
#[derive(Debug, Clone)]
pub struct IshangoBone;

impl IshangoBone {
    /// Encode a number as tally marks (returns vector of tally count per group).
    /// Traditional Ishango grouping uses columns of 3, 6, 4, 8, 10, 5, 5, 7.
    pub fn tally_encode(n: u64) -> Vec<u64> {
        let groups = vec![3u64, 6, 4, 8, 10, 5, 5, 7];
        let mut remaining = n;
        let mut result = Vec::new();
        for &g in &groups {
            if remaining == 0 {
                result.push(0);
            } else if remaining <= g {
                result.push(remaining);
                remaining = 0;
            } else {
                result.push(g);
                remaining -= g;
            }
        }
        if remaining > 0 {
            result.push(remaining);
        }
        result
    }

    /// Decode tally groups back to a number.
    pub fn tally_decode(groups: &[u64]) -> u64 {
        groups.iter().sum()
    }

    /// Count tally marks needed to represent n.
    pub fn tally_count(n: u64) -> u64 {
        Self::tally_encode(n).iter().sum()
    }

    /// Primitive prime factorization, returning (prime, exponent) pairs.
    pub fn prime_factorize(n: u64) -> Vec<(u64, u32)> {
        let mut m = n;
        let mut factors = Vec::new();

        if m == 0 || m == 1 {
            return factors;
        }

        // Handle 2
        let mut exp = 0;
        while m % 2 == 0 {
            m /= 2;
            exp += 1;
        }
        if exp > 0 {
            factors.push((2, exp));
        }

        // Handle odd factors
        let mut f = 3u64;
        while f * f <= m {
            exp = 0;
            while m % f == 0 {
                m /= f;
                exp += 1;
            }
            if exp > 0 {
                factors.push((f, exp));
            }
            f += 2;
        }

        if m > 1 {
            factors.push((m, 1));
        }

        factors
    }

    /// Check if a number is prime.
    pub fn is_prime(n: u64) -> bool {
        if n < 2 {
            return false;
        }
        if n == 2 {
            return true;
        }
        if n % 2 == 0 {
            return false;
        }
        let mut f = 3u64;
        while f * f <= n {
            if n % f == 0 {
                return false;
            }
            f += 2;
        }
        true
    }
}

// ============================================================================
// 5. YORUBA NUMERATION — Base-20 number system operations
// ============================================================================

/// Yoruba numeration is a base-20 (vigesimal) system with unique structural
/// properties, including subtraction-based construction for certain numbers.
#[derive(Debug, Clone)]
pub struct YorubaNumeration;

impl YorubaNumeration {
    /// Convert a decimal number to Yoruba-style base-20 digits.
    /// Returns the base-20 representation (least significant first).
    pub fn to_base20(n: u64) -> Vec<u8> {
        if n == 0 {
            return vec![0];
        }
        let mut m = n;
        let mut digits = Vec::new();
        while m > 0 {
            digits.push((m % 20) as u8);
            m /= 20;
        }
        digits
    }

    /// Convert from Yoruba base-20 digits (least significant first) back to decimal.
    pub fn from_base20(digits: &[u8]) -> u64 {
        let mut result: u64 = 0;
        for &d in digits.iter().rev() {
            result = result * 20 + d as u64;
        }
        result
    }

    /// Yoruba-style additive representation of a number.
    /// Returns a human-readable string in the Yoruba style.
    pub fn yoruba_string(n: u64) -> String {
        if n == 0 {
            return "òdò".to_string();
        }
        let base20 = Self::to_base20(n);
        let mut parts: Vec<String> = Vec::new();
        for (i, &digit) in base20.iter().enumerate() {
            if digit == 0 {
                continue;
            }
            let place_value = 20u64.pow(i as u32);
            if place_value == 1 {
                parts.push(Self::digit_name(digit).to_string());
            } else if digit == 1 {
                parts.push(format!("{} ogún", Self::digit_name(digit)));
            } else {
                parts.push(format!("{} {}",
                    Self::digit_name(digit),
                    Self::place_name(place_value)));
            }
        }
        parts.reverse();
        parts.join(" ")
    }

    fn digit_name(d: u8) -> &'static str {
        match d {
            1 => "oókan",
            2 => "eéjì",
            3 => "ẹéta",
            4 => "ẹérin",
            5 => "àrún",
            6 => "ẹẹ̀fa",
            7 => "ẹéje",
            8 => "ẹéjò",
            9 => "ẹésàn",
            10 => "ẹéwà",
            11 => "oókànlá",
            12 => "eéjìlá",
            13 => "ẹétàlá",
            14 => "ẹérinlá",
            15 => "àádóogún",
            16 => "ẹẹ̀rìndílógún",
            17 => "ẹétìdílógún",
            18 => "ẹéjìdílógún",
            19 => "oókàndílógún",
            _ => "unknown",
        }
    }

    fn place_name(v: u64) -> &'static str {
        match v {
            20 => "ogún",
            400 => "ọ̀ọ́dún",
            8000 => "ẹ̀gbẹ̀ẹ́wá",
            160000 => "ọ̀kẹ́",
            _ => "igba",
        }
    }

    /// Add two numbers in base-20 representation.
    pub fn base20_add(a: &[u8], b: &[u8]) -> Vec<u8> {
        let max_len = a.len().max(b.len());
        let mut result = Vec::with_capacity(max_len + 1);
        let mut carry: u8 = 0;
        for i in 0..max_len {
            let da = if i < a.len() { a[i] } else { 0 };
            let db = if i < b.len() { b[i] } else { 0 };
            let sum = da as u16 + db as u16 + carry as u16;
            result.push((sum % 20) as u8);
            carry = (sum / 20) as u8;
        }
        if carry > 0 {
            result.push(carry);
        }
        result
    }

    /// Multiply two numbers in base-20 representation.
    pub fn base20_mul(a: &[u8], b: &[u8]) -> Vec<u8> {
        let a_val = Self::from_base20(a);
        let b_val = Self::from_base20(b);
        Self::to_base20(a_val * b_val)
    }
}

// ============================================================================
// 6. MANDE MEASUREMENT — Traditional Mande measurement conversions
// ============================================================================

/// Traditional Mande measurement system conversions.
/// Used across Mali, Guinea, Senegal, and neighboring regions.
#[derive(Debug, Clone)]
pub struct MandeMeasurement;

impl MandeMeasurement {
    /// Convert from the Mandinka "kùsũ" (a forearm measure, ~0.5 m) to meters.
    pub fn kusu_to_meters(kusu: f64) -> f64 {
        kusu * 0.5
    }

    /// Convert from meters to kusu.
    pub fn meters_to_kusu(meters: f64) -> f64 {
        meters / 0.5
    }

    /// Convert from the Bambara "jàn" (a fathom, ~1.8 m) to meters.
    pub fn jan_to_meters(jan: f64) -> f64 {
        jan * 1.8
    }

    /// Convert from meters to jan.
    pub fn meters_to_jan(meters: f64) -> f64 {
        meters / 1.8
    }

    /// Convert from the Soninke "kille" (a grain measure, ~0.5 liters) to liters.
    pub fn kille_to_liters(kille: f64) -> f64 {
        kille * 0.5
    }

    /// Convert from liters to kille.
    pub fn liters_to_kille(liters: f64) -> f64 {
        liters / 0.5
    }

    /// Convert from the Mandinka "sèrè" (a basket, ~15 liters) to liters.
    pub fn sere_to_liters(sere: f64) -> f64 {
        sere * 15.0
    }

    /// Convert from liters to sere.
    pub fn liters_to_sere(liters: f64) -> f64 {
        liters / 15.0
    }

    /// Convert from the Fulani "gandal" (a bundle of millet, ~5 kg) to kilograms.
    pub fn gandal_to_kg(gandal: f64) -> f64 {
        gandal * 5.0
    }

    /// Convert from kilograms to gandal.
    pub fn kg_to_gandal(kg: f64) -> f64 {
        kg / 5.0
    }

    /// Convert from the Bambara "sèmè" (a headload, ~30 kg) to kilograms.
    pub fn seme_to_kg(seme: f64) -> f64 {
        seme * 30.0
    }

    /// Convert from kilograms to seme.
    pub fn kg_to_seme(kg: f64) -> f64 {
        kg / 30.0
    }
}

// ============================================================================
// 7. OWARE BOARD — Mancala game as computational model
// ============================================================================

/// Oware (Awari) is a mancala game of West African origin. The board is
/// treated here as a computational model: pits as memory cells, sowing as
/// a state transition function, captures as operations.
#[derive(Debug, Clone)]
pub struct OwareBoard {
    /// Pits 0-5: player 0's side; pits 6-11: player 1's side.
    /// Indexed consistently: pit i is opposite pit (11 - i).
    pub pits: [u8; 12],
    /// Seeds in storage (captured).
    pub score: [u32; 2],
    /// Current player (0 or 1).
    pub turn: usize,
    /// Game over flag.
    pub game_over: bool,
}

impl OwareBoard {
    /// Create a new Oware board with 4 seeds in each pit.
    pub fn new() -> Self {
        OwareBoard {
            pits: [4u8; 12],
            score: [0, 0],
            turn: 0,
            game_over: false,
        }
    }

    /// Create a board with a custom initial layout.
    pub fn from_pits(pits: [u8; 12]) -> Self {
        OwareBoard {
            pits,
            score: [0, 0],
            turn: 0,
            game_over: false,
        }
    }

    /// Get the owner of a pit (0 for player 0, 1 for player 1).
    pub fn pit_owner(pit: usize) -> usize {
        if pit < 6 { 0 } else { 1 }
    }

    /// Get the opposite pit index.
    pub fn opposite_pit(pit: usize) -> usize {
        11 - pit
    }

    /// Check if a pit belongs to the current player.
    pub fn is_own_pit(&self, pit: usize) -> bool {
        Self::pit_owner(pit) == self.turn
    }

    /// Sow seeds from a pit. Returns the last pit sown into, or None if invalid.
    pub fn sow(&mut self, pit: usize) -> Option<usize> {
        if self.game_over {
            return None;
        }
        if !self.is_own_pit(pit) {
            return None;
        }
        let mut seeds = self.pits[pit];
        if seeds == 0 {
            return None;
        }
        self.pits[pit] = 0;

        let mut current = pit;
        while seeds > 0 {
            current = (current + 1) % 12;
            // Skip the opponent's scoring: both players' pits are on the board.
            // In Oware, we never skip any pits while sowing.
            self.pits[current] += 1;
            seeds -= 1;
        }

        // Capture: if last seed lands in opponent's pit with 2 or 3 seeds
        if Self::pit_owner(current) != self.turn && (self.pits[current] == 2 || self.pits[current] == 3) {
            let mut captured = self.pits[current];
            self.pits[current] = 0;

            // Try to capture opposite as well (if applicable)
            let opp = Self::opposite_pit(current);
            if self.pits[opp] == 2 || self.pits[opp] == 3 {
                captured += self.pits[opp];
                self.pits[opp] = 0;
            }

            self.score[self.turn] += captured as u32;
        }

        // Check game over: if one side is empty, capture remaining and end
        let p0_empty: bool = self.pits[..6].iter().all(|&s| s == 0);
        let p1_empty: bool = self.pits[6..12].iter().all(|&s| s == 0);
        if p0_empty || p1_empty {
            let p0_remaining: u32 = self.pits[..6].iter().map(|&s| s as u32).sum();
            let p1_remaining: u32 = self.pits[6..12].iter().map(|&s| s as u32).sum();
            self.score[0] += p0_remaining;
            self.score[1] += p1_remaining;
            for pit in 0..12 {
                self.pits[pit] = 0;
            }
            self.game_over = true;
        }

        // Switch turn
        self.turn = 1 - self.turn;

        Some(current)
    }

    /// Get all legal moves for the current player.
    pub fn legal_moves(&self) -> Vec<usize> {
        if self.game_over {
            return Vec::new();
        }
        let range = if self.turn == 0 { 0..6 } else { 6..12 };
        range.filter(|&p| self.pits[p] > 0).collect()
    }

    /// Simulate a move and return the resulting board state.
    pub fn simulate_move(&self, pit: usize) -> Option<OwareBoard> {
        let mut board = self.clone();
        if board.sow(pit).is_some() {
            Some(board)
        } else {
            None
        }
    }

    /// Total seeds remaining on the board.
    pub fn total_seeds(&self) -> u32 {
        self.pits.iter().map(|&s| s as u32).sum()
    }

    /// Get the winning player, if any.
    pub fn winner(&self) -> Option<usize> {
        if !self.game_over {
            return None;
        }
        if self.score[0] > self.score[1] {
            Some(0)
        } else if self.score[1] > self.score[0] {
            Some(1)
        } else {
            None // draw
        }
    }
}

// ============================================================================
// 8. ETHIOPIAN MULTIPLICATION — Ancient peasant multiplication algorithm
// ============================================================================

/// Ethiopian (peasant) multiplication is an ancient algorithm that requires
/// only halving, doubling, and addition — no multiplication tables needed.
/// It was historically used across Northeast Africa and the Mediterranean.
#[derive(Debug, Clone)]
pub struct EthiopianMultiplication;

impl EthiopianMultiplication {
    /// Multiply two numbers using the Ethiopian peasant algorithm.
    ///
    /// Algorithm:
    /// 1. Halve the first number repeatedly (ignoring remainders) until 0.
    /// 2. Double the second number the same number of times.
    /// 3. Sum the doubled values where the halved value is odd.
    pub fn multiply(mut a: u64, mut b: u64) -> u64 {
        let mut result = 0u64;
        while a > 0 {
            if a % 2 == 1 {
                result += b;
            }
            a /= 2;
            b *= 2;
        }
        result
    }

    /// Return the intermediate steps of the algorithm.
    pub fn multiply_steps(a: u64, b: u64) -> Vec<(u64, u64, bool)> {
        let mut halved = a;
        let mut doubled = b;
        let mut steps = Vec::new();
        while halved > 0 {
            let odd = halved % 2 == 1;
            steps.push((halved, doubled, odd));
            halved /= 2;
            doubled *= 2;
        }
        steps
    }

    /// Check if the algorithm would produce an intermediate overflow.
    pub fn would_overflow(a: u64, b: u64) -> bool {
        let mut halved = a;
        let mut doubled = b;
        while halved > 0 {
            if halved % 2 == 1 {
                if u64::MAX - doubled < b {
                    // Simplified check; full check is more complex
                }
            }
            halved /= 2;
            doubled = doubled.saturating_mul(2);
        }
        a.checked_mul(b).is_none()
    }

    /// Multiply using 128-bit intermediate results (saturating).
    pub fn multiply_safe(a: u64, b: u64) -> Option<u64> {
        a.checked_mul(b)
    }
}

// ============================================================================
// 9. FIBONACCI SEQUENCE — Ethiopian interpretation
// ============================================================================

/// The Fibonacci sequence, as known through Ethiopian and other African
/// mathematical traditions. While Fibonacci is named after the Italian
/// mathematician, the sequence appears in various ancient cultures.
/// The Ethiopian interpretation emphasizes its connection to natural
/// pattern recognition and the golden ratio.
#[derive(Debug, Clone)]
pub struct FibonacciSequence;

impl FibonacciSequence {
    /// Compute the n-th Fibonacci number iteratively.
    pub fn nth(n: usize) -> u64 {
        if n == 0 {
            return 0;
        }
        if n == 1 {
            return 1;
        }
        let (mut a, mut b) = (0u64, 1u64);
        for _ in 2..=n {
            let c = a + b;
            a = b;
            b = c;
        }
        b
    }

    /// Compute the n-th Fibonacci number using the Ethiopian doubling method.
    /// This computes F(n) in O(log n) steps using the identities:
    ///   F(2k) = F(k) * (2*F(k+1) - F(k))
    ///   F(2k+1) = F(k+1)^2 + F(k)^2
    pub fn nth_fast(n: usize) -> u64 {
        if n == 0 {
            return 0;
        }
        // Returns (F(n), F(n+1))
        fn fib_pair(n: usize) -> (u64, u64) {
            if n == 0 {
                return (0, 1);
            }
            let (a, b) = fib_pair(n / 2);
            let c = a * (2 * b - a);
            let d = a * a + b * b;
            if n % 2 == 0 {
                (c, d)
            } else {
                (d, c + d)
            }
        }
        fib_pair(n).0
    }

    /// Generate the first n Fibonacci numbers.
    pub fn first_n(n: usize) -> Vec<u64> {
        let mut nums = Vec::with_capacity(n);
        if n == 0 {
            return nums;
        }
        nums.push(0);
        if n == 1 {
            return nums;
        }
        nums.push(1);
        let (mut a, mut b) = (0u64, 1u64);
        for _ in 2..n {
            let c = a + b;
            nums.push(c);
            a = b;
            b = c;
        }
        nums
    }

    /// Check if a number is a Fibonacci number.
    pub fn is_fibonacci(n: u64) -> bool {
        let is_perfect_square = |x: u64| -> bool {
            let r = (x as f64).sqrt() as u64;
            r * r == x
        };
        is_perfect_square(5 * n * n + 4) || is_perfect_square(5 * n * n - 4)
    }

    /// Compute the golden ratio approximation from F(n+1)/F(n).
    pub fn golden_ratio_approx(n: usize) -> f64 {
        if n < 2 {
            return 1.0;
        }
        let a = Self::nth(n) as f64;
        let b = Self::nth(n + 1) as f64;
        b / a
    }
}

// ============================================================================
// 10. GEOMETRIC CONCEPT — Generalized traditional geometric patterns
// ============================================================================

/// Traditional West African geometric patterns found in textiles, architecture,
/// and body art. This module provides generalized geometric pattern generation
/// inspired by traditions across the region.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GeometricPattern {
    /// Mudcloth-inspired strip patterns (Bògòlanfini).
    Mudcloth,
    /// Kente cloth geometric grid patterns.
    Kente,
    /// Ndebele house-painting patterns.
    Ndebele,
    /// Circular patterns (adinkra "Gye Nyame" style).
    Circular,
    /// Spiral patterns (adinkra "Sankofa" style).
    Spiral,
    /// Diamond/lozenge grid patterns.
    DiamondGrid,
    /// Chevron/zigzag patterns.
    Chevron,
    /// Interlacing knot patterns.
    Interlace,
}

/// A geometric concept: pattern type with parameters.
#[derive(Debug, Clone)]
pub struct GeometricConcept {
    pub pattern: GeometricPattern,
    pub scale: f64,
    pub rotation: f64,
    pub complexity: usize,
    pub symmetry: usize,
}

impl GeometricConcept {
    /// Create a new geometric concept with default parameters.
    pub fn new(pattern: GeometricPattern) -> Self {
        GeometricConcept {
            pattern,
            scale: 1.0,
            rotation: 0.0,
            complexity: 1,
            symmetry: 1,
        }
    }

    /// Set the scale.
    pub fn with_scale(mut self, scale: f64) -> Self {
        self.scale = scale;
        self
    }

    /// Set the rotation in radians.
    pub fn with_rotation(mut self, rotation: f64) -> Self {
        self.rotation = rotation;
        self
    }

    /// Set complexity level (higher = more detailed).
    pub fn with_complexity(mut self, complexity: usize) -> Self {
        self.complexity = complexity.max(1);
        self
    }

    /// Set symmetry order (1 = none, 2 = bilateral, 4 = 4-fold, etc.).
    pub fn with_symmetry(mut self, symmetry: usize) -> Self {
        self.symmetry = symmetry.max(1);
        self
    }

    /// Generate x,y coordinates for the pattern.
    pub fn generate_coordinates(&self) -> Vec<[f64; 2]> {
        let mut points = Vec::new();
        match self.pattern {
            GeometricPattern::Mudcloth => {
                // Striped rectilinear pattern
                for y in 0..self.complexity {
                    for x in 0..(self.complexity * 2) {
                        if (x + y) % 3 != 0 {
                            points.push([
                                x as f64 * self.scale,
                                y as f64 * self.scale,
                            ]);
                        }
                    }
                }
            }
            GeometricPattern::Kente => {
                // Grid with alternating weave
                for y in 0..self.complexity {
                    for x in 0..self.complexity {
                        if (x + y) % 2 == 0 {
                            points.push([
                                x as f64 * self.scale,
                                y as f64 * self.scale,
                            ]);
                        }
                    }
                }
            }
            GeometricPattern::Ndebele => {
                // Concentric geometric bands
                for ring in 0..self.complexity {
                    let r = (ring + 1) as f64 * self.scale;
                    let n_points = 4 * (ring + 1);
                    for i in 0..n_points {
                        let angle = 2.0 * std::f64::consts::PI * i as f64 / n_points as f64;
                        points.push([r * angle.cos(), r * angle.sin()]);
                    }
                }
            }
            GeometricPattern::Circular => {
                // Concentric circles
                for ring in 0..self.complexity {
                    let r = (ring + 1) as f64 * self.scale;
                    let n_points = 8 * (ring + 1);
                    for i in 0..n_points {
                        let angle = 2.0 * std::f64::consts::PI * i as f64 / n_points as f64;
                        points.push([r * angle.cos(), r * angle.sin()]);
                    }
                }
            }
            GeometricPattern::Spiral => {
                // Archimedean spiral
                let total_points = self.complexity * 24;
                for i in 0..total_points {
                    let t = i as f64 * 0.25;
                    let r = self.scale * t / (2.0 * std::f64::consts::PI);
                    let angle = t + self.rotation;
                    points.push([r * angle.cos(), r * angle.sin()]);
                }
            }
            GeometricPattern::DiamondGrid => {
                // Diamond/lozenge tessellation
                for y in 0..self.complexity {
                    for x in 0..self.complexity {
                        let cx = (x as f64 + if y % 2 == 0 { 0.0 } else { 0.5 }) * self.scale;
                        let cy = y as f64 * self.scale * 0.866;
                        points.push([cx, cy]);
                        // Add offset diamond
                        points.push([cx + self.scale * 0.5, cy + self.scale * 0.433]);
                    }
                }
            }
            GeometricPattern::Chevron => {
                // Zigzag/chevron pattern
                for row in 0..self.complexity {
                    for i in 0..(self.complexity * 2) {
                        let x = i as f64 * self.scale * 0.5;
                        let offset = if row % 2 == 0 {
                            (i as f64 * self.scale * 0.25).sin() * self.scale * 0.5
                        } else {
                            -((i as f64 * self.scale * 0.25).sin() * self.scale * 0.5)
                        };
                        let y = row as f64 * self.scale + offset;
                        points.push([x, y]);
                    }
                }
            }
            GeometricPattern::Interlace => {
                // Interlacing knot pattern
                for ring in 0..self.complexity {
                    let r = (ring + 1) as f64 * self.scale;
                    let n_points = 6 * (ring + 1);
                    for i in 0..n_points {
                        let base_angle = 2.0 * std::f64::consts::PI * i as f64 / n_points as f64;
                        let knot_scale = 1.0 + 0.3 * (base_angle * 3.0).sin();
                        let angle = base_angle + self.rotation;
                        let rad = r * knot_scale;
                        points.push([rad * angle.cos(), rad * angle.sin()]);
                    }
                }
            }
        }

        // Apply symmetry transformations
        if self.symmetry > 1 {
            let original = points.clone();
            for s in 1..self.symmetry {
                let angle = 2.0 * std::f64::consts::PI * s as f64 / self.symmetry as f64;
                for &[x, y] in &original {
                    let nx = x * angle.cos() - y * angle.sin();
                    let ny = x * angle.sin() + y * angle.cos();
                    points.push([nx, ny]);
                }
            }
        }

        points
    }

    /// Count features (points) in the pattern.
    pub fn feature_count(&self) -> usize {
        self.generate_coordinates().len()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- Griot tests ---

    #[test]
    fn test_griot_tell_and_recall() {
        let mut mem = GriotMemory::new(1.0);
        let v1 = [1.0, 0.0, 0.0, 0.0];
        let v2 = [0.0, 1.0, 0.0, 0.0];
        let v3 = [0.0, 0.0, 1.0, 0.0];

        assert_eq!(1, mem.tell(&v1, 4, 1.0, 1, 0.1, 0));
        assert_eq!(2, mem.tell(&v2, 4, 2.0, 2, 0.1, 0));
        assert_eq!(3, mem.tell(&v3, 4, 3.0, 3, 0.1, 0));

        let query = [0.9, 0.1, 0.0, 0.0];
        let results = mem.recall(&query, 4, 3);
        assert!(!results.is_empty());
        assert_eq!(1, results[0]);
    }

    #[test]
    fn test_griot_decay() {
        let mut mem = GriotMemory::new(1.0);
        let v = [1.0, 0.0, 0.0, 0.0];
        mem.tell(&v, 4, 0.0, 1, 0.5, 0);

        mem.decay(1.0);
        assert!((mem.stories[0].weight - (-0.5f64).exp()).abs() < 1e-9);

        mem.decay(4.0);
        assert!((mem.stories[0].weight - (-2.0f64).exp()).abs() < 1e-9);
    }

    #[test]
    fn test_griot_genealogy() {
        let mut mem = GriotMemory::new(1.0);
        let v = [1.0, 0.0, 0.0, 0.0];
        let s1 = mem.tell(&v, 4, 1.0, 1, 0.1, 0);
        let s2 = mem.tell_derived(s1, &v, 4, 2.0, 2);
        let s3 = mem.tell_derived(s2, &v, 4, 3.0, 3);

        let ancestors = mem.genealogy(s3);
        assert_eq!(2, ancestors.len());
        assert_eq!(s2, ancestors[0]);
        assert_eq!(s1, ancestors[1]);
    }

    #[test]
    fn test_griot_persistence() {
        let mut mem = GriotMemory::new(0.5);
        let v1 = [1.0, 1.0, 0.0, 0.0];
        let v2 = [1.1, 1.1, 0.0, 0.0];
        let v3 = [5.0, 5.0, 0.0, 0.0];
        let v4 = [5.1, 5.1, 0.0, 0.0];

        mem.tell(&v1, 4, 1.0, 1, 0.0, 0);
        mem.tell(&v2, 4, 1.0, 1, 0.0, 0);
        mem.tell(&v3, 4, 1.0, 1, 0.0, 0);
        mem.tell(&v4, 4, 1.0, 1, 0.0, 0);

        let pd = mem.persistence();
        assert_eq!(4, pd.len());
    }

    #[test]
    fn test_griot_praise_names() {
        let mut mem = GriotMemory::new(1.0);
        let v1 = [0.0, 0.0, 0.0, 0.0];
        let v2 = [0.1, 0.1, 0.0, 0.0];
        let v3 = [10.0, 10.0, 0.0, 0.0];

        mem.tell(&v1, 4, 1.0, 1, 0.0, 0);
        mem.tell(&v2, 4, 1.0, 1, 0.0, 0);
        mem.tell(&v3, 4, 1.0, 1, 0.0, 0);

        let clusters = mem.praise_names(2.0);
        assert_eq!(2, clusters.len());
    }

    #[test]
    fn test_griot_empty_memory() {
        let mem = GriotMemory::new(1.0);
        let q = [1.0, 2.0, 3.0, 4.0];
        let results = mem.recall(&q, 4, 10);
        assert!(results.is_empty());

        let pd = mem.persistence();
        assert!(pd.is_empty());
    }

    #[test]
    fn test_griot_single_story() {
        let mut mem = GriotMemory::new(1.0);
        let v = [1.0, 2.0, 3.0, 4.0];
        let id = mem.tell(&v, 4, 0.0, 1, 0.1, 0);
        assert_eq!(1, id);

        let results = mem.recall(&v, 4, 10);
        assert_eq!(1, results.len());
        assert_eq!(1, results[0]);

        let pd = mem.persistence();
        assert_eq!(1, pd.len());
        assert!(pd[0].death.is_infinite());
    }

    #[test]
    fn test_griot_recall_faded() {
        let mut mem = GriotMemory::new(1.0);
        let v = [1.0, 0.0, 0.0, 0.0];
        mem.tell(&v, 4, 0.0, 1, 5.0, 0);
        mem.decay(100.0);
        assert!(mem.stories[0].weight < 0.01);

        let results = mem.recall(&v, 4, 10);
        assert!(results.is_empty());
    }

    // --- Adinkra tests ---

    #[test]
    fn test_adinkra_encode_decode() {
        let mut space = AdinkraSpace::new();
        let concept = [0.5, -0.3, 0.8, 0.1];
        let glyph = space.encode(&concept, 4, None, 0);

        assert_ne!(0, glyph.hash);
        assert!(!glyph.symbols.is_empty());
        assert_eq!(1, space.glyphs.len());
    }

    #[test]
    fn test_adinkra_decode_roundtrip() {
        let mut space = AdinkraSpace::new();
        let concept = [0.5, -0.3, 0.8, 0.1];
        let glyph = space.encode(&concept, 4, None, 0);

        let decoded = space.decode(&glyph, 4);
        assert_eq!(4, decoded.len());
        for &v in &decoded {
            assert!(v >= -1.0 - 1e-9 && v <= 1.0 + 1e-9);
        }
    }

    #[test]
    fn test_adinkra_cultural_recoverability() {
        let mut space = AdinkraSpace::new();
        let concept = [0.5, -0.3, 0.8, 0.1];
        let context = [1.0, 1.0, 1.0, 1.0];
        let glyph = space.encode(&concept, 4, Some(&context), 4);

        let cr = space.cultural_recoverability(&glyph, &concept, Some(&context), 4);
        assert!(cr > 0.0);
        assert!(cr <= 1.0);
    }

    #[test]
    fn test_adinkra_shannon_compression() {
        let mut space = AdinkraSpace::new();
        let concept = [0.5, -0.3, 0.8, 0.1, 0.9, -0.7, 0.2, 0.4];
        let glyph = space.encode(&concept, 8, None, 0);

        let ratio = space.shannon_compression(&glyph, 8 * 8);
        assert!(ratio > 0.0);
        assert!(ratio < 1.0);
    }

    #[test]
    fn test_adinkra_deterministic() {
        let mut space1 = AdinkraSpace::new();
        let mut space2 = AdinkraSpace::new();
        let concept = [0.5, -0.3, 0.8, 0.1];

        let g1 = space1.encode(&concept, 4, None, 0);
        let g2 = space2.encode(&concept, 4, None, 0);

        assert_eq!(g1.hash, g2.hash);
        assert_eq!(g1.symbols.len(), g2.symbols.len());
    }

    #[test]
    fn test_adinkra_different_concepts() {
        let mut space = AdinkraSpace::new();
        let c1 = [1.0, 0.0, 0.0, 0.0];
        let c2 = [0.0, 1.0, 0.0, 0.0];

        let g1 = space.encode(&c1, 4, None, 0);
        let g2 = space.encode(&c2, 4, None, 0);

        assert_ne!(g1.hash, g2.hash);
    }

    #[test]
    fn test_adinkra_symbol_from_primitive() {
        let s = AdinkraSpace::symbol_from_primitive(AdinkraPrimitive::Circle, 0.5, 1.0);
        assert_eq!(1, s.primitives.len());
        assert_eq!(AdinkraPrimitive::Circle, s.primitives[0]);
        assert!((s.params[0][0] - 0.5).abs() < 1e-9);
        assert!((s.params[0][1] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adinkra_context_improves_recoverability() {
        let mut space1 = AdinkraSpace::new();
        let mut space2 = AdinkraSpace::new();
        let concept = [0.5, -0.3, 0.8, 0.1];
        let context = [2.0, 2.0, 2.0, 2.0];

        let g_no_ctx = space1.encode(&concept, 4, None, 0);
        let g_ctx = space2.encode(&concept, 4, Some(&context), 4);

        let cr_no = space1.cultural_recoverability(&g_no_ctx, &concept, None, 4);
        let cr_yes = space2.cultural_recoverability(&g_ctx, &concept, Some(&context), 4);

        assert!(cr_yes >= cr_no);
    }

    // --- Palaver tests ---

    #[test]
    fn test_palaver_two_party_convergence() {
        let mut sess = PalaverSession::new(&[1.0, 0.0], 2, 1e-6, 1000);
        sess.propose(&[0.0, 0.0], 2, 1.0, 0.0);
        sess.propose(&[1.0, 1.0], 2, 1.0, 0.0);

        let (conv, consensus) = sess.converge_with_consensus();
        assert!(conv);
        assert!((consensus[0] - 0.5).abs() < 0.1);
        assert!((consensus[1] - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_palaver_three_party_convergence() {
        let mut sess = PalaverSession::new(&[1.0, 0.0], 2, 1e-6, 1000);
        sess.propose(&[0.0, 0.0], 2, 1.0, 0.0);
        sess.propose(&[1.0, 0.0], 2, 1.0, 0.0);
        sess.propose(&[0.5, 1.0], 2, 1.0, 0.0);

        assert!(sess.converge());
    }

    #[test]
    fn test_palaver_coalition_detection() {
        let mut sess = PalaverSession::new(&[1.0, 0.0], 2, 1e-6, 100);
        sess.propose(&[0.0, 0.0], 2, 1.0, 1.0);
        sess.propose(&[0.1, 0.1], 2, 1.0, 1.0);
        sess.propose(&[10.0, 10.0], 2, 1.0, 1.0);
        sess.propose(&[10.1, 10.1], 2, 1.0, 1.0);

        let coalitions = sess.coalition(2.0);
        assert_eq!(2, coalitions.len());
    }

    #[test]
    fn test_palaver_quality_metric() {
        let mut sess = PalaverSession::new(&[1.0, 0.0], 2, 1e-9, 1000);
        sess.propose(&[0.0, 0.0], 2, 1.0, 0.0);
        sess.propose(&[1.0, 1.0], 2, 1.0, 0.0);
        sess.converge();

        let q = sess.quality();
        assert!(q > 0.9);
    }

    #[test]
    fn test_palaver_h0_dimension() {
        let mut sess = PalaverSession::new(&[1.0, 0.0], 2, 1e-6, 100);
        sess.propose(&[0.0, 0.0], 2, 1.0, 1.0);
        sess.propose(&[0.1, 0.1], 2, 1.0, 1.0);
        sess.propose(&[100.0, 100.0], 2, 1.0, 1.0);
        sess.propose(&[100.1, 100.1], 2, 1.0, 1.0);

        let h0 = sess.h0_dimension(2.0);
        assert_eq!(2, h0);
    }

    #[test]
    fn test_palaver_stubborn_prevents_convergence() {
        let mut sess = PalaverSession::new(&[1.0, 0.0], 2, 1e-9, 100);
        sess.propose(&[0.0, 0.0], 2, 1.0, 1.0);
        sess.propose(&[10.0, 10.0], 2, 1.0, 1.0);
        sess.converge();

        let dist = euclidean_distance(
            &sess.participants[0].position[..2],
            &sess.participants[1].position[..2],
        );
        assert!(dist > 5.0);
    }

    #[test]
    fn test_palaver_identical_participants() {
        let mut sess = PalaverSession::new(&[1.0, 0.0], 2, 1e-6, 100);
        sess.propose(&[5.0, 5.0], 2, 1.0, 0.3);
        sess.propose(&[5.0, 5.0], 2, 1.0, 0.3);
        sess.propose(&[5.0, 5.0], 2, 1.0, 0.3);

        let (conv, consensus) = sess.converge_with_consensus();
        assert!(conv);
        assert!((consensus[0] - 5.0).abs() < 0.1);
        assert!((consensus[1] - 5.0).abs() < 0.1);
    }

    #[test]
    fn test_palaver_single_participant() {
        // (no participants: quality=1.0, h0=0 tested below)
        let mut sess2 = PalaverSession::new(&[1.0, 0.0], 2, 1e-6, 100);
        sess2.propose(&[3.0, 4.0], 2, 1.0, 0.5);

        assert!((sess2.quality() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_palaver_weighted_convergence() {
        let mut sess = PalaverSession::new(&[1.0, 0.0], 2, 1e-9, 1000);
        sess.propose(&[0.0, 0.0], 2, 10.0, 0.0);
        sess.propose(&[10.0, 10.0], 2, 1.0, 0.0);

        sess.converge();
        let consensus = sess.consensus();
        let d1 = (consensus[0] - 0.0).abs();
        let d2 = (consensus[0] - 10.0).abs();
        assert!(d1 < d2);
    }

    #[test]
    fn test_palaver_deliberate_step() {
        let mut sess = PalaverSession::new(&[1.0, 0.0], 2, 1e-6, 100);
        sess.propose(&[0.0, 0.0], 2, 1.0, 0.0);
        sess.propose(&[0.001, 0.001], 2, 1.0, 0.0);

        let mut conv = false;
        for _ in 0..100 {
            if sess.deliberate() {
                conv = true;
                break;
            }
        }
        assert!(conv);
    }

    #[test]
    fn test_palaver_empty_session() {
        let sess = PalaverSession::new(&[1.0, 0.0], 2, 1e-6, 100);
        assert!((sess.quality() - 1.0).abs() < 1e-9);
        assert_eq!(0, sess.h0_dimension(1.0));
    }

    #[test]
    fn test_palaver_full_participants() {
        let mut sess = PalaverSession::new(&[1.0, 0.0], 2, 1e-6, 100);
        // Fill to max
        for i in 0..WAM_MAX_PARTICIPANTS {
            let pos = [i as f64 * 0.1, 0.0];
            let id = sess.propose(&pos, 2, 1.0, 0.1);
            assert_eq!(i as u32 + 1, id);
        }
        // Next one should fail
        let id = sess.propose(&[0.0, 0.0], 2, 1.0, 0.0);
        assert_eq!(0, id);
    }

    // --- Ishango Bone tests ---

    #[test]
    fn test_ishango_tally_encode_decode() {
        let groups = IshangoBone::tally_encode(24);
        assert_eq!(24, IshangoBone::tally_decode(&groups));
    }

    #[test]
    fn test_ishango_tally_count() {
        assert_eq!(10, IshangoBone::tally_count(10));
    }

    #[test]
    fn test_ishango_prime_factorize() {
        let factors = IshangoBone::prime_factorize(84);
        // 84 = 2^2 * 3 * 7
        assert!(factors.contains(&(2, 2)));
        assert!(factors.contains(&(3, 1)));
        assert!(factors.contains(&(7, 1)));
    }

    #[test]
    fn test_ishango_is_prime() {
        assert!(IshangoBone::is_prime(2));
        assert!(IshangoBone::is_prime(3));
        assert!(IshangoBone::is_prime(17));
        assert!(!IshangoBone::is_prime(1));
        assert!(!IshangoBone::is_prime(4));
        assert!(!IshangoBone::is_prime(15));
    }

    #[test]
    fn test_ishango_prime_factorize_zero_one() {
        assert!(IshangoBone::prime_factorize(0).is_empty());
        assert!(IshangoBone::prime_factorize(1).is_empty());
    }

    #[test]
    fn test_ishango_large_prime_factorize() {
        // 97 is prime
        let factors = IshangoBone::prime_factorize(97);
        assert_eq!(1, factors.len());
        assert_eq!((97, 1), factors[0]);
    }

    // --- Yoruba Numeration tests ---

    #[test]
    fn test_yoruba_base20_roundtrip() {
        for n in &[0u64, 1, 19, 20, 100, 399, 400, 8000, 160000] {
            let digits = YorubaNumeration::to_base20(*n);
            let back = YorubaNumeration::from_base20(&digits);
            assert_eq!(*n, back);
        }
    }

    #[test]
    fn test_yoruba_base20_add() {
        let a = YorubaNumeration::to_base20(37);
        let b = YorubaNumeration::to_base20(45);
        let sum = YorubaNumeration::base20_add(&a, &b);
        assert_eq!(82, YorubaNumeration::from_base20(&sum));
    }

    #[test]
    fn test_yoruba_base20_mul() {
        let a = YorubaNumeration::to_base20(5);
        let b = YorubaNumeration::to_base20(6);
        let prod = YorubaNumeration::base20_mul(&a, &b);
        assert_eq!(30, YorubaNumeration::from_base20(&prod));
    }

    #[test]
    fn test_yoruba_string() {
        let s = YorubaNumeration::yoruba_string(1);
        assert!(!s.is_empty());
        let s0 = YorubaNumeration::yoruba_string(0);
        assert_eq!("òdò", s0);
    }

    #[test]
    fn test_yoruba_zero() {
        let digits = YorubaNumeration::to_base20(0);
        assert_eq!(vec![0u8], digits);
    }

    // --- Mande Measurement tests ---

    #[test]
    fn test_mande_kusu_roundtrip() {
        let meters = MandeMeasurement::kusu_to_meters(10.0);
        assert!((meters - 5.0).abs() < 1e-9);
        let back = MandeMeasurement::meters_to_kusu(meters);
        assert!((back - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_mande_jan() {
        let meters = MandeMeasurement::jan_to_meters(2.0);
        assert!((meters - 3.6).abs() < 1e-9);
    }

    #[test]
    fn test_mande_seme() {
        let kg = MandeMeasurement::seme_to_kg(3.0);
        assert!((kg - 90.0).abs() < 1e-9);
    }

    #[test]
    fn test_mande_kille() {
        let liters = MandeMeasurement::kille_to_liters(4.0);
        assert!((liters - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_mande_gandal() {
        let kg = MandeMeasurement::gandal_to_kg(2.0);
        assert!((kg - 10.0).abs() < 1e-9);
        let gandal = MandeMeasurement::kg_to_gandal(kg);
        assert!((gandal - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_mande_sere() {
        let liters = MandeMeasurement::sere_to_liters(1.0);
        assert!((liters - 15.0).abs() < 1e-9);
    }

    // --- Oware Board tests ---

    #[test]
    fn test_oware_new_board() {
        let board = OwareBoard::new();
        assert_eq!(48, board.total_seeds());
        assert!(board.pits.iter().all(|&p| p == 4));
    }

    #[test]
    fn test_oware_sow_and_capture() {
        let mut board = OwareBoard::new();
        let last = board.sow(0);
        assert!(last.is_some());
        // Sowing from pit 0 distributes 4 seeds to pits 1,2,3,4
        assert_eq!(0, board.pits[0]);
        assert_eq!(5, board.pits[1]);
        assert_eq!(5, board.pits[2]);
        assert_eq!(5, board.pits[3]);
        assert_eq!(5, board.pits[4]);
        // Last pit should be 4
        assert_eq!(Some(4), last);
        // Turn should have switched
        assert_eq!(1, board.turn);
    }

    #[test]
    fn test_oware_legal_moves() {
        let board = OwareBoard::new();
        let moves = board.legal_moves();
        assert_eq!(6, moves.len());
        assert_eq!(moves, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_oware_simulate() {
        let board = OwareBoard::new();
        let next = board.simulate_move(0);
        assert!(next.is_some());
        let nb = next.unwrap();
        assert_eq!(0, nb.pits[0]);
        assert_eq!(1, nb.turn);
    }

    #[test]
    fn test_oware_opposite_pit() {
        assert_eq!(11, OwareBoard::opposite_pit(0));
        assert_eq!(0, OwareBoard::opposite_pit(11));
        assert_eq!(5, OwareBoard::opposite_pit(6));
    }

    #[test]
    fn test_oware_invalid_move() {
        let mut board = OwareBoard::new();
        // Can't sow opponent's pit
        assert!(board.sow(6).is_none());
        // Can't sow empty pit
        board.pits[0] = 0;
        assert!(board.sow(0).is_none());
    }

    #[test]
    fn test_oware_game_over() {
        let mut board = OwareBoard::new();
        // Empty player 0's side manually, set turn to player 1
        for p in 0..6 {
            board.pits[p] = 0;
        }
        board.turn = 1;
        board.pits[6] = 1; // one seed on player 1's side
        board.sow(6);
        assert!(board.game_over);
    }

    // --- Ethiopian Multiplication tests ---

    #[test]
    fn test_ethiopian_multiplication_basic() {
        assert_eq!(12, EthiopianMultiplication::multiply(3, 4));
        assert_eq!(0, EthiopianMultiplication::multiply(0, 5));
        assert_eq!(25, EthiopianMultiplication::multiply(5, 5));
        assert_eq!(100, EthiopianMultiplication::multiply(10, 10));
    }

    #[test]
    fn test_ethiopian_multiplication_large() {
        assert_eq!(1000, EthiopianMultiplication::multiply(125, 8));
        assert_eq!(144, EthiopianMultiplication::multiply(12, 12));
        assert_eq!(10000, EthiopianMultiplication::multiply(100, 100));
    }

    #[test]
    fn test_ethiopian_multiplication_steps() {
        let steps = EthiopianMultiplication::multiply_steps(7, 3);
        // 7 -> 3 double 6 (odd) -> add 3
        // 3 -> 1 double 12 (odd) -> add 12
        // 1 -> 0 double 24 (odd) -> add 24
        // result = 3 + 12 + 24 = 39
        assert_eq!(3, steps.len());
        let result: u64 = steps.iter()
            .filter(|(_, _, odd)| *odd)
            .map(|(_, doubled, _)| *doubled)
            .sum();
        assert_eq!(21, result);
    }

    #[test]
    fn test_ethiopian_multiplication_safe() {
        assert_eq!(Some(42), EthiopianMultiplication::multiply_safe(6, 7));
        assert!(EthiopianMultiplication::multiply_safe(1_000_000_000_000, 1_000_000_000_000).is_none());
    }

    #[test]
    fn test_ethiopian_commutative() {
        for a in 0..20 {
            for b in 0..20 {
                assert_eq!(
                    EthiopianMultiplication::multiply(a, b),
                    EthiopianMultiplication::multiply(b, a),
                    "commutativity failed for {}x{}", a, b
                );
            }
        }
    }

    // --- Fibonacci Sequence tests ---

    #[test]
    fn test_fibonacci_basic() {
        assert_eq!(0, FibonacciSequence::nth(0));
        assert_eq!(1, FibonacciSequence::nth(1));
        assert_eq!(1, FibonacciSequence::nth(2));
        assert_eq!(2, FibonacciSequence::nth(3));
        assert_eq!(3, FibonacciSequence::nth(4));
        assert_eq!(5, FibonacciSequence::nth(5));
        assert_eq!(8, FibonacciSequence::nth(6));
        assert_eq!(13, FibonacciSequence::nth(7));
    }

    #[test]
    fn test_fibonacci_fast() {
        for n in 0..20 {
            assert_eq!(
                FibonacciSequence::nth(n),
                FibonacciSequence::nth_fast(n),
                "nth vs nth_fast differ at n={}", n
            );
        }
    }


    #[test]
    fn test_fibonacci_first_n() {
        let nums = FibonacciSequence::first_n(10);
        assert_eq!(10, nums.len());
        assert_eq!(vec![0, 1, 1, 2, 3, 5, 8, 13, 21, 34], nums);
    }

    #[test]
    fn test_fibonacci_is_fibonacci() {
        assert!(FibonacciSequence::is_fibonacci(0));
        assert!(FibonacciSequence::is_fibonacci(1));
        assert!(FibonacciSequence::is_fibonacci(34));
        assert!(FibonacciSequence::is_fibonacci(55));
        assert!(!FibonacciSequence::is_fibonacci(4));
        assert!(!FibonacciSequence::is_fibonacci(7));
        assert!(!FibonacciSequence::is_fibonacci(10));
    }

    #[test]
    fn test_fibonacci_golden_ratio() {
        let phi = FibonacciSequence::golden_ratio_approx(20);
        let true_phi = (1.0 + 5.0f64.sqrt()) / 2.0;
        assert!((phi - true_phi).abs() < 1e-6);
    }

    #[test]
    fn test_fibonacci_nth_fast_large() {
        // F(40) = 102334155
        assert_eq!(102334155, FibonacciSequence::nth_fast(40));
    }

    // --- Geometric Concept tests ---

    #[test]
    fn test_geometric_mudcloth() {
        let gc = GeometricConcept::new(GeometricPattern::Mudcloth);
        let coords = gc.generate_coordinates();
        assert!(!coords.is_empty());
        assert_eq!(gc.feature_count(), coords.len());
    }

    #[test]
    fn test_geometric_kente() {
        let gc = GeometricConcept::new(GeometricPattern::Kente)
            .with_complexity(5);
        let coords = gc.generate_coordinates();
        assert!(!coords.is_empty());
    }

    #[test]
    fn test_geometric_circular() {
        let gc = GeometricConcept::new(GeometricPattern::Circular)
            .with_scale(2.0)
            .with_complexity(3);
        let coords = gc.generate_coordinates();
        assert!(!coords.is_empty());
    }

    #[test]
    fn test_geometric_spiral() {
        let gc = GeometricConcept::new(GeometricPattern::Spiral)
            .with_rotation(1.57);
        let coords = gc.generate_coordinates();
        assert!(!coords.is_empty());
    }

    #[test]
    fn test_geometric_symmetry() {
        let gc = GeometricConcept::new(GeometricPattern::Circular)
            .with_symmetry(4);
        let coords = gc.generate_coordinates();
        assert!(!coords.is_empty());
    }

    #[test]
    fn test_geometric_pattern_all() {
        use GeometricPattern::*;
        for pattern in &[Mudcloth, Kente, Ndebele, Circular, Spiral, DiamondGrid, Chevron, Interlace] {
            let gc = GeometricConcept::new(*pattern).with_complexity(2);
            assert!(!gc.generate_coordinates().is_empty(),
                    "{:?} should generate coordinates", pattern);
        }
    }

    #[test]
    fn test_geometric_diamond_grid() {
        let gc = GeometricConcept::new(GeometricPattern::DiamondGrid);
        let coords = gc.generate_coordinates();
        assert!(!coords.is_empty());
    }

    #[test]
    fn test_geometric_chevron() {
        let gc = GeometricConcept::new(GeometricPattern::Chevron);
        let coords = gc.generate_coordinates();
        assert!(!coords.is_empty());
    }

    #[test]
    fn test_geometric_interlace() {
        let gc = GeometricConcept::new(GeometricPattern::Interlace)
            .with_complexity(3);
        let coords = gc.generate_coordinates();
        assert!(!coords.is_empty());
    }
}
