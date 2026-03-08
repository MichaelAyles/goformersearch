package goformersearch

import (
	"math"
	"math/rand"
	"sort"
)

// HNSWIndex is an approximate nearest-neighbour index using the Hierarchical
// Navigable Small World algorithm (Malkov & Yashunin, 2018).
//
// Safe for concurrent Search calls once all Add calls are complete.
type HNSWIndex struct {
	dims       int
	cfg        hnswConfig
	nodes      []hnswNode
	entryPoint int // index of the entry point node
	maxLevel   int // highest layer in the graph
	mL         float64
	rng        *rand.Rand
}

// NewHNSWIndex creates an HNSW index for approximate nearest-neighbour search.
func NewHNSWIndex(dims int, opts ...HNSWOption) *HNSWIndex {
	cfg := defaultConfig()
	for _, o := range opts {
		o(&cfg)
	}
	return &HNSWIndex{
		dims:       dims,
		cfg:        cfg,
		entryPoint: -1,
		maxLevel:   -1,
		mL:         1.0 / math.Log(float64(cfg.m)),
		rng:        rand.New(rand.NewSource(42)),
	}
}

// SetEfSearch adjusts the search-time quality/speed tradeoff.
// Higher values give better recall at the cost of latency.
func (h *HNSWIndex) SetEfSearch(ef int) {
	h.cfg.efSearch = ef
}

// Add inserts a vector with the given ID. The vector is copied.
func (h *HNSWIndex) Add(id uint64, vec []float32) {
	v := make([]float32, len(vec))
	copy(v, vec)

	level := h.randomLevel()
	node := hnswNode{
		id:      id,
		vec:     v,
		level:   level,
		friends: make([][]uint32, level+1),
	}
	idx := uint32(len(h.nodes))
	h.nodes = append(h.nodes, node)

	// First node — set as entry point.
	if h.entryPoint == -1 {
		h.entryPoint = int(idx)
		h.maxLevel = level
		return
	}

	ep := uint32(h.entryPoint)

	// Phase 1: greedily traverse layers above the new node's level.
	for lc := h.maxLevel; lc > level; lc-- {
		ep = h.greedyClosest(vec, ep, lc)
	}

	// Phase 2: insert at each layer from the node's level down to 0.
	for lc := min(level, h.maxLevel); lc >= 0; lc-- {
		candidates := h.searchLayer(vec, ep, h.cfg.efConstruction, lc)
		neighbours := h.selectNeighbours(candidates, h.mForLevel(lc))

		// Connect new node to neighbours.
		h.nodes[idx].friends[lc] = neighbours

		// Connect neighbours back to new node, pruning if needed.
		mMax := h.mForLevel(lc)
		for _, nIdx := range neighbours {
			h.nodes[nIdx].friends[lc] = append(h.nodes[nIdx].friends[lc], idx)
			if len(h.nodes[nIdx].friends[lc]) > mMax {
				h.pruneConnections(nIdx, lc, mMax)
			}
		}

		if len(candidates) > 0 {
			ep = candidates[0].index
		}
	}

	// Update entry point if new node is on a higher level.
	if level > h.maxLevel {
		h.entryPoint = int(idx)
		h.maxLevel = level
	}
}

// Search returns the k nearest neighbours to the query vector, ordered by
// decreasing similarity (highest first).
func (h *HNSWIndex) Search(query []float32, k int) []Result {
	if len(h.nodes) == 0 {
		return nil
	}
	if k > len(h.nodes) {
		k = len(h.nodes)
	}

	ep := uint32(h.entryPoint)

	// Traverse upper layers greedily.
	for lc := h.maxLevel; lc > 0; lc-- {
		ep = h.greedyClosest(query, ep, lc)
	}

	// Search layer 0 with efSearch width.
	ef := h.cfg.efSearch
	if ef < k {
		ef = k
	}
	candidates := h.searchLayer(query, ep, ef, 0)

	// Return top k.
	if len(candidates) > k {
		candidates = candidates[:k]
	}
	results := make([]Result, len(candidates))
	for i, c := range candidates {
		results[i] = Result{
			ID:         h.nodes[c.index].id,
			Similarity: c.dist,
		}
	}
	return results
}

// Len returns the number of vectors in the index.
func (h *HNSWIndex) Len() int { return len(h.nodes) }

// Dims returns the dimensionality of the index.
func (h *HNSWIndex) Dims() int { return h.dims }

// candidate is a node index with its distance (similarity) to the query.
type candidate struct {
	index uint32
	dist  float32 // similarity (higher = closer)
}

// greedyClosest finds the closest node to the query on a given layer by
// greedily following the best neighbour.
func (h *HNSWIndex) greedyClosest(query []float32, ep uint32, level int) uint32 {
	bestDist := DotProduct(query, h.nodes[ep].vec)
	for {
		changed := false
		friends := h.nodes[ep].friends
		if level >= len(friends) {
			break
		}
		for _, fIdx := range friends[level] {
			d := DotProduct(query, h.nodes[fIdx].vec)
			if d > bestDist {
				bestDist = d
				ep = fIdx
				changed = true
			}
		}
		if !changed {
			break
		}
	}
	return ep
}

// searchLayer performs a beam search on a single layer, returning up to ef
// candidates sorted by decreasing similarity.
func (h *HNSWIndex) searchLayer(query []float32, ep uint32, ef int, level int) []candidate {
	visited := make(map[uint32]bool)
	visited[ep] = true

	epDist := DotProduct(query, h.nodes[ep].vec)
	candidates := []candidate{{index: ep, dist: epDist}} // max-heap candidates (best first)
	results := []candidate{{index: ep, dist: epDist}}    // collected results

	for len(candidates) > 0 {
		// Pop the best (highest similarity) candidate.
		bestIdx := 0
		for i := 1; i < len(candidates); i++ {
			if candidates[i].dist > candidates[bestIdx].dist {
				bestIdx = i
			}
		}
		current := candidates[bestIdx]
		candidates[bestIdx] = candidates[len(candidates)-1]
		candidates = candidates[:len(candidates)-1]

		// Worst result so far.
		worstResult := results[0].dist
		for _, r := range results[1:] {
			if r.dist < worstResult {
				worstResult = r.dist
			}
		}

		// If the best candidate is worse than the worst result, stop.
		if current.dist < worstResult && len(results) >= ef {
			break
		}

		// Expand the current candidate's neighbours.
		friends := h.nodes[current.index].friends
		if level >= len(friends) {
			continue
		}
		for _, fIdx := range friends[level] {
			if visited[fIdx] {
				continue
			}
			visited[fIdx] = true

			d := DotProduct(query, h.nodes[fIdx].vec)

			// Find worst result.
			worstIdx := 0
			for i := 1; i < len(results); i++ {
				if results[i].dist < results[worstIdx].dist {
					worstIdx = i
				}
			}

			if len(results) < ef || d > results[worstIdx].dist {
				candidates = append(candidates, candidate{index: fIdx, dist: d})
				results = append(results, candidate{index: fIdx, dist: d})
				if len(results) > ef {
					// Remove the worst result.
					worstIdx = 0
					for i := 1; i < len(results); i++ {
						if results[i].dist < results[worstIdx].dist {
							worstIdx = i
						}
					}
					results[worstIdx] = results[len(results)-1]
					results = results[:len(results)-1]
				}
			}
		}
	}

	// Sort by decreasing similarity.
	sort.Slice(results, func(i, j int) bool {
		return results[i].dist > results[j].dist
	})
	return results
}

// selectNeighbours picks the best m neighbours from candidates (simple heuristic: closest by similarity).
func (h *HNSWIndex) selectNeighbours(candidates []candidate, m int) []uint32 {
	if len(candidates) <= m {
		out := make([]uint32, len(candidates))
		for i, c := range candidates {
			out[i] = c.index
		}
		return out
	}
	out := make([]uint32, m)
	for i := 0; i < m; i++ {
		out[i] = candidates[i].index
	}
	return out
}

// pruneConnections trims a node's friend list on a given layer to at most m,
// keeping the closest neighbours.
func (h *HNSWIndex) pruneConnections(nodeIdx uint32, level int, m int) {
	friends := h.nodes[nodeIdx].friends[level]
	if len(friends) <= m {
		return
	}
	vec := h.nodes[nodeIdx].vec
	type scored struct {
		idx  uint32
		dist float32
	}
	scored_ := make([]scored, len(friends))
	for i, fIdx := range friends {
		scored_[i] = scored{idx: fIdx, dist: DotProduct(vec, h.nodes[fIdx].vec)}
	}
	sort.Slice(scored_, func(i, j int) bool {
		return scored_[i].dist > scored_[j].dist
	})
	pruned := make([]uint32, m)
	for i := 0; i < m; i++ {
		pruned[i] = scored_[i].idx
	}
	h.nodes[nodeIdx].friends[level] = pruned
}

func (h *HNSWIndex) mForLevel(level int) int {
	if level == 0 {
		return h.cfg.mMax0
	}
	return h.cfg.m
}

func (h *HNSWIndex) randomLevel() int {
	return int(math.Floor(-math.Log(h.rng.Float64()) * h.mL))
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
