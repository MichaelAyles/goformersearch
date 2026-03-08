package goformersearch

import "sort"

// resultHeap is a fixed-size min-heap by similarity. The root is the worst
// (lowest similarity) result, so the "should I insert?" check is O(1).
// Used by both flat and HNSW search for top-k selection.
type resultHeap struct {
	results []Result
	k       int
}

func newResultHeap(k int) *resultHeap {
	return &resultHeap{
		results: make([]Result, 0, k),
		k:       k,
	}
}

// push adds a result. If the heap is full and the new result is worse than
// the worst in the heap, it is discarded.
func (h *resultHeap) push(id uint64, similarity float32) {
	if len(h.results) < h.k {
		h.results = append(h.results, Result{ID: id, Similarity: similarity})
		h.siftUp(len(h.results) - 1)
		return
	}
	// Heap is full — only insert if better than the worst (root).
	if similarity <= h.results[0].Similarity {
		return
	}
	h.results[0] = Result{ID: id, Similarity: similarity}
	h.siftDown(0)
}

// sorted returns the heap contents sorted by decreasing similarity.
func (h *resultHeap) sorted() []Result {
	out := make([]Result, len(h.results))
	copy(out, h.results)
	sort.Slice(out, func(i, j int) bool {
		return out[i].Similarity > out[j].Similarity
	})
	return out
}

func (h *resultHeap) siftUp(i int) {
	for i > 0 {
		parent := (i - 1) / 2
		if h.results[parent].Similarity <= h.results[i].Similarity {
			break
		}
		h.results[parent], h.results[i] = h.results[i], h.results[parent]
		i = parent
	}
}

func (h *resultHeap) siftDown(i int) {
	n := len(h.results)
	for {
		smallest := i
		left := 2*i + 1
		right := 2*i + 2
		if left < n && h.results[left].Similarity < h.results[smallest].Similarity {
			smallest = left
		}
		if right < n && h.results[right].Similarity < h.results[smallest].Similarity {
			smallest = right
		}
		if smallest == i {
			break
		}
		h.results[i], h.results[smallest] = h.results[smallest], h.results[i]
		i = smallest
	}
}
