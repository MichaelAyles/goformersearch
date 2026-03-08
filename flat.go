package goformersearch

// FlatIndex is a brute-force exact nearest-neighbour index. It computes
// cosine similarity against every vector on each query. Exact results, O(n)
// per query.
//
// Safe for concurrent Search calls once all Add calls are complete.
type FlatIndex struct {
	dims    int
	vectors []float32 // contiguous row-major storage
	ids     []uint64
}

// NewFlatIndex creates a brute-force index for vectors of the given dimensionality.
func NewFlatIndex(dims int) *FlatIndex {
	return &FlatIndex{dims: dims}
}

// Add inserts a vector with the given ID. The vector is copied.
func (f *FlatIndex) Add(id uint64, vec []float32) {
	f.vectors = append(f.vectors, vec...)
	f.ids = append(f.ids, id)
}

// Search returns the k nearest neighbours to the query vector, ordered by
// decreasing similarity (highest first). For normalised vectors the similarity
// is computed as a dot product; for non-normalised vectors full cosine
// similarity is used.
func (f *FlatIndex) Search(query []float32, k int) []Result {
	n := len(f.ids)
	if n == 0 {
		return nil
	}
	if k > n {
		k = n
	}

	h := newResultHeap(k)
	for i := 0; i < n; i++ {
		start := i * f.dims
		vec := f.vectors[start : start+f.dims]
		sim := DotProduct(query, vec)
		h.push(f.ids[i], sim)
	}
	return h.sorted()
}

// Len returns the number of vectors in the index.
func (f *FlatIndex) Len() int { return len(f.ids) }

// Dims returns the dimensionality of the index.
func (f *FlatIndex) Dims() int { return f.dims }
