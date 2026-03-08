package goformersearch

// Index is the interface implemented by all index types.
type Index interface {
	// Add inserts a vector with the given ID. The vector is copied.
	Add(id uint64, vec []float32)

	// Search returns the k nearest neighbours to the query vector,
	// ordered by decreasing similarity (highest first).
	Search(query []float32, k int) []Result

	// Len returns the number of vectors in the index.
	Len() int

	// Dims returns the dimensionality of the index.
	Dims() int
}

// Result holds a search result: the vector ID and its similarity score.
type Result struct {
	ID         uint64
	Similarity float32
}
