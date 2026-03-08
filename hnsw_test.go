package goformersearch

import (
	"math/rand"
	"testing"
)

func TestHNSWEmpty(t *testing.T) {
	idx := NewHNSWIndex(4)
	results := idx.Search([]float32{1, 0, 0, 0}, 5)
	if len(results) != 0 {
		t.Fatalf("expected 0 results, got %d", len(results))
	}
}

func TestHNSWSingle(t *testing.T) {
	idx := NewHNSWIndex(3)
	idx.Add(42, normalise([]float32{1, 0, 0}))
	results := idx.Search(normalise([]float32{1, 0, 0}), 5)
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}
	if results[0].ID != 42 {
		t.Fatalf("expected ID 42, got %d", results[0].ID)
	}
}

func TestHNSWKGreaterThanN(t *testing.T) {
	idx := NewHNSWIndex(3)
	idx.Add(1, normalise([]float32{1, 0, 0}))
	idx.Add(2, normalise([]float32{0, 1, 0}))
	results := idx.Search(normalise([]float32{1, 0, 0}), 100)
	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}
}

func TestHNSWInterface(t *testing.T) {
	var _ Index = NewHNSWIndex(4)
}

func TestHNSWLenDims(t *testing.T) {
	idx := NewHNSWIndex(384)
	if idx.Len() != 0 || idx.Dims() != 384 {
		t.Fatalf("expected len=0 dims=384, got len=%d dims=%d", idx.Len(), idx.Dims())
	}
	idx.Add(1, make([]float32, 384))
	if idx.Len() != 1 {
		t.Fatalf("expected len 1, got %d", idx.Len())
	}
}

func TestHNSWRecall(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping recall test in short mode")
	}

	const (
		dims    = 64
		n       = 5000
		queries = 200
		k       = 10
	)

	rng := rand.New(rand.NewSource(123))

	// Build ground truth flat index.
	flat := NewFlatIndex(dims)
	vectors := make([][]float32, n)
	for i := 0; i < n; i++ {
		v := randomNormalisedVec(rng, dims)
		vectors[i] = v
		flat.Add(uint64(i), v)
	}

	tests := []struct {
		efSearch    int
		minRecall   float64
		efConstruct int
	}{
		{50, 0.90, 200},
		{100, 0.95, 200},
		{200, 0.98, 200},
	}

	for _, tt := range tests {
		hnsw := NewHNSWIndex(dims, WithM(16), WithEfConstruction(tt.efConstruct))
		for i, v := range vectors {
			hnsw.Add(uint64(i), v)
		}
		hnsw.SetEfSearch(tt.efSearch)

		var totalRecall float64
		for q := 0; q < queries; q++ {
			query := randomNormalisedVec(rng, dims)
			flatResults := flat.Search(query, k)
			hnswResults := hnsw.Search(query, k)

			// Count how many of the true top-k are in the HNSW results.
			trueSet := make(map[uint64]bool)
			for _, r := range flatResults {
				trueSet[r.ID] = true
			}
			hits := 0
			for _, r := range hnswResults {
				if trueSet[r.ID] {
					hits++
				}
			}
			totalRecall += float64(hits) / float64(k)
		}
		meanRecall := totalRecall / float64(queries)
		t.Logf("efSearch=%d: recall@%d = %.4f", tt.efSearch, k, meanRecall)
		if meanRecall < tt.minRecall {
			t.Errorf("efSearch=%d: recall@%d = %.4f, want >= %.2f",
				tt.efSearch, k, meanRecall, tt.minRecall)
		}
	}
}
