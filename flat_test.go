package goformersearch

import (
	"math"
	"math/rand"
	"testing"
)

func TestFlatEmpty(t *testing.T) {
	idx := NewFlatIndex(4)
	results := idx.Search([]float32{1, 0, 0, 0}, 5)
	if len(results) != 0 {
		t.Fatalf("expected 0 results, got %d", len(results))
	}
}

func TestFlatSingle(t *testing.T) {
	idx := NewFlatIndex(3)
	idx.Add(42, []float32{1, 0, 0})
	results := idx.Search([]float32{1, 0, 0}, 5)
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}
	if results[0].ID != 42 {
		t.Fatalf("expected ID 42, got %d", results[0].ID)
	}
	if results[0].Similarity < 0.999 {
		t.Fatalf("expected similarity ~1.0, got %f", results[0].Similarity)
	}
}

func TestFlatExactResults(t *testing.T) {
	idx := NewFlatIndex(3)
	// Three orthogonal unit vectors.
	idx.Add(1, []float32{1, 0, 0})
	idx.Add(2, []float32{0, 1, 0})
	idx.Add(3, []float32{0, 0, 1})

	results := idx.Search([]float32{1, 0, 0}, 2)
	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}
	if results[0].ID != 1 {
		t.Fatalf("expected first result ID 1, got %d", results[0].ID)
	}
	if results[0].Similarity < 0.999 {
		t.Fatalf("expected similarity ~1.0, got %f", results[0].Similarity)
	}
}

func TestFlatKGreaterThanN(t *testing.T) {
	idx := NewFlatIndex(2)
	idx.Add(1, []float32{1, 0})
	idx.Add(2, []float32{0, 1})

	results := idx.Search([]float32{1, 0}, 100)
	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}
}

func TestFlatOrdering(t *testing.T) {
	idx := NewFlatIndex(2)
	// Normalised vectors at different angles from the query.
	idx.Add(1, normalise([]float32{1, 0}))
	idx.Add(2, normalise([]float32{1, 1}))
	idx.Add(3, normalise([]float32{0, 1}))

	results := idx.Search(normalise([]float32{1, 0}), 3)
	if len(results) != 3 {
		t.Fatalf("expected 3 results, got %d", len(results))
	}
	// Should be ordered: ID 1 (exact match), ID 2 (45 degrees), ID 3 (90 degrees).
	if results[0].ID != 1 || results[1].ID != 2 || results[2].ID != 3 {
		t.Fatalf("unexpected ordering: %v", results)
	}
	// Verify decreasing similarity.
	for i := 1; i < len(results); i++ {
		if results[i].Similarity > results[i-1].Similarity {
			t.Fatalf("results not in decreasing similarity order at index %d", i)
		}
	}
}

func TestFlatLenDims(t *testing.T) {
	idx := NewFlatIndex(384)
	if idx.Len() != 0 {
		t.Fatalf("expected len 0, got %d", idx.Len())
	}
	if idx.Dims() != 384 {
		t.Fatalf("expected dims 384, got %d", idx.Dims())
	}
	idx.Add(1, make([]float32, 384))
	if idx.Len() != 1 {
		t.Fatalf("expected len 1, got %d", idx.Len())
	}
}

func TestFlatInterface(t *testing.T) {
	var _ Index = NewFlatIndex(4)
}

func TestFlatZeroQuery(t *testing.T) {
	idx := NewFlatIndex(3)
	idx.Add(1, normalise([]float32{1, 0, 0}))
	idx.Add(2, normalise([]float32{0, 1, 0}))

	results := idx.Search([]float32{0, 0, 0}, 2)
	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}
}

// normalise L2-normalises a vector in place and returns it.
func normalise(v []float32) []float32 {
	var norm float64
	for _, x := range v {
		norm += float64(x) * float64(x)
	}
	norm = math.Sqrt(norm)
	if norm > 0 {
		for i := range v {
			v[i] = float32(float64(v[i]) / norm)
		}
	}
	return v
}

// randomNormalisedVec returns a random L2-normalised vector.
func randomNormalisedVec(rng *rand.Rand, dims int) []float32 {
	v := make([]float32, dims)
	for i := range v {
		v[i] = rng.Float32()*2 - 1
	}
	return normalise(v)
}
