package goformersearch

import (
	"bytes"
	"math"
	"math/rand"
	"testing"
)

func TestSaveLoadFlat(t *testing.T) {
	idx := NewFlatIndex(4)
	idx.Add(10, normalise([]float32{1, 0, 0, 0}))
	idx.Add(20, normalise([]float32{0, 1, 0, 0}))
	idx.Add(30, normalise([]float32{0, 0, 1, 0}))

	var buf bytes.Buffer
	if err := Save(&buf, idx); err != nil {
		t.Fatalf("Save: %v", err)
	}

	loaded, err := LoadFlat(&buf)
	if err != nil {
		t.Fatalf("LoadFlat: %v", err)
	}

	if loaded.Len() != idx.Len() {
		t.Fatalf("len mismatch: got %d, want %d", loaded.Len(), idx.Len())
	}
	if loaded.Dims() != idx.Dims() {
		t.Fatalf("dims mismatch: got %d, want %d", loaded.Dims(), idx.Dims())
	}

	// Verify search results match.
	query := normalise([]float32{1, 0, 0, 0})
	origResults := idx.Search(query, 3)
	loadedResults := loaded.Search(query, 3)

	for i := range origResults {
		if origResults[i].ID != loadedResults[i].ID {
			t.Fatalf("result %d: ID mismatch: got %d, want %d", i, loadedResults[i].ID, origResults[i].ID)
		}
		if math.Abs(float64(origResults[i].Similarity-loadedResults[i].Similarity)) > 1e-6 {
			t.Fatalf("result %d: similarity mismatch", i)
		}
	}
}

func TestSaveLoadHNSW(t *testing.T) {
	const dims = 16
	rng := rand.New(rand.NewSource(99))

	idx := NewHNSWIndex(dims, WithM(8), WithEfConstruction(100))
	for i := 0; i < 100; i++ {
		idx.Add(uint64(i), randomNormalisedVec(rng, dims))
	}

	var buf bytes.Buffer
	if err := Save(&buf, idx); err != nil {
		t.Fatalf("Save: %v", err)
	}

	loaded, err := LoadHNSW(&buf)
	if err != nil {
		t.Fatalf("LoadHNSW: %v", err)
	}

	if loaded.Len() != idx.Len() {
		t.Fatalf("len mismatch: got %d, want %d", loaded.Len(), idx.Len())
	}
	if loaded.Dims() != idx.Dims() {
		t.Fatalf("dims mismatch: got %d, want %d", loaded.Dims(), idx.Dims())
	}

	// Both should return the same results for the same query.
	query := randomNormalisedVec(rng, dims)
	idx.SetEfSearch(100)
	loaded.SetEfSearch(100)

	origResults := idx.Search(query, 10)
	loadedResults := loaded.Search(query, 10)

	if len(origResults) != len(loadedResults) {
		t.Fatalf("result count mismatch: got %d, want %d", len(loadedResults), len(origResults))
	}
	for i := range origResults {
		if origResults[i].ID != loadedResults[i].ID {
			t.Fatalf("result %d: ID mismatch: got %d, want %d", i, loadedResults[i].ID, origResults[i].ID)
		}
	}
}

func TestSaveLoadFlatEmpty(t *testing.T) {
	idx := NewFlatIndex(8)
	var buf bytes.Buffer
	if err := Save(&buf, idx); err != nil {
		t.Fatalf("Save: %v", err)
	}
	loaded, err := LoadFlat(&buf)
	if err != nil {
		t.Fatalf("LoadFlat: %v", err)
	}
	if loaded.Len() != 0 || loaded.Dims() != 8 {
		t.Fatalf("unexpected: len=%d dims=%d", loaded.Len(), loaded.Dims())
	}
}

func TestLoadWrongType(t *testing.T) {
	idx := NewFlatIndex(4)
	idx.Add(1, []float32{1, 0, 0, 0})

	var buf bytes.Buffer
	if err := Save(&buf, idx); err != nil {
		t.Fatalf("Save: %v", err)
	}

	_, err := LoadHNSW(&buf)
	if err == nil {
		t.Fatal("expected error loading flat index as HNSW")
	}
}
