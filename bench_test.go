package goformersearch

import (
	"math/rand"
	"testing"
)

func BenchmarkDotProduct_384(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	a := randomNormalisedVec(rng, 384)
	v := randomNormalisedVec(rng, 384)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		DotProduct(a, v)
	}
}

func BenchmarkCosineSimilarity_384(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	a := randomNormalisedVec(rng, 384)
	v := randomNormalisedVec(rng, 384)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		CosineSimilarity(a, v)
	}
}

func BenchmarkFlatAdd(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	vecs := make([][]float32, b.N)
	for i := range vecs {
		vecs[i] = randomNormalisedVec(rng, 384)
	}
	idx := NewFlatIndex(384)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx.Add(uint64(i), vecs[i])
	}
}

func BenchmarkHNSWAdd(b *testing.B) {
	// Measure single-vector insertion cost into a 10k index.
	const base = 10_000
	rng := rand.New(rand.NewSource(1))
	idx := NewHNSWIndex(384, WithM(16), WithEfConstruction(200))
	for i := 0; i < base; i++ {
		idx.Add(uint64(i), randomNormalisedVec(rng, 384))
	}
	vecs := make([][]float32, b.N)
	for i := range vecs {
		vecs[i] = randomNormalisedVec(rng, 384)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx.Add(uint64(base+i), vecs[i])
	}
}

func BenchmarkFlatSearch_10k(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	idx := NewFlatIndex(384)
	for i := 0; i < 10_000; i++ {
		idx.Add(uint64(i), randomNormalisedVec(rng, 384))
	}
	query := randomNormalisedVec(rng, 384)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx.Search(query, 10)
	}
}

func BenchmarkHNSWSearch_10k(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	idx := NewHNSWIndex(384, WithM(16), WithEfConstruction(200))
	for i := 0; i < 10_000; i++ {
		idx.Add(uint64(i), randomNormalisedVec(rng, 384))
	}
	idx.SetEfSearch(50)
	query := randomNormalisedVec(rng, 384)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx.Search(query, 10)
	}
}
