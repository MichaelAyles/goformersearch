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

func benchFlatSearch(b *testing.B, n int) {
	rng := rand.New(rand.NewSource(1))
	idx := NewFlatIndex(384)
	for i := 0; i < n; i++ {
		idx.Add(uint64(i), randomNormalisedVec(rng, 384))
	}
	query := randomNormalisedVec(rng, 384)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx.Search(query, 10)
	}
}

func BenchmarkFlatSearch_1k(b *testing.B)  { benchFlatSearch(b, 1_000) }
func BenchmarkFlatSearch_10k(b *testing.B) { benchFlatSearch(b, 10_000) }
func BenchmarkFlatSearch_50k(b *testing.B) { benchFlatSearch(b, 50_000) }

func benchHNSWAdd(b *testing.B, n int) {
	rng := rand.New(rand.NewSource(1))
	vecs := make([][]float32, n)
	for i := range vecs {
		vecs[i] = randomNormalisedVec(rng, 384)
	}
	b.ResetTimer()
	idx := NewHNSWIndex(384, WithM(16), WithEfConstruction(200))
	for i := 0; i < n; i++ {
		idx.Add(uint64(i), vecs[i])
	}
}

func BenchmarkHNSWAdd_1k(b *testing.B)  { benchHNSWAdd(b, 1_000) }
func BenchmarkHNSWAdd_10k(b *testing.B) { benchHNSWAdd(b, 10_000) }

func benchHNSWSearch(b *testing.B, n int) {
	rng := rand.New(rand.NewSource(1))
	idx := NewHNSWIndex(384, WithM(16), WithEfConstruction(200))
	for i := 0; i < n; i++ {
		idx.Add(uint64(i), randomNormalisedVec(rng, 384))
	}
	idx.SetEfSearch(50)
	query := randomNormalisedVec(rng, 384)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx.Search(query, 10)
	}
}

func BenchmarkHNSWSearch_1k(b *testing.B)  { benchHNSWSearch(b, 1_000) }
func BenchmarkHNSWSearch_10k(b *testing.B) { benchHNSWSearch(b, 10_000) }
