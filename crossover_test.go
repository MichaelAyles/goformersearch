package goformersearch

import (
	"math/rand"
	"testing"
)

func BenchmarkCrossover(b *testing.B) {
	sizes := []int{100, 500, 1_000, 2_000, 5_000, 10_000}

	for _, n := range sizes {
		rng := rand.New(rand.NewSource(1))
		vecs := make([][]float32, n)
		for i := range vecs {
			vecs[i] = randomNormalisedVec(rng, 384)
		}
		query := randomNormalisedVec(rng, 384)

		b.Run("flat/"+itoa(n), func(b *testing.B) {
			idx := NewFlatIndex(384)
			for i, v := range vecs {
				idx.Add(uint64(i), v)
			}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				idx.Search(query, 10)
			}
		})

		b.Run("hnsw/"+itoa(n), func(b *testing.B) {
			idx := NewHNSWIndex(384, WithM(16), WithEfConstruction(200))
			for i, v := range vecs {
				idx.Add(uint64(i), v)
			}
			idx.SetEfSearch(50)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				idx.Search(query, 10)
			}
		})
	}
}

func itoa(n int) string {
	if n >= 1000 {
		return string(rune('0'+n/1000)) + "k"
	}
	// Simple int to string for small numbers.
	s := ""
	for n > 0 {
		s = string(rune('0'+n%10)) + s
		n /= 10
	}
	if s == "" {
		return "0"
	}
	return s
}
