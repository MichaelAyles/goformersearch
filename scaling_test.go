package goformersearch

import (
	"fmt"
	"math/rand"
	"testing"
)

func fmtSize(n int) string {
	switch {
	case n >= 1000 && n%1000 == 0:
		return fmt.Sprintf("%dk", n/1000)
	default:
		return fmt.Sprintf("%d", n)
	}
}

var scalingSizes = []int{100, 250, 500, 1_000, 2_500, 5_000, 10_000, 25_000, 50_000, 100_000}

func BenchmarkScaling(b *testing.B) {
	for _, n := range scalingSizes {
		n := n
		tag := fmtSize(n)

		b.Run("Flat/n="+tag, func(b *testing.B) {
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
		})

		for _, ef := range []int{50, 100, 200} {
			ef := ef
			b.Run(fmt.Sprintf("HNSW_ef%d/n=%s", ef, tag), func(b *testing.B) {
				rng := rand.New(rand.NewSource(1))
				idx := NewHNSWIndex(384, WithM(16), WithEfConstruction(200))
				for i := 0; i < n; i++ {
					idx.Add(uint64(i), randomNormalisedVec(rng, 384))
				}
				idx.SetEfSearch(ef)
				query := randomNormalisedVec(rng, 384)
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					idx.Search(query, 10)
				}
			})
		}
	}
}

func BenchmarkBuildTime(b *testing.B) {
	for _, n := range scalingSizes {
		n := n
		tag := fmtSize(n)

		b.Run("Flat/n="+tag, func(b *testing.B) {
			rng := rand.New(rand.NewSource(1))
			vecs := make([][]float32, n)
			for i := range vecs {
				vecs[i] = randomNormalisedVec(rng, 384)
			}
			b.ResetTimer()
			for iter := 0; iter < b.N; iter++ {
				idx := NewFlatIndex(384)
				for i, v := range vecs {
					idx.Add(uint64(i), v)
				}
			}
		})

		b.Run("HNSW/n="+tag, func(b *testing.B) {
			rng := rand.New(rand.NewSource(1))
			vecs := make([][]float32, n)
			for i := range vecs {
				vecs[i] = randomNormalisedVec(rng, 384)
			}
			b.ResetTimer()
			for iter := 0; iter < b.N; iter++ {
				idx := NewHNSWIndex(384, WithM(16), WithEfConstruction(200))
				for i, v := range vecs {
					idx.Add(uint64(i), v)
				}
			}
		})
	}
}
