package goformersearch

import (
	"fmt"
	"math/rand"
	"os"
	"testing"
)

func TestRecallAtScale(t *testing.T) {
	if os.Getenv("GOFORMERSEARCH_RECALL") == "" {
		t.Skip("skipping: set GOFORMERSEARCH_RECALL=1 to run")
	}

	const (
		dims    = 384
		queries = 100
		k       = 10
	)

	sizes := []int{1_000, 5_000, 10_000, 25_000, 50_000}
	efValues := []int{50, 100, 200}

	for _, n := range sizes {
		rng := rand.New(rand.NewSource(42))

		vecs := make([][]float32, n)
		for i := range vecs {
			vecs[i] = randomNormalisedVec(rng, dims)
		}

		flat := NewFlatIndex(dims)
		for i, v := range vecs {
			flat.Add(uint64(i), v)
		}

		hnsw := NewHNSWIndex(dims, WithM(16), WithEfConstruction(200))
		for i, v := range vecs {
			hnsw.Add(uint64(i), v)
		}

		queryVecs := make([][]float32, queries)
		for i := range queryVecs {
			queryVecs[i] = randomNormalisedVec(rng, dims)
		}

		for _, ef := range efValues {
			hnsw.SetEfSearch(ef)

			var totalRecall float64
			for _, q := range queryVecs {
				flatResults := flat.Search(q, k)
				hnswResults := hnsw.Search(q, k)

				trueSet := make(map[uint64]bool, k)
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
			t.Logf("n=%s ef=%d: recall@%d = %.4f",
				fmtSize(n), ef, k, meanRecall)
		}
	}
}

func TestRecallPareto(t *testing.T) {
	if os.Getenv("GOFORMERSEARCH_RECALL") == "" {
		t.Skip("skipping: set GOFORMERSEARCH_RECALL=1 to run")
	}

	const (
		dims    = 384
		n       = 50_000
		queries = 100
		k       = 10
	)

	rng := rand.New(rand.NewSource(42))

	vecs := make([][]float32, n)
	for i := range vecs {
		vecs[i] = randomNormalisedVec(rng, dims)
	}

	flat := NewFlatIndex(dims)
	for i, v := range vecs {
		flat.Add(uint64(i), v)
	}

	hnsw := NewHNSWIndex(dims, WithM(16), WithEfConstruction(200))
	for i, v := range vecs {
		hnsw.Add(uint64(i), v)
	}

	queryVecs := make([][]float32, queries)
	for i := range queryVecs {
		queryVecs[i] = randomNormalisedVec(rng, dims)
	}

	efValues := []int{10, 25, 50, 100, 200, 500}
	for _, ef := range efValues {
		hnsw.SetEfSearch(ef)

		var totalRecall float64
		for _, q := range queryVecs {
			flatResults := flat.Search(q, k)
			hnswResults := hnsw.Search(q, k)

			trueSet := make(map[uint64]bool, k)
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
		fmt.Printf("PARETO ef=%d recall=%.4f\n", ef, meanRecall)
		t.Logf("ef=%d: recall@%d = %.4f", ef, k, meanRecall)
	}
}
