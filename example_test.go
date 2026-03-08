package goformersearch_test

import (
	"bytes"
	"fmt"
	"math"

	"github.com/MichaelAyles/goformersearch"
)

func norm(v []float32) []float32 {
	var s float64
	for _, x := range v {
		s += float64(x) * float64(x)
	}
	s = math.Sqrt(s)
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = float32(float64(x) / s)
	}
	return out
}

func Example_flatSearch() {
	idx := goformersearch.NewFlatIndex(3)
	idx.Add(1, norm([]float32{1, 0, 0}))
	idx.Add(2, norm([]float32{0, 1, 0}))
	idx.Add(3, norm([]float32{1, 1, 0}))

	results := idx.Search(norm([]float32{1, 0, 0}), 2)
	for _, r := range results {
		fmt.Printf("ID=%d similarity=%.4f\n", r.ID, r.Similarity)
	}
	// Output:
	// ID=1 similarity=1.0000
	// ID=3 similarity=0.7071
}

func Example_hnswSearch() {
	idx := goformersearch.NewHNSWIndex(3,
		goformersearch.WithM(4),
		goformersearch.WithEfConstruction(50),
	)
	idx.Add(1, norm([]float32{1, 0, 0}))
	idx.Add(2, norm([]float32{0, 1, 0}))
	idx.Add(3, norm([]float32{1, 1, 0}))

	idx.SetEfSearch(50)
	results := idx.Search(norm([]float32{1, 0, 0}), 2)
	for _, r := range results {
		fmt.Printf("ID=%d similarity=%.4f\n", r.ID, r.Similarity)
	}
	// Output:
	// ID=1 similarity=1.0000
	// ID=3 similarity=0.7071
}

func Example_saveLoad() {
	idx := goformersearch.NewFlatIndex(3)
	idx.Add(1, norm([]float32{1, 0, 0}))
	idx.Add(2, norm([]float32{0, 1, 0}))

	var buf bytes.Buffer
	_ = goformersearch.Save(&buf, idx)

	loaded, _ := goformersearch.LoadFlat(&buf)
	fmt.Printf("Loaded %d vectors of %d dims\n", loaded.Len(), loaded.Dims())
	// Output:
	// Loaded 2 vectors of 3 dims
}
