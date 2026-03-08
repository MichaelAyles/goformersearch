package goformersearch

import (
	"math"
	"testing"
)

func TestDotProduct(t *testing.T) {
	a := []float32{1, 2, 3, 4}
	b := []float32{5, 6, 7, 8}
	got := DotProduct(a, b)
	want := float32(1*5 + 2*6 + 3*7 + 4*8) // 70
	if math.Abs(float64(got-want)) > 1e-6 {
		t.Fatalf("DotProduct = %f, want %f", got, want)
	}
}

func TestDotProductLarge(t *testing.T) {
	// Test with 384 dims to exercise the unrolled loop.
	a := make([]float32, 384)
	b := make([]float32, 384)
	var want float32
	for i := range a {
		a[i] = float32(i) * 0.01
		b[i] = float32(384-i) * 0.01
		want += a[i] * b[i]
	}
	got := DotProduct(a, b)
	if math.Abs(float64(got-want)) > 0.1 {
		t.Fatalf("DotProduct = %f, want %f", got, want)
	}
}

func TestCosineSimilarity(t *testing.T) {
	// Same direction → 1.0
	a := []float32{1, 2, 3}
	got := CosineSimilarity(a, a)
	if math.Abs(float64(got)-1.0) > 1e-6 {
		t.Fatalf("CosineSimilarity(a, a) = %f, want 1.0", got)
	}

	// Opposite direction → -1.0
	b := []float32{-1, -2, -3}
	got = CosineSimilarity(a, b)
	if math.Abs(float64(got)+1.0) > 1e-6 {
		t.Fatalf("CosineSimilarity(a, -a) = %f, want -1.0", got)
	}

	// Orthogonal → 0.0
	c := []float32{1, 0, 0}
	d := []float32{0, 1, 0}
	got = CosineSimilarity(c, d)
	if math.Abs(float64(got)) > 1e-6 {
		t.Fatalf("CosineSimilarity(orthogonal) = %f, want 0.0", got)
	}
}

func TestCosineSimilarityZero(t *testing.T) {
	a := []float32{0, 0, 0}
	b := []float32{1, 2, 3}
	got := CosineSimilarity(a, b)
	if got != 0 {
		t.Fatalf("CosineSimilarity(zero, b) = %f, want 0.0", got)
	}
}

func TestL2Distance(t *testing.T) {
	a := []float32{1, 2, 3}
	b := []float32{4, 5, 6}
	got := L2Distance(a, b)
	want := float32(27) // 9+9+9
	if math.Abs(float64(got-want)) > 1e-6 {
		t.Fatalf("L2Distance = %f, want %f", got, want)
	}
}

func TestL2DistanceSelf(t *testing.T) {
	a := []float32{1, 2, 3}
	got := L2Distance(a, a)
	if got != 0 {
		t.Fatalf("L2Distance(a, a) = %f, want 0.0", got)
	}
}
