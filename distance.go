package goformersearch

import "math"

// CosineSimilarity returns the cosine similarity between two vectors.
// For L2-normalised vectors (e.g. goformer output), this equals the dot product.
func CosineSimilarity(a, b []float32) float32 {
	var dot, normA, normB float32
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	denom := float32(math.Sqrt(float64(normA)) * math.Sqrt(float64(normB)))
	if denom == 0 {
		return 0
	}
	return dot / denom
}

// DotProduct returns the dot product of two vectors.
func DotProduct(a, b []float32) float32 {
	var sum float32
	i := 0
	// Process 8 elements at a time.
	for ; i+7 < len(a); i += 8 {
		sum += a[i]*b[i] +
			a[i+1]*b[i+1] +
			a[i+2]*b[i+2] +
			a[i+3]*b[i+3] +
			a[i+4]*b[i+4] +
			a[i+5]*b[i+5] +
			a[i+6]*b[i+6] +
			a[i+7]*b[i+7]
	}
	for ; i < len(a); i++ {
		sum += a[i] * b[i]
	}
	return sum
}

// L2Distance returns the squared L2 (Euclidean) distance between two vectors.
func L2Distance(a, b []float32) float32 {
	var sum float32
	for i := range a {
		d := a[i] - b[i]
		sum += d * d
	}
	return sum
}
