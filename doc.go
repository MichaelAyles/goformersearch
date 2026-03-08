// Package goformersearch provides pure-Go vector similarity search.
//
// It supports brute-force and HNSW (Hierarchical Navigable Small World)
// algorithms for approximate nearest-neighbour lookups. The library requires
// no CGO and has zero native dependencies, making it easy to cross-compile
// and deploy anywhere Go runs.
//
// # Quick start
//
//	// Build an index from a set of vectors.
//	idx := goformersearch.NewBruteForce(vectors, goformersearch.Cosine)
//
//	// Query the k nearest neighbours.
//	results := idx.Search(query, k)
//
// # Key types
//
//   - Index: interface satisfied by all search backends.
//   - BruteForce: exact nearest-neighbour search via exhaustive comparison.
//   - HNSW: approximate nearest-neighbour search using a navigable small-world graph.
//   - DistanceFunc: pluggable distance/similarity metric (Cosine, Euclidean, DotProduct).
package goformersearch
