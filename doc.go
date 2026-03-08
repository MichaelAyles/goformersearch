// Package goformersearch provides pure-Go vector similarity search.
//
// It supports brute-force and HNSW (Hierarchical Navigable Small World)
// algorithms for nearest-neighbour lookups. The library requires no CGO
// and has zero native dependencies.
//
// # Quick start
//
//	// Build a brute-force index (exact results).
//	idx := goformersearch.NewFlatIndex(384)
//	idx.Add(1, embedding)
//
//	// Or an HNSW index (approximate, much faster at scale).
//	idx := goformersearch.NewHNSWIndex(384)
//	idx.Add(1, embedding)
//
//	// Query the k nearest neighbours.
//	results := idx.Search(query, 10)
//
// # Key types
//
//   - [Index]: interface satisfied by all search backends.
//   - [FlatIndex]: exact nearest-neighbour search via exhaustive comparison.
//   - [HNSWIndex]: approximate nearest-neighbour search using a navigable small-world graph.
//   - [Result]: a search result containing the vector ID and similarity score.
package goformersearch
