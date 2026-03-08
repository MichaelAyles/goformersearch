package goformersearch

// hnswNode represents a single node in the HNSW graph.
type hnswNode struct {
	id      uint64
	vec     []float32  // copy of the vector
	level   int        // max layer this node appears in
	friends [][]uint32 // friends[layer] = neighbour node indices
}

// hnswConfig holds HNSW construction and search parameters.
type hnswConfig struct {
	m              int // max connections per node per layer
	mMax0          int // max connections on layer 0 (2*M)
	efConstruction int // build-time search width
	efSearch       int // query-time search width
}

func defaultConfig() hnswConfig {
	return hnswConfig{
		m:              16,
		mMax0:          32,
		efConstruction: 200,
		efSearch:       50,
	}
}

// HNSWOption configures HNSW index parameters.
type HNSWOption func(*hnswConfig)

// WithM sets the maximum number of connections per node per layer.
// Default 16. Higher values improve recall at the cost of memory and build time.
func WithM(m int) HNSWOption {
	return func(c *hnswConfig) {
		c.m = m
		c.mMax0 = 2 * m
	}
}

// WithEfConstruction sets the build-time search width. Default 200.
// Higher values produce a better graph at the cost of slower insertion.
func WithEfConstruction(ef int) HNSWOption {
	return func(c *hnswConfig) {
		c.efConstruction = ef
	}
}

// WithEfSearch sets the query-time search width. Default 50.
// Higher values give better recall at the cost of latency.
func WithEfSearch(ef int) HNSWOption {
	return func(c *hnswConfig) {
		c.efSearch = ef
	}
}
