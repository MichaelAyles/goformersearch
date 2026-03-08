# goformersearch

[![Go Reference](https://pkg.go.dev/badge/github.com/MichaelAyles/goformersearch.svg)](https://pkg.go.dev/github.com/MichaelAyles/goformersearch)
[![CI](https://github.com/MichaelAyles/goformersearch/actions/workflows/ci.yml/badge.svg)](https://github.com/MichaelAyles/goformersearch/actions/workflows/ci.yml)

Pure Go vector similarity search. Brute-force and HNSW. No CGO. No native dependencies.

```go
import "github.com/MichaelAyles/goformersearch"

// Build an index
index := goformersearch.NewHNSWIndex(384)
for _, doc := range documents {
    index.Add(doc.ID, doc.Embedding)
}

// Search
results := index.Search(queryVec, 10)
for _, r := range results {
    fmt.Printf("ID: %d, similarity: %.4f\n", r.ID, r.Similarity)
}
```

## What This Is

A Go library that indexes float32 vectors and returns the k nearest neighbours by cosine similarity. Two index types: brute-force (exact) and HNSW (approximate). Designed to pair with [goformer](https://github.com/MichaelAyles/goformer) but works with any source of float32 vectors.

Reference workload: 10k-50k document chunks at 384 dimensions, tens of queries per second on a single core.

## Why

FAISS is the standard for vector search, but using it from Go requires CGO. chromem-go bundles embedding, storage, and persistence into one package — more surface area than you need if you just want an index. The pure Go ANN libraries (goannoy, gann) use different algorithms or are archived.

goformersearch does one thing: vectors in, neighbours out. It ships both exact and approximate search behind the same interface, with zero dependencies.

## API

```go
// Index is the interface implemented by all index types.
type Index interface {
    Add(id uint64, vec []float32)
    Search(query []float32, k int) []Result
    Len() int
    Dims() int
}

// Result holds a search result.
type Result struct {
    ID         uint64
    Similarity float32
}

// Brute-force: exact results, O(n) per query.
func NewFlatIndex(dims int) *FlatIndex

// HNSW: approximate results, O(log n) per query.
func NewHNSWIndex(dims int, opts ...HNSWOption) *HNSWIndex

// HNSW tuning options.
func WithM(m int) HNSWOption              // connections per node (default 16)
func WithEfConstruction(ef int) HNSWOption // build-time search width (default 200)
func WithEfSearch(ef int) HNSWOption       // query-time search width (default 50)

// Serialisation.
func Save(w io.Writer, idx Index) error
func LoadFlat(r io.Reader) (*FlatIndex, error)
func LoadHNSW(r io.Reader) (*HNSWIndex, error)
```

The `Index` interface is the key design decision. Code that doesn't care about exact vs approximate uses `Index`. Code that needs to tune HNSW parameters uses `*HNSWIndex` directly.

## Usage

### Flat index (exact search)

```go
index := goformersearch.NewFlatIndex(384)

// Add vectors
for id, vec := range vectors {
    index.Add(uint64(id), vec)
}

// Search — returns exact k nearest neighbours
results := index.Search(query, 10)
```

### HNSW index (approximate search)

```go
index := goformersearch.NewHNSWIndex(384,
    goformersearch.WithM(16),
    goformersearch.WithEfConstruction(200),
)

// Add vectors
for id, vec := range vectors {
    index.Add(uint64(id), vec)
}

// Tune search quality vs speed
index.SetEfSearch(100)

// Search — returns approximate k nearest neighbours
results := index.Search(query, 10)
```

### With goformer

```go
model, _ := goformer.Load("./bge-small-en-v1.5")

index := goformersearch.NewHNSWIndex(model.Dims())
for _, doc := range documents {
    vec, _ := model.Embed(doc.Text)
    index.Add(doc.ID, vec)
}

queryVec, _ := model.Embed("DMA channel configuration")
results := index.Search(queryVec, 10)
```

### Save and load

```go
// Save
f, _ := os.Create("index.bin")
goformersearch.Save(f, index)
f.Close()

// Load
f, _ = os.Open("index.bin")
index, _ := goformersearch.LoadHNSW(f)
f.Close()
```

## Performance Targets

At 384 dimensions (BGE-small-en-v1.5 output):

| Operation | Target |
|---|---|
| Flat search, 10k vectors | < 1ms |
| Flat search, 50k vectors | < 5ms |
| HNSW search, 50k vectors (ef=50) | < 0.5ms |
| HNSW build, 10k vectors | < 5s |
| HNSW build, 50k vectors | < 30s |

### HNSW Recall

| efSearch | Expected recall@10 |
|---|---|
| 50 | > 0.95 |
| 100 | > 0.98 |
| 200 | > 0.99 |

## Concurrency

Safe for concurrent reads (Search) once all writes (Add) are complete. Not safe for concurrent Add. This matches the expected pattern: build the index, then serve queries.

## Limitations

- **In-memory only.** No disk-backed indexes.
- **No deletion.** HNSW deletion is complex; rebuild the index instead.
- **No filtering.** This is a vector index, not a database.
- **Cosine similarity only.** Assumes L2-normalised input vectors.

## License

MIT
