package goformersearch

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
)

var (
	magic   = [4]byte{'G', 'V', 'S', 'C'}
	version = byte(1)
	order   = binary.LittleEndian
)

const (
	typeFlat byte = 0
	typeHNSW byte = 1
)

// Save writes the index to w in a binary format.
func Save(w io.Writer, idx Index) error {
	switch v := idx.(type) {
	case *FlatIndex:
		return saveFlat(w, v)
	case *HNSWIndex:
		return saveHNSW(w, v)
	default:
		return fmt.Errorf("goformersearch: unsupported index type %T", idx)
	}
}

// LoadFlat reads a FlatIndex from r.
func LoadFlat(r io.Reader) (*FlatIndex, error) {
	dims, count, err := readHeader(r, typeFlat)
	if err != nil {
		return nil, err
	}

	ids := make([]uint64, count)
	if err := binary.Read(r, order, ids); err != nil {
		return nil, fmt.Errorf("goformersearch: reading IDs: %w", err)
	}
	vectors := make([]float32, count*dims)
	if err := binary.Read(r, order, vectors); err != nil {
		return nil, fmt.Errorf("goformersearch: reading vectors: %w", err)
	}

	return &FlatIndex{dims: int(dims), vectors: vectors, ids: ids}, nil
}

// LoadHNSW reads an HNSWIndex from r.
func LoadHNSW(r io.Reader) (*HNSWIndex, error) {
	dims, count, err := readHeader(r, typeHNSW)
	if err != nil {
		return nil, err
	}

	var m, efConstruction, maxLevel, entryPoint uint32
	for _, p := range []*uint32{&m, &efConstruction, &maxLevel, &entryPoint} {
		if err := binary.Read(r, order, p); err != nil {
			return nil, fmt.Errorf("goformersearch: reading HNSW params: %w", err)
		}
	}

	nodes := make([]hnswNode, count)
	for i := uint32(0); i < count; i++ {
		var id uint64
		var level uint32
		if err := binary.Read(r, order, &id); err != nil {
			return nil, fmt.Errorf("goformersearch: reading node %d: %w", i, err)
		}
		if err := binary.Read(r, order, &level); err != nil {
			return nil, fmt.Errorf("goformersearch: reading node %d level: %w", i, err)
		}
		vec := make([]float32, dims)
		if err := binary.Read(r, order, vec); err != nil {
			return nil, fmt.Errorf("goformersearch: reading node %d vector: %w", i, err)
		}

		friends := make([][]uint32, level+1)
		for l := uint32(0); l <= level; l++ {
			var numFriends uint32
			if err := binary.Read(r, order, &numFriends); err != nil {
				return nil, fmt.Errorf("goformersearch: reading node %d layer %d: %w", i, l, err)
			}
			if numFriends > 0 {
				f := make([]uint32, numFriends)
				if err := binary.Read(r, order, f); err != nil {
					return nil, fmt.Errorf("goformersearch: reading node %d friends: %w", i, err)
				}
				friends[l] = f
			}
		}

		nodes[i] = hnswNode{id: id, vec: vec, level: int(level), friends: friends}
	}

	cfg := hnswConfig{
		m:              int(m),
		mMax0:          int(2 * m),
		efConstruction: int(efConstruction),
		efSearch:       50,
	}

	return &HNSWIndex{
		dims:       int(dims),
		cfg:        cfg,
		nodes:      nodes,
		entryPoint: int(entryPoint),
		maxLevel:   int(maxLevel),
		mL:         1.0 / math.Log(float64(m)),
	}, nil
}

func saveFlat(w io.Writer, f *FlatIndex) error {
	if err := writeHeader(w, typeFlat, uint32(f.dims), uint32(f.Len())); err != nil {
		return err
	}
	if err := binary.Write(w, order, f.ids); err != nil {
		return fmt.Errorf("goformersearch: writing IDs: %w", err)
	}
	if err := binary.Write(w, order, f.vectors); err != nil {
		return fmt.Errorf("goformersearch: writing vectors: %w", err)
	}
	return nil
}

func saveHNSW(w io.Writer, h *HNSWIndex) error {
	if err := writeHeader(w, typeHNSW, uint32(h.dims), uint32(h.Len())); err != nil {
		return err
	}
	for _, v := range []uint32{uint32(h.cfg.m), uint32(h.cfg.efConstruction), uint32(h.maxLevel), uint32(h.entryPoint)} {
		if err := binary.Write(w, order, v); err != nil {
			return fmt.Errorf("goformersearch: writing HNSW params: %w", err)
		}
	}
	for i, node := range h.nodes {
		if err := binary.Write(w, order, node.id); err != nil {
			return fmt.Errorf("goformersearch: writing node %d: %w", i, err)
		}
		if err := binary.Write(w, order, uint32(node.level)); err != nil {
			return fmt.Errorf("goformersearch: writing node %d level: %w", i, err)
		}
		if err := binary.Write(w, order, node.vec); err != nil {
			return fmt.Errorf("goformersearch: writing node %d vector: %w", i, err)
		}
		for l := 0; l <= node.level; l++ {
			friends := node.friends[l]
			if err := binary.Write(w, order, uint32(len(friends))); err != nil {
				return fmt.Errorf("goformersearch: writing node %d layer %d: %w", i, l, err)
			}
			if len(friends) > 0 {
				if err := binary.Write(w, order, friends); err != nil {
					return fmt.Errorf("goformersearch: writing node %d friends: %w", i, err)
				}
			}
		}
	}
	return nil
}

func writeHeader(w io.Writer, idxType byte, dims, count uint32) error {
	if _, err := w.Write(magic[:]); err != nil {
		return fmt.Errorf("goformersearch: writing magic: %w", err)
	}
	if _, err := w.Write([]byte{version, idxType}); err != nil {
		return fmt.Errorf("goformersearch: writing version: %w", err)
	}
	if err := binary.Write(w, order, dims); err != nil {
		return fmt.Errorf("goformersearch: writing dims: %w", err)
	}
	if err := binary.Write(w, order, count); err != nil {
		return fmt.Errorf("goformersearch: writing count: %w", err)
	}
	return nil
}

func readHeader(r io.Reader, expectedType byte) (dims, count uint32, err error) {
	var hdr [6]byte
	if _, err = io.ReadFull(r, hdr[:]); err != nil {
		return 0, 0, fmt.Errorf("goformersearch: reading header: %w", err)
	}
	if hdr[0] != magic[0] || hdr[1] != magic[1] || hdr[2] != magic[2] || hdr[3] != magic[3] {
		return 0, 0, fmt.Errorf("goformersearch: invalid magic number")
	}
	if hdr[4] != version {
		return 0, 0, fmt.Errorf("goformersearch: unsupported version %d", hdr[4])
	}
	if hdr[5] != expectedType {
		return 0, 0, fmt.Errorf("goformersearch: expected index type %d, got %d", expectedType, hdr[5])
	}
	if err = binary.Read(r, order, &dims); err != nil {
		return 0, 0, fmt.Errorf("goformersearch: reading dims: %w", err)
	}
	if err = binary.Read(r, order, &count); err != nil {
		return 0, 0, fmt.Errorf("goformersearch: reading count: %w", err)
	}
	return dims, count, nil
}
