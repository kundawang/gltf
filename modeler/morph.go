package modeler

import (
	"errors"
	"fmt"
	"math"

	"github.com/qmuntal/gltf"
)

// MorphGeometry is the result of evaluating the morph targets of a primitive.
// Both slices are independent from the data stored in the document:
// modifying them does not alter the underlying glTF buffers.
type MorphGeometry struct {
	// Position holds the morphed POSITION attribute, one [3]float32 per vertex.
	Position [][3]float32
	// Normal holds the morphed NORMAL attribute, one normalized [3]float32
	// per vertex. It is nil if the primitive has no NORMAL attribute.
	Normal [][3]float32
}

// ReadMorphGeometry evaluates the POSITION and NORMAL morph targets of
// the primitive of node and returns the resulting geometry as independent slices.
//
// The morph weights are node.Weights when the node defines any, otherwise
// the default weights of the mesh referenced by the node are used, and when
// neither of them defines weights every target is evaluated with a zero weight
// (the glTF default). An explicitly provided, non-empty weight list must have
// exactly one entry per morph target.
//
// The displacement of every target is added as POSITION' = POSITION +
// Σ(weight_i * POSITION_i) and, when the primitive has a NORMAL attribute,
// NORMAL' = normalize(NORMAL + Σ(weight_i * NORMAL_i)). Target attributes that
// are not present in a target leave the base value untouched and TANGENT
// targets, if any, are ignored.
//
// ReadMorphGeometry supports sparse, non-interleaved and interleaved accessors
// and never modifies the document.
//
// It is safe to use even with malformed documents. It returns an error instead
// of panicking when a node, mesh, primitive or accessor index is invalid, when
// the number of weights does not match the number of morph targets, when a
// weight is not finite, when a target has a different vertex count than the
// base primitive, or when a morphed normal has zero length and cannot be
// normalized.
func ReadMorphGeometry(doc *gltf.Document, node *gltf.Node, primitive *gltf.Primitive) (*MorphGeometry, error) {
	if node == nil {
		return nil, errors.New("gltf: node is nil")
	}
	if node.Mesh == nil {
		return nil, errors.New("gltf: node does not reference a mesh")
	}
	if len(doc.Meshes) <= *node.Mesh {
		return nil, fmt.Errorf("gltf: mesh index %d overflows", *node.Mesh)
	}
	mesh := doc.Meshes[*node.Mesh]
	if mesh == nil {
		return nil, fmt.Errorf("gltf: mesh index %d is nil", *node.Mesh)
	}

	belongs := false
	for _, p := range mesh.Primitives {
		if p == primitive {
			belongs = true
			break
		}
	}
	if !belongs {
		return nil, errors.New("gltf: primitive does not belong to the node's mesh")
	}

	weights := node.Weights
	if len(weights) == 0 {
		weights = mesh.Weights
	}
	// When neither the node nor the mesh defines weights, the glTF spec
	// assigns a zero weight to every target.
	if len(weights) == 0 && len(primitive.Targets) > 0 {
		weights = make([]float64, len(primitive.Targets))
	}
	if len(weights) != len(primitive.Targets) {
		return nil, fmt.Errorf(
			"gltf: morph weights count %d does not match targets count %d",
			len(weights), len(primitive.Targets))
	}
	for i, w := range weights {
		if math.IsNaN(w) || math.IsInf(w, 0) {
			return nil, fmt.Errorf("gltf: morph weight %d is not finite: %v", i, w)
		}
	}

	posIndex, ok := primitive.Attributes[gltf.POSITION]
	if !ok {
		return nil, errors.New("gltf: primitive has no POSITION attribute")
	}
	position, err := readMorphAttribute(doc, posIndex)
	if err != nil {
		return nil, fmt.Errorf("POSITION: %w", err)
	}
	count := len(position)

	var normal [][3]float32
	normalIndex, hasNormal := primitive.Attributes[gltf.NORMAL]
	if hasNormal {
		normal, err = readMorphAttribute(doc, normalIndex)
		if err != nil {
			return nil, fmt.Errorf("NORMAL: %w", err)
		}
		if len(normal) != count {
			return nil, fmt.Errorf(
				"gltf: NORMAL has %d vertices, want %d", len(normal), count)
		}
	}

	morphedPosition := make([][3]float32, count)
	copy(morphedPosition, position)
	morphedNormal := normal
	if hasNormal {
		morphedNormal = make([][3]float32, count)
		copy(morphedNormal, normal)
	}

	for targetIndex, target := range primitive.Targets {
		weight := float32(weights[targetIndex])
		if index, ok := target[gltf.POSITION]; ok {
			displacement, err := readMorphAttribute(doc, index)
			if err != nil {
				return nil, fmt.Errorf("target %d POSITION: %w", targetIndex, err)
			}
			if len(displacement) != count {
				return nil, fmt.Errorf(
					"gltf: target %d POSITION has %d vertices, want %d",
					targetIndex, len(displacement), count)
			}
			for i, d := range displacement {
				morphedPosition[i][0] += weight * d[0]
				morphedPosition[i][1] += weight * d[1]
				morphedPosition[i][2] += weight * d[2]
			}
		}

		if index, ok := target[gltf.NORMAL]; ok {
			if !hasNormal {
				return nil, fmt.Errorf(
					"gltf: target %d has NORMAL but primitive has no base NORMAL", targetIndex)
			}
			displacement, err := readMorphAttribute(doc, index)
			if err != nil {
				return nil, fmt.Errorf("target %d NORMAL: %w", targetIndex, err)
			}
			if len(displacement) != count {
				return nil, fmt.Errorf(
					"gltf: target %d NORMAL has %d vertices, want %d",
					targetIndex, len(displacement), count)
			}
			for i, d := range displacement {
				morphedNormal[i][0] += weight * d[0]
				morphedNormal[i][1] += weight * d[1]
				morphedNormal[i][2] += weight * d[2]
			}
		}
	}

	if hasNormal {
		for i, n := range morphedNormal {
			length := float32(math.Sqrt(float64(n[0]*n[0] + n[1]*n[1] + n[2]*n[2])))
			if length == 0 {
				return nil, fmt.Errorf("gltf: morphed normal %d has zero length", i)
			}
			morphedNormal[i] = [3]float32{n[0] / length, n[1] / length, n[2] / length}
		}
	}

	return &MorphGeometry{Position: morphedPosition, Normal: morphedNormal}, nil
}

// readMorphAttribute reads a VEC3 FLOAT accessor by its index, including any
// sparse or interleaved layout.
func readMorphAttribute(doc *gltf.Document, index int) ([][3]float32, error) {
	if index < 0 || len(doc.Accessors) <= index {
		return nil, fmt.Errorf("gltf: accessor index %d overflows", index)
	}
	acr := doc.Accessors[index]
	if acr == nil {
		return nil, fmt.Errorf("gltf: accessor index %d is nil", index)
	}
	return ReadPosition(doc, acr, nil)
}
