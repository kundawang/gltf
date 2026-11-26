package modeler_test

import (
	"encoding/binary"
	"math"
	"reflect"
	"testing"

	"github.com/qmuntal/gltf"
	"github.com/qmuntal/gltf/modeler"
)

// morphTargetData is the raw VEC3 float data of one morph target.
// A nil slice means the target does not provide that attribute, in which
// case the base value must be preserved.
type morphTargetData struct {
	Position [][3]float32
	Normal   [][3]float32
}

// newMorphDoc builds a one-primitive, one-node document whose base geometry
// and morph targets are stored in plain, non-interleaved float32 accessors.
// meshWeights is used as the mesh default weights; nodeWeights, when not nil,
// is set as the node weights and takes precedence.
func newMorphDoc(meshWeights, nodeWeights []float64,
	basePosition, baseNormal [][3]float32,
	targets ...morphTargetData,
) (doc *gltf.Document, node *gltf.Node, primitive *gltf.Primitive) {
	doc = gltf.NewDocument()
	attrs := gltf.PrimitiveAttributes{}
	attrs[gltf.POSITION] = modeler.WritePosition(doc, basePosition)
	if baseNormal != nil {
		attrs[gltf.NORMAL] = modeler.WriteAccessor(doc, gltf.TargetArrayBuffer, baseNormal)
	}

	targetAttrs := make([]gltf.PrimitiveAttributes, len(targets))
	for i, target := range targets {
		targetAttrs[i] = gltf.PrimitiveAttributes{}
		if target.Position != nil {
			targetAttrs[i][gltf.POSITION] = modeler.WriteAccessor(doc, gltf.TargetArrayBuffer, target.Position)
		}
		if target.Normal != nil {
			targetAttrs[i][gltf.NORMAL] = modeler.WriteAccessor(doc, gltf.TargetArrayBuffer, target.Normal)
		}
	}

	primitive = &gltf.Primitive{Attributes: attrs, Targets: targetAttrs}
	mesh := &gltf.Mesh{Primitives: []*gltf.Primitive{primitive}}
	if meshWeights != nil {
		mesh.Weights = meshWeights
	}
	doc.Meshes = append(doc.Meshes, mesh)
	node = &gltf.Node{Mesh: gltf.Index(0)}
	if nodeWeights != nil {
		node.Weights = nodeWeights
	}
	doc.Nodes = append(doc.Nodes, node)
	return doc, node, primitive
}

func approxVec3(got, want [3]float32) bool {
	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > 1e-6 {
			return false
		}
	}
	return true
}

func TestReadMorphGeometry_Position(t *testing.T) {
	basePosition := [][3]float32{{1, 0, 0}, {0, 0, 0}}
	targets := []morphTargetData{
		{Position: [][3]float32{{2, 0, 0}, {2, 0, 0}}},
		{Position: [][3]float32{{0, 4, 0}, {0, 4, 0}}},
	}

	t.Run("mesh default weights", func(t *testing.T) {
		doc, node, primitive := newMorphDoc(
			[]float64{0.25, 0.5}, nil, basePosition, nil, targets...)
		got, err := modeler.ReadMorphGeometry(doc, node, primitive)
		if err != nil {
			t.Fatal(err)
		}
		want := [][3]float32{{1.5, 2, 0}, {0.5, 2, 0}}
		if !reflect.DeepEqual(got.Position, want) {
			t.Errorf("Position = %v, want %v", got.Position, want)
		}
		if got.Normal != nil {
			t.Errorf("Normal = %v, want nil", got.Normal)
		}
	})

	t.Run("node weights override mesh weights", func(t *testing.T) {
		doc, node, primitive := newMorphDoc(
			[]float64{1, 1}, []float64{0.25, 0.5}, basePosition, nil, targets...)
		got, err := modeler.ReadMorphGeometry(doc, node, primitive)
		if err != nil {
			t.Fatal(err)
		}
		want := [][3]float32{{1.5, 2, 0}, {0.5, 2, 0}}
		if !reflect.DeepEqual(got.Position, want) {
			t.Errorf("Position = %v, want %v", got.Position, want)
		}
	})

	t.Run("zero weights leave the base untouched", func(t *testing.T) {
		doc, node, primitive := newMorphDoc(nil, nil, basePosition, nil, targets...)
		got, err := modeler.ReadMorphGeometry(doc, node, primitive)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got.Position, basePosition) {
			t.Errorf("Position = %v, want %v", got.Position, basePosition)
		}
	})
}

func TestReadMorphGeometry_Normal(t *testing.T) {
	basePosition := [][3]float32{{0, 0, 0}, {0, 0, 0}}
	baseNormal := [][3]float32{{1, 0, 0}, {0, 0, 1}}
	targets := []morphTargetData{
		{Position: [][3]float32{{2, 0, 0}, {2, 0, 0}}},
		{Normal: [][3]float32{{0, 1, 0}, {0, 1, 0}}},
	}

	doc, node, primitive := newMorphDoc(
		[]float64{0.25, 0.5}, nil, basePosition, baseNormal, targets...)
	got, err := modeler.ReadMorphGeometry(doc, node, primitive)
	if err != nil {
		t.Fatal(err)
	}

	wantPosition := [][3]float32{{0.5, 0, 0}, {0.5, 0, 0}}
	if !reflect.DeepEqual(got.Position, wantPosition) {
		t.Errorf("Position = %v, want %v", got.Position, wantPosition)
	}

	// First normal: (1,0,0) + 0.5*(0,1,0) = (1,0.5,0), normalized.
	// Second normal: (0,0,1) + 0.5*(0,1,0) = (0,0.5,1), normalized.
	wantNormal := [][3]float32{
		{float32(1 / math.Hypot(1, 0.5)), float32(0.5 / math.Hypot(1, 0.5)), 0},
		{0, float32(0.5 / math.Hypot(1, 0.5)), float32(1 / math.Hypot(1, 0.5))},
	}
	if len(got.Normal) != len(wantNormal) {
		t.Fatalf("Normal len = %d, want %d", len(got.Normal), len(wantNormal))
	}
	for i := range wantNormal {
		if !approxVec3(got.Normal[i], wantNormal[i]) {
			t.Errorf("Normal[%d] = %v, want %v", i, got.Normal[i], wantNormal[i])
		}
		length := math.Sqrt(float64(got.Normal[i][0]*got.Normal[i][0] +
			got.Normal[i][1]*got.Normal[i][1] + got.Normal[i][2]*got.Normal[i][2]))
		if math.Abs(length-1) > 1e-6 {
			t.Errorf("Normal[%d] length = %v, want 1", i, length)
		}
	}
}

func TestReadMorphGeometry_MissingTargetAttributeKeepsBase(t *testing.T) {
	basePosition := [][3]float32{{1, 0, 0}, {1, 0, 0}}
	baseNormal := [][3]float32{{1, 0, 0}, {1, 0, 0}}
	targets := []morphTargetData{
		{Position: [][3]float32{{2, 0, 0}, {2, 0, 0}}},
		{Normal: [][3]float32{{0, 1, 0}, {0, 1, 0}}},
	}

	doc, node, primitive := newMorphDoc(
		[]float64{0.25, 0.5}, nil, basePosition, baseNormal, targets...)
	got, err := modeler.ReadMorphGeometry(doc, node, primitive)
	if err != nil {
		t.Fatal(err)
	}
	wantPosition := [][3]float32{{1.5, 0, 0}, {1.5, 0, 0}}
	if !reflect.DeepEqual(got.Position, wantPosition) {
		t.Errorf("Position = %v, want %v", got.Position, wantPosition)
	}
	wantNormal := [][3]float32{
		{float32(1 / math.Hypot(1, 0.5)), float32(0.5 / math.Hypot(1, 0.5)), 0},
		{float32(1 / math.Hypot(1, 0.5)), float32(0.5 / math.Hypot(1, 0.5)), 0},
	}
	for i := range wantNormal {
		if !approxVec3(got.Normal[i], wantNormal[i]) {
			t.Errorf("Normal[%d] = %v, want %v", i, got.Normal[i], wantNormal[i])
		}
	}
}

func TestReadMorphGeometry_SparseAccessor(t *testing.T) {
	doc := gltf.NewDocument()
	basePosition := [][3]float32{{1, 0, 0}, {7, 7, 7}}
	baseIndex := modeler.WritePosition(doc, basePosition)

	// Sparse-only target: vertex 0 is displaced to (2,0,0), vertex 1 keeps
	// the zero-initialized base value of an accessor without buffer view.
	indexData := make([]byte, 2)
	binary.LittleEndian.PutUint16(indexData, 0)
	valueData := float32Bytes(2, 0, 0)
	bufferData := append(append([]byte{}, indexData...), valueData...)
	doc.Buffers = append(doc.Buffers, &gltf.Buffer{Data: bufferData, ByteLength: len(bufferData)})
	bufferIndex := len(doc.Buffers) - 1
	doc.BufferViews = append(doc.BufferViews,
		&gltf.BufferView{Buffer: bufferIndex, ByteLength: 2},
		&gltf.BufferView{Buffer: bufferIndex, ByteOffset: 2, ByteLength: len(valueData)},
	)
	sparseTarget := &gltf.Accessor{
		ComponentType: gltf.ComponentFloat,
		Type:          gltf.AccessorVec3,
		Count:         2,
		Sparse: &gltf.Sparse{
			Count:   1,
			Indices: gltf.SparseIndices{BufferView: len(doc.BufferViews) - 2, ComponentType: gltf.ComponentUshort},
			Values:  gltf.SparseValues{BufferView: len(doc.BufferViews) - 1},
		},
	}
	doc.Accessors = append(doc.Accessors, sparseTarget)
	targetIndex := len(doc.Accessors) - 1

	primitive := &gltf.Primitive{
		Attributes: gltf.PrimitiveAttributes{gltf.POSITION: baseIndex},
		Targets:    []gltf.PrimitiveAttributes{{gltf.POSITION: targetIndex}},
	}
	doc.Meshes = append(doc.Meshes, &gltf.Mesh{
		Primitives: []*gltf.Primitive{primitive},
		Weights:    []float64{0.25},
	})
	node := &gltf.Node{Mesh: gltf.Index(0)}
	doc.Nodes = append(doc.Nodes, node)

	got, err := modeler.ReadMorphGeometry(doc, node, primitive)
	if err != nil {
		t.Fatal(err)
	}
	want := [][3]float32{{1.5, 0, 0}, {7, 7, 7}}
	if !reflect.DeepEqual(got.Position, want) {
		t.Errorf("Position = %v, want %v", got.Position, want)
	}
}

func TestReadMorphGeometry_InterleavedAccessor(t *testing.T) {
	doc := gltf.NewDocument()

	baseInterleaved, err := modeler.WriteAccessorsInterleaved(doc,
		[][3]float32{{1, 0, 0}, {1, 0, 0}}, // POSITION
		[][3]float32{{1, 0, 0}, {1, 0, 0}}, // NORMAL
	)
	if err != nil {
		t.Fatal(err)
	}
	targetInterleaved, err := modeler.WriteAccessorsInterleaved(doc,
		[][3]float32{{2, 0, 0}, {2, 0, 0}}, // POSITION displacement
		[][3]float32{{0, 1, 0}, {0, 1, 0}}, // NORMAL displacement
	)
	if err != nil {
		t.Fatal(err)
	}

	primitive := &gltf.Primitive{
		Attributes: gltf.PrimitiveAttributes{
			gltf.POSITION: baseInterleaved[0],
			gltf.NORMAL:   baseInterleaved[1],
		},
		Targets: []gltf.PrimitiveAttributes{{
			gltf.POSITION: targetInterleaved[0],
			gltf.NORMAL:   targetInterleaved[1],
		}},
	}
	doc.Meshes = append(doc.Meshes, &gltf.Mesh{
		Primitives: []*gltf.Primitive{primitive},
		Weights:    []float64{0.25},
	})
	node := &gltf.Node{Mesh: gltf.Index(0)}
	doc.Nodes = append(doc.Nodes, node)

	got, err := modeler.ReadMorphGeometry(doc, node, primitive)
	if err != nil {
		t.Fatal(err)
	}
	wantPosition := [][3]float32{{1.5, 0, 0}, {1.5, 0, 0}}
	if !reflect.DeepEqual(got.Position, wantPosition) {
		t.Errorf("Position = %v, want %v", got.Position, wantPosition)
	}
	wantNormal := [][3]float32{
		{float32(1 / math.Hypot(1, 0.25)), float32(0.25 / math.Hypot(1, 0.25)), 0},
		{float32(1 / math.Hypot(1, 0.25)), float32(0.25 / math.Hypot(1, 0.25)), 0},
	}
	for i := range wantNormal {
		if !approxVec3(got.Normal[i], wantNormal[i]) {
			t.Errorf("Normal[%d] = %v, want %v", i, got.Normal[i], wantNormal[i])
		}
	}
}

func TestReadMorphGeometry_DocumentIsNotModified(t *testing.T) {
	basePosition := [][3]float32{{1, 0, 0}, {1, 0, 0}}
	targets := []morphTargetData{
		{Position: [][3]float32{{2, 0, 0}, {2, 0, 0}}},
		{Position: [][3]float32{{0, 4, 0}, {0, 4, 0}}},
	}
	doc, node, primitive := newMorphDoc(
		[]float64{0.25, 0.5}, nil, basePosition, nil, targets...)

	before := append([]byte(nil), doc.Buffers[0].Data...)
	got, err := modeler.ReadMorphGeometry(doc, node, primitive)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(doc.Buffers[0].Data, before) {
		t.Error("the document buffer was modified")
	}

	// Mutating the result must not affect the accessors of the document.
	got.Position[0] = [3]float32{99, 99, 99}
	reread, err := modeler.ReadMorphGeometry(doc, node, primitive)
	if err != nil {
		t.Fatal(err)
	}
	want := [][3]float32{{1.5, 2, 0}, {1.5, 2, 0}}
	if !reflect.DeepEqual(reread.Position, want) {
		t.Errorf("Position after mutating previous result = %v, want %v", reread.Position, want)
	}
}

func TestReadMorphGeometry_Errors(t *testing.T) {
	basePosition := [][3]float32{{1, 0, 0}}
	standardTargets := []morphTargetData{
		{Position: [][3]float32{{2, 0, 0}}},
		{Position: [][3]float32{{0, 4, 0}}},
	}

	tests := []struct {
		name string
		doc  *gltf.Document
		node *gltf.Node
		prim *gltf.Primitive
	}{
		{
			"nil node",
			gltf.NewDocument(), nil, &gltf.Primitive{},
		},
		{
			"node without mesh",
			gltf.NewDocument(), &gltf.Node{}, &gltf.Primitive{},
		},
		{
			"mesh index overflows",
			gltf.NewDocument(), &gltf.Node{Mesh: gltf.Index(3)}, &gltf.Primitive{},
		},
		{
			"primitive not in mesh",
			func() *gltf.Document {
				d, _, _ := newMorphDoc(nil, nil, basePosition, nil, standardTargets[0])
				return d
			}(),
			nil, &gltf.Primitive{},
		},
		{
			"weights count mismatch",
			func() *gltf.Document {
				d, _, _ := newMorphDoc([]float64{0.25}, nil, basePosition, nil, standardTargets...)
				return d
			}(),
			nil, nil,
		},
		{
			"node weights count mismatch",
			func() *gltf.Document {
				d, _, _ := newMorphDoc(nil, []float64{0.25}, basePosition, nil, standardTargets...)
				return d
			}(),
			nil, nil,
		},
		{
			"NaN weight",
			func() *gltf.Document {
				d, _, _ := newMorphDoc([]float64{math.NaN()}, nil, basePosition, nil, standardTargets[0])
				return d
			}(),
			nil, nil,
		},
		{
			"infinite weight",
			func() *gltf.Document {
				d, _, _ := newMorphDoc([]float64{math.Inf(1)}, nil, basePosition, nil, standardTargets[0])
				return d
			}(),
			nil, nil,
		},
		{
			"invalid accessor index",
			func() *gltf.Document {
				d, _, _ := newMorphDoc([]float64{0.25}, nil, basePosition, nil, standardTargets[0])
				d.Meshes[0].Primitives[0].Targets[0][gltf.POSITION] = 42
				return d
			}(),
			nil, nil,
		},
		{
			"target vertex count mismatch",
			func() *gltf.Document {
				d := gltf.NewDocument()
				base := modeler.WritePosition(d, [][3]float32{{1, 0, 0}, {1, 0, 0}})
				displaced := modeler.WritePosition(d, [][3]float32{{2, 0, 0}})
				p := &gltf.Primitive{
					Attributes: gltf.PrimitiveAttributes{gltf.POSITION: base},
					Targets:    []gltf.PrimitiveAttributes{{gltf.POSITION: displaced}},
				}
				d.Meshes = append(d.Meshes, &gltf.Mesh{
					Primitives: []*gltf.Primitive{p}, Weights: []float64{0.25},
				})
				d.Nodes = append(d.Nodes, &gltf.Node{Mesh: gltf.Index(0)})
				return d
			}(),
			nil, nil,
		},
		{
			"zero length normal",
			func() *gltf.Document {
				// Base normal (1,0,0) plus displacement (-1,0,0) at weight 1 -> (0,0,0).
				d, _, _ := newMorphDoc(
					[]float64{1}, nil,
					basePosition,
					[][3]float32{{1, 0, 0}},
					morphTargetData{Normal: [][3]float32{{-1, 0, 0}}},
				)
				return d
			}(),
			nil, nil,
		},
		{
			"target normal without base normal",
			func() *gltf.Document {
				d, _, _ := newMorphDoc(
					[]float64{1}, nil, basePosition, nil,
					morphTargetData{Normal: [][3]float32{{0, 1, 0}}},
				)
				return d
			}(),
			nil, nil,
		},
		{
			"primitive without position",
			func() *gltf.Document {
				d := gltf.NewDocument()
				p := &gltf.Primitive{Attributes: gltf.PrimitiveAttributes{}}
				d.Meshes = append(d.Meshes, &gltf.Mesh{Primitives: []*gltf.Primitive{p}})
				d.Nodes = append(d.Nodes, &gltf.Node{Mesh: gltf.Index(0)})
				return d
			}(),
			nil, nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			node := tt.node
			prim := tt.prim
			if node == nil && len(tt.doc.Nodes) > 0 {
				node = tt.doc.Nodes[0]
			}
			if prim == nil && len(tt.doc.Meshes) > 0 && len(tt.doc.Meshes[0].Primitives) > 0 {
				prim = tt.doc.Meshes[0].Primitives[0]
			}
			if _, err := modeler.ReadMorphGeometry(tt.doc, node, prim); err == nil {
				t.Error("ReadMorphGeometry() expected error, got nil")
			}
		})
	}
}
