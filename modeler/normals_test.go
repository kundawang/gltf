package modeler_test

import (
	"errors"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/qmuntal/gltf"
	"github.com/qmuntal/gltf/modeler"
)

const normalEpsilon = 1e-6

func approxVec3(t *testing.T, got, want [3]float32) {
	t.Helper()
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > normalEpsilon {
			t.Fatalf("normal = %v, want %v", got, want)
		}
	}
}

func assertUnitNormals(t *testing.T, normals [][3]float32) {
	t.Helper()
	for i, n := range normals {
		length := math.Sqrt(float64(n[0])*float64(n[0]) + float64(n[1])*float64(n[1]) + float64(n[2])*float64(n[2]))
		if math.Abs(length-1) > normalEpsilon {
			t.Fatalf("normal[%d] = %v has length %f, want 1", i, n, length)
		}
	}
}

// newTriangleDoc returns a document with a single indexed TRIANGLES primitive
// and a material and texture coordinate attribute, so preservation can be
// checked.
func newTriangleDoc(t *testing.T, positions [][3]float32, indices []uint16) (*gltf.Document, int, int) {
	t.Helper()
	doc := gltf.NewDocument()
	positionAccessor := modeler.WritePosition(doc, positions)
	indexAccessor := modeler.WriteIndices(doc, indices)
	uvAccessor := modeler.WriteTextureCoord(doc, make([][2]float32, len(positions)))
	material := &gltf.Material{Name: "keep"}
	doc.Materials = []*gltf.Material{material}
	doc.Meshes = []*gltf.Mesh{{
		Name: "mesh",
		Primitives: []*gltf.Primitive{{
			Indices:  gltf.Index(indexAccessor),
			Material: gltf.Index(0),
			Mode:     gltf.PrimitiveTriangles,
			Attributes: gltf.PrimitiveAttributes{
				gltf.POSITION:   positionAccessor,
				gltf.TEXCOORD_0: uvAccessor,
			},
		}},
	}}
	return doc, 0, 0
}

func TestComputeSmoothNormalsCounterClockwiseIsPlusZ(t *testing.T) {
	// Counter-clockwise in the XY plane (viewed from +Z).
	positions := [][3]float32{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}
	normals, err := modeler.ComputeSmoothNormals(positions, []uint32{0, 1, 2})
	if err != nil {
		t.Fatal(err)
	}
	if len(normals) != 3 {
		t.Fatalf("len(normals) = %d, want 3", len(normals))
	}
	for i, n := range normals {
		approxVec3(t, n, [3]float32{0, 0, 1})
		_ = i
	}

	// Non-indexed form of the same triangle.
	normals, err = modeler.ComputeSmoothNormals(positions, nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, n := range normals {
		approxVec3(t, n, [3]float32{0, 0, 1})
	}
}

func TestComputeSmoothNormalsReversedWindingIsMinusZ(t *testing.T) {
	positions := [][3]float32{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}
	normals, err := modeler.ComputeSmoothNormals(positions, []uint32{0, 2, 1})
	if err != nil {
		t.Fatal(err)
	}
	for _, n := range normals {
		approxVec3(t, n, [3]float32{0, 0, -1})
	}
}

func TestComputeSmoothNormalsAreaWeighted(t *testing.T) {
	// Triangle A (area 0.5) lies in the XY plane with normal +Z.
	// Triangle B (area 2) lies in the YZ plane with normal +X.
	// They share vertex 0, whose normal is the normalized sum of the two
	// unnormalized area vectors: (4, 0, 1).
	positions := [][3]float32{
		{0, 0, 0},            // shared
		{1, 0, 0}, {0, 1, 0}, // triangle A
		{0, 2, 0}, {0, 0, 2}, // triangle B (with vertex 0)
	}
	indices := []uint32{0, 1, 2, 0, 3, 4}
	normals, err := modeler.ComputeSmoothNormals(positions, indices)
	if err != nil {
		t.Fatal(err)
	}
	assertUnitNormals(t, normals)

	inv := 1 / math.Sqrt(17)
	approxVec3(t, normals[0], [3]float32{float32(4 * inv), 0, float32(inv)})
	approxVec3(t, normals[1], [3]float32{0, 0, 1})
	approxVec3(t, normals[2], [3]float32{0, 0, 1})
	approxVec3(t, normals[3], [3]float32{1, 0, 0})
	approxVec3(t, normals[4], [3]float32{1, 0, 0})

	// Shrinking triangle B changes the shared normal according to the new
	// area ratio (area 0.5 vs 0.5): normalized (1, 0, 1).
	smaller := [][3]float32{
		{0, 0, 0},
		{1, 0, 0}, {0, 1, 0},
		{0, 1, 0}, {0, 0, 1},
	}
	normals, err = modeler.ComputeSmoothNormals(smaller, indices)
	if err != nil {
		t.Fatal(err)
	}
	inv = 1 / math.Sqrt(2)
	approxVec3(t, normals[0], [3]float32{float32(inv), 0, float32(inv)})
}

func TestComputeSmoothNormalsSharedVertices(t *testing.T) {
	// Two coplanar triangles of a unit quad sharing the diagonal 0-2.
	// Every vertex normal must be +Z, despite the triangles being
	// accumulated twice at 0 and 2.
	positions := [][3]float32{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}}
	indices := []uint32{0, 1, 2, 0, 2, 3}
	normals, err := modeler.ComputeSmoothNormals(positions, indices)
	if err != nil {
		t.Fatal(err)
	}
	assertUnitNormals(t, normals)
	for _, n := range normals {
		approxVec3(t, n, [3]float32{0, 0, 1})
	}
}

func TestComputeSmoothNormalsDegenerateAndUnused(t *testing.T) {
	positions := [][3]float32{
		{0, 0, 0}, {1, 0, 0}, {2, 0, 0}, // collinear, degenerate
		{2, 0, 0},                       // referenced only by the degenerate triangle
		{5, 5, 5},                       // never referenced
		{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, // valid triangle
	}
	indices := []uint32{0, 1, 2, 0, 3, 1, 5, 6, 7}
	normals, err := modeler.ComputeSmoothNormals(positions, indices)
	if err != nil {
		t.Fatal(err)
	}
	assertUnitNormals(t, normals)
	approxVec3(t, normals[0], [3]float32{0, 0, 1})
	// Vertices without a contribution fall back to the deterministic unit +Z.
	approxVec3(t, normals[2], [3]float32{0, 0, 1})
	approxVec3(t, normals[3], [3]float32{0, 0, 1})
	approxVec3(t, normals[4], [3]float32{0, 0, 1})
}

func TestComputeSmoothNormalsErrors(t *testing.T) {
	valid := [][3]float32{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}

	tests := []struct {
		name      string
		positions [][3]float32
		indices   []uint32
		want      error
	}{
		{"empty", nil, nil, modeler.ErrNoValidFaces},
		{"index out of range", valid, []uint32{0, 1, 3}, modeler.ErrIndexOutOfRange},
		{"index count not multiple of 3", valid, []uint32{0, 1}, modeler.ErrInvalidIndexCount},
		{"position count not multiple of 3", [][3]float32{{0, 0, 0}, {1, 0, 0}}, nil, modeler.ErrInvalidIndexCount},
		{"NaN position", [][3]float32{{0, 0, 0}, {float32(math.NaN()), 0, 0}, {0, 1, 0}}, nil, modeler.ErrNonFinitePosition},
		{"Inf position", [][3]float32{{0, 0, 0}, {1, 0, 0}, {0, float32(math.Inf(1)), 0}}, nil, modeler.ErrNonFinitePosition},
		{"unused NaN position", [][3]float32{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {float32(math.NaN()), 0, 0}}, []uint32{0, 1, 2}, modeler.ErrNonFinitePosition},
		{"all degenerate", [][3]float32{{0, 0, 0}, {1, 0, 0}, {2, 0, 0}}, nil, modeler.ErrNoValidFaces},
		{"repeated index", [][3]float32{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}, []uint32{0, 0, 1}, modeler.ErrNoValidFaces},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := modeler.ComputeSmoothNormals(tt.positions, tt.indices)
			if !errors.Is(err, tt.want) {
				t.Fatalf("error = %v, want %v", err, tt.want)
			}
		})
	}
}

func TestGenerateNormalsIndexed(t *testing.T) {
	positions := [][3]float32{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}
	doc, meshIndex, primitiveIndex := newTriangleDoc(t, positions, []uint16{0, 1, 2})

	out, normalAccessor, err := modeler.GenerateNormals(doc, meshIndex, primitiveIndex)
	if err != nil {
		t.Fatal(err)
	}
	if normalAccessor != len(doc.Accessors) {
		t.Fatalf("normal accessor = %d, want %d", normalAccessor, len(doc.Accessors))
	}

	primitive := out.Meshes[meshIndex].Primitives[primitiveIndex]
	gotIndex, ok := primitive.Attributes[gltf.NORMAL]
	if !ok || gotIndex != normalAccessor {
		t.Fatalf("NORMAL attribute = %d (present %v), want %d", gotIndex, ok, normalAccessor)
	}

	normals, err := modeler.ReadNormal(out, out.Accessors[normalAccessor], nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(normals) != len(positions) {
		t.Fatalf("len(normals) = %d, want %d", len(normals), len(positions))
	}
	for _, n := range normals {
		approxVec3(t, n, [3]float32{0, 0, 1})
	}

	// The accessor itself has the glTF shape of a NORMAL attribute.
	acr := out.Accessors[normalAccessor]
	if acr.ComponentType != gltf.ComponentFloat || acr.Type != gltf.AccessorVec3 || acr.Count != len(positions) {
		t.Fatalf("unexpected NORMAL accessor: %+v", acr)
	}
}

func TestGenerateNormalsNonIndexed(t *testing.T) {
	positions := [][3]float32{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}
	doc := gltf.NewDocument()
	positionAccessor := modeler.WritePosition(doc, positions)
	doc.Meshes = []*gltf.Mesh{{Primitives: []*gltf.Primitive{{
		Mode:       gltf.PrimitiveTriangles,
		Attributes: gltf.PrimitiveAttributes{gltf.POSITION: positionAccessor},
	}}}}

	out, normalAccessor, err := modeler.GenerateNormals(doc, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	normals, err := modeler.ReadNormal(out, out.Accessors[normalAccessor], nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, n := range normals {
		approxVec3(t, n, [3]float32{0, 0, 1})
	}

	// Non-indexed positions that are not a multiple of three are rejected.
	badDoc := gltf.NewDocument()
	badAccessor := modeler.WritePosition(badDoc, [][3]float32{{0, 0, 0}, {1, 0, 0}})
	badDoc.Meshes = []*gltf.Mesh{{Primitives: []*gltf.Primitive{{
		Mode:       gltf.PrimitiveTriangles,
		Attributes: gltf.PrimitiveAttributes{gltf.POSITION: badAccessor},
	}}}}
	if _, _, err := modeler.GenerateNormals(badDoc, 0, 0); !errors.Is(err, modeler.ErrInvalidIndexCount) {
		t.Fatalf("error = %v, want ErrInvalidIndexCount", err)
	}
}

func TestGenerateNormalsSparsePosition(t *testing.T) {
	doc := gltf.NewDocument()
	valuesAccessor := modeler.WriteAccessor(doc, gltf.TargetArrayBuffer, [][3]float32{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}})
	indicesAccessor := modeler.WriteAccessor(doc, gltf.TargetArrayBuffer, []uint16{0, 1, 2})
	triangleIndices := modeler.WriteIndices(doc, []uint16{0, 1, 2})

	positionAccessor := &gltf.Accessor{
		ComponentType: gltf.ComponentFloat,
		Type:          gltf.AccessorVec3,
		Count:         4,
		Sparse: &gltf.Sparse{
			Count: 3,
			Indices: gltf.SparseIndices{
				BufferView:    *doc.Accessors[indicesAccessor].BufferView,
				ComponentType: gltf.ComponentUshort,
			},
			Values: gltf.SparseValues{
				BufferView: *doc.Accessors[valuesAccessor].BufferView,
			},
		},
	}
	doc.Accessors = append(doc.Accessors, positionAccessor)
	positionIndex := len(doc.Accessors) - 1

	doc.Meshes = []*gltf.Mesh{{Primitives: []*gltf.Primitive{{
		Indices:    gltf.Index(triangleIndices),
		Mode:       gltf.PrimitiveTriangles,
		Attributes: gltf.PrimitiveAttributes{gltf.POSITION: positionIndex},
	}}}}

	out, normalAccessor, err := modeler.GenerateNormals(doc, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	normals, err := modeler.ReadNormal(out, out.Accessors[normalAccessor], nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(normals) != 4 {
		t.Fatalf("len(normals) = %d, want 4", len(normals))
	}
	for _, n := range normals {
		approxVec3(t, n, [3]float32{0, 0, 1})
	}
}

func TestGenerateNormalsInterleavedPosition(t *testing.T) {
	doc := gltf.NewDocument()
	attributes, err := modeler.WritePrimitiveAttributes(doc,
		modeler.PrimitiveAttribute{Name: gltf.POSITION, Data: [][3]float32{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}},
		modeler.PrimitiveAttribute{Name: gltf.TEXCOORD_0, Data: [][2]float32{{0, 0}, {1, 0}, {0, 1}}},
	)
	if err != nil {
		t.Fatal(err)
	}
	indexAccessor := modeler.WriteIndices(doc, []uint16{0, 1, 2})
	doc.Meshes = []*gltf.Mesh{{Primitives: []*gltf.Primitive{{
		Indices:    gltf.Index(indexAccessor),
		Mode:       gltf.PrimitiveTriangles,
		Attributes: attributes,
	}}}}

	out, normalAccessor, err := modeler.GenerateNormals(doc, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	normals, err := modeler.ReadNormal(out, out.Accessors[normalAccessor], nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, n := range normals {
		approxVec3(t, n, [3]float32{0, 0, 1})
	}
	// The interleaved POSITION and TEXCOORD_0 accessors are preserved.
	if out.Meshes[0].Primitives[0].Attributes[gltf.POSITION] != attributes[gltf.POSITION] ||
		out.Meshes[0].Primitives[0].Attributes[gltf.TEXCOORD_0] != attributes[gltf.TEXCOORD_0] {
		t.Fatalf("interleaved attributes changed: %v", out.Meshes[0].Primitives[0].Attributes)
	}
}

func TestGenerateNormalsPreservesDocument(t *testing.T) {
	positions := [][3]float32{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}}
	doc, meshIndex, primitiveIndex := newTriangleDoc(t, positions, []uint16{0, 1, 2, 1, 3, 2})

	// Add a second untouched primitive and capture the original state.
	otherPosition := modeler.WritePosition(doc, positions)
	doc.Meshes[0].Primitives = append(doc.Meshes[0].Primitives, &gltf.Primitive{
		Mode:       gltf.PrimitiveTriangles,
		Attributes: gltf.PrimitiveAttributes{gltf.POSITION: otherPosition},
	})
	originalAccessorCount := len(doc.Accessors)
	originalBufferViews := append([]*gltf.BufferView(nil), doc.BufferViews...)
	originalBuffer := append([]byte(nil), doc.Buffers[0].Data...)
	originalByteLength := doc.Buffers[0].ByteLength
	originalMesh := doc.Meshes[0]
	originalPrimitive := doc.Meshes[0].Primitives[primitiveIndex]
	originalOther := doc.Meshes[0].Primitives[1]
	originalIndices := *originalPrimitive.Indices
	originalMaterial := *originalPrimitive.Material
	originalAttributes := make(gltf.PrimitiveAttributes, len(originalPrimitive.Attributes))
	for k, v := range originalPrimitive.Attributes {
		originalAttributes[k] = v
	}

	out, normalAccessor, err := modeler.GenerateNormals(doc, meshIndex, primitiveIndex)
	if err != nil {
		t.Fatal(err)
	}

	// The original document is untouched.
	if len(doc.Accessors) != originalAccessorCount {
		t.Fatalf("original accessor count = %d, want %d", len(doc.Accessors), originalAccessorCount)
	}
	if doc.Meshes[0] != originalMesh || doc.Meshes[0].Primitives[primitiveIndex] != originalPrimitive {
		t.Fatal("original mesh or primitive pointer changed")
	}
	if hasNormal(doc.Meshes[0].Primitives[0].Attributes) {
		t.Fatal("original primitive gained a NORMAL attribute")
	}
	if !reflect.DeepEqual(doc.Meshes[0].Primitives[0].Attributes, originalAttributes) {
		t.Fatalf("original attributes changed: %v", doc.Meshes[0].Primitives[0].Attributes)
	}
	if *doc.Meshes[0].Primitives[0].Indices != originalIndices || *doc.Meshes[0].Primitives[0].Material != originalMaterial {
		t.Fatal("original indices or material changed")
	}
	if !reflect.DeepEqual(doc.BufferViews, originalBufferViews) {
		t.Fatal("original buffer views changed")
	}
	if !reflect.DeepEqual(doc.Buffers[0].Data, originalBuffer) || doc.Buffers[0].ByteLength != originalByteLength {
		t.Fatal("original buffer data changed")
	}
	if doc.Meshes[0].Primitives[1] != originalOther {
		t.Fatal("other primitive changed")
	}

	// The returned document preserves positions, indices, UV, material and
	// the other primitive.
	outPrimitive := out.Meshes[meshIndex].Primitives[primitiveIndex]
	if *outPrimitive.Indices != originalIndices || *outPrimitive.Material != originalMaterial {
		t.Fatal("indices or material not preserved in the returned document")
	}
	if outPrimitive.Attributes[gltf.POSITION] != originalAttributes[gltf.POSITION] ||
		outPrimitive.Attributes[gltf.TEXCOORD_0] != originalAttributes[gltf.TEXCOORD_0] {
		t.Fatal("position or UV not preserved in the returned document")
	}
	if out.Meshes[meshIndex].Primitives[1] != originalOther ||
		out.Meshes[meshIndex].Primitives[1].Attributes[gltf.POSITION] != otherPosition {
		t.Fatal("other primitive not preserved in the returned document")
	}
	if out.Materials[0] != doc.Materials[0] {
		t.Fatal("materials changed")
	}
	if out.Accessors[normalAccessor] == nil {
		t.Fatal("normal accessor was not written")
	}

	// Mutating the returned document's buffer cannot reach doc.
	out.Buffers[0].Data[0] ^= 0xff
	if reflect.DeepEqual(out.Buffers[0].Data[:len(originalBuffer)], originalBuffer) {
		t.Fatal("the returned document shares its last buffer storage with the original")
	}
	if !reflect.DeepEqual(doc.Buffers[0].Data, originalBuffer) {
		t.Fatal("mutating the returned document modified the original buffer")
	}
}

func hasNormal(attrs gltf.PrimitiveAttributes) bool {
	_, ok := attrs[gltf.NORMAL]
	return ok
}

func TestGenerateNormalsExistingNormal(t *testing.T) {
	positions := [][3]float32{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}
	doc, _, _ := newTriangleDoc(t, positions, []uint16{0, 1, 2})
	normalAccessor := modeler.WriteNormal(doc, [][3]float32{{0, 0, 1}, {0, 0, 1}, {0, 0, 1}})
	doc.Meshes[0].Primitives[0].Attributes[gltf.NORMAL] = normalAccessor

	if _, _, err := modeler.GenerateNormals(doc, 0, 0); !errors.Is(err, modeler.ErrNormalsExist) {
		t.Fatalf("error = %v, want ErrNormalsExist", err)
	}

	out, index, err := modeler.GenerateNormals(doc, 0, 0, modeler.OverwriteNormals())
	if err != nil {
		t.Fatal(err)
	}
	if index == normalAccessor {
		t.Fatal("OverwriteNormals should write a new accessor")
	}
	if got := out.Meshes[0].Primitives[0].Attributes[gltf.NORMAL]; got != index {
		t.Fatalf("NORMAL attribute = %d, want %d", got, index)
	}
	// The old NORMAL accessor is still present for other users.
	if index >= len(out.Accessors) || out.Accessors[normalAccessor] == nil {
		t.Fatal("old accessor storage was not preserved")
	}
	normals, err := modeler.ReadNormal(out, out.Accessors[index], nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, n := range normals {
		approxVec3(t, n, [3]float32{0, 0, 1})
	}
}

func TestGenerateNormalsErrors(t *testing.T) {
	positions := [][3]float32{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}

	t.Run("nil document", func(t *testing.T) {
		if _, _, err := modeler.GenerateNormals(nil, 0, 0); !errors.Is(err, modeler.ErrNilDocument) {
			t.Fatalf("error = %v, want ErrNilDocument", err)
		}
	})

	t.Run("missing mesh", func(t *testing.T) {
		doc, _, _ := newTriangleDoc(t, positions, []uint16{0, 1, 2})
		if _, _, err := modeler.GenerateNormals(doc, 1, 0); !errors.Is(err, modeler.ErrPrimitiveNotFound) {
			t.Fatalf("error = %v, want ErrPrimitiveNotFound", err)
		}
		if _, _, err := modeler.GenerateNormals(doc, 0, 1); !errors.Is(err, modeler.ErrPrimitiveNotFound) {
			t.Fatalf("error = %v, want ErrPrimitiveNotFound", err)
		}
	})

	t.Run("missing position", func(t *testing.T) {
		doc := gltf.NewDocument()
		doc.Meshes = []*gltf.Mesh{{Primitives: []*gltf.Primitive{{Mode: gltf.PrimitiveTriangles}}}}
		if _, _, err := modeler.GenerateNormals(doc, 0, 0); !errors.Is(err, modeler.ErrMissingPosition) {
			t.Fatalf("error = %v, want ErrMissingPosition", err)
		}
	})

	t.Run("not triangles", func(t *testing.T) {
		doc, _, _ := newTriangleDoc(t, positions, []uint16{0, 1, 2})
		doc.Meshes[0].Primitives[0].Mode = gltf.PrimitiveTriangleStrip
		if _, _, err := modeler.GenerateNormals(doc, 0, 0); !errors.Is(err, modeler.ErrNotTriangles) {
			t.Fatalf("error = %v, want ErrNotTriangles", err)
		}
	})

	t.Run("index out of range", func(t *testing.T) {
		doc, _, _ := newTriangleDoc(t, positions, []uint16{0, 1, 2})
		doc.Meshes[0].Primitives[0].Indices = gltf.Index(modeler.WriteIndices(doc, []uint16{0, 1, 3}))
		if _, _, err := modeler.GenerateNormals(doc, 0, 0); !errors.Is(err, modeler.ErrIndexOutOfRange) {
			t.Fatalf("error = %v, want ErrIndexOutOfRange", err)
		}
	})

	t.Run("non finite position", func(t *testing.T) {
		doc, _, _ := newTriangleDoc(t, positions, []uint16{0, 1, 2})
		doc.Meshes[0].Primitives[0].Attributes[gltf.POSITION] = modeler.WritePosition(doc, [][3]float32{{0, 0, 0}, {1, 0, 0}, {0, float32(math.NaN()), 0}})
		if _, _, err := modeler.GenerateNormals(doc, 0, 0); !errors.Is(err, modeler.ErrNonFinitePosition) {
			t.Fatalf("error = %v, want ErrNonFinitePosition", err)
		}
	})

	t.Run("no valid faces", func(t *testing.T) {
		doc, _, _ := newTriangleDoc(t, [][3]float32{{0, 0, 0}, {1, 0, 0}, {2, 0, 0}}, []uint16{0, 1, 2})
		if _, _, err := modeler.GenerateNormals(doc, 0, 0); !errors.Is(err, modeler.ErrNoValidFaces) {
			t.Fatalf("error = %v, want ErrNoValidFaces", err)
		}
	})
}

func TestGenerateNormalsRoundTripGLB(t *testing.T) {
	positions := [][3]float32{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}
	doc, meshIndex, primitiveIndex := newTriangleDoc(t, positions, []uint16{0, 1, 2})

	out, normalAccessor, err := modeler.GenerateNormals(doc, meshIndex, primitiveIndex)
	if err != nil {
		t.Fatal(err)
	}
	want, err := modeler.ReadNormal(out, out.Accessors[normalAccessor], nil)
	if err != nil {
		t.Fatal(err)
	}

	name := filepath.Join(t.TempDir(), "normals.glb")
	if err := gltf.SaveBinary(out, name); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(name); err != nil {
		t.Fatal(err)
	}

	reloaded, err := gltf.Open(name)
	if err != nil {
		t.Fatal(err)
	}
	if len(reloaded.Meshes) != 1 || len(reloaded.Meshes[0].Primitives) != 1 {
		t.Fatalf("unexpected mesh structure after reload: %+v", reloaded.Meshes)
	}
	primitive := reloaded.Meshes[0].Primitives[0]
	gotAccessor, ok := primitive.Attributes[gltf.NORMAL]
	if !ok {
		t.Fatal("NORMAL attribute missing after reload")
	}
	// Positions, indices, UV and the material survive the round trip.
	if _, ok := primitive.Attributes[gltf.POSITION]; !ok {
		t.Fatal("POSITION attribute missing after reload")
	}
	if _, ok := primitive.Attributes[gltf.TEXCOORD_0]; !ok {
		t.Fatal("TEXCOORD_0 attribute missing after reload")
	}
	if primitive.Indices == nil || primitive.Material == nil {
		t.Fatal("indices or material missing after reload")
	}
	if len(reloaded.Materials) != 1 || reloaded.Materials[0].Name != "keep" {
		t.Fatal("material missing after reload")
	}

	got, err := modeler.ReadNormal(reloaded, reloaded.Accessors[gotAccessor], nil)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("normals after reload = %v, want %v", got, want)
	}
	for _, n := range got {
		approxVec3(t, n, [3]float32{0, 0, 1})
	}
	// The reloaded positions and indices still describe the same triangle.
	gotPositions, err := modeler.ReadPosition(reloaded, reloaded.Accessors[primitive.Attributes[gltf.POSITION]], nil)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(gotPositions, positions) {
		t.Fatalf("positions after reload = %v, want %v", gotPositions, positions)
	}
	gotIndices, err := modeler.ReadIndices(reloaded, reloaded.Accessors[*primitive.Indices], nil)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(gotIndices, []uint32{0, 1, 2}) {
		t.Fatalf("indices after reload = %v", gotIndices)
	}
}

// TestGenerateNormalsSampleModel runs the feature against the official
// glTF-Sample-Models Triangle assets, which only store POSITION (and indices
// for one of them), exercising external-buffer loading as well.
func TestGenerateNormalsSampleModel(t *testing.T) {
	cases := []struct {
		name string
		path string
	}{
		{"indexed", "../testdata/Triangle/glTF/Triangle.gltf"},
		{"non-indexed", "../testdata/TriangleWithoutIndices/glTF/TriangleWithoutIndices.gltf"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			doc, err := gltf.Open(tc.path)
			if err != nil {
				t.Fatal(err)
			}
			primitive := doc.Meshes[0].Primitives[0]
			if _, ok := primitive.Attributes[gltf.NORMAL]; ok {
				t.Skip("sample model already has normals")
			}
			out, accessor, err := modeler.GenerateNormals(doc, 0, 0)
			if err != nil {
				t.Fatal(err)
			}
			if _, ok := doc.Meshes[0].Primitives[0].Attributes[gltf.NORMAL]; ok {
				t.Fatal("GenerateNormals modified the source document")
			}
			normals, err := modeler.ReadNormal(out, out.Accessors[accessor], nil)
			if err != nil {
				t.Fatal(err)
			}
			if len(normals) != 3 {
				t.Fatalf("len(normals) = %d, want 3", len(normals))
			}
			// The sample triangle is counter-clockwise in the XY plane.
			for _, n := range normals {
				approxVec3(t, n, [3]float32{0, 0, 1})
			}
		})
	}
}

func TestComputeSmoothNormalsCancellingContributionFallsBack(t *testing.T) {
	// Two opposite-facing triangles share vertex 0. Its contributions cancel
	// exactly, so it has no meaningful direction and gets the fallback normal;
	// the other vertices keep their respective face normals.
	positions := [][3]float32{
		{0, 0, 0},
		{1, 0, 0}, {0, 1, 0},
		{1, 0, 0}, {0, 1, 0},
	}
	indices := []uint32{0, 1, 2, 0, 4, 3}
	normals, err := modeler.ComputeSmoothNormals(positions, indices)
	if err != nil {
		t.Fatal(err)
	}
	assertUnitNormals(t, normals)
	approxVec3(t, normals[0], [3]float32{0, 0, 1})
	approxVec3(t, normals[1], [3]float32{0, 0, 1})
	approxVec3(t, normals[2], [3]float32{0, 0, 1})
	approxVec3(t, normals[3], [3]float32{0, 0, -1})
	approxVec3(t, normals[4], [3]float32{0, 0, -1})
}

func TestGenerateNormalsNoBufferAndDefaultMode(t *testing.T) {
	// A zero-value Mode is TRIANGLES per the glTF spec, and a document with
	// no buffers at all must still gain its first buffer in the copy.
	doc := gltf.NewDocument()
	positionAccessor := modeler.WritePosition(doc, [][3]float32{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}})
	doc.Meshes = []*gltf.Mesh{{Primitives: []*gltf.Primitive{{
		Attributes: gltf.PrimitiveAttributes{gltf.POSITION: positionAccessor},
	}}}}

	out, accessor, err := modeler.GenerateNormals(doc, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(out.Buffers) != len(doc.Buffers) {
		t.Fatalf("buffer count = %d, want %d", len(out.Buffers), len(doc.Buffers))
	}
	normals, err := modeler.ReadNormal(out, out.Accessors[accessor], nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, n := range normals {
		approxVec3(t, n, [3]float32{0, 0, 1})
	}
}

func TestGenerateNormalsAlias(t *testing.T) {
	positions := [][3]float32{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}
	doc, _, _ := newTriangleDoc(t, positions, []uint16{0, 1, 2})
	out, accessor, err := modeler.GenerateMeshNormals(doc, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if out.Meshes[0].Primitives[0].Attributes[gltf.NORMAL] != accessor {
		t.Fatal("GenerateMeshNormals did not write the NORMAL attribute")
	}
}
