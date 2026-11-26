package modeler_test

// Verification suite for UID 117 primitive topology conversion. Every
// identifier is prefixed uid117 to avoid colliding with any externally
// supplied test files in package modeler_test.

import (
	"bytes"
	"errors"
	"math"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/qmuntal/gltf"
	"github.com/qmuntal/gltf/modeler"
)

func uid117Positions4() [][3]float32 {
	return [][3]float32{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}}
}

func uid117Doc(mode gltf.PrimitiveMode, indices any, positions [][3]float32) *gltf.Document {
	doc := gltf.NewDocument()
	doc.Materials = []*gltf.Material{{Name: "uid117-mat"}}
	doc.Meshes = []*gltf.Mesh{{
		Name: "uid117-mesh",
		Primitives: []*gltf.Primitive{{
			Attributes: gltf.PrimitiveAttributes{},
			Mode:       mode,
			Material:   gltf.Index(0),
			Extras:     map[string]any{"k": "v"},
		}},
	}}
	p := doc.Meshes[0].Primitives[0]
	if positions != nil {
		p.Attributes[gltf.POSITION] = modeler.WritePosition(doc, positions)
		// Add a second attribute to prove all attributes survive.
		uv := make([][2]float32, len(positions))
		for i := range uv {
			uv[i] = [2]float32{float32(i), 0.5}
		}
		p.Attributes[gltf.TEXCOORD_0] = modeler.WriteTextureCoord(doc, uv)
	}
	if indices != nil {
		p.Indices = gltf.Index(modeler.WriteIndices(doc, indices))
	}
	return doc
}

func uid117Read(t *testing.T, doc *gltf.Document, accIndex *int) []uint32 {
	t.Helper()
	if accIndex == nil {
		t.Fatal("nil indices accessor")
	}
	got, err := modeler.ReadIndices(doc, doc.Accessors[*accIndex], nil)
	if err != nil {
		t.Fatalf("ReadIndices: %v", err)
	}
	return got
}

func TestUID117_ExactVectors(t *testing.T) {
	if got := modeler.StripTriangleIndices([]uint32{0, 1, 2, 3}); !reflect.DeepEqual(got, []uint32{0, 1, 2, 2, 1, 3}) {
		t.Fatalf("strip = %v", got)
	}
	if got := modeler.FanTriangleIndices([]uint32{0, 1, 2, 3}); !reflect.DeepEqual(got, []uint32{0, 1, 2, 0, 2, 3}) {
		t.Fatalf("fan = %v", got)
	}
}

func TestUID117_IndexedConversions(t *testing.T) {
	cases := []struct {
		name string
		mode gltf.PrimitiveMode
		want []uint32
	}{
		{"strip", gltf.PrimitiveTriangleStrip, []uint32{0, 1, 2, 2, 1, 3}},
		{"fan", gltf.PrimitiveTriangleFan, []uint32{0, 1, 2, 0, 2, 3}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			src := uid117Doc(tc.mode, []uint32{0, 1, 2, 3}, uid117Positions4())
			out, err := modeler.ConvertPrimitiveTopology(src, 0, 0)
			if err != nil {
				t.Fatal(err)
			}
			p := out.Meshes[0].Primitives[0]
			if p.Mode != gltf.PrimitiveTriangles {
				t.Fatalf("mode = %v, want TRIANGLES", p.Mode)
			}
			if got := uid117Read(t, out, p.Indices); !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("indices = %v, want %v", got, tc.want)
			}
			if doc := out.Accessors[*p.Indices]; doc.ComponentType != gltf.ComponentUint || doc.Type != gltf.AccessorScalar || doc.Count != len(tc.want) {
				t.Fatalf("bad accessor: %+v", doc)
			}
		})
	}
}

func TestUID117_UnindexedConversions(t *testing.T) {
	for _, tc := range []struct {
		name string
		mode gltf.PrimitiveMode
		want []uint32
	}{
		{"strip", gltf.PrimitiveTriangleStrip, []uint32{0, 1, 2, 2, 1, 3}},
		{"fan", gltf.PrimitiveTriangleFan, []uint32{0, 1, 2, 0, 2, 3}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			src := uid117Doc(tc.mode, nil, uid117Positions4())
			if src.Meshes[0].Primitives[0].Indices != nil {
				t.Fatal("precondition: unindexed")
			}
			out, err := modeler.ToTriangles(src, src.Meshes[0].Primitives[0])
			if err != nil {
				t.Fatal(err)
			}
			p := out.Meshes[0].Primitives[0]
			if p.Indices == nil {
				t.Fatal("expected a valid indices accessor")
			}
			if got := uid117Read(t, out, p.Indices); !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("indices = %v, want %v", got, tc.want)
			}
			if src.Meshes[0].Primitives[0].Indices != nil || src.Meshes[0].Primitives[0].Mode != tc.mode {
				t.Fatal("source document was modified")
			}
		})
	}
}

func TestUID117_PreservesMaterialAndAttributes(t *testing.T) {
	src := uid117Doc(gltf.PrimitiveTriangleStrip, []uint32{0, 1, 2, 3}, uid117Positions4())
	out, err := modeler.ConvertPrimitiveMode(src, modeler.PrimitiveRef{Mesh: 0, Primitive: 0})
	if err != nil {
		t.Fatal(err)
	}
	got := out.Meshes[0].Primitives[0]
	want := src.Meshes[0].Primitives[0]
	if !reflect.DeepEqual(got.Attributes, want.Attributes) {
		t.Fatalf("attributes = %v, want %v", got.Attributes, want.Attributes)
	}
	if got.Material == nil || want.Material == nil || *got.Material != *want.Material {
		t.Fatalf("material not preserved: %v vs %v", got.Material, want.Material)
	}
	if !reflect.DeepEqual(got.Targets, want.Targets) {
		t.Fatal("targets not preserved")
	}
	if !reflect.DeepEqual(got.Extras, want.Extras) {
		t.Fatalf("extras not preserved: %#v", got.Extras)
	}
	if len(out.Materials) != 1 || out.Materials[0].Name != "uid117-mat" {
		t.Fatal("material asset not preserved")
	}
	// Attribute data still readable through the output document.
	pos, err := modeler.ReadPosition(out, out.Accessors[got.Attributes[gltf.POSITION]], nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(pos) != 4 || pos[2] != [3]float32{1, 1, 0} {
		t.Fatalf("position data wrong: %v", pos)
	}
}

func TestUID117_SourceUnchangedAndCloneIndependent(t *testing.T) {
	src := uid117Doc(gltf.PrimitiveTriangleFan, []uint32{0, 1, 2, 3}, uid117Positions4())
	before := modeler.CloneDocument(src)
	out, err := modeler.ConvertPrimitiveTopology(src, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(src, before) {
		t.Fatal("source document changed by conversion")
	}
	if out == src {
		t.Fatal("expected an independent document")
	}
	// Mutating the output must not reach the source.
	out.Meshes[0].Name = "changed"
	if src.Meshes[0].Name == "changed" {
		t.Fatal("output shares mesh storage with source")
	}
}

func TestUID117_SharedIndicesAccessorIsolation(t *testing.T) {
	src := uid117Doc(gltf.PrimitiveTriangleStrip, []uint32{0, 1, 2, 3}, uid117Positions4())
	p0 := src.Meshes[0].Primitives[0]
	shared := *p0.Indices
	src.Meshes = append(src.Meshes, &gltf.Mesh{Primitives: []*gltf.Primitive{{
		Attributes: gltf.PrimitiveAttributes{gltf.POSITION: p0.Attributes[gltf.POSITION]},
		Indices:    gltf.Index(shared),
		Mode:       gltf.PrimitiveTriangleFan,
	}}})

	out, err := modeler.ConvertPrimitiveTopology(src, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	o0 := out.Meshes[0].Primitives[0]
	o1 := out.Meshes[1].Primitives[0]
	if o0.Indices == nil || *o0.Indices == shared {
		t.Fatal("converted primitive must reference a fresh accessor")
	}
	if o1.Indices == nil || *o1.Indices != shared {
		t.Fatalf("other primitive must keep shared accessor %d, got %v", shared, o1.Indices)
	}
	if o1.Mode != gltf.PrimitiveTriangleFan {
		t.Fatal("other primitive mode changed")
	}
	if got := uid117Read(t, out, o1.Indices); !reflect.DeepEqual(got, []uint32{0, 1, 2, 3}) {
		t.Fatalf("shared accessor content changed: %v", got)
	}
	if src.Meshes[1].Primitives[0].Mode != gltf.PrimitiveTriangleFan {
		t.Fatal("source other primitive changed")
	}

	// Strict option turns sharing into a clear error.
	_, err = modeler.ConvertPrimitiveAt(src, 0, 0, modeler.RejectSharedAccessor())
	if !errors.Is(err, modeler.ErrSharedAccessor) {
		t.Fatalf("strict shared err = %v", err)
	}
}

func TestUID117_OtherPrimitiveUnchanged(t *testing.T) {
	src := uid117Doc(gltf.PrimitiveTriangleStrip, []uint32{0, 1, 2, 3}, uid117Positions4())
	src.Meshes[0].Primitives = append(src.Meshes[0].Primitives, &gltf.Primitive{
		Attributes: gltf.PrimitiveAttributes{gltf.POSITION: 0},
		Mode:       gltf.PrimitivePoints,
	})
	out, err := modeler.ConvertPrimitiveTopology(src, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(out.Meshes[0].Primitives[1], src.Meshes[0].Primitives[1]) {
		t.Fatal("unselected primitive changed")
	}
}

func TestUID117_DegenerateTrianglesSkipped(t *testing.T) {
	// Repeated indices create zero-area slots that must not emit a face.
	if got := modeler.StripTriangleIndices([]uint32{0, 0, 1, 2, 3}); !reflect.DeepEqual(got, []uint32{1, 0, 2, 1, 2, 3}) {
		t.Fatalf("strip degenerates = %v", got)
	}
	if got := modeler.FanTriangleIndices([]uint32{0, 1, 1, 2, 3}); !reflect.DeepEqual(got, []uint32{0, 1, 2, 0, 2, 3}) {
		// Fan triples: (0,1,1) skip; (0,1,2) emit; (0,2,3) emit.
		t.Fatalf("fan degenerates = %v", got)
	}

	// Geometric zero-area with distinct indices is also removed.
	col := [][3]float32{{0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {0, 1, 0}}
	doc := uid117Doc(gltf.PrimitiveTriangleFan, []uint32{0, 1, 2, 3}, col)
	out, err := modeler.ConvertPrimitiveTopology(doc, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	p := out.Meshes[0].Primitives[0]
	if got := uid117Read(t, out, p.Indices); !reflect.DeepEqual(got, []uint32{0, 2, 3}) {
		t.Fatalf("geometric degenerate result = %v", got)
	}
}

func TestUID117_ErrorCases(t *testing.T) {
	pos3 := [][3]float32{{}, {}, {}}

	small := uid117Doc(gltf.PrimitiveTriangleStrip, []uint32{0, 1}, pos3[:2])
	if _, err := modeler.ConvertPrimitiveTopology(small, 0, 0); !errors.Is(err, modeler.ErrTooFewVertices) {
		t.Fatalf("small err = %v", err)
	}
	// Permissive option yields an empty triangle list instead.
	empty, err := modeler.ConvertPrimitiveAt(small, 0, 0, modeler.AllowEmptyTopology())
	if err != nil {
		t.Fatalf("allow-empty err = %v", err)
	}
	if got := uid117Read(t, empty, empty.Meshes[0].Primitives[0].Indices); len(got) != 0 {
		t.Fatalf("expected empty, got %v", got)
	}

	oob := uid117Doc(gltf.PrimitiveTriangleStrip, []uint32{0, 1, 9}, pos3)
	if _, err := modeler.ConvertPrimitiveTopology(oob, 0, 0); !errors.Is(err, modeler.ErrIndexOutOfRange) {
		t.Fatalf("oob err = %v", err)
	}

	for _, mode := range []gltf.PrimitiveMode{gltf.PrimitivePoints, gltf.PrimitiveLines, gltf.PrimitiveMode(42)} {
		bad := uid117Doc(mode, []uint32{0, 1, 2}, pos3)
		if _, err := modeler.ConvertPrimitiveTopology(bad, 0, 0); !errors.Is(err, modeler.ErrInvalidTopology) {
			t.Fatalf("mode %v err = %v", mode, err)
		}
	}

	missing := uid117Doc(gltf.PrimitiveTriangleStrip, []uint32{0, 1, 2}, pos3)
	if _, err := modeler.ConvertPrimitiveTopology(missing, 7, 0); !errors.Is(err, modeler.ErrPrimitiveNotFound) {
		t.Fatalf("missing mesh err = %v", err)
	}
	if _, err := modeler.ConvertPrimitiveTopology(nil, 0, 0); !errors.Is(err, modeler.ErrInvalidDocument) {
		t.Fatalf("nil doc err = %v", err)
	}

	huge := gltf.NewDocument()
	huge.Accessors = []*gltf.Accessor{{ComponentType: gltf.ComponentFloat, Type: gltf.AccessorVec3, Count: math.MaxInt32}}
	huge.Meshes = []*gltf.Mesh{{Primitives: []*gltf.Primitive{{
		Attributes: gltf.PrimitiveAttributes{gltf.POSITION: 0},
		Mode:       gltf.PrimitiveTriangleStrip,
	}}}}
	if _, err := modeler.ConvertPrimitiveTopology(huge, 0, 0); !errors.Is(err, modeler.ErrOutputTooLarge) {
		t.Fatalf("oversized err = %v", err)
	}
}

func TestUID117_GLBRoundTrip(t *testing.T) {
	src := uid117Doc(gltf.PrimitiveTriangleStrip, []uint32{0, 1, 2, 3}, uid117Positions4())
	out, err := modeler.ConvertPrimitiveTopology(src, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	if err := gltf.NewEncoder(&buf).Encode(out); err != nil {
		t.Fatal(err)
	}
	rt := gltf.NewDocument()
	if err := gltf.NewDecoder(bytes.NewReader(buf.Bytes())).Decode(rt); err != nil {
		t.Fatal(err)
	}
	p := rt.Meshes[0].Primitives[0]
	if p.Mode != gltf.PrimitiveTriangles {
		t.Fatalf("rt mode %v", p.Mode)
	}
	if got := uid117Read(t, rt, p.Indices); !reflect.DeepEqual(got, []uint32{0, 1, 2, 2, 1, 3}) {
		t.Fatalf("rt indices %v", got)
	}
}

func TestUID117_SaveAndOpen(t *testing.T) {
	src := uid117Doc(gltf.PrimitiveTriangleFan, []uint32{0, 1, 2, 3}, uid117Positions4())
	out, err := modeler.ConvertPrimitiveTopology(src, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "conv.glb")
	if err := gltf.SaveBinary(out, path); err != nil {
		t.Fatal(err)
	}
	doc, err := gltf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	p := doc.Meshes[0].Primitives[0]
	if got := uid117Read(t, doc, p.Indices); !reflect.DeepEqual(got, []uint32{0, 1, 2, 0, 2, 3}) {
		t.Fatalf("saved indices %v", got)
	}
}

func TestUID117_ApplyInPlace(t *testing.T) {
	src := uid117Doc(gltf.PrimitiveTriangleStrip, []uint32{0, 1, 2, 3}, uid117Positions4())
	if err := modeler.ApplyPrimitiveTopology(src, 0, 0); err != nil {
		t.Fatal(err)
	}
	p := src.Meshes[0].Primitives[0]
	if p.Mode != gltf.PrimitiveTriangles {
		t.Fatal("not converted in place")
	}
	if got := uid117Read(t, src, p.Indices); !reflect.DeepEqual(got, []uint32{0, 1, 2, 2, 1, 3}) {
		t.Fatalf("apply indices %v", got)
	}
}

func TestUID117_OfficialMeshPrimitiveModes(t *testing.T) {
	// Exact index data decoded from the official Khronos
	// glTF-Sample-Assets/Models/MeshPrimitiveModes sample:
	//   TRIANGLES accessor 4, STRIP accessor 5, FAN accessor 6.
	officialTriangles := []uint32{0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 5, 0, 5, 6, 0, 6, 1}
	officialStrip := []uint32{2, 3, 1, 4, 6, 5}
	officialFan := []uint32{0, 1, 2, 3, 4, 5, 6, 1}
	positions := [][3]float32{
		{0, 0, 0}, {0.866, -0.5, 0}, {0.866, 0.5, 0}, {0, 1, 0},
		{-0.866, 0.5, 0}, {-0.866, -0.5, 0}, {0, -1, 0},
	}

	doc := gltf.NewDocument()
	posAcc := modeler.WritePosition(doc, positions)
	triAcc := modeler.WriteIndices(doc, officialTriangles)
	stripAcc := modeler.WriteIndices(doc, officialStrip)
	fanAcc := modeler.WriteIndices(doc, officialFan)
	mk := func(mode gltf.PrimitiveMode, acc int) *gltf.Primitive {
		return &gltf.Primitive{
			Attributes: gltf.PrimitiveAttributes{gltf.POSITION: posAcc},
			Indices:    gltf.Index(acc),
			Mode:       mode,
		}
	}
	doc.Meshes = []*gltf.Mesh{{Primitives: []*gltf.Primitive{
		mk(gltf.PrimitiveTriangles, triAcc),
		mk(gltf.PrimitiveTriangleStrip, stripAcc),
		mk(gltf.PrimitiveTriangleFan, fanAcc),
	}}}

	// Fan expansion must equal the official TRIANGLES index list exactly.
	out, err := modeler.ConvertPrimitiveTopology(doc, 0, 2)
	if err != nil {
		t.Fatal(err)
	}
	if got := uid117Read(t, out, out.Meshes[0].Primitives[2].Indices); !reflect.DeepEqual(got, officialTriangles) {
		t.Fatalf("fan != official triangles:\n got %v\nwant %v", got, officialTriangles)
	}

	// Strip has the known alternating expansion.
	out2, err := modeler.ConvertPrimitiveTopology(doc, 0, 1)
	if err != nil {
		t.Fatal(err)
	}
	if got := uid117Read(t, out2, out2.Meshes[0].Primitives[1].Indices); !reflect.DeepEqual(got, []uint32{2, 3, 1, 1, 3, 4, 1, 4, 6, 6, 4, 5}) {
		t.Fatalf("strip expansion = %v", got)
	}

	// The TRIANGLES primitive stays a no-op and outputs encode.
	out3, err := modeler.ConvertPrimitiveTopology(doc, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if out3.Meshes[0].Primitives[0].Mode != gltf.PrimitiveTriangles {
		t.Fatal("triangles primitive should stay triangles")
	}
	var buf bytes.Buffer
	if err := gltf.NewEncoder(&buf).Encode(out2); err != nil {
		t.Fatalf("encode strip output: %v", err)
	}
}
