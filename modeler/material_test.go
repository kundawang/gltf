package modeler_test

import (
	"math"
	"reflect"
	"testing"

	"github.com/qmuntal/gltf"
	"github.com/qmuntal/gltf/ext/texturetransform"
	"github.com/qmuntal/gltf/modeler"
)

// uvDoc builds a document with two TEXCOORD sets.
// TEXCOORD_0 is a non-interleaved FLOAT VEC2 accessor,
// TEXCOORD_1 is a FLOAT VEC2 accessor interleaved with a padding component.
func uvDoc() (*gltf.Document, *gltf.Primitive) {
	uv0 := []float32{1, 0, 0, 1}
	uv1 := []float32{0.5, 0.5, 0.25, 0.75}
	interleaved := make([]byte, 0, 16*len(uv1)/2)
	// Each vertex occupies 16 bytes: u, v and a 2-float padding gap.
	for i := 0; i < len(uv1); i += 2 {
		interleaved = append(interleaved, float32Bytes(uv1[i], uv1[i+1], 0, 0)...)
	}
	doc := &gltf.Document{
		Buffers: []*gltf.Buffer{
			{ByteLength: len(float32Bytes(uv0...)), Data: float32Bytes(uv0...)},
			{ByteLength: len(interleaved), Data: interleaved},
		},
		BufferViews: []*gltf.BufferView{
			{Buffer: 0, ByteLength: len(float32Bytes(uv0...))},
			{Buffer: 1, ByteLength: len(interleaved), ByteStride: 16},
		},
		Accessors: []*gltf.Accessor{
			{BufferView: gltf.Index(0), ComponentType: gltf.ComponentFloat, Type: gltf.AccessorVec2, Count: 2},
			{BufferView: gltf.Index(1), ComponentType: gltf.ComponentFloat, Type: gltf.AccessorVec2, Count: 2},
		},
	}
	primitive := &gltf.Primitive{
		Attributes: gltf.PrimitiveAttributes{
			gltf.TEXCOORD_0: 0,
			gltf.TEXCOORD_1: 1,
		},
		Material: gltf.Index(0),
	}
	return doc, primitive
}

func baseColorMaterial(ti *gltf.TextureInfo) *gltf.Material {
	return &gltf.Material{
		PBRMetallicRoughness: &gltf.PBRMetallicRoughness{BaseColorTexture: ti},
	}
}

func TestReadMaterialTextureCoord(t *testing.T) {
	doc, primitive := uvDoc()
	tests := []struct {
		name     string
		material *gltf.Material
		slot     modeler.MaterialTextureSlot
		buffer   [][2]float32
		want     [][2]float32
		wantErr  bool
	}{
		{
			name:     "no-extension-keeps-uv",
			material: baseColorMaterial(&gltf.TextureInfo{Index: 0}),
			slot:     modeler.MaterialBaseColorTexture,
			want:     [][2]float32{{1, 0}, {0, 1}},
		},
		{
			name:     "texcoord-selects-second-set",
			material: baseColorMaterial(&gltf.TextureInfo{Index: 0, TexCoord: 1}),
			slot:     modeler.MaterialBaseColorTexture,
			want:     [][2]float32{{0.5, 0.5}, {0.25, 0.75}},
		},
		{
			name: "full-transform-first-set",
			material: baseColorMaterial(&gltf.TextureInfo{
				Index: 0,
				Extensions: gltf.Extensions{
					texturetransform.ExtensionName: &texturetransform.TextureTranform{
						Offset: [2]float64{0.1, 0.2}, Rotation: math.Pi / 2, Scale: [2]float64{2, 3},
					},
				},
			}),
			slot: modeler.MaterialBaseColorTexture,
			want: [][2]float32{{0.1, 2.2}, {-2.9, 0.2}},
		},
		{
			name: "extension-texcoord-overrides-slot-texcoord",
			material: baseColorMaterial(&gltf.TextureInfo{
				Index: 1, TexCoord: 0,
				Extensions: gltf.Extensions{
					texturetransform.ExtensionName: &texturetransform.TextureTranform{
						TexCoord: gltf.Index(1), Scale: texturetransform.DefaultScale,
					},
				},
			}),
			slot: modeler.MaterialBaseColorTexture,
			want: [][2]float32{{0.5, 0.5}, {0.25, 0.75}},
		},
		{
			name: "transform-applied-on-second-set",
			material: baseColorMaterial(&gltf.TextureInfo{
				Index: 0, TexCoord: 1,
				Extensions: gltf.Extensions{
					texturetransform.ExtensionName: &texturetransform.TextureTranform{
						Offset: [2]float64{0.1, 0.2}, Scale: [2]float64{2, 2},
					},
				},
			}),
			slot: modeler.MaterialBaseColorTexture,
			want: [][2]float32{{1.1, 1.2}, {0.6, 1.7}},
		},
		{
			name:     "reuses-buffer",
			material: baseColorMaterial(&gltf.TextureInfo{Index: 0}),
			slot:     modeler.MaterialBaseColorTexture,
			buffer:   make([][2]float32, 2),
			want:     [][2]float32{{1, 0}, {0, 1}},
		},
		{
			name:     "missing-slot",
			material: &gltf.Material{},
			slot:     modeler.MaterialBaseColorTexture,
			wantErr:  true,
		},
		{
			name:     "missing-uv-set",
			material: baseColorMaterial(&gltf.TextureInfo{Index: 0, TexCoord: 7}),
			slot:     modeler.MaterialBaseColorTexture,
			wantErr:  true,
		},
		{
			name: "non-finite-transform",
			material: baseColorMaterial(&gltf.TextureInfo{
				Index: 0,
				Extensions: gltf.Extensions{
					texturetransform.ExtensionName: &texturetransform.TextureTranform{
						Rotation: math.NaN(), Scale: texturetransform.DefaultScale,
					},
				},
			}),
			slot:    modeler.MaterialBaseColorTexture,
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := modeler.ReadMaterialTextureCoord(doc, primitive, tt.material, tt.slot, tt.buffer)
			if (err != nil) != tt.wantErr {
				t.Errorf("ReadMaterialTextureCoord() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("ReadMaterialTextureCoord() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestReadMaterialTextureCoord_Slots(t *testing.T) {
	doc, primitive := uvDoc()
	transform := gltf.Extensions{
		texturetransform.ExtensionName: &texturetransform.TextureTranform{
			Offset: [2]float64{0.1, 0.2}, Rotation: math.Pi / 2, Scale: [2]float64{2, 3},
		},
	}
	material := &gltf.Material{
		PBRMetallicRoughness: &gltf.PBRMetallicRoughness{
			BaseColorTexture:         &gltf.TextureInfo{Index: 0, Extensions: transform},
			MetallicRoughnessTexture: &gltf.TextureInfo{Index: 0, TexCoord: 1},
		},
		NormalTexture:    &gltf.NormalTexture{Index: gltf.Index(0), TexCoord: 1},
		OcclusionTexture: &gltf.OcclusionTexture{Index: gltf.Index(0)},
		EmissiveTexture:  &gltf.TextureInfo{Index: 0, TexCoord: 1, Extensions: transform},
	}
	want := map[modeler.MaterialTextureSlot][][2]float32{
		modeler.MaterialBaseColorTexture:         {{0.1, 2.2}, {-2.9, 0.2}},
		modeler.MaterialMetallicRoughnessTexture: {{0.5, 0.5}, {0.25, 0.75}},
		modeler.MaterialNormalTexture:            {{0.5, 0.5}, {0.25, 0.75}},
		modeler.MaterialOcclusionTexture:         {{1, 0}, {0, 1}},
		modeler.MaterialEmissiveTexture:          {{-1.4, 1.2}, {-2.15, 0.7}},
	}
	for slot, wantUVs := range want {
		t.Run("slot", func(t *testing.T) {
			got, err := modeler.ReadMaterialTextureCoord(doc, primitive, material, slot, nil)
			if err != nil {
				t.Fatalf("slot %d: ReadMaterialTextureCoord() error = %v", slot, err)
			}
			if !reflect.DeepEqual(got, wantUVs) {
				t.Errorf("slot %d: ReadMaterialTextureCoord() = %v, want %v", slot, got, wantUVs)
			}
		})
	}
}

func TestReadMaterialTextureCoord_Sparse(t *testing.T) {
	// Base accessor: all zeroes. Sparse values override the first vertex
	// with (1, 0), matching the UV=(1,0) of the golden case.
	baseData := make([]byte, 16)  // 4 zero float32 components -> 2 VEC2 vertices
	sparseIndices := []byte{0, 0} // uint16 index 0
	sparseValues := float32Bytes(1, 0)
	doc := &gltf.Document{
		Buffers: []*gltf.Buffer{
			{ByteLength: len(baseData), Data: baseData},
			{ByteLength: len(sparseIndices), Data: sparseIndices},
			{ByteLength: len(sparseValues), Data: sparseValues},
		},
		BufferViews: []*gltf.BufferView{
			{Buffer: 0, ByteLength: len(baseData)},
			{Buffer: 1, ByteLength: len(sparseIndices)},
			{Buffer: 2, ByteLength: len(sparseValues)},
		},
		Accessors: []*gltf.Accessor{{
			BufferView:    gltf.Index(0),
			ComponentType: gltf.ComponentFloat,
			Type:          gltf.AccessorVec2,
			Count:         2,
			Sparse: &gltf.Sparse{
				Count:   1,
				Indices: gltf.SparseIndices{BufferView: 1, ComponentType: gltf.ComponentUshort},
				Values:  gltf.SparseValues{BufferView: 2},
			},
		}},
	}
	primitive := &gltf.Primitive{Attributes: gltf.PrimitiveAttributes{gltf.TEXCOORD_0: 0}}
	material := baseColorMaterial(&gltf.TextureInfo{
		Index: 0,
		Extensions: gltf.Extensions{
			texturetransform.ExtensionName: &texturetransform.TextureTranform{
				Offset: [2]float64{0.1, 0.2}, Rotation: math.Pi / 2, Scale: [2]float64{2, 3},
			},
		},
	})
	got, err := modeler.ReadMaterialTextureCoord(doc, primitive, material, modeler.MaterialBaseColorTexture, nil)
	if err != nil {
		t.Fatalf("ReadMaterialTextureCoord() error = %v", err)
	}
	want := [][2]float32{{0.1, 2.2}, {0.1, 0.2}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("ReadMaterialTextureCoord() = %v, want %v", got, want)
	}
}

func TestReadMaterialTextureCoord_NormalizedInteger(t *testing.T) {
	// Unsigned normalized byte UVs: 255 -> 1.0, 0 -> 0.0.
	// VEC2 byte elements are padded to 4 bytes per vertex.
	data := []byte{255, 0, 0, 0, 0, 255, 0, 0}
	doc := &gltf.Document{
		Buffers:     []*gltf.Buffer{{ByteLength: len(data), Data: data}},
		BufferViews: []*gltf.BufferView{{Buffer: 0, ByteLength: len(data)}},
		Accessors:   []*gltf.Accessor{{BufferView: gltf.Index(0), ComponentType: gltf.ComponentUbyte, Type: gltf.AccessorVec2, Count: 2, Normalized: true}},
	}
	primitive := &gltf.Primitive{Attributes: gltf.PrimitiveAttributes{gltf.TEXCOORD_0: 0}}
	material := baseColorMaterial(&gltf.TextureInfo{Index: 0})
	got, err := modeler.ReadMaterialTextureCoord(doc, primitive, material, modeler.MaterialBaseColorTexture, nil)
	if err != nil {
		t.Fatalf("ReadMaterialTextureCoord() error = %v", err)
	}
	want := [][2]float32{{1, 0}, {0, 1}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("ReadMaterialTextureCoord() = %v, want %v", got, want)
	}
}

func TestReadMaterialTextureCoord_Errors(t *testing.T) {
	doc, primitive := uvDoc()
	goodMaterial := baseColorMaterial(&gltf.TextureInfo{Index: 0})
	tests := []struct {
		name      string
		doc       *gltf.Document
		primitive *gltf.Primitive
		material  *gltf.Material
		slot      modeler.MaterialTextureSlot
	}{
		{"nil-primitive", doc, nil, goodMaterial, modeler.MaterialBaseColorTexture},
		{"nil-material", doc, primitive, nil, modeler.MaterialBaseColorTexture},
		{"nil-pbr", doc, primitive, &gltf.Material{}, modeler.MaterialBaseColorTexture},
		{"missing-normal", doc, primitive, &gltf.Material{}, modeler.MaterialNormalTexture},
		{"missing-occlusion", doc, primitive, &gltf.Material{}, modeler.MaterialOcclusionTexture},
		{"missing-emissive", doc, primitive, &gltf.Material{}, modeler.MaterialEmissiveTexture},
		{"unknown-slot", doc, primitive, goodMaterial, modeler.MaterialTextureSlot(255)},
		{"wrong-component-type", doc, &gltf.Primitive{
			Attributes: gltf.PrimitiveAttributes{gltf.TEXCOORD_0: 0},
		}, func() *gltf.Material {
			doc.Accessors[0].ComponentType = gltf.ComponentByte
			return goodMaterial
		}(), modeler.MaterialBaseColorTexture},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := modeler.ReadMaterialTextureCoord(tt.doc, tt.primitive, tt.material, tt.slot, nil); err == nil {
				t.Errorf("ReadMaterialTextureCoord() expected an error, got nil")
			}
		})
	}
}

func TestReadMaterialTextureCoord_NonFiniteUV(t *testing.T) {
	// (NaN, 0) and (1, +Inf) source coordinates must be rejected even without
	// a texture transform.
	data := append(float32Bytes(float32(math.NaN()), 0), float32Bytes(1, float32(math.Inf(1)))...)
	doc := &gltf.Document{
		Buffers:     []*gltf.Buffer{{ByteLength: len(data), Data: data}},
		BufferViews: []*gltf.BufferView{{Buffer: 0, ByteLength: len(data)}},
		Accessors:   []*gltf.Accessor{{BufferView: gltf.Index(0), ComponentType: gltf.ComponentFloat, Type: gltf.AccessorVec2, Count: 2}},
	}
	primitive := &gltf.Primitive{Attributes: gltf.PrimitiveAttributes{gltf.TEXCOORD_0: 0}}
	if _, err := modeler.ReadMaterialTextureCoord(doc, primitive, baseColorMaterial(&gltf.TextureInfo{Index: 0}), modeler.MaterialBaseColorTexture, nil); err == nil {
		t.Error("ReadMaterialTextureCoord() expected an error for non-finite UVs, got nil")
	}
}

func TestReadMaterialTextureCoord_DoesNotMutateDocument(t *testing.T) {
	doc, primitive := uvDoc()
	rawBefore := append([]byte(nil), doc.Buffers[0].Data...)
	material := baseColorMaterial(&gltf.TextureInfo{
		Index: 0,
		Extensions: gltf.Extensions{
			texturetransform.ExtensionName: &texturetransform.TextureTranform{
				Offset: [2]float64{0.1, 0.2}, Rotation: math.Pi / 2, Scale: [2]float64{2, 3},
			},
		},
	})
	got, err := modeler.ReadMaterialTextureCoord(doc, primitive, material, modeler.MaterialBaseColorTexture, nil)
	if err != nil {
		t.Fatalf("ReadMaterialTextureCoord() error = %v", err)
	}
	if !reflect.DeepEqual(doc.Buffers[0].Data, rawBefore) {
		t.Error("ReadMaterialTextureCoord() mutated the shared buffer data")
	}
	// Calling again on a different slot sharing the same accessor must still
	// see the original coordinates.
	again, err := modeler.ReadMaterialTextureCoord(doc, primitive, baseColorMaterial(&gltf.TextureInfo{Index: 0}), modeler.MaterialBaseColorTexture, nil)
	if err != nil {
		t.Fatalf("ReadMaterialTextureCoord() second call error = %v", err)
	}
	if !reflect.DeepEqual(again, [][2]float32{{1, 0}, {0, 1}}) {
		t.Errorf("shared UVs were mutated: got %v", again)
	}
	if reflect.DeepEqual(got, again) {
		t.Error("transformed and raw UVs are unexpectedly equal")
	}
}
