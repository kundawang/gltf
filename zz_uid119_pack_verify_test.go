package gltf_test

// Verification suite for the UID 119 self-contained GLB packer. Every local
// identifier and test name carries the uid119 prefix so an externally supplied
// test file dropped into package gltf_test cannot collide with this suite.

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"io/fs"
	"math"
	"os"
	"strings"
	"testing"
	"testing/fstest"

	"github.com/qmuntal/gltf"
)

// uid119FixedPNG is a deterministic 1x1 RGB PNG (73 bytes, length mod 4 == 1), so
// packing it at an 8-byte boundary also exercises BIN chunk padding.
var uid119FixedPNG = []byte{
	0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52,
	0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
	0xde, 0x00, 0x00, 0x00, 0x10, 0x49, 0x44, 0x41, 0x54, 0x78, 0xda, 0x62, 0xaa, 0x6f, 0x68, 0x04,
	0x04, 0x00, 0x00, 0xff, 0xff, 0x03, 0x0a, 0x01, 0x83, 0xfd, 0x53, 0xfc, 0x3a, 0x00, 0x00, 0x00,
	0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
}

// uid119PackFixture builds a document with two 4-byte external buffers and one
// external PNG, wired through accessors, a mesh, a texture and a material.
func uid119PackFixture() (*gltf.Document, fstest.MapFS, []byte, []byte) {
	first := []byte{0x10, 0x20, 0x30, 0x40}
	second := []byte{0xa0, 0xa1, 0xa2, 0xa3}
	doc := &gltf.Document{
		Asset: gltf.Asset{Version: "2.0"},
		Buffers: []*gltf.Buffer{
			{URI: "buffers/first.bin", ByteLength: len(first)},
			{URI: "second.bin", ByteLength: len(second)},
		},
		BufferViews: []*gltf.BufferView{
			{Buffer: 0, ByteOffset: 0, ByteLength: 4, Target: gltf.TargetArrayBuffer},
			{Buffer: 1, ByteOffset: 0, ByteLength: 4, Target: gltf.TargetElementArrayBuffer},
		},
		Accessors: []*gltf.Accessor{
			{BufferView: gltf.Index(0), ComponentType: gltf.ComponentUbyte, Normalized: true, Count: 1, Type: gltf.AccessorVec4},
			{BufferView: gltf.Index(1), ComponentType: gltf.ComponentUbyte, Count: 1, Type: gltf.AccessorVec4},
		},
		Images:   []*gltf.Image{{Name: "pixel", URI: "textures/pixel.png", MimeType: "image/png"}},
		Samplers: []*gltf.Sampler{{MagFilter: gltf.MagLinear, MinFilter: gltf.MinLinearMipMapLinear}},
		Textures: []*gltf.Texture{{Sampler: gltf.Index(0), Source: gltf.Index(0)}},
		Materials: []*gltf.Material{{
			Name: "packed-mat",
			PBRMetallicRoughness: &gltf.PBRMetallicRoughness{
				BaseColorTexture: &gltf.TextureInfo{Index: 0},
			},
		}},
		Meshes: []*gltf.Mesh{{Primitives: []*gltf.Primitive{{
			Attributes: gltf.PrimitiveAttributes{gltf.COLOR_0: 0, gltf.JOINTS_0: 1},
			Material:   gltf.Index(0),
		}}}},
		Nodes:  []*gltf.Node{{Mesh: gltf.Index(0), Name: "node"}},
		Scenes: []*gltf.Scene{{Nodes: []int{0}}},
		Scene:  gltf.Index(0),
	}
	fsys := fstest.MapFS{
		"buffers/first.bin":  {Data: first},
		"second.bin":         {Data: second},
		"textures/pixel.png": {Data: uid119FixedPNG},
	}
	return doc, fsys, first, second
}

type uid119GLBChunks struct {
	headerLength uint32
	json         []byte
	bin          []byte
}

func uid119ParseGLB(t *testing.T, blob []byte) uid119GLBChunks {
	t.Helper()
	if len(blob) < 12 {
		t.Fatalf("GLB too short: %d bytes", len(blob))
	}
	magic := binary.LittleEndian.Uint32(blob[0:4])
	if magic != 0x46546c67 {
		t.Fatalf("bad glTF magic: %#x", magic)
	}
	version := binary.LittleEndian.Uint32(blob[4:8])
	if version != 2 {
		t.Fatalf("bad GLB version: %d", version)
	}
	total := binary.LittleEndian.Uint32(blob[8:12])
	if int(total) != len(blob) {
		t.Fatalf("GLB header length %d does not match bytes %d", total, len(blob))
	}
	out := uid119GLBChunks{headerLength: total}
	off := 12
	for off+8 <= len(blob) {
		n := int(binary.LittleEndian.Uint32(blob[off : off+4]))
		typ := binary.LittleEndian.Uint32(blob[off+4 : off+8])
		off += 8
		if off+n > len(blob) {
			t.Fatalf("chunk %#x length %d overruns GLB", typ, n)
		}
		chunk := blob[off : off+n]
		switch typ {
		case 0x4e4f534a: // JSON
			if out.json != nil {
				t.Fatal("duplicate JSON chunk")
			}
			out.json = chunk
		case 0x004e4942: // BIN
			if out.bin != nil {
				t.Fatal("duplicate BIN chunk")
			}
			out.bin = chunk
		default:
			t.Fatalf("unexpected chunk type %#x", typ)
		}
		off += n
	}
	if off != len(blob) {
		t.Fatalf("trailing bytes: consumed %d of %d", off, len(blob))
	}
	if out.json == nil {
		t.Fatal("missing JSON chunk")
	}
	return out
}

func uid119CollectURIs(v any, uris *[]string) {
	switch x := v.(type) {
	case map[string]any:
		for k, val := range x {
			if k == "uri" {
				if s, ok := val.(string); ok {
					*uris = append(*uris, s)
				}
			}
			uid119CollectURIs(val, uris)
		}
	case []any:
		for _, e := range x {
			uid119CollectURIs(e, uris)
		}
	}
}

// The headline verification: two 4-byte external buffers plus one fixed PNG
// land at deterministic, byte-exact boundaries in one aligned BIN chunk, no
// external URI survives, and the GLB is re-readable with references intact.
func TestUID119Pack_TwoBuffersAndFixedPNG(t *testing.T) {
	doc, fsys, first, second := uid119PackFixture()

	beforeJSON, err := json.Marshal(doc)
	if err != nil {
		t.Fatal(err)
	}

	blob, err := gltf.PackGLB(doc, fsys)
	if err != nil {
		t.Fatalf("Pack() error = %v", err)
	}

	// The input document must be untouched.
	afterJSON, err := json.Marshal(doc)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(beforeJSON, afterJSON) {
		t.Error("Pack() modified the input document")
	}
	if doc.Buffers[0].URI != "buffers/first.bin" || doc.Buffers[1].URI != "second.bin" {
		t.Error("Pack() rewrote URIs on the input document")
	}
	if doc.BufferViews[0].Buffer != 0 || doc.BufferViews[1].Buffer != 1 {
		t.Error("Pack() rewrote buffer indices on the input document")
	}
	if doc.Images[0].BufferView != nil {
		t.Error("Pack() rewrote the input image")
	}
	// External files are only read, never written.
	if !bytes.Equal(fsys["buffers/first.bin"].Data, first) || !bytes.Equal(fsys["second.bin"].Data, second) {
		t.Error("Pack() modified external file system entries")
	}

	chunks := uid119ParseGLB(t, blob)

	// No URI of any kind points outside the GLB; in this fixture every uri
	// property must be gone.
	var raw map[string]any
	if err := json.Unmarshal(chunks.json, &raw); err != nil {
		t.Fatalf("JSON chunk not valid JSON: %v", err)
	}
	var uris []string
	uid119CollectURIs(raw, &uris)
	if len(uris) != 0 {
		t.Errorf("packed GLB still contains uri properties: %v", uris)
	}
	for _, name := range []string{"first.bin", "second.bin", "pixel.png", "buffers/", "textures/"} {
		if bytes.Contains(chunks.json, []byte(name)) {
			t.Errorf("packed JSON still references external resource %q", name)
		}
	}

	// Expected BIN layout:
	//   [0,4)  first buffer
	//   [4,8)  second buffer
	//   [8,81) fixed PNG (73 bytes)
	//   [81,84) zero padding to 4-byte boundary
	const pngOffset = 8
	pngEnd := pngOffset + len(uid119FixedPNG) // 81
	binPadded := (pngEnd + 3) &^ 3            // 84
	if len(chunks.bin) != binPadded {
		t.Fatalf("BIN chunk length = %d, want aligned %d", len(chunks.bin), binPadded)
	}
	if got := chunks.bin[0:4]; !bytes.Equal(got, first) {
		t.Errorf("first buffer boundary bytes = %v, want %v", got, first)
	}
	if got := chunks.bin[4:8]; !bytes.Equal(got, second) {
		t.Errorf("second buffer boundary bytes = %v, want %v", got, second)
	}
	if got := chunks.bin[pngOffset:pngEnd]; !bytes.Equal(got, uid119FixedPNG) {
		t.Errorf("PNG boundary bytes mismatch (%d bytes)", len(got))
	}
	for i := pngEnd; i < binPadded; i++ {
		if chunks.bin[i] != 0 {
			t.Errorf("BIN padding at %d = %#x, want 0", i, chunks.bin[i])
		}
	}

	// Re-read the GLB and cross-check document-level boundaries and content.
	reread := uid119DecodeGLB(t, blob)
	if len(reread.Buffers) != 1 {
		t.Fatalf("buffers count = %d, want 1", len(reread.Buffers))
	}
	if reread.Buffers[0].URI != "" {
		t.Errorf("merged buffer uri = %q, want empty", reread.Buffers[0].URI)
	}
	if reread.Buffers[0].ByteLength != pngEnd {
		t.Errorf("merged buffer byteLength = %d, want %d (padding excluded)", reread.Buffers[0].ByteLength, pngEnd)
	}
	bin := reread.Buffers[0].Data
	if len(bin) != pngEnd {
		t.Fatalf("decoded BIN data length = %d, want %d", len(bin), pngEnd)
	}
	wantViews := []struct {
		buffer int
		offset int
		length int
	}{
		{0, 0, 4},
		{0, 4, 4},
		{0, pngOffset, len(uid119FixedPNG)}, // image view appended last
	}
	if len(reread.BufferViews) != len(wantViews) {
		t.Fatalf("bufferView count = %d, want %d", len(reread.BufferViews), len(wantViews))
	}
	for i, w := range wantViews {
		bv := reread.BufferViews[i]
		if bv.Buffer != w.buffer || bv.ByteOffset != w.offset || bv.ByteLength != w.length {
			t.Errorf("bufferViews[%d] = {buffer:%d offset:%d len:%d}, want %+v", i, bv.Buffer, bv.ByteOffset, bv.ByteLength, w)
		}
		if got := bin[bv.ByteOffset : bv.ByteOffset+bv.ByteLength]; !bytes.Equal(got, [][]byte{first, second, uid119FixedPNG}[i]) {
			t.Errorf("bufferViews[%d] bytes mismatch", i)
		}
	}

	// Image was rewritten to the appended view.
	img := reread.Images[0]
	if img.URI != "" || img.BufferView == nil || *img.BufferView != 2 {
		t.Fatalf("image not retargeted to bufferView 2: %+v", img)
	}
	if img.MimeType != "image/png" {
		t.Errorf("image mimeType = %q, want image/png", img.MimeType)
	}

	// Vertex bytes survive through the accessors.
	for i, want := range [][]byte{first, second} {
		acc := reread.Accessors[i]
		bv := reread.BufferViews[*acc.BufferView]
		start := bv.ByteOffset + acc.ByteOffset
		if got := bin[start : start+4]; !bytes.Equal(got, want) {
			t.Errorf("accessor %d vertex bytes = %v, want %v", i, got, want)
		}
	}

	// Material -> texture -> image reference chain is intact.
	if reread.Materials[0].PBRMetallicRoughness == nil ||
		reread.Materials[0].PBRMetallicRoughness.BaseColorTexture == nil ||
		reread.Materials[0].PBRMetallicRoughness.BaseColorTexture.Index != 0 {
		t.Fatal("material baseColorTexture reference lost")
	}
	if reread.Textures[0].Source == nil || *reread.Textures[0].Source != 0 {
		t.Fatal("texture source reference lost")
	}
	if reread.Meshes[0].Primitives[0].Material == nil || *reread.Meshes[0].Primitives[0].Material != 0 {
		t.Fatal("primitive material reference lost")
	}
	if got := reread.Meshes[0].Primitives[0].Attributes[gltf.COLOR_0]; got != 0 {
		t.Errorf("COLOR_0 attribute = %d, want 0", got)
	}
	if got := reread.Meshes[0].Primitives[0].Attributes[gltf.JOINTS_0]; got != 1 {
		t.Errorf("JOINTS_0 attribute = %d, want 1", got)
	}
}

func TestUID119Pack_DocumentFormAndWriter(t *testing.T) {
	doc, fsys, _, _ := uid119PackFixture()

	// PackDocument exposes the rewritten in-memory document.
	packedDoc, err := gltf.NewPacker(fsys).PackDocument(doc)
	if err != nil {
		t.Fatal(err)
	}
	if len(packedDoc.Buffers) != 1 || packedDoc.Buffers[0].URI != "" {
		t.Fatalf("PackDocument buffer = %+v", packedDoc.Buffers)
	}
	if packedDoc.BufferViews[1].Buffer != 0 || packedDoc.BufferViews[1].ByteOffset != 4 {
		t.Errorf("second view not retargeted: %+v", packedDoc.BufferViews[1])
	}
	if packedDoc.Images[0].BufferView == nil || *packedDoc.Images[0].BufferView != 2 {
		t.Errorf("image not retargeted: %+v", packedDoc.Images[0])
	}

	// Packer.Pack is the byte-returning entry point.
	packed, err := gltf.NewPacker(fsys).Pack(doc)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.HasPrefix(packed, []byte{0x67, 0x6c, 0x54, 0x46}) { // "glTF"
		t.Error("Packer.Pack did not write a GLB")
	}

	// Pack and PackGLB package functions are identical.
	if also, err := gltf.Pack(doc, fsys); err != nil || !bytes.Equal(also, packed) {
		t.Error("Pack and PackGLB produced different output")
	}

	// PackBinary is the streaming package entry point.
	var buf bytes.Buffer
	if err := gltf.PackBinary(&buf, doc, fsys); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(packed, buf.Bytes()) {
		t.Error("Pack and PackBinary produced different output")
	}

	// Packer.Encode matches as well.
	buf.Reset()
	if err := gltf.NewPacker(fsys).Encode(&buf, doc); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(packed, buf.Bytes()) {
		t.Error("Packer.Pack and Packer.Encode produced different output")
	}
}

func TestUID119Pack_RejectBadPathsAndResources(t *testing.T) {
	jpegLike := []byte{0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10, 'J', 'F', 'I', 'F'}

	tests := []struct {
		name      string
		mutate    func(d *gltf.Document, fsys fstest.MapFS)
		fsys      fstest.MapFS
		nilFS     bool
		wantIs    error
		errSubstr string
	}{
		{
			name: "parent traversal",
			mutate: func(d *gltf.Document, _ fstest.MapFS) {
				d.Buffers[0].URI = "../outside.bin"
			},
			errSubstr: "escapes",
		},
		{
			name: "nested traversal escaping root",
			mutate: func(d *gltf.Document, _ fstest.MapFS) {
				d.Buffers[0].URI = "a/b/../../../../outside.bin"
			},
			errSubstr: "escapes",
		},
		{
			name: "absolute posix path",
			mutate: func(d *gltf.Document, _ fstest.MapFS) {
				d.Buffers[0].URI = "/etc/passwd"
			},
			errSubstr: "absolute path",
		},
		{
			name: "windows drive path",
			mutate: func(d *gltf.Document, _ fstest.MapFS) {
				d.Buffers[0].URI = `C:\Windows\x.bin`
			},
			errSubstr: "scheme",
		},
		{
			name: "backslash traversal",
			mutate: func(d *gltf.Document, _ fstest.MapFS) {
				d.Buffers[0].URI = `..\outside.bin`
			},
			errSubstr: "escapes",
		},
		{
			name: "unc path",
			mutate: func(d *gltf.Document, _ fstest.MapFS) {
				d.Buffers[0].URI = `\\server\share\x.bin`
			},
			errSubstr: "not allowed",
		},
		{
			name: "http scheme",
			mutate: func(d *gltf.Document, _ fstest.MapFS) {
				d.Buffers[0].URI = "http://example.com/a.bin"
			},
			errSubstr: "scheme",
		},
		{
			name:      "missing buffer file",
			mutate:    func(d *gltf.Document, fsys fstest.MapFS) { delete(fsys, "buffers/first.bin") },
			wantIs:    fs.ErrNotExist,
			errSubstr: "first.bin",
		},
		{
			name: "nil fsys for external buffer",
			mutate: func(_ *gltf.Document, _ fstest.MapFS) {
			},
			nilFS:     true,
			wantIs:    fs.ErrNotExist,
			errSubstr: "first.bin",
		},
		{
			name: "buffer too short",
			mutate: func(d *gltf.Document, _ fstest.MapFS) {
				d.Buffers[0].ByteLength = 5
			},
			errSubstr: "length mismatch",
		},
		{
			name: "buffer too long",
			mutate: func(d *gltf.Document, _ fstest.MapFS) {
				d.Buffers[0].ByteLength = 3
			},
			errSubstr: "length mismatch",
		},
		{
			name: "missing image",
			mutate: func(_ *gltf.Document, fsys fstest.MapFS) {
				delete(fsys, "textures/pixel.png")
			},
			wantIs:    fs.ErrNotExist,
			errSubstr: "pixel.png",
		},
		{
			name: "wrong image mime png declared jpeg bytes",
			fsys: fstest.MapFS{
				"buffers/first.bin":  {Data: []byte{1, 2, 3, 4}},
				"second.bin":         {Data: []byte{5, 6, 7, 8}},
				"textures/pixel.png": {Data: jpegLike},
			},
			errSubstr: "wrong image mimeType",
		},
		{
			name: "unsupported image mime",
			mutate: func(d *gltf.Document, _ fstest.MapFS) {
				d.Images[0].MimeType = "image/gif"
			},
			errSubstr: "unsupported image mimeType",
		},
		{
			name: "declared png with non image bytes",
			fsys: fstest.MapFS{
				"buffers/first.bin":  {Data: []byte{1, 2, 3, 4}},
				"second.bin":         {Data: []byte{5, 6, 7, 8}},
				"textures/pixel.png": {Data: []byte("this is not an image")},
			},
			errSubstr: "mimeType",
		},
		{
			name: "bufferView out of bounds",
			mutate: func(d *gltf.Document, _ fstest.MapFS) {
				d.BufferViews[0].ByteOffset = 2
			},
			errSubstr: "out of buffer",
		},
		{
			name: "bufferView buffer index out of range",
			mutate: func(d *gltf.Document, _ fstest.MapFS) {
				d.BufferViews[0].Buffer = 9
			},
			errSubstr: "out of range",
		},
		{
			name: "negative buffer length",
			mutate: func(d *gltf.Document, _ fstest.MapFS) {
				d.Buffers[0].ByteLength = -1
			},
			errSubstr: "negative byteLength",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			doc, defaultFS, _, _ := uid119PackFixture()
			useFS := defaultFS
			if tt.fsys != nil {
				useFS = tt.fsys
			}
			if tt.mutate != nil {
				tt.mutate(doc, defaultFS)
			}
			var fsysArg fs.FS
			if !tt.nilFS {
				fsysArg = useFS
			}
			_, err := gltf.PackGLB(doc, fsysArg)
			if err == nil {
				t.Fatal("Pack() expected error, got nil")
			}
			if tt.errSubstr != "" && !strings.Contains(err.Error(), tt.errSubstr) {
				t.Errorf("error %q does not contain %q", err.Error(), tt.errSubstr)
			}
			if tt.wantIs != nil && !errors.Is(err, tt.wantIs) {
				t.Errorf("error %v does not wrap %v", err, tt.wantIs)
			}
		})
	}

	if _, err := gltf.PackGLB(nil, fstest.MapFS{}); err == nil {
		t.Error("Pack(nil) expected error")
	}
}

func TestUID119Pack_EmbeddedDataURIs(t *testing.T) {
	bin := []byte{0xde, 0xad, 0xbe, 0xef}
	doc := &gltf.Document{
		Asset: gltf.Asset{Version: "2.0"},
		Buffers: []*gltf.Buffer{
			func() *gltf.Buffer {
				b := &gltf.Buffer{ByteLength: 4, Data: bin}
				b.EmbeddedResource()
				return b
			}(),
		},
		BufferViews: []*gltf.BufferView{{Buffer: 0, ByteLength: 4}},
		Images: []*gltf.Image{{
			URI: "data:image/png;base64," + base64.StdEncoding.EncodeToString(uid119FixedPNG),
		}},
	}
	blob, err := gltf.PackGLB(doc, nil) // no fsys: everything is embedded
	if err != nil {
		t.Fatalf("Pack() error = %v", err)
	}
	chunks := uid119ParseGLB(t, blob)
	if !bytes.Equal(chunks.bin[0:4], bin) {
		t.Errorf("data-URI buffer bytes = %v", chunks.bin[0:4])
	}
	const pngOff = 4
	if !bytes.Equal(chunks.bin[pngOff:pngOff+len(uid119FixedPNG)], uid119FixedPNG) {
		t.Error("data-URI image bytes not consolidated into BIN")
	}
	reread := uid119DecodeGLB(t, blob)
	for _, b := range reread.Buffers {
		if b.URI != "" && strings.HasPrefix(b.URI, "data:") {
			t.Error("embedded data URI was not moved into BIN")
		}
	}
	if reread.Images[0].BufferView == nil || reread.Images[0].MimeType != "image/png" || reread.Images[0].URI != "" {
		t.Errorf("data-URI image not retargeted: %+v", reread.Images[0])
	}
}

func uid119DecodeGLB(t *testing.T, blob []byte) *gltf.Document {
	t.Helper()
	doc := new(gltf.Document)
	if err := gltf.NewDecoder(bytes.NewReader(blob)).Decode(doc); err != nil {
		t.Fatalf("re-read failed: %v", err)
	}
	return doc
}

func TestUID119Pack_EmptyResources(t *testing.T) {
	// Completely resource-less document packs to a header+JSON-only GLB that
	// stays re-readable.
	doc := &gltf.Document{
		Asset:  gltf.Asset{Version: "2.0"},
		Images: []*gltf.Image{{Name: "placeholder"}}, // neither uri nor bufferView
	}
	blob, err := gltf.PackGLB(doc, nil)
	if err != nil {
		t.Fatalf("Pack() empty error = %v", err)
	}
	chunks := uid119ParseGLB(t, blob)
	if chunks.bin != nil {
		t.Errorf("empty pack should have no BIN chunk, got %d bytes", len(chunks.bin))
	}
	reread := uid119DecodeGLB(t, blob)
	if len(reread.Buffers) != 0 {
		t.Errorf("empty pack buffers = %d, want 0", len(reread.Buffers))
	}
	if reread.Images[0].Name != "placeholder" {
		t.Error("empty image entry was not preserved")
	}

	// A zero-length placeholder buffer followed by a real one maps correctly.
	real := []byte{9, 8, 7, 6}
	doc2 := &gltf.Document{
		Asset: gltf.Asset{Version: "2.0"},
		Buffers: []*gltf.Buffer{
			{ByteLength: 0},
			{URI: "real.bin", ByteLength: 4},
		},
		BufferViews: []*gltf.BufferView{{Buffer: 1, ByteLength: 4}},
	}
	fsys := fstest.MapFS{"real.bin": {Data: real}}
	blob2, err := gltf.PackGLB(doc2, fsys)
	if err != nil {
		t.Fatalf("Pack() mixed empty error = %v", err)
	}
	reread2 := uid119DecodeGLB(t, blob2)
	bv := reread2.BufferViews[0]
	if bv.Buffer != 0 || bv.ByteOffset != 0 {
		t.Errorf("view = {buffer:%d offset:%d}, want {0 0}", bv.Buffer, bv.ByteOffset)
	}
	if got := reread2.Buffers[0].Data[bv.ByteOffset : bv.ByteOffset+4]; !bytes.Equal(got, real) {
		t.Errorf("real bytes = %v, want %v", got, real)
	}
}

func TestUID119Pack_CubeFixture(t *testing.T) {
	dir := "testdata/Cube/glTF"
	f, err := os.Open(dir + "/Cube.gltf")
	if err != nil {
		t.Skipf("Cube fixture unavailable: %v", err)
	}
	defer f.Close()
	doc := new(gltf.Document)
	if err := gltf.NewDecoderFS(f, os.DirFS(dir)).Decode(doc); err != nil {
		t.Fatalf("decode Cube: %v", err)
	}
	blob, err := gltf.PackGLB(doc, os.DirFS(dir))
	if err != nil {
		t.Fatalf("Pack(Cube) error = %v", err)
	}
	reread := uid119DecodeGLB(t, blob)
	if len(reread.Buffers) != 1 || reread.Buffers[0].URI != "" {
		t.Fatalf("Cube buffers = %+v", reread.Buffers)
	}
	for i, img := range reread.Images {
		if img.URI != "" || img.BufferView == nil {
			t.Errorf("Cube image %d not packed: %+v", i, img)
		}
		if img.MimeType != "image/png" {
			t.Errorf("Cube image %d mime = %q", i, img.MimeType)
		}
	}
	// Every original bufferView still resolves to the exact original bytes.
	orig := doc.Buffers[0].Data
	merged := reread.Buffers[0].Data
	for i := range doc.BufferViews {
		oldBV := doc.BufferViews[i]
		newBV := reread.BufferViews[i]
		want := orig[oldBV.ByteOffset : oldBV.ByteOffset+oldBV.ByteLength]
		got := merged[newBV.ByteOffset : newBV.ByteOffset+newBV.ByteLength]
		if !bytes.Equal(got, want) {
			t.Fatalf("Cube bufferView %d bytes differ after pack", i)
		}
	}
}

// Guard against int overflow in the offset arithmetic helpers through the
// public surface: an astronomical view must be rejected, not panic.
func TestUID119Pack_OversizedBufferViewRejected(t *testing.T) {
	doc, fsys, _, _ := uid119PackFixture()
	doc.BufferViews[0].ByteOffset = math.MaxInt
	doc.BufferViews[0].ByteLength = math.MaxInt
	if _, err := gltf.PackGLB(doc, fsys); err == nil {
		t.Fatal("expected overflow/out-of-bounds error")
	}
}

// A document whose external buffer bytes were already loaded in memory can be
// packed without an fs.FS; a mismatched in-memory length still fails.
func TestUID119Pack_LoadedBufferDataWithoutFS(t *testing.T) {
	data := []byte{1, 2, 3, 4}
	doc := &gltf.Document{
		Asset:       gltf.Asset{Version: "2.0"},
		Buffers:     []*gltf.Buffer{{URI: "a.bin", ByteLength: 4, Data: data}},
		BufferViews: []*gltf.BufferView{{Buffer: 0, ByteLength: 4}},
	}
	blob, err := gltf.PackGLB(doc, nil)
	if err != nil {
		t.Fatalf("Pack() with loaded Data error = %v", err)
	}
	reread := uid119DecodeGLB(t, blob)
	if !bytes.Equal(reread.Buffers[0].Data, data) {
		t.Error("loaded buffer bytes not preserved")
	}

	bad := *doc
	bad.Buffers = []*gltf.Buffer{{URI: "a.bin", ByteLength: 4, Data: []byte{1, 2}}}
	if _, err := gltf.PackGLB(&bad, nil); err == nil {
		t.Error("expected error when neither FS nor matching Data is available")
	}
}
