package gltf

// Internal checks for UID 119 packer helpers. Test names carry the uid119
// prefix to avoid collisions with externally supplied test files.

import (
	"bytes"
	"math"
	"testing"
)

func TestUID119Align4(t *testing.T) {
	cases := []struct {
		in   int
		want int
	}{
		{0, 0},
		{1, 4},
		{3, 4},
		{4, 4},
		{5, 8},
		{73, 76},
	}
	for _, c := range cases {
		got, err := align4(c.in)
		if err != nil || got != c.want {
			t.Errorf("align4(%d) = (%d, %v), want %d", c.in, got, err, c.want)
		}
	}
	if _, err := align4(-1); err == nil {
		t.Error("align4(-1) expected error")
	}
	if _, err := align4(math.MaxInt); err == nil {
		t.Error("align4(MaxInt) expected overflow error")
	}
}

func TestUID119AddOverflow(t *testing.T) {
	if v, bad := addOverflow(4, 8); bad || v != 12 {
		t.Errorf("addOverflow(4,8) = %d,%v", v, bad)
	}
	if _, bad := addOverflow(math.MaxInt, 1); !bad {
		t.Error("addOverflow(MaxInt,1) expected overflow")
	}
	if _, bad := addOverflow(1, -1); !bad {
		t.Error("addOverflow(1,-1) expected invalid")
	}
}

func TestUID119ResolveRelativePath(t *testing.T) {
	ok := map[string]string{
		"a.bin":          "a.bin",
		"./a/b.bin":      "a/b.bin",
		"textures/x.png": "textures/x.png",
		"a/../b.bin":     "b.bin",
		"a%2fb.bin":      "a/b.bin",
	}
	for in, want := range ok {
		got, err := resolveRelativePath(in)
		if err != nil {
			t.Errorf("resolveRelativePath(%q) unexpected error: %v", in, err)
			continue
		}
		if got != want {
			t.Errorf("resolveRelativePath(%q) = %q, want %q", in, got, want)
		}
	}
	bad := []string{
		"",
		"../a.bin",
		"a/../../b.bin",
		"/abs/a.bin",
		`..\a.bin`,
		"C:/x/a.bin",
		"http://x/a.bin",
		"//server/share/a.bin",
		"%2e%2e/a.bin",
	}
	for _, in := range bad {
		if _, err := resolveRelativePath(in); err == nil {
			t.Errorf("resolveRelativePath(%q) expected error", in)
		}
	}
}

func TestUID119SniffImageMIME(t *testing.T) {
	png := []byte{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n', 0, 0}
	jpeg := []byte{0xff, 0xd8, 0xff, 0xe0}
	if got := sniffImageMIME(png); got != packMIMEPNG {
		t.Errorf("png sniff = %q", got)
	}
	if got := sniffImageMIME(jpeg); got != packMIMEJPEG {
		t.Errorf("jpeg sniff = %q", got)
	}
	if got := sniffImageMIME([]byte("not image")); got != "" {
		t.Errorf("text sniff = %q, want empty", got)
	}
}

func TestUID119PackDocumentCompatibleWithStockEncoder(t *testing.T) {
	doc := &Document{
		Asset:       Asset{Version: "2.0"},
		Buffers:     []*Buffer{{ByteLength: 4, Data: []byte{1, 2, 3, 4}}},
		BufferViews: []*BufferView{{Buffer: 0, ByteLength: 4}},
	}
	packed, bin, err := NewPacker(nil).packDocument(doc)
	if err != nil {
		t.Fatalf("packDocument error: %v", err)
	}
	if len(bin) != 4 || len(packed.Buffers) != 1 {
		t.Fatalf("unexpected packed document: bin=%d buffers=%d", len(bin), len(packed.Buffers))
	}
	var buf bytes.Buffer
	if err := NewEncoder(&buf).Encode(packed); err != nil {
		t.Fatalf("stock Encoder failed: %v", err)
	}
	out := new(Document)
	if err := NewDecoder(&buf).Decode(out); err != nil {
		t.Fatalf("stock Decoder failed on encoded packed doc: %v", err)
	}
	if len(out.Buffers) != 1 || !bytes.Equal(out.Buffers[0].Data, []byte{1, 2, 3, 4}) {
		t.Fatalf("round-trip buffer mismatch: %+v", out.Buffers)
	}
}

func TestUID119WritePackedGLBRejectsOversizedLength(t *testing.T) {
	// A buffer claiming more than uint32 cannot be framed as GLB.
	packed := &Document{Buffers: []*Buffer{{ByteLength: math.MaxUint32 + 1}}}
	if err := writePackedGLB(&bytes.Buffer{}, packed, nil); err == nil {
		t.Fatal("expected uint32 overflow error")
	}
}
