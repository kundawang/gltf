package gltf_test

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"strings"
	"testing"

	"github.com/qmuntal/gltf"
)

// ---------- independent matrix helpers used to build expectations ----------

func rpMul(a, b [16]float64) [16]float64 {
	var out [16]float64
	for c := 0; c < 4; c++ {
		for r := 0; r < 4; r++ {
			v := 0.0
			for k := 0; k < 4; k++ {
				v += a[k*4+r] * b[c*4+k]
			}
			out[c*4+r] = v
		}
	}
	return out
}

func rpTRS(t [3]float64, q [4]float64, s [3]float64) [16]float64 {
	x, y, z, w := q[0], q[1], q[2], q[3]
	return [16]float64{
		(1 - 2*(y*y+z*z)) * s[0], (2 * (x*y + w*z)) * s[0], (2 * (x*z - w*y)) * s[0], 0,
		(2 * (x*y - w*z)) * s[1], (1 - 2*(x*x+z*z)) * s[1], (2 * (y*z + w*x)) * s[1], 0,
		(2 * (x*z + w*y)) * s[2], (2 * (y*z - w*x)) * s[2], (1 - 2*(x*x+y*y)) * s[2], 0,
		t[0], t[1], t[2], 1,
	}
}

func rpMatNear(a, b [16]float64, eps float64) bool {
	for i := range a {
		scale := math.Max(1, math.Max(math.Abs(a[i]), math.Abs(b[i])))
		if math.Abs(a[i]-b[i]) > eps*scale {
			return false
		}
	}
	return true
}

func rpVecNear(a, b [3]float64, eps float64) bool {
	for i := 0; i < 3; i++ {
		if math.Abs(a[i]-b[i]) > eps {
			return false
		}
	}
	return true
}

func rpY90() [4]float64 {
	s := math.Sqrt2 / 2
	return [4]float64{0, s, 0, s}
}

func rpPtr(v *int) string {
	if v == nil {
		return "<nil>"
	}
	return fmt.Sprint(*v)
}

// rpFingerprint captures the full hierarchy and every transform field using
// raw float bit patterns, so documents containing NaN/Inf can still be
// compared for atomicity.
func rpFingerprint(doc *gltf.Document) string {
	var b strings.Builder
	for i, n := range doc.Nodes {
		if n == nil {
			b.WriteString("nil;")
			continue
		}
		fmt.Fprintf(&b, "%d %v m:", i, n.Children)
		for _, v := range n.Matrix {
			fmt.Fprintf(&b, "%016x", math.Float64bits(v))
		}
		b.WriteString(" t:")
		for _, v := range n.Translation {
			fmt.Fprintf(&b, "%016x", math.Float64bits(v))
		}
		b.WriteString(" r:")
		for _, v := range n.Rotation {
			fmt.Fprintf(&b, "%016x", math.Float64bits(v))
		}
		b.WriteString(" s:")
		for _, v := range n.Scale {
			fmt.Fprintf(&b, "%016x", math.Float64bits(v))
		}
		fmt.Fprintf(&b, " mesh=%s cam=%s skin=%s name=%s;", rpPtr(n.Mesh), rpPtr(n.Camera), rpPtr(n.Skin), n.Name)
	}
	for _, s := range doc.Scenes {
		if s == nil {
			b.WriteString("nil-scene;")
			continue
		}
		fmt.Fprintf(&b, "scene %v;", s.Nodes)
	}
	return b.String()
}

func rpAssertUnchanged(t *testing.T, doc *gltf.Document, before string) {
	t.Helper()
	if after := rpFingerprint(doc); after != before {
		t.Fatalf("document changed after failed ReparentNode\nbefore: %s\nafter:  %s", before, after)
	}
}

// rpStoredTRS reports whether the node currently stores its transform as TRS
// rather than a raw matrix.
func rpStoredTRS(n *gltf.Node) bool {
	return n.Matrix == gltf.DefaultMatrix || n.Matrix == [16]float64{}
}

// The exact scenario from the task: old parent x=2, child local x=1, new parent
// x=-3. After reparenting the child local x must be 6 and world x must be 3.
func TestReparentNode_TranslationExample(t *testing.T) {
	mesh := gltf.Index(0)
	cam := gltf.Index(0)
	doc := &gltf.Document{
		Scenes: []*gltf.Scene{{Nodes: []int{0, 2}}},
		Nodes: []*gltf.Node{
			{Name: "old", Translation: [3]float64{2, 0, 0}, Children: []int{1}},
			{Name: "child", Translation: [3]float64{1, 0, 0}, Mesh: mesh, Camera: cam},
			{Name: "new", Translation: [3]float64{-3, 0, 0}},
		},
		Meshes:  []*gltf.Mesh{{Name: "m"}},
		Cameras: []*gltf.Camera{{Name: "c"}},
	}

	beforeWorld, err := doc.NodeWorldMatrix(1)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(beforeWorld[12]-3) > 1e-12 {
		t.Fatalf("precondition world x = %v, want 3", beforeWorld[12])
	}

	if err := doc.ReparentNode(0, 1, 2); err != nil {
		t.Fatalf("ReparentNode: %v", err)
	}

	if got := doc.Nodes[0].Children; len(got) != 0 {
		t.Fatalf("old parent children = %v, want empty", got)
	}
	if got := doc.Nodes[2].Children; !reflect.DeepEqual(got, []int{1}) {
		t.Fatalf("new parent children = %v, want [1]", got)
	}
	if got := doc.Scenes[0].Nodes; !reflect.DeepEqual(got, []int{0, 2}) {
		t.Fatalf("scene roots = %v, want [0 2]; the child must not become a root", got)
	}

	child := doc.Nodes[1]
	if math.Abs(child.Translation[0]-6) > 1e-12 {
		t.Fatalf("local x = %v, want 6", child.Translation[0])
	}
	if !rpStoredTRS(child) {
		t.Fatalf("expected TRS storage, got matrix %v", child.Matrix)
	}
	if child.Mesh != mesh || child.Camera != cam {
		t.Fatalf("resource references lost: mesh=%v camera=%v", child.Mesh, child.Camera)
	}
	if child.Name != "child" {
		t.Fatalf("name lost: %q", child.Name)
	}

	afterWorld, err := doc.NodeWorldMatrix(1)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(afterWorld[12]-3) > 1e-12 {
		t.Fatalf("world x after = %v, want 3", afterWorld[12])
	}
	if !rpMatNear(beforeWorld, afterWorld, 1e-12) {
		t.Fatalf("world matrix changed:\nbefore %v\nafter  %v", beforeWorld, afterWorld)
	}

	// The document must still be encodable and the child must serialize as TRS.
	encoded, err := json.Marshal(child)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(encoded), `"matrix"`) {
		t.Fatalf("child encoded with matrix: %s", encoded)
	}
	if !strings.Contains(string(encoded), `"translation":[6,0,0]`) {
		t.Fatalf("child encoded without translation [6,0,0]: %s", encoded)
	}
}

func TestReparentNode_ToRoot(t *testing.T) {
	doc := &gltf.Document{
		Scenes: []*gltf.Scene{{Nodes: []int{0}}},
		Nodes: []*gltf.Node{
			{Translation: [3]float64{2, 0, 0}, Children: []int{1}},
			{Translation: [3]float64{1, 0, 0}},
		},
	}
	if err := doc.ReparentNode(0, 1, gltf.RootParent); err != nil {
		t.Fatal(err)
	}
	if got := doc.Nodes[0].Children; len(got) != 0 {
		t.Fatalf("old parent children = %v, want empty", got)
	}
	if got := doc.Scenes[0].Nodes; !reflect.DeepEqual(got, []int{0, 1}) {
		t.Fatalf("roots = %v, want [0 1]", got)
	}
	if math.Abs(doc.Nodes[1].Translation[0]-3) > 1e-12 {
		t.Fatalf("root local x = %v, want 3 (world preserved)", doc.Nodes[1].Translation[0])
	}
	world, err := doc.NodeWorldMatrix(1)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(world[12]-3) > 1e-12 {
		t.Fatalf("world x = %v, want 3", world[12])
	}

	// Making an existing root a root again is a no-op.
	before := rpFingerprint(doc)
	if err := doc.ReparentNode(0, 1, gltf.RootParent); err != nil {
		t.Fatal(err)
	}
	rpAssertUnchanged(t, doc, before)
}

// The old parent is rotated 90 degrees about Y, so the child at local (1,0,0)
// sits at world (0,0,-1) with a Ry90 orientation. Reparenting under a parent at
// (-3,0,0) keeps the Ry90 orientation and gives a local (3,0,-1).
func TestReparentNode_RotatedParent(t *testing.T) {
	doc := &gltf.Document{
		Scenes: []*gltf.Scene{{Nodes: []int{0, 2}}},
		Nodes: []*gltf.Node{
			{Rotation: rpY90(), Children: []int{1}},
			{Translation: [3]float64{1, 0, 0}},
			{Translation: [3]float64{-3, 0, 0}},
		},
	}
	wantWorld := rpTRS([3]float64{0, 0, -1}, rpY90(), gltf.DefaultScale)

	before, err := doc.NodeWorldMatrix(1)
	if err != nil {
		t.Fatal(err)
	}
	if !rpMatNear(before, wantWorld, 1e-12) {
		t.Fatalf("precondition world = %v, want %v", before, wantWorld)
	}
	if pos := [3]float64{before[12], before[13], before[14]}; !rpVecNear(pos, [3]float64{0, 0, -1}, 1e-12) {
		t.Fatalf("precondition position = %v, want (0,0,-1)", pos)
	}

	if err := doc.ReparentNode(0, 1, 2); err != nil {
		t.Fatal(err)
	}
	child := doc.Nodes[1]
	if !rpVecNear(child.Translation, [3]float64{3, 0, -1}, 1e-12) {
		t.Fatalf("local translation = %v, want (3,0,-1)", child.Translation)
	}
	if math.Abs(child.Rotation[1]-math.Sqrt2/2) > 1e-12 || math.Abs(child.Rotation[3]-math.Sqrt2/2) > 1e-12 {
		t.Fatalf("rotation = %v, want Ry90", child.Rotation)
	}
	after, err := doc.NodeWorldMatrix(1)
	if err != nil {
		t.Fatal(err)
	}
	if !rpMatNear(after, before, 1e-12) {
		t.Fatalf("world changed after reparent:\nbefore %v\nafter  %v", before, after)
	}
}

// Both parents rotated and scaled, the child itself also rotated. The local
// result may need a raw matrix; either representation must keep the world
// matrix intact.
func TestReparentNode_RotatedAndScaledParents(t *testing.T) {
	h := math.Pi / 8
	q45 := [4]float64{0, 0, math.Sin(h), math.Cos(h)}
	doc := &gltf.Document{
		Scenes: []*gltf.Scene{{Nodes: []int{0, 2}}},
		Nodes: []*gltf.Node{
			{Translation: [3]float64{1, 2, 3}, Rotation: rpY90(), Scale: [3]float64{2, 2, 2}, Children: []int{1}},
			{Translation: [3]float64{4, -1, 2}, Rotation: q45},
			{Translation: [3]float64{-5, 0, 1}, Rotation: q45, Scale: [3]float64{1.5, 1.5, 1.5}},
		},
	}
	before, err := doc.NodeWorldMatrix(1)
	if err != nil {
		t.Fatal(err)
	}
	if err := doc.ReparentNode(0, 1, 2); err != nil {
		t.Fatal(err)
	}
	after, err := doc.NodeWorldMatrix(1)
	if err != nil {
		t.Fatal(err)
	}
	if !rpMatNear(after, before, 1e-9) {
		t.Fatalf("world changed:\nbefore %v\nafter  %v", before, after)
	}

	// The local matrix must be exactly inv(worldNewParent) * worldNode.
	local, err := doc.Nodes[1].LocalMatrix()
	if err != nil {
		t.Fatal(err)
	}
	wantLocal := rpMul(rpInverse(t, rpTRS([3]float64{-5, 0, 1}, q45, [3]float64{1.5, 1.5, 1.5})), before)
	if !rpMatNear(local, wantLocal, 1e-9) {
		t.Fatalf("local matrix = %v, want %v", local, wantLocal)
	}
}

func rpInverse(t *testing.T, m [16]float64) [16]float64 {
	t.Helper()
	var a [4][8]float64
	for r := 0; r < 4; r++ {
		for c := 0; c < 4; c++ {
			a[r][c] = m[c*4+r]
		}
		a[r][r+4] = 1
	}
	for c := 0; c < 4; c++ {
		p := c
		for r := c + 1; r < 4; r++ {
			if math.Abs(a[r][c]) > math.Abs(a[p][c]) {
				p = r
			}
		}
		if a[p][c] == 0 {
			t.Fatalf("singular matrix in test inverse")
		}
		a[c], a[p] = a[p], a[c]
		d := a[c][c]
		for j := 0; j < 8; j++ {
			a[c][j] /= d
		}
		for r := 0; r < 4; r++ {
			if r == c {
				continue
			}
			f := a[r][c]
			for j := 0; j < 8; j++ {
				a[r][j] -= f * a[c][j]
			}
		}
	}
	var out [16]float64
	for r := 0; r < 4; r++ {
		for c := 0; c < 4; c++ {
			out[c*4+r] = a[r][c+4]
		}
	}
	return out
}

func TestReparentNode_PreservesSubtreeAndReferences(t *testing.T) {
	doc := &gltf.Document{
		Scenes: []*gltf.Scene{{Nodes: []int{0, 4}}},
		Nodes: []*gltf.Node{
			{Translation: [3]float64{2, 0, 0}, Children: []int{1}},
			{Translation: [3]float64{1, 0, 0}, Rotation: rpY90(), Children: []int{2, 3},
				Weights: []float64{0.25}, Skin: gltf.Index(0), Extras: map[string]any{"k": "v"}},
			{Translation: [3]float64{0, 1, 0}, Mesh: gltf.Index(0)},
			{Translation: [3]float64{0, 0, 1}, Camera: gltf.Index(0)},
			{Translation: [3]float64{-3, 4, 0}},
		},
		Meshes:  []*gltf.Mesh{{}},
		Cameras: []*gltf.Camera{{}},
		Skins:   []*gltf.Skin{{Joints: []int{1}}},
	}
	before := map[int][16]float64{}
	for _, i := range []int{1, 2, 3} {
		world, err := doc.NodeWorldMatrix(i)
		if err != nil {
			t.Fatal(err)
		}
		before[i] = world
	}
	if err := doc.ReparentNode(0, 1, 4); err != nil {
		t.Fatal(err)
	}
	for _, i := range []int{1, 2, 3} {
		after, err := doc.NodeWorldMatrix(i)
		if err != nil {
			t.Fatal(err)
		}
		if !rpMatNear(after, before[i], 1e-9) {
			t.Fatalf("node %d world changed:\n%v\n%v", i, before[i], after)
		}
	}
	if doc.Nodes[2].Mesh == nil || *doc.Nodes[2].Mesh != 0 {
		t.Fatal("mesh reference lost")
	}
	if doc.Nodes[3].Camera == nil || *doc.Nodes[3].Camera != 0 {
		t.Fatal("camera reference lost")
	}
	moved := doc.Nodes[1]
	if moved.Skin == nil || *moved.Skin != 0 {
		t.Fatal("skin reference lost")
	}
	if !reflect.DeepEqual(moved.Weights, []float64{0.25}) {
		t.Fatalf("morph weights lost: %v", moved.Weights)
	}
	if got := moved.Extras.(map[string]any)["k"]; got != "v" {
		t.Fatalf("extras lost: %v", moved.Extras)
	}
	if got := doc.Nodes[0].Children; len(got) != 0 {
		t.Fatalf("old parent children = %v, want empty", got)
	}
}

// A child whose local transform contains shear cannot be expressed as TRS.
// After reparenting it must be stored as a raw matrix with TRS reset, and its
// world transform must be preserved.
func TestReparentNode_MatrixFallbackForShear(t *testing.T) {
	sheared := [16]float64{
		1, 0, 0, 0,
		0.5, 1, 0, 0, // second column sheared into x
		0, 0, 1, 0,
		2, 3, 4, 1,
	}
	doc := &gltf.Document{
		Scenes: []*gltf.Scene{{Nodes: []int{0, 2}}},
		Nodes: []*gltf.Node{
			{Translation: [3]float64{7, 0, 0}, Children: []int{1}},
			{Matrix: sheared},
			{Rotation: rpY90()},
		},
	}
	before, err := doc.NodeWorldMatrix(1)
	if err != nil {
		t.Fatal(err)
	}
	if err := doc.ReparentNode(0, 1, 2); err != nil {
		t.Fatalf("sheared reparent should be allowed via matrix: %v", err)
	}
	n := doc.Nodes[1]
	if rpStoredTRS(n) {
		t.Fatalf("expected raw Matrix storage, got %v", n.Matrix)
	}
	if n.Translation != gltf.DefaultTranslation || n.Rotation != gltf.DefaultRotation || n.Scale != gltf.DefaultScale {
		t.Fatalf("TRS must be reset when storing a matrix: t=%v r=%v s=%v", n.Translation, n.Rotation, n.Scale)
	}
	after, err := doc.NodeWorldMatrix(1)
	if err != nil {
		t.Fatal(err)
	}
	if !rpMatNear(after, before, 1e-9) {
		t.Fatalf("sheared world changed:\n%v\n%v", before, after)
	}

	// The document must encode: a node with both matrix and TRS would be invalid.
	encoded, err := json.Marshal(n)
	if err != nil {
		t.Fatalf("marshal after matrix fallback: %v", err)
	}
	if !strings.Contains(string(encoded), `"matrix"`) {
		t.Fatalf("expected matrix in JSON: %s", encoded)
	}
	for _, field := range []string{`"translation"`, `"rotation"`, `"scale"`} {
		if strings.Contains(string(encoded), field) {
			t.Fatalf("matrix node also encoded %s: %s", field, encoded)
		}
	}

	var decoded gltf.Document
	full, err := json.Marshal(doc)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(full, &decoded); err != nil {
		t.Fatal(err)
	}
	if !rpMatNear(decoded.Nodes[1].Matrix, n.Matrix, 1e-12) {
		t.Fatalf("decoded matrix %v, want %v", decoded.Nodes[1].Matrix, n.Matrix)
	}
}

func TestReparentNode_MovesMatrixNode(t *testing.T) {
	// A node with a raw matrix is moved under a rotated parent.
	raw := [16]float64{
		0, 1, 0, 0,
		-1, 0, 0, 0,
		0, 0, 1, 0,
		1, 2, 3, 1,
	}
	doc := &gltf.Document{
		Scenes: []*gltf.Scene{{Nodes: []int{0, 2}}},
		Nodes: []*gltf.Node{
			{Rotation: rpY90(), Children: []int{1}},
			{Matrix: raw},
			{Translation: [3]float64{3, -2, 5}, Scale: [3]float64{2, 2, 2}},
		},
	}
	before, err := doc.NodeWorldMatrix(1)
	if err != nil {
		t.Fatal(err)
	}
	if err := doc.ReparentNode(0, 1, 2); err != nil {
		t.Fatal(err)
	}
	after, err := doc.NodeWorldMatrix(1)
	if err != nil {
		t.Fatal(err)
	}
	if !rpMatNear(after, before, 1e-9) {
		t.Fatalf("world changed:\n%v\n%v", before, after)
	}
}

// TestReparentNode_ErrorsAreAtomic verifies that invalid inputs never mutate
// the document.
func TestReparentNode_ErrorsAreAtomic(t *testing.T) {
	basic := func() *gltf.Document {
		return &gltf.Document{
			Scenes: []*gltf.Scene{{Nodes: []int{0, 2}}},
			Nodes: []*gltf.Node{
				{Translation: [3]float64{2, 0, 0}, Children: []int{1}},
				{Translation: [3]float64{1, 0, 0}},
				{Translation: [3]float64{-3, 0, 0}},
			},
		}
	}
	chain := func() *gltf.Document {
		return &gltf.Document{
			Scenes: []*gltf.Scene{{Nodes: []int{0}}},
			Nodes: []*gltf.Node{
				{Children: []int{1}},
				{Children: []int{2}},
				{},
			},
		}
	}
	multiParent := func() *gltf.Document {
		return &gltf.Document{
			Scenes: []*gltf.Scene{{Nodes: []int{0, 2}}},
			Nodes: []*gltf.Node{
				{Children: []int{1}},
				{Translation: [3]float64{1, 0, 0}},
				{Children: []int{1}},
			},
		}
	}
	rootAndChild := func() *gltf.Document {
		return &gltf.Document{
			Scenes: []*gltf.Scene{{Nodes: []int{0, 1}}},
			Nodes: []*gltf.Node{
				{Children: []int{1}},
				{Translation: [3]float64{1, 0, 0}},
			},
		}
	}
	singular := func() *gltf.Document {
		return &gltf.Document{
			Scenes: []*gltf.Scene{{Nodes: []int{0, 2}}},
			Nodes: []*gltf.Node{
				{Children: []int{1}},
				{Translation: [3]float64{1, 0, 0}},
				{Scale: [3]float64{0, 1, 1}},
			},
		}
	}
	nonFiniteParent := func() *gltf.Document {
		return &gltf.Document{
			Scenes: []*gltf.Scene{{Nodes: []int{0, 2}}},
			Nodes: []*gltf.Node{
				{Children: []int{1}},
				{Translation: [3]float64{1, 0, 0}},
				{Rotation: [4]float64{math.NaN(), 0, 0, 1}},
			},
		}
	}
	nonFiniteResult := func() *gltf.Document {
		return &gltf.Document{
			Scenes: []*gltf.Scene{{Nodes: []int{0, 2}}},
			Nodes: []*gltf.Node{
				{Children: []int{1}},
				{Translation: [3]float64{1e200, 0, 0}},
				{Scale: [3]float64{1e-300, 1, 1}},
			},
		}
	}
	tests := []struct {
		name   string
		doc    func() *gltf.Document
		scene  int
		node   int
		parent int
	}{
		{"scene negative", basic, -2, 1, 2},
		{"scene too large", basic, 4, 1, 2},
		{"node negative", basic, 0, -1, 2},
		{"node too large", basic, 0, 9, 2},
		{"parent too large", basic, 0, 1, 9},
		{"parent negative", basic, 0, 1, -5},
		{"parent is self", chain, 0, 1, 1},
		{"parent is child", chain, 0, 0, 1},
		{"parent is grandchild", chain, 0, 0, 2},
		{"multiple node parents", multiParent, 0, 1, 2},
		{"root and child", rootAndChild, 0, 1, gltf.RootParent},
		{"singular parent", singular, 0, 1, 2},
		{"non finite parent", nonFiniteParent, 0, 1, 2},
		{"non finite result", nonFiniteResult, 0, 1, 2},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			doc := tt.doc()
			before := rpFingerprint(doc)
			if err := doc.ReparentNode(tt.scene, tt.node, tt.parent); err == nil {
				t.Fatal("expected error, got nil")
			}
			rpAssertUnchanged(t, doc, before)
		})
	}

	if err := gltf.ReparentNode(nil, 0, 1, 2); err == nil {
		t.Fatal("expected error for a nil document")
	}
}

func TestReparentNode_NoOpSameParent(t *testing.T) {
	doc := &gltf.Document{
		Scenes: []*gltf.Scene{{Nodes: []int{0}}},
		Nodes: []*gltf.Node{
			{Translation: [3]float64{2, 0, 0}, Children: []int{1, 4}},
			{Translation: [3]float64{1, 0, 0}},
			{},
			{},
			{Translation: [3]float64{9, 0, 0}},
		},
	}
	before := rpFingerprint(doc)
	if err := doc.ReparentNode(0, 1, 0); err != nil {
		t.Fatal(err)
	}
	rpAssertUnchanged(t, doc, before)
	if got := doc.Nodes[0].Children; !reflect.DeepEqual(got, []int{1, 4}) {
		t.Fatalf("children reordered by no-op: %v", got)
	}
}

// Sibling order must be preserved when one child is removed.
func TestReparentNode_PreservesSiblingOrder(t *testing.T) {
	doc := &gltf.Document{
		Scenes: []*gltf.Scene{{Nodes: []int{0, 4}}},
		Nodes: []*gltf.Node{
			{Children: []int{1, 2, 3}},
			{Translation: [3]float64{1, 0, 0}},
			{Translation: [3]float64{2, 0, 0}},
			{Translation: [3]float64{3, 0, 0}},
			{Translation: [3]float64{-3, 0, 0}},
		},
	}
	if err := doc.ReparentNode(0, 2, 4); err != nil {
		t.Fatal(err)
	}
	if got := doc.Nodes[0].Children; !reflect.DeepEqual(got, []int{1, 3}) {
		t.Fatalf("remaining children = %v, want [1 3]", got)
	}
	if got := doc.Nodes[4].Children; !reflect.DeepEqual(got, []int{2}) {
		t.Fatalf("new parent children = %v, want [2]", got)
	}
}

// Moving a node under one of its ancestors (going up the hierarchy) must work.
func TestReparentNode_UnderAncestor(t *testing.T) {
	doc := &gltf.Document{
		Scenes: []*gltf.Scene{{Nodes: []int{0}}},
		Nodes: []*gltf.Node{
			{Translation: [3]float64{1, 0, 0}, Children: []int{1}},
			{Translation: [3]float64{2, 0, 0}, Rotation: rpY90(), Children: []int{2}},
			{Translation: [3]float64{3, 1, 0}, Mesh: gltf.Index(0)},
		},
		Meshes: []*gltf.Mesh{{}},
	}
	before, err := doc.NodeWorldMatrix(2)
	if err != nil {
		t.Fatal(err)
	}
	if err := doc.ReparentNode(0, 2, 0); err != nil {
		t.Fatal(err)
	}
	if got := doc.Nodes[1].Children; len(got) != 0 {
		t.Fatalf("middle node children = %v, want empty", got)
	}
	if got := doc.Nodes[0].Children; !reflect.DeepEqual(got, []int{1, 2}) {
		t.Fatalf("ancestor children = %v, want [1 2]", got)
	}
	after, err := doc.NodeWorldMatrix(2)
	if err != nil {
		t.Fatal(err)
	}
	if !rpMatNear(after, before, 1e-9) {
		t.Fatalf("world changed:\n%v\n%v", before, after)
	}
}

func TestReparentNode_FromSceneRootToParent(t *testing.T) {
	doc := &gltf.Document{
		Scenes: []*gltf.Scene{{Nodes: []int{0, 1}}},
		Nodes: []*gltf.Node{
			{Translation: [3]float64{5, 0, 0}},
			{Translation: [3]float64{1, 0, 0}},
		},
	}
	if err := doc.ReparentNode(0, 1, 0); err != nil {
		t.Fatal(err)
	}
	if got := doc.Scenes[0].Nodes; !reflect.DeepEqual(got, []int{0}) {
		t.Fatalf("roots = %v, want [0]", got)
	}
	if got := doc.Nodes[0].Children; !reflect.DeepEqual(got, []int{1}) {
		t.Fatalf("parent children = %v, want [1]", got)
	}
	if math.Abs(doc.Nodes[1].Translation[0]+4) > 1e-12 {
		t.Fatalf("local x = %v, want -4", doc.Nodes[1].Translation[0])
	}
	world, err := doc.NodeWorldMatrix(1)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(world[12]-1) > 1e-12 {
		t.Fatalf("world x = %v, want 1", world[12])
	}
}

// Attaching a previous orphan preserves its world transform and registers it
// under the new parent.
func TestReparentNode_OrphanToParent(t *testing.T) {
	doc := &gltf.Document{
		Scenes: []*gltf.Scene{{Nodes: []int{0}}},
		Nodes: []*gltf.Node{
			{Translation: [3]float64{1, 2, 3}},
			{Translation: [3]float64{-1, 0, 4}, Rotation: rpY90()},
		},
	}
	before, err := doc.NodeWorldMatrix(1)
	if err != nil {
		t.Fatal(err)
	}
	if err := doc.ReparentNode(0, 1, 0); err != nil {
		t.Fatal(err)
	}
	after, err := doc.NodeWorldMatrix(1)
	if err != nil {
		t.Fatal(err)
	}
	if !rpMatNear(after, before, 1e-12) {
		t.Fatalf("world changed:\n%v\n%v", before, after)
	}
	if got := doc.Nodes[0].Children; !reflect.DeepEqual(got, []int{1}) {
		t.Fatalf("parent children = %v, want [1]", got)
	}
}

// Reparenting must work on documents decoded from JSON, where Node.Matrix is
// pre-filled with the identity default.
func TestReparentNode_EncodedDocument(t *testing.T) {
	src := `{
		"asset": {"version": "2.0"},
		"scenes": [{"nodes": [0, 2]}],
		"nodes": [
			{"translation": [2, 0, 0], "children": [1]},
			{"translation": [1, 0, 0], "mesh": 0},
			{"translation": [-3, 0, 0]}
		],
		"meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}]
	}`
	var doc gltf.Document
	if err := json.Unmarshal([]byte(src), &doc); err != nil {
		t.Fatal(err)
	}
	before, err := doc.NodeWorldMatrix(1)
	if err != nil {
		t.Fatal(err)
	}
	if err := doc.ReparentNode(0, 1, 2); err != nil {
		t.Fatal(err)
	}
	after, err := doc.NodeWorldMatrix(1)
	if err != nil {
		t.Fatal(err)
	}
	if !rpMatNear(after, before, 1e-12) {
		t.Fatalf("world changed:\n%v\n%v", before, after)
	}

	out, err := json.Marshal(&doc)
	if err != nil {
		t.Fatal(err)
	}
	var again gltf.Document
	if err := json.Unmarshal(out, &again); err != nil {
		t.Fatalf("re-decode %s: %v", out, err)
	}
	if math.Abs(again.Nodes[1].Translation[0]-6) > 1e-12 {
		t.Fatalf("decoded local x = %v, want 6; json=%s", again.Nodes[1].Translation[0], out)
	}
	if again.Nodes[1].Mesh == nil || *again.Nodes[1].Mesh != 0 {
		t.Fatal("mesh reference lost through JSON")
	}
}

func TestNodeWorldMatrix_Errors(t *testing.T) {
	if err := gltf.ReparentNode(nil, 0, 0, 0); err == nil {
		t.Fatal("expected error from nil document")
	}
	var nilDoc *gltf.Document
	if _, err := nilDoc.NodeWorldMatrix(0); err == nil {
		t.Fatal("expected error from nil document world matrix")
	}
	var nilNode *gltf.Node
	if _, err := nilNode.LocalMatrix(); err == nil {
		t.Fatal("expected error from nil node local matrix")
	}
	doc := &gltf.Document{Nodes: []*gltf.Node{{}, {}, {}}}
	if _, err := doc.NodeWorldMatrix(-1); err == nil {
		t.Fatal("expected error for negative node index")
	}
	if _, err := doc.NodeWorldMatrix(3); err == nil {
		t.Fatal("expected error for out of range node index")
	}

	cycle := &gltf.Document{Nodes: []*gltf.Node{{Children: []int{1}}, {Children: []int{0}}}}
	if _, err := cycle.NodeWorldMatrix(0); err == nil {
		t.Fatal("expected error for hierarchy cycle")
	}

	multi := &gltf.Document{Nodes: []*gltf.Node{{Children: []int{2}}, {Children: []int{2}}, {}}}
	if _, err := multi.NodeWorldMatrix(2); err == nil {
		t.Fatal("expected error for multiple parents")
	}

	bad := &gltf.Document{Nodes: []*gltf.Node{{Translation: [3]float64{math.Inf(1), 0, 0}}}}
	if _, err := bad.NodeWorldMatrix(0); err == nil {
		t.Fatal("expected error for non-finite transform")
	}
}

// Property test: for random hierarchies the world matrix of the moved node and
// of its subtree is invariant.
func TestReparentNode_RandomizedWorldPreservation(t *testing.T) {
	r := rand.New(rand.NewSource(42))
	randQuat := func() [4]float64 {
		ax, ay, az := r.NormFloat64(), r.NormFloat64(), r.NormFloat64()
		l := math.Sqrt(ax*ax + ay*ay + az*az)
		if l < 1e-9 {
			return gltf.DefaultRotation
		}
		h := r.Float64() * math.Pi
		s := math.Sin(h)
		return [4]float64{ax / l * s, ay / l * s, az / l * s, math.Cos(h)}
	}
	randPos := func() [3]float64 {
		return [3]float64{(r.Float64() - .5) * 20, (r.Float64() - .5) * 20, (r.Float64() - .5) * 20}
	}
	randScale := func() [3]float64 {
		return [3]float64{.5 + r.Float64()*2, .5 + r.Float64()*2, .5 + r.Float64()*2}
	}

	for iter := 0; iter < 300; iter++ {
		doc := &gltf.Document{
			Scenes: []*gltf.Scene{{Nodes: []int{0, 2}}},
			Nodes: []*gltf.Node{
				{Translation: randPos(), Rotation: randQuat(), Scale: randScale(), Children: []int{1}},
				{Translation: randPos(), Rotation: randQuat(), Scale: randScale(), Children: []int{3}},
				{Translation: randPos(), Rotation: randQuat(), Scale: randScale()},
				{Translation: randPos(), Rotation: randQuat(), Scale: randScale(), Mesh: gltf.Index(0)},
			},
			Meshes: []*gltf.Mesh{{}},
		}
		before, err := doc.NodeWorldMatrix(1)
		if err != nil {
			t.Fatalf("iter %d: %v", iter, err)
		}
		beforeChild, err := doc.NodeWorldMatrix(3)
		if err != nil {
			t.Fatalf("iter %d: %v", iter, err)
		}

		if err := doc.ReparentNode(0, 1, 2); err != nil {
			t.Fatalf("iter %d: %v", iter, err)
		}
		after, err := doc.NodeWorldMatrix(1)
		if err != nil {
			t.Fatalf("iter %d: %v", iter, err)
		}
		afterChild, err := doc.NodeWorldMatrix(3)
		if err != nil {
			t.Fatalf("iter %d: %v", iter, err)
		}
		if !rpMatNear(after, before, 1e-9) {
			t.Fatalf("iter %d world mismatch:\nbefore %v\nafter  %v", iter, before, after)
		}
		if !rpMatNear(afterChild, beforeChild, 1e-9) {
			t.Fatalf("iter %d subtree world mismatch:\nbefore %v\nafter  %v", iter, beforeChild, afterChild)
		}

		// The stored local matrix must be inverse(worldNewParent) * worldNode.
		local, err := doc.Nodes[1].LocalMatrix()
		if err != nil {
			t.Fatalf("iter %d: %v", iter, err)
		}
		wantLocal := rpMul(rpInverse(t, rpTRS(doc.Nodes[2].Translation, doc.Nodes[2].Rotation, doc.Nodes[2].Scale)), before)
		if !rpMatNear(local, wantLocal, 1e-9) {
			t.Fatalf("iter %d local mismatch:\nwant %v\ngot  %v", iter, wantLocal, local)
		}
		if doc.Nodes[3].Mesh == nil {
			t.Fatalf("iter %d: mesh reference lost", iter)
		}

		// Move it back to the root; the world must still be preserved.
		if err := doc.ReparentNode(0, 1, gltf.RootParent); err != nil {
			t.Fatalf("iter %d to root: %v", iter, err)
		}
		rootWorld, err := doc.NodeWorldMatrix(1)
		if err != nil {
			t.Fatalf("iter %d: %v", iter, err)
		}
		if !rpMatNear(rootWorld, before, 1e-9) {
			t.Fatalf("iter %d root world mismatch:\nbefore %v\nafter  %v", iter, before, rootWorld)
		}
	}
}

// A node that is also a root of another scene still has a single parent in the
// target scene, so reparenting it there is well defined.
func TestReparentNode_NodeAlsoRootOfAnotherScene(t *testing.T) {
	doc := &gltf.Document{
		Scenes: []*gltf.Scene{{Nodes: []int{0}}, {Nodes: []int{1}}},
		Nodes: []*gltf.Node{
			{Translation: [3]float64{5, 0, 0}},
			{Translation: [3]float64{1, 0, 0}, Children: []int{2}},
			{Translation: [3]float64{2, 0, 0}, Mesh: gltf.Index(0)},
		},
		Meshes: []*gltf.Mesh{{}},
	}
	before, err := doc.NodeWorldMatrix(2)
	if err != nil {
		t.Fatal(err)
	}
	if err := doc.ReparentNode(0, 2, 0); err != nil {
		t.Fatal(err)
	}
	if got := doc.Nodes[1].Children; len(got) != 0 {
		t.Fatalf("old parent children = %v, want empty", got)
	}
	if got := doc.Nodes[0].Children; !reflect.DeepEqual(got, []int{2}) {
		t.Fatalf("new parent children = %v, want [2]", got)
	}
	if got := doc.Scenes[1].Nodes; !reflect.DeepEqual(got, []int{1}) {
		t.Fatalf("scene 1 roots = %v, want [1] unchanged", got)
	}
	after, err := doc.NodeWorldMatrix(2)
	if err != nil {
		t.Fatal(err)
	}
	if !rpMatNear(after, before, 1e-12) {
		t.Fatalf("world changed:\n%v\n%v", before, after)
	}
}

func TestReparentNode_NegativeScale(t *testing.T) {
	doc := &gltf.Document{
		Scenes: []*gltf.Scene{{Nodes: []int{0, 2}}},
		Nodes: []*gltf.Node{
			{Translation: [3]float64{2, 0, 0}, Scale: [3]float64{-1, 2, 3}, Children: []int{1}},
			{Translation: [3]float64{1, 0, 0}, Rotation: rpY90(), Scale: [3]float64{2, -1, 1}},
			{Translation: [3]float64{-3, 1, 4}, Rotation: rpY90(), Scale: [3]float64{1, 1, -2}},
		},
	}
	before, err := doc.NodeWorldMatrix(1)
	if err != nil {
		t.Fatal(err)
	}
	if err := doc.ReparentNode(0, 1, 2); err != nil {
		t.Fatal(err)
	}
	after, err := doc.NodeWorldMatrix(1)
	if err != nil {
		t.Fatal(err)
	}
	if !rpMatNear(after, before, 1e-9) {
		t.Fatalf("world changed:\n%v\n%v", before, after)
	}
	// Whatever representation is chosen must decode back to the same transform.
	encoded, err := json.Marshal(doc.Nodes[1])
	if err != nil {
		t.Fatal(err)
	}
	var decoded gltf.Node
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatalf("re-decode %s: %v", encoded, err)
	}
	local, err := decoded.LocalMatrix()
	if err != nil {
		t.Fatal(err)
	}
	wantLocal, err := doc.Nodes[1].LocalMatrix()
	if err != nil {
		t.Fatal(err)
	}
	if !rpMatNear(local, wantLocal, 1e-12) {
		t.Fatalf("local after JSON round trip = %v, want %v", local, wantLocal)
	}
}

// Reparenting under a parent with a very large translation must still work:
// the translate part is exactly invertible, so it must not be mistaken for a
// singular transform.
func TestReparentNode_LargeTranslationParent(t *testing.T) {
	for _, tx := range []float64{1e9, 1e12, 1e15} {
		doc := &gltf.Document{
			Scenes: []*gltf.Scene{{Nodes: []int{0, 1}}},
			Nodes: []*gltf.Node{
				{Translation: [3]float64{tx, 0, 0}},
				{Translation: [3]float64{1, 2, 3}, Rotation: rpY90(), Scale: [3]float64{2, 3, 4}},
			},
		}
		before, err := doc.NodeWorldMatrix(1)
		if err != nil {
			t.Fatalf("tx=%v: %v", tx, err)
		}
		if err := doc.ReparentNode(0, 1, 0); err != nil {
			t.Fatalf("tx=%v: ReparentNode failed: %v", tx, err)
		}
		after, err := doc.NodeWorldMatrix(1)
		if err != nil {
			t.Fatalf("tx=%v: %v", tx, err)
		}
		if !rpMatNear(after, before, 1e-6) {
			t.Fatalf("tx=%v world changed:\n%v\n%v", tx, before, after)
		}
	}
}

func TestReparentNode_OptionalParentVariant(t *testing.T) {
	doc := &gltf.Document{
		Scenes: []*gltf.Scene{{Nodes: []int{0, 2}}},
		Nodes: []*gltf.Node{
			{Translation: [3]float64{2, 0, 0}, Children: []int{1}},
			{Translation: [3]float64{1, 0, 0}},
			{Translation: [3]float64{-3, 0, 0}},
		},
	}
	if err := doc.SetNodeParent(0, 1, gltf.Index(2)); err != nil {
		t.Fatal(err)
	}
	if math.Abs(doc.Nodes[1].Translation[0]-6) > 1e-12 {
		t.Fatalf("SetNodeParent local x = %v, want 6", doc.Nodes[1].Translation[0])
	}

	// A nil parent means root.
	doc2 := &gltf.Document{
		Scenes: []*gltf.Scene{{Nodes: []int{0}}},
		Nodes: []*gltf.Node{
			{Translation: [3]float64{2, 0, 0}, Children: []int{1}},
			{Translation: [3]float64{1, 0, 0}},
		},
	}
	if err := gltf.ReparentNode(doc2, 0, 1, gltf.RootParent); err != nil {
		t.Fatal(err)
	}
	if got := doc2.Scenes[0].Nodes; !reflect.DeepEqual(got, []int{0, 1}) {
		t.Fatalf("package ReparentNode roots = %v, want [0 1]", got)
	}

	doc3 := &gltf.Document{
		Scenes: []*gltf.Scene{{Nodes: []int{0}}},
		Nodes: []*gltf.Node{
			{Translation: [3]float64{2, 0, 0}, Children: []int{1}},
			{Translation: [3]float64{1, 0, 0}},
		},
	}
	if err := doc3.SetNodeParent(0, 1, nil); err != nil {
		t.Fatal(err)
	}
	if got := doc3.Scenes[0].Nodes; !reflect.DeepEqual(got, []int{0, 1}) {
		t.Fatalf("SetNodeParent(nil) roots = %v, want [0 1]", got)
	}
}

func ExampleDocument_ReparentNode() {
	doc := &gltf.Document{
		Scenes: []*gltf.Scene{{Nodes: []int{0, 2}}},
		Nodes: []*gltf.Node{
			{Translation: [3]float64{2, 0, 0}, Children: []int{1}},
			{Translation: [3]float64{1, 0, 0}},
			{Translation: [3]float64{-3, 0, 0}},
		},
	}
	if err := doc.ReparentNode(0, 1, 2); err != nil {
		panic(err)
	}
	fmt.Printf("local x = %.0f\n", doc.Nodes[1].Translation[0])
	// Output: local x = 6
}
