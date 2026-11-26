package gltf_test

import (
	"encoding/binary"
	"math"
	"strings"
	"testing"

	"github.com/qmuntal/gltf"
)

func mat4Float32Bytes(m [16]float64) []byte {
	buf := make([]byte, 64)
	for i, v := range m {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(float32(v)))
	}
	return buf
}

func translationMatrix(x, y, z float64) [16]float64 {
	m := gltf.DefaultMatrix
	m[12], m[13], m[14] = x, y, z
	return m
}

func rotationZ(angle float64) [4]float64 {
	return [4]float64{0, 0, math.Sin(angle / 2), math.Cos(angle / 2)}
}

func nodeWithTRS(t [3]float64, r [4]float64, s [3]float64) *gltf.Node {
	return &gltf.Node{Translation: t, Rotation: r, Scale: s}
}

func approxMatrix(t *testing.T, got, want [16]float64, epsilon float64) {
	t.Helper()
	for i := 0; i < 16; i++ {
		if math.Abs(got[i]-want[i]) > epsilon {
			t.Fatalf("matrix mismatch at [%d]: got %v, want %v\nfull got:  %v\nfull want: %v",
				i, got[i], want[i], got, want)
		}
	}
}

func approxTranslation(t *testing.T, got [16]float64, x, y, z, epsilon float64) {
	t.Helper()
	want := [3]float64{x, y, z}
	gotT := [3]float64{got[12], got[13], got[14]}
	for i := 0; i < 3; i++ {
		if math.Abs(gotT[i]-want[i]) > epsilon {
			t.Fatalf("translation mismatch: got %v, want %v (matrix %v)", gotT, want, got)
		}
	}
}

// When inverseBindMatrices are not passed, JointMatrices reads them from the
// skin's FLOAT MAT4 accessor and still yields translation x=2.
func TestJointMatricesReadsInverseBindAccessor(t *testing.T) {
	ibmBytes := mat4Float32Bytes(translationMatrix(-1, 0, 0))
	doc := &gltf.Document{
		Accessors: []*gltf.Accessor{{
			BufferView:    gltf.Index(0),
			ComponentType: gltf.ComponentFloat,
			Count:         1,
			Type:          gltf.AccessorMat4,
		}},
		BufferViews: []*gltf.BufferView{{Buffer: 0, ByteLength: len(ibmBytes)}},
		Buffers:     []*gltf.Buffer{{ByteLength: len(ibmBytes), Data: ibmBytes}},
		Nodes: []*gltf.Node{
			{Skin: gltf.Index(0), Translation: [3]float64{2, 0, 0}},
			{Translation: [3]float64{5, 0, 0}},
		},
		Skins: []*gltf.Skin{{Joints: []int{1}, InverseBindMatrices: gltf.Index(0)}},
	}
	matrices, err := doc.JointMatrices(&gltf.Scene{Nodes: []int{0, 1}}, 0, nil)
	if err != nil {
		t.Fatalf("JointMatrices with accessor IBM: %v", err)
	}
	approxTranslation(t, matrices[0], 2, 0, 0, 1e-6)

	// An accessor whose count differs from the joint count is an error.
	bad := *doc
	bad.Accessors = []*gltf.Accessor{{
		BufferView:    gltf.Index(0),
		ComponentType: gltf.ComponentFloat,
		Count:         2,
		Type:          gltf.AccessorMat4,
	}}
	_, err = gltf.JointMatrices(&bad, &gltf.Scene{Nodes: []int{0, 1}}, 0, nil)
	if err == nil || !strings.Contains(err.Error(), "inverse bind matrix count 2 does not match joint count 1") {
		t.Fatalf("expected IBM count mismatch error, got %v", err)
	}
}

// The task's fixed scenario: mesh node at x=2, joint at x=5 and an inverse
// bind translation of x=-1 must yield a mesh-local joint translation of x=2.
func TestJointMatricesFixedScenario(t *testing.T) {
	doc := &gltf.Document{
		Nodes: []*gltf.Node{
			{Name: "mesh", Skin: gltf.Index(0), Translation: [3]float64{2, 0, 0}},
			{Name: "joint", Translation: [3]float64{5, 0, 0}},
		},
		Skins: []*gltf.Skin{{Joints: []int{1}}},
	}
	scene := &gltf.Scene{Nodes: []int{0, 1}}
	ibm := [][16]float64{translationMatrix(-1, 0, 0)}

	matrices, err := doc.JointMatrices(scene, 0, nil, ibm...)
	if err != nil {
		t.Fatalf("JointMatrices returned error: %v", err)
	}
	if len(matrices) != 1 {
		t.Fatalf("expected 1 matrix, got %d", len(matrices))
	}
	t.Logf("fixed scenario mesh-local joint translation = (%.6f, %.6f, %.6f)",
		matrices[0][12], matrices[0][13], matrices[0][14])
	approxTranslation(t, matrices[0], 2, 0, 0, 1e-9)

	// The package-level entry point must behave identically.
	pkgMatrices, err := gltf.JointMatrices(doc, scene, 0, nil, ibm...)
	if err != nil {
		t.Fatalf("gltf.JointMatrices returned error: %v", err)
	}
	approxMatrix(t, pkgMatrices[0], matrices[0], 0)
}

// With no inverse bind matrices every inverse bind defaults to identity,
// so the same scenario gives -2 + 5 = 3.
func TestJointMatricesDefaultInverseBind(t *testing.T) {
	doc := &gltf.Document{
		Nodes: []*gltf.Node{
			{Skin: gltf.Index(0), Translation: [3]float64{2, 0, 0}},
			{Translation: [3]float64{5, 0, 0}},
		},
		Skins: []*gltf.Skin{{Joints: []int{1}}},
	}
	matrices, err := gltf.JointMatrices(doc, &gltf.Scene{Nodes: []int{0, 1}}, 0, nil)
	if err != nil {
		t.Fatalf("JointMatrices returned error: %v", err)
	}
	approxTranslation(t, matrices[0], 3, 0, 0, 1e-9)
}

// Parent rotations must be composed both above the joint and above the mesh.
func TestJointMatricesParentRotation(t *testing.T) {
	const eps = 1e-9
	halfPi := math.Pi / 2
	rotZ90 := rotationZ(halfPi)

	t.Run("joint parent rotates joint translation", func(t *testing.T) {
		// 0: mesh at origin, 1: rotated parent, 2: joint at x=1.
		doc := &gltf.Document{
			Nodes: []*gltf.Node{
				{Skin: gltf.Index(0)},
				{Rotation: rotZ90, Children: []int{2}},
				{Translation: [3]float64{1, 0, 0}},
			},
			Skins: []*gltf.Skin{{Joints: []int{2}}},
		}
		matrices, err := gltf.JointMatrices(doc, &gltf.Scene{Nodes: []int{0, 1}}, 0, nil)
		if err != nil {
			t.Fatalf("JointMatrices returned error: %v", err)
		}
		m := matrices[0]
		// Rz(+90) maps +X to +Y.
		approxTranslation(t, m, 0, 1, 0, eps)
		if math.Abs(m[0]-0) > eps || math.Abs(m[1]-1) > eps ||
			math.Abs(m[4]-(-1)) > eps || math.Abs(m[5]-0) > eps {
			t.Fatalf("rotation part mismatch: %v", m)
		}
	})

	t.Run("mesh parent rotation is removed by inverse mesh world", func(t *testing.T) {
		// 0: Rz90 parent, 1: mesh at x=2 with skin, 2: joint at x=5.
		// IBM translates x=-1 (effective joint offset x=4).
		// Expected: T(-2) * Rz(-90) * T(4), translation (-2, -4, 0).
		doc := &gltf.Document{
			Nodes: []*gltf.Node{
				{Rotation: rotZ90, Children: []int{1}},
				{Skin: gltf.Index(0), Translation: [3]float64{2, 0, 0}},
				{Translation: [3]float64{5, 0, 0}},
			},
			Skins: []*gltf.Skin{{Joints: []int{2}}},
		}
		ibm := [][16]float64{translationMatrix(-1, 0, 0)}
		matrices, err := gltf.JointMatrices(doc, &gltf.Scene{Nodes: []int{0, 2}}, 1, nil, ibm...)
		if err != nil {
			t.Fatalf("JointMatrices returned error: %v", err)
		}
		m := matrices[0]
		t.Logf("mesh-parent rotation joint translation = (%.6f, %.6f, %.6f)", m[12], m[13], m[14])
		approxTranslation(t, m, -2, -4, 0, eps)
		// Linear part must be Rz(-90): column 0 ~ (0,-1,0), column 1 ~ (1,0,0).
		if math.Abs(m[0]-0) > eps || math.Abs(m[1]-(-1)) > eps ||
			math.Abs(m[4]-1) > eps || math.Abs(m[5]-0) > eps {
			t.Fatalf("inverse rotation part mismatch: %v", m)
		}
	})
}

// One skin referenced by two mesh nodes in the same document must yield
// mesh-local results for each node without cross-call state.
func TestJointMatricesSharedSkinMultipleMeshes(t *testing.T) {
	doc := &gltf.Document{
		Nodes: []*gltf.Node{
			{Name: "mesh-a", Skin: gltf.Index(0), Translation: [3]float64{2, 0, 0}},
			{Name: "mesh-b", Skin: gltf.Index(0), Translation: [3]float64{3, 0, 0}},
			{Name: "joint", Translation: [3]float64{5, 0, 0}},
		},
		Skins: []*gltf.Skin{{Joints: []int{2}}},
	}
	ibm := [][16]float64{translationMatrix(-1, 0, 0)}
	scene := &gltf.Scene{Nodes: []int{0, 1, 2}}

	first, err := doc.JointMatrices(scene, 0, nil, ibm...)
	if err != nil {
		t.Fatalf("mesh-a at x=2: %v", err)
	}
	approxTranslation(t, first[0], 2, 0, 0, 1e-9)

	second, err := doc.JointMatrices(scene, 1, nil, ibm...)
	if err != nil {
		t.Fatalf("mesh-b at x=3: %v", err)
	}
	approxTranslation(t, second[0], 1, 0, 0, 1e-9)

	// Re-evaluating mesh-a must still give exactly the first result.
	again, err := doc.JointMatrices(scene, 0, nil, ibm...)
	if err != nil {
		t.Fatalf("mesh-a repeat: %v", err)
	}
	approxMatrix(t, again[0], first[0], 0)
}

// Joints with several entries come back in skin.Joints order.
func TestJointMatricesJointOrder(t *testing.T) {
	doc := &gltf.Document{
		Nodes: []*gltf.Node{
			{Skin: gltf.Index(0)},
			{Translation: [3]float64{1, 0, 0}},
			{Translation: [3]float64{2, 0, 0}},
		},
		Skins: []*gltf.Skin{{Joints: []int{2, 1}}},
	}
	matrices, err := gltf.JointMatrices(doc, &gltf.Scene{Nodes: []int{0, 1, 2}}, 0, nil)
	if err != nil {
		t.Fatalf("JointMatrices returned error: %v", err)
	}
	if len(matrices) != 2 {
		t.Fatalf("expected 2 matrices, got %d", len(matrices))
	}
	approxTranslation(t, matrices[0], 2, 0, 0, 1e-9) // Joints[0] = node 2 at x=2
	approxTranslation(t, matrices[1], 1, 0, 0, 1e-9) // Joints[1] = node 1 at x=1
}

// Pose overrides current node local matrices and must not mutate the document.
func TestJointMatricesPoseOverrideReadOnly(t *testing.T) {
	jointNode := &gltf.Node{Translation: [3]float64{5, 0, 0}}
	meshNode := &gltf.Node{Skin: gltf.Index(0), Translation: [3]float64{2, 0, 0}}
	doc := &gltf.Document{
		Nodes: []*gltf.Node{meshNode, jointNode},
		Skins: []*gltf.Skin{{Joints: []int{1}}},
	}
	ibm := [][16]float64{translationMatrix(-1, 0, 0)}
	scene := &gltf.Scene{Nodes: []int{0, 1}}

	// Animate the joint to x=6: -2 + 6 - 1 = 3.
	pose := map[int][16]float64{1: translationMatrix(6, 0, 0)}
	matrices, err := gltf.JointMatrices(doc, scene, 0, pose, ibm...)
	if err != nil {
		t.Fatalf("JointMatrices with pose: %v", err)
	}
	approxTranslation(t, matrices[0], 3, 0, 0, 1e-9)

	// The document node must be untouched.
	if jointNode.Translation[0] != 5 {
		t.Fatalf("joint node mutated by evaluation: translation = %v", jointNode.Translation)
	}
	if _, ok := pose[0]; ok {
		t.Fatalf("pose map was modified")
	}

	// Nil pose keeps using the document transform.
	rest, err := gltf.JointMatrices(doc, scene, 0, nil, ibm...)
	if err != nil {
		t.Fatalf("JointMatrices without pose: %v", err)
	}
	approxTranslation(t, rest[0], 2, 0, 0, 1e-9)
}

// TRS poses can be composed with Node.LocalMatrix before evaluation.
func TestNodeLocalMatrixTRS(t *testing.T) {
	n := nodeWithTRS(
		[3]float64{2, 3, 4},
		rotationZ(math.Pi/2),
		[3]float64{1, 1, 1},
	)
	m, err := n.LocalMatrix()
	if err != nil {
		t.Fatalf("LocalMatrix error: %v", err)
	}
	approxTranslation(t, m, 2, 3, 4, 1e-9)
	// +X rotated by Rz90 becomes +Y.
	if math.Abs(m[1]-1) > 1e-9 {
		t.Fatalf("expected rotated X basis to point +Y, matrix %v", m)
	}

	// Explicit matrices take precedence and are returned verbatim.
	explicit := translationMatrix(7, 8, 9)
	n2 := &gltf.Node{Matrix: explicit, Translation: [3]float64{1, 1, 1}}
	m2, err := n2.LocalMatrix()
	if err != nil {
		t.Fatalf("LocalMatrix explicit error: %v", err)
	}
	approxMatrix(t, m2, explicit, 0)

	if _, err := (*gltf.Node)(nil).LocalMatrix(); err == nil {
		t.Fatal("nil node LocalMatrix expected error")
	}
}

func TestJointMatricesErrors(t *testing.T) {
	singularMatrix := gltf.DefaultMatrix
	singularMatrix[5] = 0 // zero Y scale -> determinant 0

	baseDoc := func() *gltf.Document {
		return &gltf.Document{
			Nodes: []*gltf.Node{
				{Skin: gltf.Index(0), Translation: [3]float64{2, 0, 0}},
				{Translation: [3]float64{5, 0, 0}},
			},
			Skins: []*gltf.Skin{{Joints: []int{1}}},
		}
	}
	ibmOK := [][16]float64{translationMatrix(-1, 0, 0)}

	tests := []struct {
		name    string
		doc     *gltf.Document
		scene   *gltf.Scene
		mesh    int
		pose    map[int][16]float64
		ibm     [][16]float64
		wantSub string
	}{
		{
			name:    "nil document",
			doc:     nil,
			mesh:    0,
			wantSub: "nil document",
		},
		{
			name:    "mesh index out of range",
			doc:     baseDoc(),
			mesh:    9,
			wantSub: "invalid skinned mesh node",
		},
		{
			name: "mesh node without skin",
			doc: &gltf.Document{
				Nodes: []*gltf.Node{{}},
			},
			mesh:    0,
			wantSub: "does not reference a skin",
		},
		{
			name: "invalid skin index",
			doc: &gltf.Document{
				Nodes: []*gltf.Node{{Skin: gltf.Index(4)}},
				Skins: []*gltf.Skin{},
			},
			mesh:    0,
			wantSub: "invalid skin index",
		},
		{
			name: "skin without joints",
			doc: &gltf.Document{
				Nodes: []*gltf.Node{{Skin: gltf.Index(0)}},
				Skins: []*gltf.Skin{{}},
			},
			mesh:    0,
			wantSub: "no joints",
		},
		{
			name: "joint index out of range",
			doc: &gltf.Document{
				Nodes: []*gltf.Node{{Skin: gltf.Index(0)}},
				Skins: []*gltf.Skin{{Joints: []int{8}}},
			},
			mesh:    0,
			wantSub: "invalid node index 8",
		},
		{
			name: "negative joint index",
			doc: &gltf.Document{
				Nodes: []*gltf.Node{{Skin: gltf.Index(0)}},
				Skins: []*gltf.Skin{{Joints: []int{-1}}},
			},
			mesh:    0,
			wantSub: "invalid node index -1",
		},
		{
			name: "nil joint node",
			doc: &gltf.Document{
				Nodes: []*gltf.Node{{Skin: gltf.Index(0)}, nil},
				Skins: []*gltf.Skin{{Joints: []int{1}}},
			},
			mesh:    0,
			wantSub: "invalid node index 1",
		},
		{
			name:    "inverse bind matrix count mismatch",
			doc:     baseDoc(),
			mesh:    0,
			ibm:     [][16]float64{translationMatrix(0, 0, 0), translationMatrix(1, 0, 0)},
			wantSub: "inverse bind matrix count 2 does not match joint count 1",
		},
		{
			name: "hierarchy cycle",
			doc: &gltf.Document{
				Nodes: []*gltf.Node{
					{Skin: gltf.Index(0), Children: []int{1}},
					{Children: []int{0}},
				},
				Skins: []*gltf.Skin{{Joints: []int{0}}},
			},
			scene:   &gltf.Scene{Nodes: []int{0}},
			mesh:    0,
			wantSub: "cycle",
		},
		{
			name: "invalid child index",
			doc: &gltf.Document{
				Nodes: []*gltf.Node{
					{Skin: gltf.Index(0), Children: []int{5}},
					{Translation: [3]float64{1, 0, 0}},
				},
				Skins: []*gltf.Skin{{Joints: []int{1}}},
			},
			mesh:    0,
			wantSub: "invalid child index 5",
		},
		{
			name: "node with multiple parents",
			doc: &gltf.Document{
				Nodes: []*gltf.Node{
					{Skin: gltf.Index(0), Children: []int{2}},
					{Children: []int{2}},
					{},
				},
				Skins: []*gltf.Skin{{Joints: []int{2}}},
			},
			mesh:    0,
			wantSub: "multiple parents",
		},
		{
			name: "non invertible mesh matrix",
			doc: &gltf.Document{
				Nodes: []*gltf.Node{
					{Skin: gltf.Index(0), Matrix: singularMatrix},
					{Translation: [3]float64{5, 0, 0}},
				},
				Skins: []*gltf.Skin{{Joints: []int{1}}},
			},
			mesh:    0,
			ibm:     ibmOK,
			wantSub: "not invertible",
		},
		{
			name: "non invertible mesh TRS scale",
			doc: &gltf.Document{
				Nodes: []*gltf.Node{
					{Skin: gltf.Index(0), Scale: [3]float64{1, 0, 1}},
					{Translation: [3]float64{5, 0, 0}},
				},
				Skins: []*gltf.Skin{{Joints: []int{1}}},
			},
			mesh:    0,
			wantSub: "not invertible",
		},
		{
			name: "non invertible inherited parent scale",
			doc: &gltf.Document{
				Nodes: []*gltf.Node{
					{Scale: [3]float64{1, 0, 1}, Children: []int{1}},
					{Skin: gltf.Index(0)},
					{Translation: [3]float64{5, 0, 0}},
				},
				Skins: []*gltf.Skin{{Joints: []int{2}}},
			},
			scene:   &gltf.Scene{Nodes: []int{0, 2}},
			mesh:    1,
			wantSub: "not invertible",
		},
		{
			name: "non finite node matrix",
			doc: &gltf.Document{
				Nodes: []*gltf.Node{
					{Skin: gltf.Index(0), Matrix: func() [16]float64 {
						m := gltf.DefaultMatrix
						m[0] = math.NaN()
						return m
					}()},
					{Translation: [3]float64{5, 0, 0}},
				},
				Skins: []*gltf.Skin{{Joints: []int{1}}},
			},
			mesh:    0,
			wantSub: "non-finite",
		},
		{
			name: "non finite translation",
			doc: &gltf.Document{
				Nodes: []*gltf.Node{
					{Skin: gltf.Index(0), Translation: [3]float64{math.Inf(1), 0, 0}},
					{Translation: [3]float64{5, 0, 0}},
				},
				Skins: []*gltf.Skin{{Joints: []int{1}}},
			},
			mesh:    0,
			wantSub: "non-finite",
		},
		{
			name: "non finite pose matrix",
			doc:  baseDoc(),
			mesh: 0,
			pose: map[int][16]float64{1: func() [16]float64 {
				m := gltf.DefaultMatrix
				m[12] = math.Inf(-1)
				return m
			}()},
			wantSub: "pose matrix of node 1",
		},
		{
			name:    "non finite inverse bind matrix",
			doc:     baseDoc(),
			mesh:    0,
			ibm:     [][16]float64{{math.NaN(), 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}},
			wantSub: "inverse bind matrix 0",
		},
		{
			name:    "invalid scene root",
			doc:     baseDoc(),
			scene:   &gltf.Scene{Nodes: []int{42}},
			mesh:    0,
			wantSub: "invalid root node index 42",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := gltf.JointMatrices(tt.doc, tt.scene, tt.mesh, tt.pose, tt.ibm...)
			if err == nil {
				t.Fatalf("expected error containing %q, got nil", tt.wantSub)
			}
			if tt.wantSub != "" && !strings.Contains(err.Error(), tt.wantSub) {
				t.Fatalf("error %q does not contain %q", err.Error(), tt.wantSub)
			}
		})
	}
}
