package modeler_test

import (
	"math"
	"reflect"
	"testing"

	"github.com/qmuntal/gltf"
	"github.com/qmuntal/gltf/modeler"
)

const matrixTolerance = 1e-9

// transformPoint applies a column-major 4x4 matrix to a 3D point, performing
// the homogeneous divide, and returns the resulting clip-space coordinates.
func transformPoint(m [16]float64, x, y, z float64) [3]float64 {
	cx := m[0]*x + m[4]*y + m[8]*z + m[12]
	cy := m[1]*x + m[5]*y + m[9]*z + m[13]
	cz := m[2]*x + m[6]*y + m[10]*z + m[14]
	cw := m[3]*x + m[7]*y + m[11]*z + m[15]
	if cw != 0 && cw != 1 {
		cx, cy, cz = cx/cw, cy/cw, cz/cw
	}
	return [3]float64{cx, cy, cz}
}

func assertMatrix(t *testing.T, name string, got, want [16]float64) {
	t.Helper()
	for i := range want {
		if math.Abs(got[i]-want[i]) > matrixTolerance {
			t.Errorf("%s[%d] = %v, want %v (matrix %v)", name, i, got[i], want[i], got)
			return
		}
	}
}

func perspectiveCamera(yfov float64, znear float64, zfar *float64) *gltf.Camera {
	return &gltf.Camera{
		Type: "perspective",
		Perspective: &gltf.Perspective{
			Yfov:  yfov,
			Znear: znear,
			Zfar:  zfar,
		},
	}
}

func TestReadCameraMatrices_DepthMapping(t *testing.T) {
	// Camera at the origin, 90 degrees vertical FOV, 1:1 aspect ratio,
	// near plane at 1 and far plane at 3.
	doc := &gltf.Document{
		Nodes:   []*gltf.Node{{Camera: gltf.Index(0)}},
		Cameras: []*gltf.Camera{perspectiveCamera(math.Pi/2, 1, gltf.Float(3))},
	}
	matrices, err := modeler.ReadCameraMatrices(doc, 0, 1)
	if err != nil {
		t.Fatalf("ReadCameraMatrices() error = %v", err)
	}

	assertMatrix(t, "View", matrices.View, gltf.DefaultMatrix)

	// A point on the -Z axis at the near plane must map to clip depth -1,
	// and the far plane to +1.
	near := transformPoint(matrices.Projection, 0, 0, -1)
	if math.Abs(near[2]-(-1)) > matrixTolerance {
		t.Errorf("near plane clip depth = %v, want -1", near[2])
	}
	far := transformPoint(matrices.Projection, 0, 0, -3)
	if math.Abs(far[2]-1) > matrixTolerance {
		t.Errorf("far plane clip depth = %v, want 1", far[2])
	}

	// With a 90 degree vertical FOV and a square viewport, a point on the
	// plane z=-1 at x=1 or y=1 must land exactly on the clip boundary.
	right := transformPoint(matrices.Projection, 1, 0, -1)
	if math.Abs(right[0]-1) > matrixTolerance {
		t.Errorf("right clip x = %v, want 1", right[0])
	}
	top := transformPoint(matrices.Projection, 0, 1, -1)
	if math.Abs(top[1]-1) > matrixTolerance {
		t.Errorf("top clip y = %v, want 1", top[1])
	}
}

func TestReadCameraMatrices_TranslatedCamera(t *testing.T) {
	// A camera looking straight down -Z from (0, 0, 5) sees the world
	// origin 5 units in front of it.
	doc := &gltf.Document{
		Nodes: []*gltf.Node{{
			Camera:      gltf.Index(0),
			Translation: [3]float64{0, 0, 5},
		}},
		Cameras: []*gltf.Camera{perspectiveCamera(math.Pi/2, 1, gltf.Float(3))},
	}
	matrices, err := modeler.ReadCameraMatrices(doc, 0, 1)
	if err != nil {
		t.Fatalf("ReadCameraMatrices() error = %v", err)
	}

	want := [16]float64{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, -5, 1,
	}
	assertMatrix(t, "View", matrices.View, want)

	// The world point (0, 0, 4) is 1 unit in front of the camera, on the
	// near plane, so its clip depth must be -1.
	front := transformPoint(matrices.View, 0, 0, 4)
	if math.Abs(front[2]-(-1)) > matrixTolerance {
		t.Errorf("translated camera projected z = %v, want -1", front[2])
	}
}

func TestReadCameraMatrices_ParentedCamera(t *testing.T) {
	// The camera node is a child of a translated parent. Its view matrix
	// must combine both transforms.
	doc := &gltf.Document{
		Nodes: []*gltf.Node{
			{Children: []int{1}, Translation: [3]float64{0, 0, 10}},
			{Camera: gltf.Index(0), Translation: [3]float64{0, 0, 2}},
		},
		Cameras: []*gltf.Camera{perspectiveCamera(math.Pi/2, 1, gltf.Float(3))},
	}
	matrices, err := modeler.ReadCameraMatrices(doc, 1, 1)
	if err != nil {
		t.Fatalf("ReadCameraMatrices() error = %v", err)
	}

	want := [16]float64{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, -12, 1,
	}
	assertMatrix(t, "View", matrices.View, want)
}

func TestReadCameraMatrices_Orthographic(t *testing.T) {
	doc := &gltf.Document{
		Nodes: []*gltf.Node{{Camera: gltf.Index(0)}},
		Cameras: []*gltf.Camera{{
			Type: "orthographic",
			Orthographic: &gltf.Orthographic{
				Xmag: 2, Ymag: 1, Znear: 1, Zfar: 3,
			},
		}},
	}
	matrices, err := modeler.ReadCameraMatrices(doc, 0, 1)
	if err != nil {
		t.Fatalf("ReadCameraMatrices() error = %v", err)
	}

	want := [16]float64{
		0.5, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, -1, 0,
		0, 0, -2, 1,
	}
	assertMatrix(t, "Projection", matrices.Projection, want)

	// The orthographic extents are defined by xmag/ymag, so points at the
	// edge at the near plane land on the clip boundary.
	right := transformPoint(matrices.Projection, 2, 0, -1)
	if math.Abs(right[0]-1) > matrixTolerance {
		t.Errorf("orthographic right clip x = %v, want 1", right[0])
	}
	near := transformPoint(matrices.Projection, 0, 0, -1)
	if math.Abs(near[2]-(-1)) > matrixTolerance {
		t.Errorf("orthographic near clip depth = %v, want -1", near[2])
	}
	far := transformPoint(matrices.Projection, 0, 0, -3)
	if math.Abs(far[2]-1) > matrixTolerance {
		t.Errorf("orthographic far clip depth = %v, want 1", far[2])
	}
}

func TestReadCameraMatrices_InfinitePerspective(t *testing.T) {
	doc := &gltf.Document{
		Nodes:   []*gltf.Node{{Camera: gltf.Index(0)}},
		Cameras: []*gltf.Camera{perspectiveCamera(math.Pi/2, 1, nil)},
	}
	matrices, err := modeler.ReadCameraMatrices(doc, 0, 1)
	if err != nil {
		t.Fatalf("ReadCameraMatrices() error = %v", err)
	}

	// Near plane maps to -1 and a point at infinity maps to +1.
	near := transformPoint(matrices.Projection, 0, 0, -1)
	if math.Abs(near[2]-(-1)) > matrixTolerance {
		t.Errorf("infinite near clip depth = %v, want -1", near[2])
	}
	// Far away points approach the +1 clip depth without ever reaching it.
	far := transformPoint(matrices.Projection, 0, 0, -1e9)
	if math.Abs(far[2]-1) > 1e-6 {
		t.Errorf("infinite projection at z=-1e9 clip depth = %v, want ~1", far[2])
	}
}

func TestReadCameraMatrices_AspectRatio(t *testing.T) {
	// Explicit aspect ratio wins over the supplied one.
	doc := &gltf.Document{
		Nodes: []*gltf.Node{{Camera: gltf.Index(0)}},
		Cameras: []*gltf.Camera{{
			Type: "perspective",
			Perspective: &gltf.Perspective{
				Yfov: math.Pi / 2, Znear: 1, Zfar: gltf.Float(3),
				AspectRatio: gltf.Float(2),
			},
		}},
	}
	matrices, err := modeler.ReadCameraMatrices(doc, 0, 1)
	if err != nil {
		t.Fatalf("ReadCameraMatrices() error = %v", err)
	}
	// x scale is halved in clip space relative to the square case, so a
	// point at x=2 on the near plane lands on the right boundary.
	right := transformPoint(matrices.Projection, 2, 0, -1)
	if math.Abs(right[0]-1) > matrixTolerance {
		t.Errorf("aspect=2 right clip x = %v, want 1", right[0])
	}

	// Without an explicit ratio the externally supplied one is used.
	doc.Cameras[0].Perspective.AspectRatio = nil
	matrices, err = modeler.ReadCameraMatrices(doc, 0, 2)
	if err != nil {
		t.Fatalf("ReadCameraMatrices() error = %v", err)
	}
	right = transformPoint(matrices.Projection, 2, 0, -1)
	if math.Abs(right[0]-1) > matrixTolerance {
		t.Errorf("external aspect=2 right clip x = %v, want 1", right[0])
	}
}

func TestReadCameraMatrices_RotatedCamera(t *testing.T) {
	// Rotate the camera 90 degrees around +Y so it looks towards -X instead
	// of -Z. A world point in front of it must land on the -Z axis in view
	// space.
	sin, cos := math.Sin(math.Pi/4), math.Cos(math.Pi/4)
	doc := &gltf.Document{
		Nodes: []*gltf.Node{{
			Camera:   gltf.Index(0),
			Rotation: [4]float64{0, sin, 0, cos},
		}},
		Cameras: []*gltf.Camera{perspectiveCamera(math.Pi/2, 1, gltf.Float(3))},
	}
	matrices, err := modeler.ReadCameraMatrices(doc, 0, 1)
	if err != nil {
		t.Fatalf("ReadCameraMatrices() error = %v", err)
	}
	// The camera local -Z now points to world -X, so the world point
	// (-4, 0, 0) is 4 units in front of it.
	front := transformPoint(matrices.View, -4, 0, 0)
	if math.Abs(front[2]-(-4)) > 1e-9 || math.Abs(front[0]) > 1e-9 || math.Abs(front[1]) > 1e-9 {
		t.Errorf("rotated view of (-4,0,0) = %v, want (0,0,-4)", front)
	}
}

func TestReadCameraMatrices_MatrixProperty(t *testing.T) {
	// A camera defined with an explicit translation matrix.
	doc := &gltf.Document{
		Nodes: []*gltf.Node{{
			Camera: gltf.Index(0),
			Matrix: [16]float64{
				1, 0, 0, 0,
				0, 1, 0, 0,
				0, 0, 1, 0,
				3, 4, 5, 1,
			},
		}},
		Cameras: []*gltf.Camera{perspectiveCamera(math.Pi/2, 1, gltf.Float(3))},
	}
	matrices, err := modeler.ReadCameraMatrices(doc, 0, 1)
	if err != nil {
		t.Fatalf("ReadCameraMatrices() error = %v", err)
	}
	want := [16]float64{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		-3, -4, -5, 1,
	}
	assertMatrix(t, "View", matrices.View, want)
}

func TestReadCameraMatrices_DoesNotMutateDocument(t *testing.T) {
	doc := &gltf.Document{
		Nodes: []*gltf.Node{{
			Camera:      gltf.Index(0),
			Translation: [3]float64{1, 2, 3},
			Children:    []int{},
		}},
		Cameras: []*gltf.Camera{perspectiveCamera(math.Pi/2, 1, gltf.Float(3))},
	}
	want := *doc.Nodes[0]
	if _, err := modeler.ReadCameraMatrices(doc, 0, 1); err != nil {
		t.Fatalf("ReadCameraMatrices() error = %v", err)
	}
	if !reflect.DeepEqual(*doc.Nodes[0], want) {
		t.Errorf("ReadCameraMatrices() modified the node: got %+v, want %+v", *doc.Nodes[0], want)
	}
}

func TestReadCameraMatrices_InfersCameraType(t *testing.T) {
	// When Type is empty the projection is inferred from the populated
	// property, matching how a Camera is decoded from JSON.
	doc := &gltf.Document{
		Nodes: []*gltf.Node{{Camera: gltf.Index(0)}},
		Cameras: []*gltf.Camera{{
			Perspective: &gltf.Perspective{Yfov: math.Pi / 2, Znear: 1, Zfar: gltf.Float(3)},
		}},
	}
	matrices, err := modeler.ReadCameraMatrices(doc, 0, 1)
	if err != nil {
		t.Fatalf("ReadCameraMatrices() error = %v", err)
	}
	near := transformPoint(matrices.Projection, 0, 0, -1)
	if math.Abs(near[2]-(-1)) > matrixTolerance {
		t.Errorf("inferred perspective near clip depth = %v, want -1", near[2])
	}
}

func TestReadCameraMatrices_TestdataScene(t *testing.T) {
	doc, err := gltf.Open("../testdata/Cameras/glTF/Cameras.gltf")
	if err != nil {
		t.Fatalf("gltf.Open() error = %v", err)
	}

	// Both camera nodes are translated by the same amount, so their view
	// matrices only differ from the identity by that translation.
	wantView := [16]float64{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		-0.5, -0.5, -3, 1,
	}
	for _, nodeIndex := range []int{1, 2} {
		matrices, err := modeler.ReadCameraMatrices(doc, nodeIndex, 1)
		if err != nil {
			t.Fatalf("node %d: ReadCameraMatrices() error = %v", nodeIndex, err)
		}
		assertMatrix(t, "View", matrices.View, wantView)

		// The world origin sits at (0, 0, -3) in the camera space.
		origin := transformPoint(matrices.View, 0, 0, 0)
		if math.Abs(origin[2]-(-3)) > 1e-9 {
			t.Errorf("node %d: origin view z = %v, want -3", nodeIndex, origin[2])
		}
	}

	// The orthographic camera maps the near plane (0.01) to -1 and the far
	// plane (100) to +1 along -Z.
	ortho, err := modeler.ReadCameraMatrices(doc, 2, 1)
	if err != nil {
		t.Fatalf("ReadCameraMatrices() error = %v", err)
	}
	near := transformPoint(ortho.Projection, 0, 0, -0.01)
	if math.Abs(near[2]-(-1)) > 1e-9 {
		t.Errorf("orthographic near clip depth = %v, want -1", near[2])
	}
	far := transformPoint(ortho.Projection, 0, 0, -100)
	if math.Abs(far[2]-1) > 1e-9 {
		t.Errorf("orthographic far clip depth = %v, want 1", far[2])
	}
}

func TestReadCameraMatrices_ScaledCamera(t *testing.T) {
	// An invertible scale does not make the transform non-invertible, so
	// the view matrix is simply its inverse.
	doc := &gltf.Document{
		Nodes: []*gltf.Node{{
			Camera:      gltf.Index(0),
			Translation: [3]float64{0, 0, 5},
			Scale:       [3]float64{2, 2, 2},
		}},
		Cameras: []*gltf.Camera{perspectiveCamera(math.Pi/2, 1, gltf.Float(3))},
	}
	matrices, err := modeler.ReadCameraMatrices(doc, 0, 1)
	if err != nil {
		t.Fatalf("ReadCameraMatrices() error = %v", err)
	}
	want := [16]float64{
		0.5, 0, 0, 0,
		0, 0.5, 0, 0,
		0, 0, 0.5, 0,
		0, 0, -2.5, 1,
	}
	assertMatrix(t, "View", matrices.View, want)
}

func TestReadCameraMatrices_Errors(t *testing.T) {
	valid := func() *gltf.Document {
		return &gltf.Document{
			Nodes:   []*gltf.Node{{Camera: gltf.Index(0)}},
			Cameras: []*gltf.Camera{perspectiveCamera(math.Pi/2, 1, gltf.Float(3))},
		}
	}
	tests := []struct {
		name      string
		doc       *gltf.Document
		nodeIndex int
		aspect    float64
	}{
		{"nil-document", nil, 0, 1},
		{"node-index-negative", valid(), -1, 1},
		{"node-index-overflow", valid(), 5, 1},
		{"node-nil", &gltf.Document{
			Nodes:   []*gltf.Node{nil},
			Cameras: valid().Cameras,
		}, 0, 1},
		{"no-camera", &gltf.Document{
			Nodes:   []*gltf.Node{{}},
			Cameras: valid().Cameras,
		}, 0, 1},
		{"camera-index-overflow", &gltf.Document{
			Nodes:   []*gltf.Node{{Camera: gltf.Index(3)}},
			Cameras: valid().Cameras,
		}, 0, 1},
		{"camera-nil", &gltf.Document{
			Nodes:   []*gltf.Node{{Camera: gltf.Index(0)}},
			Cameras: []*gltf.Camera{nil},
		}, 0, 1},
		{"hierarchy-cycle", &gltf.Document{
			Nodes: []*gltf.Node{
				{Camera: gltf.Index(0), Children: []int{1}},
				{Children: []int{0}},
			},
			Cameras: valid().Cameras,
		}, 0, 1},
		{"multiple-parents", &gltf.Document{
			Nodes: []*gltf.Node{
				{Camera: gltf.Index(0)},
				{Children: []int{0}},
				{Children: []int{0}},
			},
			Cameras: valid().Cameras,
		}, 0, 1},
		{"znear-zero", func() *gltf.Document {
			d := valid()
			d.Cameras[0].Perspective.Znear = 0
			return d
		}(), 0, 1},
		{"znear-negative", func() *gltf.Document {
			d := valid()
			d.Cameras[0].Perspective.Znear = -1
			return d
		}(), 0, 1},
		{"zfar-equals-znear", func() *gltf.Document {
			d := valid()
			d.Cameras[0].Perspective.Zfar = gltf.Float(1)
			return d
		}(), 0, 1},
		{"zfar-less-than-znear", func() *gltf.Document {
			d := valid()
			d.Cameras[0].Perspective.Zfar = gltf.Float(0.5)
			return d
		}(), 0, 1},
		{"yfov-zero", func() *gltf.Document {
			d := valid()
			d.Cameras[0].Perspective.Yfov = 0
			return d
		}(), 0, 1},
		{"yfov-pi", func() *gltf.Document {
			d := valid()
			d.Cameras[0].Perspective.Yfov = math.Pi
			return d
		}(), 0, 1},
		{"yfov-nan", func() *gltf.Document {
			d := valid()
			d.Cameras[0].Perspective.Yfov = math.NaN()
			return d
		}(), 0, 1},
		{"explicit-aspect-zero", func() *gltf.Document {
			d := valid()
			d.Cameras[0].Perspective.AspectRatio = gltf.Float(0)
			return d
		}(), 0, 1},
		{"external-aspect-negative", valid(), 0, -1},
		{"external-aspect-nan", valid(), 0, math.NaN()},
		{"orthographic-znear-zero", &gltf.Document{
			Nodes: []*gltf.Node{{Camera: gltf.Index(0)}},
			Cameras: []*gltf.Camera{{
				Type:         "orthographic",
				Orthographic: &gltf.Orthographic{Xmag: 1, Ymag: 1, Znear: 0, Zfar: 3},
			}},
		}, 0, 1},
		{"orthographic-xmag-zero", &gltf.Document{
			Nodes: []*gltf.Node{{Camera: gltf.Index(0)}},
			Cameras: []*gltf.Camera{{
				Type:         "orthographic",
				Orthographic: &gltf.Orthographic{Xmag: 0, Ymag: 1, Znear: 1, Zfar: 3},
			}},
		}, 0, 1},
		{"orthographic-zfar-less-than-znear", &gltf.Document{
			Nodes: []*gltf.Node{{Camera: gltf.Index(0)}},
			Cameras: []*gltf.Camera{{
				Type:         "orthographic",
				Orthographic: &gltf.Orthographic{Xmag: 1, Ymag: 1, Znear: 2, Zfar: 1},
			}},
		}, 0, 1},
		{"singular-matrix", &gltf.Document{
			Nodes: []*gltf.Node{{
				Camera: gltf.Index(0),
				Matrix: [16]float64{0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1},
			}},
			Cameras: valid().Cameras,
		}, 0, 1},
		{"non-finite-matrix", &gltf.Document{
			Nodes: []*gltf.Node{{
				Camera: gltf.Index(0),
				Matrix: [16]float64{math.NaN(), 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1},
			}},
			Cameras: valid().Cameras,
		}, 0, 1},
		{"non-finite-translation", &gltf.Document{
			Nodes: []*gltf.Node{{
				Camera:      gltf.Index(0),
				Translation: [3]float64{math.Inf(1), 0, 0},
			}},
			Cameras: valid().Cameras,
		}, 0, 1},
		{"non-finite-scale", &gltf.Document{
			Nodes: []*gltf.Node{{
				Camera: gltf.Index(0),
				Scale:  [3]float64{1, math.NaN(), 1},
			}},
			Cameras: valid().Cameras,
		}, 0, 1},
		{"zero-scale", &gltf.Document{
			Nodes: []*gltf.Node{{
				Camera: gltf.Index(0),
				Scale:  [3]float64{1, 0, 1},
			}},
			Cameras: valid().Cameras,
		}, 0, 1},
		{"unknown-camera-type", &gltf.Document{
			Nodes: []*gltf.Node{{Camera: gltf.Index(0)}},
			Cameras: []*gltf.Camera{{
				Type:        "fisheye",
				Perspective: valid().Cameras[0].Perspective,
			}},
		}, 0, 1},
		{"no-projection", &gltf.Document{
			Nodes:   []*gltf.Node{{Camera: gltf.Index(0)}},
			Cameras: []*gltf.Camera{{Type: "perspective"}},
		}, 0, 1},
		{"perspective-property-missing", &gltf.Document{
			Nodes: []*gltf.Node{{Camera: gltf.Index(0)}},
			Cameras: []*gltf.Camera{{
				Type:         "perspective",
				Orthographic: &gltf.Orthographic{Xmag: 1, Ymag: 1, Znear: 1, Zfar: 3},
			}},
		}, 0, 1},
		{"orthographic-property-missing", &gltf.Document{
			Nodes: []*gltf.Node{{Camera: gltf.Index(0)}},
			Cameras: []*gltf.Camera{{
				Type:        "orthographic",
				Perspective: valid().Cameras[0].Perspective,
			}},
		}, 0, 1},
		{"both-projections", &gltf.Document{
			Nodes: []*gltf.Node{{Camera: gltf.Index(0)}},
			Cameras: []*gltf.Camera{{
				Type:         "perspective",
				Perspective:  valid().Cameras[0].Perspective,
				Orthographic: &gltf.Orthographic{Xmag: 1, Ymag: 1, Znear: 1, Zfar: 3},
			}},
		}, 0, 1},
		{"zfar-nan", func() *gltf.Document {
			d := valid()
			d.Cameras[0].Perspective.Zfar = gltf.Float(math.NaN())
			return d
		}(), 0, 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := modeler.ReadCameraMatrices(tt.doc, tt.nodeIndex, tt.aspect); err == nil {
				t.Errorf("ReadCameraMatrices() error = nil, want non-nil")
			}
		})
	}
}
