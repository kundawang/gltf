package modeler

import (
	"errors"
	"fmt"
	"math"

	"github.com/qmuntal/gltf"
)

// CameraMatrices contains the view and projection matrices of a camera
// instantiated by a node. Both matrices are 4x4 and stored in column-major
// order, the same convention used by gltf.Node.Matrix, so they can be
// uploaded to a graphics API without further transposition.
//
// The view matrix transforms world coordinates into the camera space defined
// by the node: +X points right, +Y points up and the camera looks towards -Z.
// The projection matrix transforms camera space into clip space using the
// glTF conventions: depth is -1 at the near clipping plane and +1 at the far
// clipping plane (or at infinity for a perspective camera without zfar).
type CameraMatrices struct {
	View       [16]float64
	Projection [16]float64
}

// ReadCameraMatrices returns the view and projection matrices of the camera
// instantiated by doc.Nodes[nodeIndex].
//
// The view matrix is the inverse of the global transformation of the camera
// node, i.e. the composition of the local transforms of the node and all of
// its ancestors. A node whose matrix property is set uses that matrix,
// otherwise its translation, rotation and scale are composed as specified by
// glTF. The projection matrix is derived from the camera properties following
// the glTF conventions and supports finite and infinite perspective
// projections as well as orthographic projections.
//
// For perspective cameras the effective aspect ratio is taken from the
// camera's aspectRatio property when defined. Otherwise aspectRatio is used,
// which is typically the width over height of the viewport. Orthographic
// cameras define their extents with xmag and ymag, so aspectRatio is ignored
// for them.
//
// ReadCameraMatrices never modifies doc. It returns a descriptive error when
// the document, the node or the camera is missing or malformed, when the
// node hierarchy contains a cycle or a node has several parents, when the
// camera parameters (near/far planes, yfov, aspect ratio, xmag, ymag) are out
// of range or non-finite, or when the global transformation of the camera
// node is not invertible.
func ReadCameraMatrices(doc *gltf.Document, nodeIndex int, aspectRatio float64) (CameraMatrices, error) {
	var matrices CameraMatrices
	if doc == nil {
		return matrices, errors.New("gltf: document is nil")
	}
	if nodeIndex < 0 || nodeIndex >= len(doc.Nodes) {
		return matrices, fmt.Errorf("gltf: node index %d overflows", nodeIndex)
	}
	node := doc.Nodes[nodeIndex]
	if node == nil {
		return matrices, fmt.Errorf("gltf: node %d is nil", nodeIndex)
	}
	if node.Camera == nil {
		return matrices, fmt.Errorf("gltf: node %d does not define a camera", nodeIndex)
	}
	cameraIndex := *node.Camera
	if cameraIndex < 0 || cameraIndex >= len(doc.Cameras) {
		return matrices, fmt.Errorf("gltf: camera index %d overflows", cameraIndex)
	}
	camera := doc.Cameras[cameraIndex]
	if camera == nil {
		return matrices, errors.New("gltf: camera is nil")
	}

	global, err := cameraGlobalTransform(doc, nodeIndex)
	if err != nil {
		return matrices, err
	}
	view, err := invertMatrix4(global)
	if err != nil {
		return matrices, err
	}
	projection, err := cameraProjection(camera, aspectRatio)
	if err != nil {
		return matrices, err
	}
	matrices.View = view
	matrices.Projection = projection
	return matrices, nil
}

var zeroMatrix4 [16]float64

// cameraGlobalTransform composes the local transforms from the root of the
// hierarchy down to nodeIndex, returning the node's global transformation in
// column-major order.
func cameraGlobalTransform(doc *gltf.Document, nodeIndex int) ([16]float64, error) {
	// glTF nodes only store their children, so build a child -> parent map
	// to be able to walk up the hierarchy. Nodes with several parents are
	// recorded as ambiguous and only rejected if they are part of the chain
	// that leads to the camera, so unrelated malformed nodes do not prevent
	// querying a well-formed camera.
	const ambiguousParent = -1
	parents := make(map[int]int, len(doc.Nodes))
	for i, node := range doc.Nodes {
		if node == nil {
			continue
		}
		for _, child := range node.Children {
			if child < 0 || child >= len(doc.Nodes) || doc.Nodes[child] == nil {
				// Not a valid edge, so it cannot be part of the camera's
				// ancestry. Indexing errors are reported when the camera
				// itself points to a missing node.
				continue
			}
			if parent, ok := parents[child]; ok && parent != i {
				parents[child] = ambiguousParent
			} else {
				parents[child] = i
			}
		}
	}

	// Collect the chain from the node up to its root. Revisiting a node is
	// only possible if the hierarchy contains a cycle.
	chain := make([]int, 0, len(doc.Nodes))
	visited := make(map[int]bool, len(doc.Nodes))
	for current := nodeIndex; ; {
		if visited[current] {
			return zeroMatrix4, errors.New("gltf: node hierarchy contains a cycle")
		}
		visited[current] = true
		chain = append(chain, current)
		parent, ok := parents[current]
		if !ok {
			break
		}
		if parent == ambiguousParent {
			return zeroMatrix4, fmt.Errorf("gltf: node %d has multiple parents", current)
		}
		current = parent
	}

	global := gltf.DefaultMatrix
	for i := len(chain) - 1; i >= 0; i-- {
		local, err := cameraLocalTransform(doc.Nodes[chain[i]], chain[i])
		if err != nil {
			return zeroMatrix4, err
		}
		global = multiplyMatrix4(global, local)
	}
	return global, nil
}

// cameraLocalTransform returns the local transformation of a node in
// column-major order. Matrix takes precedence over the TRS properties, using
// the same rules as the encoder: an empty or identity matrix means that the
// node is defined by its translation, rotation and scale.
func cameraLocalTransform(node *gltf.Node, index int) ([16]float64, error) {
	if node == nil {
		return zeroMatrix4, fmt.Errorf("gltf: node %d is nil", index)
	}
	if node.Matrix != gltf.DefaultMatrix && node.Matrix != zeroMatrix4 {
		for _, v := range node.Matrix {
			if !isFinite(v) {
				return zeroMatrix4, fmt.Errorf("gltf: node %d matrix contains a non-finite value", index)
			}
		}
		return node.Matrix, nil
	}

	translation := node.TranslationOrDefault()
	rotation := node.RotationOrDefault()
	scale := node.ScaleOrDefault()
	for _, v := range translation {
		if !isFinite(v) {
			return zeroMatrix4, fmt.Errorf("gltf: node %d translation contains a non-finite value", index)
		}
	}
	for _, v := range rotation {
		if !isFinite(v) {
			return zeroMatrix4, fmt.Errorf("gltf: node %d rotation contains a non-finite value", index)
		}
	}
	for _, v := range scale {
		if !isFinite(v) {
			return zeroMatrix4, fmt.Errorf("gltf: node %d scale contains a non-finite value", index)
		}
	}
	return composeTRS(translation, rotation, scale), nil
}

// composeTRS builds the local transformation matrix from translation,
// rotation and scale, as defined by the glTF specification. The result is in
// column-major order.
func composeTRS(translation [3]float64, rotation [4]float64, scale [3]float64) [16]float64 {
	x, y, z, w := rotation[0], rotation[1], rotation[2], rotation[3]
	var m [16]float64
	m[0] = scale[0] * (1 - 2*(y*y+z*z))
	m[1] = scale[0] * (2 * (x*y + z*w))
	m[2] = scale[0] * (2 * (x*z - y*w))
	m[4] = scale[1] * (2 * (x*y - z*w))
	m[5] = scale[1] * (1 - 2*(x*x+z*z))
	m[6] = scale[1] * (2 * (y*z + x*w))
	m[8] = scale[2] * (2 * (x*z + y*w))
	m[9] = scale[2] * (2 * (y*z - x*w))
	m[10] = scale[2] * (1 - 2*(x*x+y*y))
	m[12] = translation[0]
	m[13] = translation[1]
	m[14] = translation[2]
	m[15] = 1
	return m
}

// multiplyMatrix4 returns a*b, where both matrices are in column-major order.
func multiplyMatrix4(a, b [16]float64) [16]float64 {
	var product [16]float64
	for col := 0; col < 4; col++ {
		for row := 0; row < 4; row++ {
			var sum float64
			for k := 0; k < 4; k++ {
				sum += a[k*4+row] * b[col*4+k]
			}
			product[col*4+row] = sum
		}
	}
	return product
}

// invertMatrix4 returns the inverse of a column-major 4x4 matrix.
//
// The view matrix is the inverse of the camera's global transformation, so a
// singular transformation (for example one with an all-zero column) cannot
// produce a valid view and yields an error. Matrices whose entries are not
// finite, or whose inverse cannot be represented, are also rejected.
func invertMatrix4(m [16]float64) ([16]float64, error) {
	for _, v := range m {
		if !isFinite(v) {
			return zeroMatrix4, errors.New("gltf: camera transformation matrix contains a non-finite value")
		}
	}
	// Gauss-Jordan elimination on the augmented matrix [m | I].
	var aug [4][8]float64
	for row := 0; row < 4; row++ {
		for col := 0; col < 4; col++ {
			aug[row][col] = m[col*4+row]
		}
		aug[row][4+row] = 1
	}
	for col := 0; col < 4; col++ {
		// Partial pivoting keeps the elimination stable.
		pivot := col
		for row := col + 1; row < 4; row++ {
			if math.Abs(aug[row][col]) > math.Abs(aug[pivot][col]) {
				pivot = row
			}
		}
		// A zero pivot means the matrix is singular.
		if aug[pivot][col] == 0 {
			return zeroMatrix4, errors.New("gltf: camera transformation matrix is not invertible")
		}
		if pivot != col {
			aug[col], aug[pivot] = aug[pivot], aug[col]
		}
		scale := aug[col][col]
		for k := 0; k < 8; k++ {
			aug[col][k] /= scale
		}
		for row := 0; row < 4; row++ {
			if row == col {
				continue
			}
			factor := aug[row][col]
			if factor == 0 {
				continue
			}
			for k := 0; k < 8; k++ {
				aug[row][k] -= factor * aug[col][k]
			}
		}
	}

	var inverse [16]float64
	for row := 0; row < 4; row++ {
		for col := 0; col < 4; col++ {
			v := aug[row][4+col]
			if !isFinite(v) {
				return zeroMatrix4, errors.New("gltf: camera transformation matrix is not invertible")
			}
			inverse[col*4+row] = v
		}
	}
	return inverse, nil
}

// cameraProjection builds the projection matrix of camera in column-major
// order.
func cameraProjection(camera *gltf.Camera, aspectRatio float64) ([16]float64, error) {
	if camera == nil {
		return zeroMatrix4, errors.New("gltf: camera is nil")
	}
	if camera.Perspective != nil && camera.Orthographic != nil {
		return zeroMatrix4, errors.New("gltf: camera must not define both perspective and orthographic properties")
	}

	cameraType := camera.Type
	if cameraType == "" {
		switch {
		case camera.Perspective != nil:
			cameraType = "perspective"
		case camera.Orthographic != nil:
			cameraType = "orthographic"
		default:
			return zeroMatrix4, errors.New("gltf: camera does not define a projection")
		}
	}
	switch cameraType {
	case "perspective":
		return perspectiveProjection(camera.Perspective, aspectRatio)
	case "orthographic":
		return orthographicProjection(camera.Orthographic)
	default:
		return zeroMatrix4, fmt.Errorf("gltf: unknown camera type %q", cameraType)
	}
}

func perspectiveProjection(p *gltf.Perspective, aspectRatio float64) ([16]float64, error) {
	if p == nil {
		return zeroMatrix4, errors.New("gltf: perspective camera does not define the perspective property")
	}
	if !isFinite(p.Yfov) || p.Yfov <= 0 || p.Yfov >= math.Pi {
		return zeroMatrix4, errors.New("gltf: camera yfov must be finite, greater than zero and less than pi")
	}
	if !isFinite(p.Znear) || p.Znear <= 0 {
		return zeroMatrix4, errors.New("gltf: camera znear must be finite and greater than zero")
	}
	aspect, err := effectiveAspectRatio(p.AspectRatio, aspectRatio)
	if err != nil {
		return zeroMatrix4, err
	}
	var zfar float64
	infinite := p.Zfar == nil
	if !infinite {
		zfar = *p.Zfar
		if !isFinite(zfar) || zfar <= p.Znear {
			return zeroMatrix4, errors.New("gltf: camera zfar must be finite and greater than znear")
		}
	}

	f := 1 / math.Tan(p.Yfov/2)
	var m [16]float64
	m[0] = f / aspect
	m[5] = f
	m[11] = -1
	if infinite {
		// Map the near plane to -1 and the far plane at infinity to +1.
		m[10] = -1
		m[14] = -2 * p.Znear
	} else {
		m[10] = (zfar + p.Znear) / (p.Znear - zfar)
		m[14] = 2 * zfar * p.Znear / (p.Znear - zfar)
	}
	for _, v := range m {
		if !isFinite(v) {
			return zeroMatrix4, errors.New("gltf: camera projection contains a non-finite value")
		}
	}
	return m, nil
}

func orthographicProjection(o *gltf.Orthographic) ([16]float64, error) {
	if o == nil {
		return zeroMatrix4, errors.New("gltf: orthographic camera does not define the orthographic property")
	}
	if !isFinite(o.Xmag) || o.Xmag <= 0 {
		return zeroMatrix4, errors.New("gltf: camera xmag must be finite and greater than zero")
	}
	if !isFinite(o.Ymag) || o.Ymag <= 0 {
		return zeroMatrix4, errors.New("gltf: camera ymag must be finite and greater than zero")
	}
	if !isFinite(o.Znear) || o.Znear <= 0 {
		return zeroMatrix4, errors.New("gltf: camera znear must be finite and greater than zero")
	}
	if !isFinite(o.Zfar) || o.Zfar <= o.Znear {
		return zeroMatrix4, errors.New("gltf: camera zfar must be finite and greater than znear")
	}

	var m [16]float64
	m[0] = 1 / o.Xmag
	m[5] = 1 / o.Ymag
	m[10] = -2 / (o.Zfar - o.Znear)
	m[14] = -(o.Zfar + o.Znear) / (o.Zfar - o.Znear)
	m[15] = 1
	for _, v := range m {
		if !isFinite(v) {
			return zeroMatrix4, errors.New("gltf: camera projection contains a non-finite value")
		}
	}
	return m, nil
}

// effectiveAspectRatio returns the aspect ratio defined by the camera, if
// any, or the externally supplied viewport aspect ratio otherwise.
func effectiveAspectRatio(explicit *float64, external float64) (float64, error) {
	if explicit != nil {
		if !isFinite(*explicit) || *explicit <= 0 {
			return 0, errors.New("gltf: camera aspect ratio must be finite and greater than zero")
		}
		return *explicit, nil
	}
	if !isFinite(external) || external <= 0 {
		return 0, errors.New("gltf: camera aspect ratio must be finite and greater than zero")
	}
	return external, nil
}

func isFinite(v float64) bool {
	return !math.IsNaN(v) && !math.IsInf(v, 0)
}
