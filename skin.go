package gltf

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
)

// JointMatrices evaluates the skinning matrices of the skinned mesh node
// meshNode in scene and returns one matrix per entry of the referenced
// Skin.Joints, in that same order.
//
// Every returned matrix transforms a position from joint local space into
// the local space of meshNode:
//
//	J = inverse(world(meshNode)) * world(joint) * inverseBindMatrix[joint]
//
// World transforms combine the node Matrix (when present) or its
// Translation/Rotation/Scale properties with all ancestors through the
// node hierarchy.
//
// Inverse bind matrices are resolved in one of two ways. Callers that have
// already decoded them pass the matrices in Skin.Joints order as extra
// arguments (or with ibm... for a slice). When none are passed, the skin's
// inverseBindMatrices accessor is read from the document; if the skin does
// not define inverseBindMatrices every inverse-bind matrix defaults to
// identity.
//
// pose optionally overrides the current local matrix of individual nodes,
// keyed by node index. It only affects this evaluation and never mutates
// the document, which makes the function safe to call every frame with a
// fresh pose. Nodes missing from pose use their document-defined transform.
//
// If scene is nil the active scene declared by Document.Scene is used.
//
// The same skin can be evaluated for several mesh nodes: the result only
// depends on the meshNode and pose arguments of each call, so every mesh
// gets matrices expressed in its own local space.
//
// An error is returned for invalid document indices, joints pointing
// outside Document.Nodes, hierarchy cycles, inverse-bind/joint count
// mismatches, a mesh world transform that cannot be inverted, and any
// non-finite transform value.
func (doc *Document) JointMatrices(scene *Scene, meshNode int, pose map[int][16]float64, inverseBindMatrices ...[16]float64) ([][16]float64, error) {
	return JointMatrices(doc, scene, meshNode, pose, inverseBindMatrices...)
}

// JointMatrices is the package-level entry point for evaluating joint
// matrices; see Document.JointMatrices for the full contract.
func JointMatrices(doc *Document, scene *Scene, meshNode int, pose map[int][16]float64, inverseBindMatrices ...[16]float64) ([][16]float64, error) {
	if doc == nil {
		return nil, errors.New("gltf: cannot evaluate joint matrices of a nil document")
	}
	if meshNode < 0 || meshNode >= len(doc.Nodes) || doc.Nodes[meshNode] == nil {
		return nil, fmt.Errorf("gltf: invalid skinned mesh node index %d", meshNode)
	}
	meshNodeValue := doc.Nodes[meshNode]
	if meshNodeValue.Skin == nil {
		return nil, fmt.Errorf("gltf: node %d does not reference a skin", meshNode)
	}
	skinIndex := *meshNodeValue.Skin
	if skinIndex < 0 || skinIndex >= len(doc.Skins) || doc.Skins[skinIndex] == nil {
		return nil, fmt.Errorf("gltf: node %d references invalid skin index %d", meshNode, skinIndex)
	}
	skin := doc.Skins[skinIndex]
	joints := skin.Joints
	if len(joints) == 0 {
		return nil, fmt.Errorf("gltf: skin %d defines no joints", skinIndex)
	}
	for order, joint := range joints {
		if joint < 0 || joint >= len(doc.Nodes) || doc.Nodes[joint] == nil {
			return nil, fmt.Errorf("gltf: skin %d joints[%d] references invalid node index %d",
				skinIndex, order, joint)
		}
	}

	ibm := inverseBindMatrices
	if len(ibm) == 0 {
		decoded, err := readSkinInverseBindMatrices(doc, skin, len(joints))
		if err != nil {
			return nil, err
		}
		ibm = decoded
	} else if len(ibm) != len(joints) {
		return nil, fmt.Errorf("gltf: inverse bind matrix count %d does not match joint count %d",
			len(ibm), len(joints))
	}
	for i, matrix := range ibm {
		if !finiteMat4(matrix) {
			return nil, fmt.Errorf("gltf: inverse bind matrix %d contains a non-finite value", i)
		}
	}
	if err := validateScene(doc, scene); err != nil {
		return nil, err
	}

	parents, err := buildParents(doc)
	if err != nil {
		return nil, err
	}
	if err := detectHierarchyCycles(doc.Nodes, parents); err != nil {
		return nil, err
	}

	world := make([]*[16]float64, len(doc.Nodes))
	var worldOf func(nodeIndex int) ([16]float64, error)
	worldOf = func(nodeIndex int) ([16]float64, error) {
		if cached := world[nodeIndex]; cached != nil {
			return *cached, nil
		}
		local, err := poseOrNodeMatrix(doc.Nodes[nodeIndex], nodeIndex, pose)
		if err != nil {
			return DefaultMatrix, err
		}
		w := local
		if parent := parents[nodeIndex]; parent >= 0 {
			parentWorld, err := worldOf(parent)
			if err != nil {
				return DefaultMatrix, err
			}
			w = mulMat4(parentWorld, local)
		}
		if !finiteMat4(w) {
			return DefaultMatrix, fmt.Errorf("gltf: world matrix of node %d contains a non-finite value", nodeIndex)
		}
		world[nodeIndex] = &w
		return w, nil
	}

	meshWorld, err := worldOf(meshNode)
	if err != nil {
		return nil, err
	}
	meshInverse, ok := inverseMat4(meshWorld)
	if !ok {
		return nil, fmt.Errorf("gltf: world transform of mesh node %d is not invertible", meshNode)
	}
	if !finiteMat4(meshInverse) {
		return nil, fmt.Errorf("gltf: inverse world transform of mesh node %d contains a non-finite value", meshNode)
	}

	matrices := make([][16]float64, len(joints))
	for i, joint := range joints {
		jointWorld, err := worldOf(joint)
		if err != nil {
			return nil, err
		}
		jointMatrix := mulMat4(meshInverse, jointWorld)
		if i < len(ibm) {
			jointMatrix = mulMat4(jointMatrix, ibm[i])
		}
		if !finiteMat4(jointMatrix) {
			return nil, fmt.Errorf("gltf: joint %d (node %d) matrix contains a non-finite value", i, joint)
		}
		matrices[i] = jointMatrix
	}
	return matrices, nil
}

// inverseBindMatrixFloatCount is the number of float32 values in a mat4.
const inverseBindMatrixFloatCount = 16

// readSkinInverseBindMatrices decodes the FLOAT MAT4 accessor referenced by
// skin.InverseBindMatrices. It returns nil without error when the skin does
// not declare inverse bind matrices, in which case every matrix is treated
// as identity. The accessor count must equal jointCount.
func readSkinInverseBindMatrices(doc *Document, skin *Skin, jointCount int) ([][16]float64, error) {
	accessorIndex := skin.InverseBindMatrices
	if accessorIndex == nil {
		return nil, nil
	}
	if *accessorIndex < 0 || *accessorIndex >= len(doc.Accessors) || doc.Accessors[*accessorIndex] == nil {
		return nil, fmt.Errorf("gltf: skin inverseBindMatrices references invalid accessor %d", *accessorIndex)
	}
	accessor := doc.Accessors[*accessorIndex]
	if accessor.Type != AccessorMat4 {
		return nil, fmt.Errorf("gltf: inverseBindMatrices accessor %d must have MAT4 type, got %v",
			*accessorIndex, accessor.Type)
	}
	if accessor.ComponentType != ComponentFloat {
		return nil, fmt.Errorf("gltf: inverseBindMatrices accessor %d must use FLOAT components, got %v",
			*accessorIndex, accessor.ComponentType)
	}
	if accessor.Count < 0 {
		return nil, fmt.Errorf("gltf: inverseBindMatrices accessor %d has negative count %d",
			*accessorIndex, accessor.Count)
	}
	if accessor.Count != jointCount {
		return nil, fmt.Errorf("gltf: inverse bind matrix count %d does not match joint count %d",
			accessor.Count, jointCount)
	}

	matrices := make([][16]float64, accessor.Count)
	if accessor.BufferView == nil {
		// An accessor without a buffer view is zero-initialized per glTF 2.0.
		return matrices, nil
	}
	viewIndex := *accessor.BufferView
	if viewIndex < 0 || viewIndex >= len(doc.BufferViews) || doc.BufferViews[viewIndex] == nil {
		return nil, fmt.Errorf("gltf: inverseBindMatrices accessor %d references invalid bufferView %d",
			*accessorIndex, viewIndex)
	}
	view := doc.BufferViews[viewIndex]
	if view.Buffer < 0 || view.Buffer >= len(doc.Buffers) || doc.Buffers[view.Buffer] == nil {
		return nil, fmt.Errorf("gltf: inverseBindMatrices bufferView %d references invalid buffer %d",
			viewIndex, view.Buffer)
	}
	data := doc.Buffers[view.Buffer].Data

	stride := view.ByteStride
	if stride == 0 {
		stride = inverseBindMatrixFloatCount * 4
	}
	matrixBytes := inverseBindMatrixFloatCount * 4
	if stride < matrixBytes {
		return nil, fmt.Errorf("gltf: inverseBindMatrices bufferView %d stride %d is smaller than a MAT4",
			viewIndex, stride)
	}
	base := view.ByteOffset + accessor.ByteOffset
	if base < 0 {
		return nil, fmt.Errorf("gltf: inverseBindMatrices accessor %d has a negative byte offset", *accessorIndex)
	}
	for i := 0; i < accessor.Count; i++ {
		offset := base + i*stride
		if offset < 0 || offset+matrixBytes > len(data) {
			return nil, fmt.Errorf("gltf: inverseBindMatrices accessor %d is out of buffer bounds at matrix %d",
				*accessorIndex, i)
		}
		for component := 0; component < inverseBindMatrixFloatCount; component++ {
			start := offset + component*4
			bits := binary.LittleEndian.Uint32(data[start : start+4])
			matrices[i][component] = float64(math.Float32frombits(bits))
		}
	}
	return matrices, nil
}

// LocalMatrix returns the local transformation matrix of n.
//
// A non-default Matrix is returned as-is. Otherwise the matrix is composed
// as Translation * Rotation * Scale using the glTF defaults for missing
// values. The returned matrix is column-major, matching Node.Matrix.
// An error is returned if any component is not finite.
func (n *Node) LocalMatrix() ([16]float64, error) {
	if n == nil {
		return DefaultMatrix, errors.New("gltf: cannot compute the local matrix of a nil node")
	}
	return composeNodeMatrix(n)
}

func poseOrNodeMatrix(n *Node, nodeIndex int, pose map[int][16]float64) ([16]float64, error) {
	if pose != nil {
		if m, ok := pose[nodeIndex]; ok {
			if !finiteMat4(m) {
				return DefaultMatrix, fmt.Errorf("gltf: pose matrix of node %d contains a non-finite value", nodeIndex)
			}
			return m, nil
		}
	}
	m, err := composeNodeMatrix(n)
	if err != nil {
		return DefaultMatrix, fmt.Errorf("gltf: node %d transform is invalid: %w", nodeIndex, err)
	}
	return m, nil
}

func composeNodeMatrix(n *Node) ([16]float64, error) {
	if n.Matrix != emptyMatrix && n.Matrix != DefaultMatrix {
		if !finiteMat4(n.Matrix) {
			return DefaultMatrix, errors.New("matrix contains a non-finite value")
		}
		return n.Matrix, nil
	}
	translation := n.TranslationOrDefault()
	rotation := n.RotationOrDefault()
	scale := n.ScaleOrDefault()
	for _, v := range translation {
		if !isFinite(v) {
			return DefaultMatrix, errors.New("translation contains a non-finite value")
		}
	}
	for _, v := range rotation {
		if !isFinite(v) {
			return DefaultMatrix, errors.New("rotation contains a non-finite value")
		}
	}
	for _, v := range scale {
		if !isFinite(v) {
			return DefaultMatrix, errors.New("scale contains a non-finite value")
		}
	}
	m := mulMat4(
		translationMat4(translation),
		mulMat4(quaternionMat4(rotation), scaleMat4(scale)),
	)
	if !finiteMat4(m) {
		return DefaultMatrix, errors.New("composed transform contains a non-finite value")
	}
	return m, nil
}

// validateScene resolves a nil scene against Document.Scene and checks that
// every declared root node is a valid reference.
func validateScene(doc *Document, scene *Scene) error {
	if scene == nil && doc.Scene != nil {
		sceneIndex := *doc.Scene
		if sceneIndex >= 0 && sceneIndex < len(doc.Scenes) {
			scene = doc.Scenes[sceneIndex]
		}
	}
	if scene == nil {
		return nil
	}
	for _, root := range scene.Nodes {
		if root < 0 || root >= len(doc.Nodes) || doc.Nodes[root] == nil {
			return fmt.Errorf("gltf: scene references invalid root node index %d", root)
		}
	}
	return nil
}

// buildParents maps every node index to its parent index (-1 for roots).
// Hierarchy links are the Node.Children references.
func buildParents(doc *Document) ([]int, error) {
	parents := make([]int, len(doc.Nodes))
	for i := range parents {
		parents[i] = -1
	}
	for parentIndex, parent := range doc.Nodes {
		if parent == nil {
			continue
		}
		for _, child := range parent.Children {
			if child < 0 || child >= len(doc.Nodes) || doc.Nodes[child] == nil {
				return nil, fmt.Errorf("gltf: node %d references invalid child index %d", parentIndex, child)
			}
			if parents[child] >= 0 && parents[child] != parentIndex {
				return nil, fmt.Errorf("gltf: node %d has multiple parents (%d and %d)",
					child, parents[child], parentIndex)
			}
			parents[child] = parentIndex
		}
	}
	return parents, nil
}

const (
	hierarchyUnvisited = iota
	hierarchyActive
	hierarchyDone
)

func detectHierarchyCycles(nodes []*Node, parents []int) error {
	state := make([]uint8, len(nodes))
	var visit func(nodeIndex int) error
	visit = func(nodeIndex int) error {
		switch state[nodeIndex] {
		case hierarchyActive:
			return fmt.Errorf("gltf: node hierarchy contains a cycle at node %d", nodeIndex)
		case hierarchyDone:
			return nil
		}
		state[nodeIndex] = hierarchyActive
		if parent := parents[nodeIndex]; parent >= 0 {
			if err := visit(parent); err != nil {
				return err
			}
		}
		state[nodeIndex] = hierarchyDone
		return nil
	}
	for i := range nodes {
		if nodes[i] == nil {
			continue
		}
		if err := visit(i); err != nil {
			return err
		}
	}
	return nil
}

func isFinite(v float64) bool {
	return !math.IsNaN(v) && !math.IsInf(v, 0)
}

func finiteMat4(m [16]float64) bool {
	for _, v := range m {
		if !isFinite(v) {
			return false
		}
	}
	return true
}

// mulMat4 returns a*b. Matrices are column-major.
func mulMat4(a, b [16]float64) [16]float64 {
	var out [16]float64
	for column := 0; column < 4; column++ {
		for row := 0; row < 4; row++ {
			out[column*4+row] =
				a[0*4+row]*b[column*4+0] +
					a[1*4+row]*b[column*4+1] +
					a[2*4+row]*b[column*4+2] +
					a[3*4+row]*b[column*4+3]
		}
	}
	return out
}

func translationMat4(t [3]float64) [16]float64 {
	m := DefaultMatrix
	m[12], m[13], m[14] = t[0], t[1], t[2]
	return m
}

func scaleMat4(s [3]float64) [16]float64 {
	return [16]float64{
		s[0], 0, 0, 0,
		0, s[1], 0, 0,
		0, 0, s[2], 0,
		0, 0, 0, 1,
	}
}

// quaternionMat4 converts an (x, y, z, w) quaternion to a column-major
// rotation matrix.
func quaternionMat4(q [4]float64) [16]float64 {
	x, y, z, w := q[0], q[1], q[2], q[3]
	xx, yy, zz := x*x, y*y, z*z
	xy, xz, yz := x*y, x*z, y*z
	wx, wy, wz := w*x, w*y, w*z
	return [16]float64{
		1 - 2*(yy+zz), 2 * (xy + wz), 2 * (xz - wy), 0,
		2 * (xy - wz), 1 - 2*(xx+zz), 2 * (yz + wx), 0,
		2 * (xz + wy), 2 * (yz - wx), 1 - 2*(xx+yy), 0,
		0, 0, 0, 1,
	}
}

// inverseMat4 returns the inverse of a general 4x4 matrix using Gauss
// Jordan elimination with partial pivoting. The second result is false when
// m is singular. Matrices are column-major.
func inverseMat4(m [16]float64) ([16]float64, bool) {
	// Original per-column largest magnitude, used as relative pivot scale so
	// uniformly tiny but valid transforms are not rejected as singular.
	var columnScale [4]float64
	for c := 0; c < 4; c++ {
		for r := 0; r < 4; r++ {
			columnScale[c] = math.Max(columnScale[c], math.Abs(m[c*4+r]))
		}
	}

	var a [4][8]float64
	for r := 0; r < 4; r++ {
		for c := 0; c < 4; c++ {
			a[r][c] = m[c*4+r]
		}
		a[r][r+4] = 1
	}

	for col := 0; col < 4; col++ {
		pivot := col
		for r := col + 1; r < 4; r++ {
			if math.Abs(a[r][col]) > math.Abs(a[pivot][col]) {
				pivot = r
			}
		}
		pivotMagnitude := math.Abs(a[pivot][col])
		if pivotMagnitude == 0 {
			return DefaultMatrix, false
		}
		if columnScale[col] > 0 && pivotMagnitude <= 1e-12*columnScale[col] {
			return DefaultMatrix, false
		}
		if pivot != col {
			a[col], a[pivot] = a[pivot], a[col]
		}
		divisor := a[col][col]
		for c := 0; c < 8; c++ {
			a[col][c] /= divisor
		}
		for r := 0; r < 4; r++ {
			if r == col {
				continue
			}
			factor := a[r][col]
			if factor == 0 {
				continue
			}
			for c := 0; c < 8; c++ {
				a[r][c] -= factor * a[col][c]
			}
		}
	}

	var inv [16]float64
	for r := 0; r < 4; r++ {
		for c := 0; c < 4; c++ {
			inv[c*4+r] = a[r][c+4]
		}
	}
	return inv, true
}
