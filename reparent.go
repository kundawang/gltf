package gltf

import (
	"errors"
	"fmt"
	"math"
)

// RootParent is the parent index that places a node at the root of a scene.
// It is the value to pass as newParentIndex to (*Document).ReparentNode when
// the node must be detached from its current parent.
const RootParent = -1

// epsMatrix is the relative tolerance used when deciding whether a matrix can
// be represented losslessly by translation/rotation/scale.
const epsMatrix = 1e-9

// ReparentNode moves nodeIndex in the scene sceneIndex so that its new parent
// is newParentIndex, preserving the world transform of the node and of its
// whole subtree.
//
// Pass RootParent (-1) as newParentIndex to detach the node from its current
// parent and make it a root of the scene. The scene root list and the old and
// new parent Children lists are updated accordingly. Only the hierarchy slices
// and the moved node's local transform are modified; Name, Mesh, Camera, Skin,
// Weights, Extensions, Extras and every other resource reference are kept.
//
// The new local transform is computed as:
//
//	localNew = inverse(worldNewParent) * worldNode
//
// where worldNode is evaluated before any mutation. When localNew can be
// represented losslessly as Translation * Rotation * Scale the node stores TRS
// properties; otherwise the general 4x4 result is stored in Node.Matrix, so
// sheared or reflected transforms are preserved as well.
//
// ReparentNode is atomic: if sceneIndex, nodeIndex or newParentIndex is out of
// range, newParentIndex is nodeIndex or one of its descendants, the node is
// claimed by more than one parent, the new parent world matrix is not
// invertible, or any transform value involved or resulting is not finite, the
// document is left untouched and a non-nil error is returned. Reparenting a
// node under its current parent, or making an existing scene root a root
// again, is a no-op.
func (doc *Document) ReparentNode(sceneIndex, nodeIndex, newParentIndex int) error {
	if doc == nil {
		return errors.New("gltf: cannot reparent a node of a nil document")
	}
	if sceneIndex < 0 || sceneIndex >= len(doc.Scenes) || doc.Scenes[sceneIndex] == nil {
		return fmt.Errorf("gltf: reparent node: scene index %d out of range", sceneIndex)
	}
	if nodeIndex < 0 || nodeIndex >= len(doc.Nodes) || doc.Nodes[nodeIndex] == nil {
		return fmt.Errorf("gltf: reparent node: node index %d out of range", nodeIndex)
	}
	toRoot := newParentIndex == RootParent
	if !toRoot {
		if newParentIndex < 0 || newParentIndex >= len(doc.Nodes) || doc.Nodes[newParentIndex] == nil {
			return fmt.Errorf("gltf: reparent node %d: parent index %d out of range", nodeIndex, newParentIndex)
		}
		if newParentIndex == nodeIndex {
			return fmt.Errorf("gltf: reparent node %d: cannot parent a node to itself", nodeIndex)
		}
		if nodeIsDescendant(doc.Nodes, nodeIndex, newParentIndex) {
			return fmt.Errorf("gltf: reparent node %d: new parent %d is a descendant of the node", nodeIndex, newParentIndex)
		}
	}

	scene := doc.Scenes[sceneIndex]

	// Count every ownership slot within this scene: each appearance of the node
	// in a Node.Children list plus each appearance in the scene root list. Valid
	// glTF gives a node at most one parent and at most one root slot; anything
	// else is ambiguous and must be rejected instead of guessing a world
	// transform.
	claims := make([]int, 0, 2)
	for parent, node := range doc.Nodes {
		if node == nil {
			continue
		}
		for _, child := range node.Children {
			if child == nodeIndex {
				claims = append(claims, parent)
			}
		}
	}
	rootSlots := 0
	for _, root := range scene.Nodes {
		if root == nodeIndex {
			rootSlots++
		}
	}
	if len(claims)+rootSlots > 1 {
		return fmt.Errorf("gltf: reparent node %d: node has multiple parents", nodeIndex)
	}
	oldParent := RootParent
	if len(claims) == 1 {
		oldParent = claims[0]
	}

	// Re-parenting into the exact current slot is a no-op. Returning before
	// doing any math also avoids introducing floating point drift.
	if toRoot {
		if oldParent == RootParent && rootSlots == 1 {
			return nil
		}
	} else if oldParent == newParentIndex {
		return nil
	}

	parentOf, ambiguousParent := parentIndex(doc.Nodes)

	worldNode, err := nodeWorldMatrix(doc.Nodes, parentOf, ambiguousParent, nodeIndex)
	if err != nil {
		return fmt.Errorf("gltf: reparent node %d: %w", nodeIndex, err)
	}

	worldParent := DefaultMatrix
	if !toRoot {
		worldParent, err = nodeWorldMatrix(doc.Nodes, parentOf, ambiguousParent, newParentIndex)
		if err != nil {
			return fmt.Errorf("gltf: reparent node %d: new parent %d: %w", nodeIndex, newParentIndex, err)
		}
	}

	inverseParent, invertible := invertMat4(worldParent)
	if !invertible || !inverseIsAccurate(worldParent, inverseParent) {
		if toRoot {
			return fmt.Errorf("gltf: reparent node %d: root transform is not invertible", nodeIndex)
		}
		return fmt.Errorf("gltf: reparent node %d: new parent %d transform is not invertible", nodeIndex, newParentIndex)
	}

	newLocal := mulMat4(inverseParent, worldNode)
	if !finiteMat4(newLocal) {
		return fmt.Errorf("gltf: reparent node %d: resulting local transform is not finite", nodeIndex)
	}

	// All validation and math succeeded; mutate the document as a last step.
	if oldParent >= 0 {
		old := doc.Nodes[oldParent]
		children := make([]int, 0, len(old.Children))
		for _, child := range old.Children {
			if child != nodeIndex {
				children = append(children, child)
			}
		}
		old.Children = children
	}

	if rootSlots == 1 {
		roots := make([]int, 0, len(scene.Nodes))
		for _, root := range scene.Nodes {
			if root != nodeIndex {
				roots = append(roots, root)
			}
		}
		scene.Nodes = roots
	}

	if toRoot {
		scene.Nodes = append(scene.Nodes, nodeIndex)
	} else {
		parent := doc.Nodes[newParentIndex]
		parent.Children = append(parent.Children, nodeIndex)
	}

	storeNodeTransform(doc.Nodes[nodeIndex], newLocal)
	return nil
}

// ReparentNode is the package-level version of (*Document).ReparentNode.
func ReparentNode(doc *Document, sceneIndex, nodeIndex, newParentIndex int) error {
	return doc.ReparentNode(sceneIndex, nodeIndex, newParentIndex)
}

// SetNodeParent reparents nodeIndex like ReparentNode, taking the new parent as
// an optional index. A nil newParent places the node at the root of the scene,
// mirroring the optional-index convention used elsewhere in this package.
func (doc *Document) SetNodeParent(sceneIndex, nodeIndex int, newParent *int) error {
	parent := RootParent
	if newParent != nil {
		parent = *newParent
	}
	return doc.ReparentNode(sceneIndex, nodeIndex, parent)
}

// LocalMatrix returns the local transformation matrix of n in column-major
// order, matching the layout of Node.Matrix.
//
// A non-default Node.Matrix is returned as-is. Otherwise the matrix is composed
// as Translation * Rotation * Scale, using the glTF defaults for missing
// values. An error is returned if n is nil or any component is not finite.
func (n *Node) LocalMatrix() ([16]float64, error) {
	if n == nil {
		return DefaultMatrix, errors.New("gltf: cannot compute the local matrix of a nil node")
	}
	return composeNodeMatrix(n)
}

// NodeWorldMatrix returns the world transformation matrix of nodeIndex in
// column-major order.
//
// World transforms combine the node local matrix with all its ancestors,
// following the Node.Children links. Scene root lists are not needed for this
// computation. An error is returned for invalid indices, multiple parents,
// hierarchy cycles, non-finite transform values or a nil document.
func (doc *Document) NodeWorldMatrix(nodeIndex int) ([16]float64, error) {
	if doc == nil {
		return DefaultMatrix, errors.New("gltf: cannot compute a world matrix of a nil document")
	}
	if nodeIndex < 0 || nodeIndex >= len(doc.Nodes) || doc.Nodes[nodeIndex] == nil {
		return DefaultMatrix, fmt.Errorf("gltf: node index %d out of range", nodeIndex)
	}
	parentOf, ambiguousParent := parentIndex(doc.Nodes)
	world, err := nodeWorldMatrix(doc.Nodes, parentOf, ambiguousParent, nodeIndex)
	if err != nil {
		return DefaultMatrix, err
	}
	if !finiteMat4(world) {
		return DefaultMatrix, fmt.Errorf("gltf: world matrix of node %d is not finite", nodeIndex)
	}
	return world, nil
}

// parentIndex maps every child node to its parent (RootParent for nodes without
// one). Nodes claimed by more than one parent are flagged in ambiguous.
func parentIndex(nodes []*Node) (parent []int, ambiguous []bool) {
	parent = make([]int, len(nodes))
	ambiguous = make([]bool, len(nodes))
	for i := range parent {
		parent[i] = RootParent
	}
	for parentIdx, n := range nodes {
		if n == nil {
			continue
		}
		for _, child := range n.Children {
			if child < 0 || child >= len(nodes) || nodes[child] == nil {
				continue
			}
			switch {
			case parent[child] == RootParent:
				parent[child] = parentIdx
			case parent[child] != parentIdx:
				ambiguous[child] = true
			}
		}
	}
	return parent, ambiguous
}

// nodeIsDescendant reports whether candidate is reachable from ancestor by
// following Node.Children links. Invalid references and cycles are ignored.
func nodeIsDescendant(nodes []*Node, ancestor, candidate int) bool {
	if ancestor < 0 || ancestor >= len(nodes) || nodes[ancestor] == nil {
		return false
	}
	visited := make(map[int]struct{}, 8)
	stack := []int{ancestor}
	for len(stack) > 0 {
		current := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if current < 0 || current >= len(nodes) || nodes[current] == nil {
			continue
		}
		if _, seen := visited[current]; seen {
			continue
		}
		visited[current] = struct{}{}
		for _, child := range nodes[current].Children {
			if child == candidate {
				return true
			}
			if child >= 0 && child < len(nodes) && nodes[child] != nil {
				stack = append(stack, child)
			}
		}
	}
	return false
}

// nodeWorldMatrix composes the local matrices from the topmost ancestor down
// to nodeIndex.
func nodeWorldMatrix(nodes []*Node, parentOf []int, ambiguousParent []bool, nodeIndex int) ([16]float64, error) {
	chain := make([]int, 0, 8)
	seen := make(map[int]struct{}, 8)
	for current := nodeIndex; current != RootParent; current = parentOf[current] {
		if current < 0 || current >= len(nodes) || nodes[current] == nil {
			return DefaultMatrix, fmt.Errorf("node index %d out of range", current)
		}
		if _, ok := seen[current]; ok {
			return DefaultMatrix, fmt.Errorf("node hierarchy contains a cycle at node %d", current)
		}
		seen[current] = struct{}{}
		if ambiguousParent[current] {
			return DefaultMatrix, fmt.Errorf("node %d has multiple parents", current)
		}
		chain = append(chain, current)
	}

	world := DefaultMatrix
	for i := len(chain) - 1; i >= 0; i-- {
		local, err := composeNodeMatrix(nodes[chain[i]])
		if err != nil {
			return DefaultMatrix, fmt.Errorf("node %d: %w", chain[i], err)
		}
		world = mulMat4(world, local)
	}
	if !finiteMat4(world) {
		return DefaultMatrix, fmt.Errorf("world transform of node %d is not finite", nodeIndex)
	}
	return world, nil
}

// composeNodeMatrix returns the effective local matrix of n, following the
// glTF rule that a non-default Node.Matrix takes precedence over TRS.
func composeNodeMatrix(n *Node) ([16]float64, error) {
	if n == nil {
		return DefaultMatrix, errors.New("cannot compute the local matrix of a nil node")
	}
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
		if !isFiniteFloat(v) {
			return DefaultMatrix, errors.New("translation contains a non-finite value")
		}
	}
	for _, v := range rotation {
		if !isFiniteFloat(v) {
			return DefaultMatrix, errors.New("rotation contains a non-finite value")
		}
	}
	for _, v := range scale {
		if !isFiniteFloat(v) {
			return DefaultMatrix, errors.New("scale contains a non-finite value")
		}
	}

	// glTF requires unit quaternions; normalize so slightly denormalized input
	// still produces a proper rotation matrix.
	norm := math.Sqrt(rotation[0]*rotation[0] + rotation[1]*rotation[1] +
		rotation[2]*rotation[2] + rotation[3]*rotation[3])
	if norm == 0 {
		rotation = DefaultRotation
	} else {
		rotation[0], rotation[1], rotation[2], rotation[3] =
			rotation[0]/norm, rotation[1]/norm, rotation[2]/norm, rotation[3]/norm
	}

	m := composeTRSMatrix(translation, rotation, scale)
	if !finiteMat4(m) {
		return DefaultMatrix, errors.New("composed transform contains a non-finite value")
	}
	return m, nil
}

// composeTRSMatrix builds the column-major matrix Translation * Rotation *
// Scale from a translation, an (x, y, z, w) quaternion and a scale.
func composeTRSMatrix(t [3]float64, q [4]float64, s [3]float64) [16]float64 {
	x, y, z, w := q[0], q[1], q[2], q[3]
	xx, yy, zz := x*x, y*y, z*z
	xy, xz, yz := x*y, x*z, y*z
	wx, wy, wz := w*x, w*y, w*z

	// Rotation matrix entries indexed by (row, column).
	r00, r01, r02 := 1-2*(yy+zz), 2*(xy-wz), 2*(xz+wy)
	r10, r11, r12 := 2*(xy+wz), 1-2*(xx+zz), 2*(yz-wx)
	r20, r21, r22 := 2*(xz-wy), 2*(yz+wx), 1-2*(xx+yy)

	return [16]float64{
		r00 * s[0], r10 * s[0], r20 * s[0], 0,
		r01 * s[1], r11 * s[1], r21 * s[1], 0,
		r02 * s[2], r12 * s[2], r22 * s[2], 0,
		t[0], t[1], t[2], 1,
	}
}

// storeNodeTransform writes m into the node as TRS when it can be decomposed
// losslessly and as a raw Matrix otherwise. When a matrix is stored the TRS
// fields are reset to their glTF defaults so the node never carries both
// representations, which would make it invalid.
func storeNodeTransform(n *Node, m [16]float64) {
	if t, q, s, ok := decomposeTRSMatrix(m); ok {
		n.Matrix = DefaultMatrix
		n.Translation = t
		n.Rotation = q
		n.Scale = s
		return
	}
	n.Matrix = m
	n.Translation = DefaultTranslation
	n.Rotation = DefaultRotation
	n.Scale = DefaultScale
}

// decomposeTRSMatrix splits an affine column-major matrix into translation, a
// unit quaternion and scale. The boolean is false when the matrix cannot be
// reproduced exactly by T * R * S with an orthonormal rotation, for instance
// when it contains shear, a reflection that requires a signed scale plus a
// rotation, a zero column or a non-affine bottom row. In that case the caller
// should keep the raw matrix.
func decomposeTRSMatrix(m [16]float64) ([3]float64, [4]float64, [3]float64, bool) {
	var t [3]float64
	var q [4]float64
	var s [3]float64

	// Affine matrices have a [0, 0, 0, 1] bottom row.
	if math.Abs(m[3]) > epsMatrix || math.Abs(m[7]) > epsMatrix ||
		math.Abs(m[11]) > epsMatrix || math.Abs(m[15]-1) > epsMatrix {
		return t, q, s, false
	}

	t = [3]float64{m[12], m[13], m[14]}
	c0 := [3]float64{m[0], m[1], m[2]}
	c1 := [3]float64{m[4], m[5], m[6]}
	c2 := [3]float64{m[8], m[9], m[10]}

	l0 := math.Sqrt(c0[0]*c0[0] + c0[1]*c0[1] + c0[2]*c0[2])
	l1 := math.Sqrt(c1[0]*c1[0] + c1[1]*c1[1] + c1[2]*c1[2])
	l2 := math.Sqrt(c2[0]*c2[0] + c2[1]*c2[1] + c2[2]*c2[2])
	if l0 < epsMatrix || l1 < epsMatrix || l2 < epsMatrix {
		return t, q, s, false
	}
	s = [3]float64{l0, l1, l2}

	r0 := [3]float64{c0[0] / l0, c0[1] / l0, c0[2] / l0}
	r1 := [3]float64{c1[0] / l1, c1[1] / l1, c1[2] / l1}
	r2 := [3]float64{c2[0] / l2, c2[1] / l2, c2[2] / l2}

	// An odd reflection has a negative determinant. Moving the sign into the
	// third scale component leaves a proper rotation that still reproduces the
	// original columns.
	det := r0[0]*(r1[1]*r2[2]-r1[2]*r2[1]) -
		r0[1]*(r1[0]*r2[2]-r1[2]*r2[0]) +
		r0[2]*(r1[0]*r2[1]-r1[1]*r2[0])
	if det < 0 {
		s[2] = -s[2]
		r2[0], r2[1], r2[2] = -r2[0], -r2[1], -r2[2]
	}

	// Orthonormal columns are required; a non-zero dot product means shear.
	dot01 := r0[0]*r1[0] + r0[1]*r1[1] + r0[2]*r1[2]
	dot02 := r0[0]*r2[0] + r0[1]*r2[1] + r0[2]*r2[2]
	dot12 := r1[0]*r2[0] + r1[1]*r2[1] + r1[2]*r2[2]
	if math.Abs(dot01) > epsMatrix || math.Abs(dot02) > epsMatrix || math.Abs(dot12) > epsMatrix {
		return t, q, s, false
	}

	// Rotation entries by (row, column); the columns were normalized above.
	r00, r10, r20 := r0[0], r0[1], r0[2]
	r01, r11, r21 := r1[0], r1[1], r1[2]
	r02, r12, r22 := r2[0], r2[1], r2[2]

	// Shepperd's method: choose the branch with the largest denominator for
	// numerical stability.
	switch trace := r00 + r11 + r22; {
	case trace > 0:
		scale := 2 * math.Sqrt(trace+1)
		q[3] = 0.25 * scale
		q[0] = (r21 - r12) / scale
		q[1] = (r02 - r20) / scale
		q[2] = (r10 - r01) / scale
	case r00 > r11 && r00 > r22:
		scale := 2 * math.Sqrt(1+r00-r11-r22)
		q[0] = 0.25 * scale
		q[3] = (r21 - r12) / scale
		q[1] = (r01 + r10) / scale
		q[2] = (r02 + r20) / scale
	case r11 > r22:
		scale := 2 * math.Sqrt(1+r11-r00-r22)
		q[1] = 0.25 * scale
		q[3] = (r02 - r20) / scale
		q[0] = (r01 + r10) / scale
		q[2] = (r12 + r21) / scale
	default:
		scale := 2 * math.Sqrt(1+r22-r00-r11)
		q[2] = 0.25 * scale
		q[3] = (r10 - r01) / scale
		q[0] = (r02 + r20) / scale
		q[1] = (r12 + r21) / scale
	}

	norm := math.Sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3])
	if norm < epsMatrix || !isFiniteFloat(norm) {
		return t, q, s, false
	}
	q[0], q[1], q[2], q[3] = q[0]/norm, q[1]/norm, q[2]/norm, q[3]/norm

	// Strip floating point noise so axis-aligned transforms keep clean values
	// such as 0 and 1 instead of 1.0000000000000002.
	for i := range t {
		t[i] = snapComponent(t[i])
	}
	for i := range s {
		s[i] = snapComponent(s[i])
	}
	for i := range q {
		q[i] = snapComponent(q[i])
	}

	// Only accept the decomposition if it reproduces the input, which guards
	// against accumulated floating point error in the checks above.
	recomposed := composeTRSMatrix(t, q, s)
	for i := 0; i < 16; i++ {
		scale := math.Max(1, math.Abs(m[i]))
		if math.Abs(recomposed[i]-m[i]) > epsMatrix*scale {
			return t, q, s, false
		}
	}
	return t, q, s, true
}

func isFiniteFloat(v float64) bool {
	return !math.IsNaN(v) && !math.IsInf(v, 0)
}

// snapComponent rounds a value that is within tolerance of an integer to that
// integer, removing floating point noise from decomposed TRS components.
func snapComponent(v float64) float64 {
	r := math.Round(v)
	if math.Abs(v-r) <= 1e-12 {
		return r
	}
	return v
}

// finiteMat4 reports whether every element of m is finite.
func finiteMat4(m [16]float64) bool {
	for _, v := range m {
		if !isFiniteFloat(v) {
			return false
		}
	}
	return true
}

// mulMat4 returns the product a*b of two column-major matrices.
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

// invertMat4 returns the inverse of a column-major 4x4 matrix. The second
// result is false when m is singular or not finite.
//
// Affine matrices, which is what glTF node transforms and any product of them
// are, are inverted through their 3x3 linear part. This keeps the result exact
// for arbitrarily large translations, which a generic elimination would
// otherwise mistake for a near-singular pivot.
func invertMat4(m [16]float64) ([16]float64, bool) {
	if !finiteMat4(m) {
		return DefaultMatrix, false
	}
	if inverse, ok := invertAffineMat4(m); ok {
		return inverse, true
	}
	if isAffineMat4(m) {
		// An affine matrix has an invertible affine inverse or none at all;
		// never fall back to the generic path for it.
		return DefaultMatrix, false
	}

	var a [4][8]float64
	for row := 0; row < 4; row++ {
		for col := 0; col < 4; col++ {
			a[row][col] = m[col*4+row]
		}
		a[row][row+4] = 1
	}

	for col := 0; col < 4; col++ {
		pivot := col
		for row := col + 1; row < 4; row++ {
			if math.Abs(a[row][col]) > math.Abs(a[pivot][col]) {
				pivot = row
			}
		}
		if a[pivot][col] == 0 {
			return DefaultMatrix, false
		}
		if pivot != col {
			a[col], a[pivot] = a[pivot], a[col]
		}
		divisor := a[col][col]
		for j := 0; j < 8; j++ {
			a[col][j] /= divisor
		}
		for row := 0; row < 4; row++ {
			if row == col {
				continue
			}
			factor := a[row][col]
			if factor == 0 {
				continue
			}
			for j := 0; j < 8; j++ {
				a[row][j] -= factor * a[col][j]
			}
		}
	}

	var inverse [16]float64
	for row := 0; row < 4; row++ {
		for col := 0; col < 4; col++ {
			inverse[col*4+row] = a[row][col+4]
		}
	}
	if !finiteMat4(inverse) {
		return DefaultMatrix, false
	}
	return inverse, true
}

// isAffineMat4 reports whether m is an affine transform, i.e. its bottom row is
// [0, 0, 0, 1].
func isAffineMat4(m [16]float64) bool {
	return math.Abs(m[3]) <= epsMatrix && math.Abs(m[7]) <= epsMatrix &&
		math.Abs(m[11]) <= epsMatrix && math.Abs(m[15]-1) <= epsMatrix
}

// invertAffineMat4 inverts an affine column-major matrix by inverting its 3x3
// linear part and folding the translation. The boolean is false when the
// matrix is not affine or its linear part is singular.
func invertAffineMat4(m [16]float64) ([16]float64, bool) {
	if !isAffineMat4(m) {
		return DefaultMatrix, false
	}

	// Linear part, row-major: column c of m starts at index c*4.
	a, b, c := m[0], m[4], m[8]
	d, e, f := m[1], m[5], m[9]
	g, h, i := m[2], m[6], m[10]

	// Cofactors of the 3x3 matrix.
	cofA := e*i - f*h
	cofB := f*g - d*i
	cofC := d*h - e*g

	det := a*cofA + b*cofB + c*cofC
	if !isFiniteFloat(det) || det == 0 {
		return DefaultMatrix, false
	}

	// Reject matrices that are singular up to floating point precision. The
	// threshold is relative to the product of the column norms, so uniformly
	// scaled but well-conditioned transforms are accepted.
	norm0 := math.Sqrt(a*a + d*d + g*g)
	norm1 := math.Sqrt(b*b + e*e + h*h)
	norm2 := math.Sqrt(c*c + f*f + i*i)
	scale := norm0 * norm1 * norm2
	if !isFiniteFloat(scale) || scale == 0 || math.Abs(det) <= 1e-12*scale {
		return DefaultMatrix, false
	}

	invDet := 1 / det
	var linear [9]float64 // row-major 3x3 inverse
	linear[0] = cofA * invDet
	linear[1] = (c*h - b*i) * invDet
	linear[2] = (b*f - c*e) * invDet
	linear[3] = cofB * invDet
	linear[4] = (a*i - c*g) * invDet
	linear[5] = (c*d - a*f) * invDet
	linear[6] = cofC * invDet
	linear[7] = (b*g - a*h) * invDet
	linear[8] = (a*e - b*d) * invDet

	tx, ty, tz := m[12], m[13], m[14]
	inverse := [16]float64{
		linear[0], linear[3], linear[6], 0,
		linear[1], linear[4], linear[7], 0,
		linear[2], linear[5], linear[8], 0,
		-(linear[0]*tx + linear[1]*ty + linear[2]*tz),
		-(linear[3]*tx + linear[4]*ty + linear[5]*tz),
		-(linear[6]*tx + linear[7]*ty + linear[8]*tz),
		1,
	}
	if !finiteMat4(inverse) {
		return DefaultMatrix, false
	}
	return inverse, true
}

// inverseIsAccurate guards against numerically singular matrices whose
// elimination still yields finite values. It checks that inv*m is close to the
// identity, with a tolerance scaled by each entry's accumulated term magnitude
// so large translations do not trip the check through cancellation residual.
func inverseIsAccurate(m, inverse [16]float64) bool {
	const eps = 1e-9
	for row := 0; row < 4; row++ {
		for col := 0; col < 4; col++ {
			var sum, magnitude float64
			for k := 0; k < 4; k++ {
				term := inverse[k*4+row] * m[col*4+k]
				sum += term
				magnitude += math.Abs(term)
			}
			want := 0.0
			if row == col {
				want = 1
			}
			if math.Abs(sum-want) > eps*math.Max(1, magnitude) {
				return false
			}
		}
	}
	return true
}
