package lightspunctual

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"

	"github.com/qmuntal/gltf"
)

// WorldLight is a KHR_lights_punctual light instance resolved into the world
// space of a scene.
//
// glTF stores lights in node-local space: a node references a light by index
// and the node transform places and orients it. A WorldLight is the result of
// resolving that reference through the node hierarchy of a scene and exposes
// the values a renderer needs, with the glTF defaults already applied.
type WorldLight struct {
	// Node is the index in Document.Nodes of the node that instantiates the
	// light.
	Node int
	// Light is the index in the scene's KHR_lights_punctual lights array of
	// the referenced light.
	Light int
	// Name is the light's name, if it defines one.
	Name string
	// Type is one of TypeDirectional, TypePoint or TypeSpot.
	Type string
	// Position is the world-space origin of the light node. It is the
	// relevant quantity for point and spot lights; directional lights are
	// infinitely far away and are only characterized by Direction.
	Position [3]float64
	// Direction is the world-space unit vector the light points at. It is
	// defined for directional and spot lights, following the glTF convention
	// that they emit along the node's local -Z axis. It is the zero vector for
	// point lights, which emit in every direction.
	//
	// Only the rotations of the node and its ancestors affect Direction.
	// Translation and scale, including non-uniform scale, never skew it.
	Direction [3]float64
	// Color is the light color, (1, 1, 1) when the light does not define one.
	Color [3]float64
	// Intensity is the light intensity, 1 when the light does not define one.
	Intensity float64
	// Range is the distance at which the light intensity reaches zero, +Inf
	// when the light does not define one.
	Range float64
	// InnerConeAngle is the angle, in radians, of the fully lit cone of a spot
	// light. It is 0 for other light types.
	InnerConeAngle float64
	// OuterConeAngle is the angle, in radians, at which the cone of a spot
	// light ends. It is 0 for other light types. A spot light without an
	// explicit outer cone angle uses math.Pi / 4.
	OuterConeAngle float64
}

// ReadSceneLights returns the world-space evaluation of every
// KHR_lights_punctual light instantiated by the nodes of scene, in depth-first
// traversal order of the scene graph.
//
// A node instantiates a light when it defines the KHR_lights_punctual node
// extension with a light index. The referenced light is read from the root
// KHR_lights_punctual extension of doc, which may store either decoded
// lightspunctual values or raw JSON. The node transform and the transforms of
// all its ancestors are composed to compute Position, while Direction only
// takes rotations into account, so non-uniform scale never distorts the light
// axis.
//
// Missing light properties are resolved using the glTF defaults: white color,
// intensity 1, infinite range, and cone angles 0 (inner) and math.Pi / 4
// (outer). Lights of an unknown type, or with out-of-range values, are
// reported as errors rather than silently clamped. The document is never
// modified.
//
// A nil scene means the active scene declared by Document.Scene. If doc has no
// active scene, or the requested scene is not part of a well-formed node
// graph, an error is returned.
//
// ReadSceneLights is safe to use with malformed documents. It returns a
// descriptive error instead of panicking when a scene root, child reference or
// light index is invalid, when a node is referenced more than once or the
// hierarchy contains a cycle, when a light has an invalid type, range,
// intensity, color or cone angles, when a KHR_lights_punctual extension value
// has the wrong shape, or when a transform contains a non-finite value or
// produces a degenerate light direction. Errors name the offending node and,
// when applicable, light index.
func ReadSceneLights(doc *gltf.Document, scene *gltf.Scene) ([]WorldLight, error) {
	if doc == nil {
		return nil, errors.New("gltf: document is nil")
	}
	resolvedScene, err := resolveLightScene(doc, scene)
	if err != nil {
		return nil, err
	}

	// The root lights array is only decoded when a node actually references a
	// light, so documents without lights do not pay for it and a malformed
	// array does not fail an otherwise valid scene query.
	var (
		lights       Lights
		lightsLoaded bool
		lightsErr    error
	)
	loadLights := func() (Lights, error) {
		if !lightsLoaded {
			lights, lightsErr = readDocumentLights(doc)
			lightsLoaded = true
		}
		return lights, lightsErr
	}

	const (
		lightActive = iota + 1
		lightDone
	)
	state := make([]uint8, len(doc.Nodes))
	worldLights := make([]WorldLight, 0)

	var walk func(nodeIndex int, parentMatrix [16]float64, parentRotation [9]float64) error
	walk = func(nodeIndex int, parentMatrix [16]float64, parentRotation [9]float64) error {
		switch state[nodeIndex] {
		case lightActive:
			return fmt.Errorf("gltf: node hierarchy contains a cycle at node %d", nodeIndex)
		case lightDone:
			return fmt.Errorf("gltf: node %d is referenced more than once in the scene", nodeIndex)
		}
		state[nodeIndex] = lightActive

		node := doc.Nodes[nodeIndex]
		if node == nil {
			return fmt.Errorf("gltf: scene references nil node %d", nodeIndex)
		}
		localMatrix, localRotation, err := localNodeTransform(node, nodeIndex)
		if err != nil {
			return err
		}
		worldMatrix := mulMat4(parentMatrix, localMatrix)
		if !finiteMat4(worldMatrix) {
			return fmt.Errorf("gltf: world transform of node %d contains a non-finite value", nodeIndex)
		}
		worldRotation := mulMat3(parentRotation, localRotation)
		if !finiteMat3(worldRotation) {
			return fmt.Errorf("gltf: world rotation of node %d contains a non-finite value", nodeIndex)
		}

		lightIndex, hasLight, err := nodeLightIndex(node, nodeIndex)
		if err != nil {
			return err
		}
		if hasLight {
			definedLights, err := loadLights()
			if err != nil {
				return err
			}
			instance, err := evaluateSceneLight(
				nodeIndex, lightIndex, definedLights, worldMatrix, worldRotation,
			)
			if err != nil {
				return err
			}
			worldLights = append(worldLights, instance)
		}

		for _, child := range node.Children {
			if child < 0 || child >= len(doc.Nodes) {
				return fmt.Errorf("gltf: node %d references invalid child index %d", nodeIndex, child)
			}
			if doc.Nodes[child] == nil {
				return fmt.Errorf("gltf: node %d references nil child node %d", nodeIndex, child)
			}
			if err := walk(child, worldMatrix, worldRotation); err != nil {
				return err
			}
		}
		state[nodeIndex] = lightDone
		return nil
	}

	for _, root := range resolvedScene.Nodes {
		if root < 0 || root >= len(doc.Nodes) {
			return nil, fmt.Errorf("gltf: scene references invalid root node index %d", root)
		}
		if doc.Nodes[root] == nil {
			return nil, fmt.Errorf("gltf: scene references nil root node %d", root)
		}
		if err := walk(root, gltf.DefaultMatrix, identityMat3); err != nil {
			return nil, err
		}
	}
	return worldLights, nil
}

// resolveLightScene returns scene, or the document's active scene when scene
// is nil.
func resolveLightScene(doc *gltf.Document, scene *gltf.Scene) (*gltf.Scene, error) {
	if scene != nil {
		return scene, nil
	}
	if doc.Scene == nil {
		return nil, errors.New("gltf: document does not define an active scene")
	}
	index := *doc.Scene
	if index < 0 || index >= len(doc.Scenes) {
		return nil, fmt.Errorf("gltf: active scene index %d overflows", index)
	}
	if doc.Scenes[index] == nil {
		return nil, fmt.Errorf("gltf: active scene %d is nil", index)
	}
	return doc.Scenes[index], nil
}

// nodeLightExtension is the node-level KHR_lights_punctual payload.
type nodeLightExtension struct {
	Light *int `json:"light"`
}

// lightsExtension is the document-level KHR_lights_punctual payload.
type lightsExtension struct {
	Lights Lights `json:"lights"`
}

// nodeLightIndex extracts the light index referenced by node. The second
// result reports whether the node defines the extension at all.
func nodeLightIndex(node *gltf.Node, nodeIndex int) (int, bool, error) {
	value, ok := node.Extensions[ExtensionName]
	if !ok {
		return 0, false, nil
	}
	switch v := value.(type) {
	case LightIndex:
		return int(v), true, nil
	case *LightIndex:
		if v == nil {
			return 0, true, fmt.Errorf("gltf: node %d has a nil KHR_lights_punctual light index", nodeIndex)
		}
		return int(*v), true, nil
	}
	// Raw values and values built by hand are normalized through JSON so they
	// follow the same validation and default rules as decoded ones.
	data, err := json.Marshal(value)
	if err != nil {
		return 0, true, fmt.Errorf("gltf: node %d has an invalid KHR_lights_punctual extension: %w", nodeIndex, err)
	}
	var env nodeLightExtension
	if err := json.Unmarshal(data, &env); err != nil {
		return 0, true, fmt.Errorf("gltf: node %d has an invalid KHR_lights_punctual extension: %w", nodeIndex, err)
	}
	if env.Light == nil {
		return 0, true, fmt.Errorf("gltf: node %d does not define a KHR_lights_punctual light index", nodeIndex)
	}
	return *env.Light, true, nil
}

// readDocumentLights decodes the root KHR_lights_punctual extension.
func readDocumentLights(doc *gltf.Document) (Lights, error) {
	value, ok := doc.Extensions[ExtensionName]
	if !ok {
		return nil, nil
	}
	switch v := value.(type) {
	case Lights:
		return v, nil
	case []*Light:
		return Lights(v), nil
	case *Lights:
		if v == nil {
			return nil, nil
		}
		return *v, nil
	}
	data, err := json.Marshal(value)
	if err != nil {
		return nil, fmt.Errorf("gltf: invalid KHR_lights_punctual extension: %w", err)
	}
	var env lightsExtension
	if err := json.Unmarshal(data, &env); err != nil {
		return nil, fmt.Errorf("gltf: invalid KHR_lights_punctual extension: %w", err)
	}
	return env.Lights, nil
}

// evaluateSceneLight validates the referenced light and combines its
// parameters with the world transform of the node that instantiates it.
func evaluateSceneLight(nodeIndex, lightIndex int, lights Lights, worldMatrix [16]float64, worldRotation [9]float64) (WorldLight, error) {
	if lightIndex < 0 || lightIndex >= len(lights) {
		return WorldLight{}, fmt.Errorf(
			"gltf: node %d references light %d, but the document defines %d lights",
			nodeIndex, lightIndex, len(lights),
		)
	}
	light := lights[lightIndex]
	if light == nil {
		return WorldLight{}, fmt.Errorf("gltf: light %d referenced by node %d is nil", lightIndex, nodeIndex)
	}

	switch light.Type {
	case TypeDirectional, TypePoint, TypeSpot:
	default:
		return WorldLight{}, fmt.Errorf(
			"gltf: light %d referenced by node %d has invalid type %q",
			lightIndex, nodeIndex, light.Type,
		)
	}

	color := light.ColorOrDefault()
	for _, c := range color {
		if !isFinite(c) {
			return WorldLight{}, fmt.Errorf(
				"gltf: light %d referenced by node %d has a non-finite color", lightIndex, nodeIndex,
			)
		}
	}
	intensity := light.IntensityOrDefault()
	if !isFinite(intensity) || intensity < 0 {
		return WorldLight{}, fmt.Errorf(
			"gltf: light %d referenced by node %d has invalid intensity %v", lightIndex, nodeIndex, intensity,
		)
	}
	lightRange := math.Inf(1)
	if light.Range != nil {
		lightRange = *light.Range
		// An explicit +Inf encodes the same "no distance cutoff" as the
		// default, so it is accepted. NaN, -Inf and non-positive ranges are
		// invalid.
		if math.IsNaN(lightRange) || math.IsInf(lightRange, -1) || lightRange <= 0 {
			return WorldLight{}, fmt.Errorf(
				"gltf: light %d referenced by node %d has invalid range %v", lightIndex, nodeIndex, lightRange,
			)
		}
	}

	instance := WorldLight{
		Node:      nodeIndex,
		Light:     lightIndex,
		Name:      light.Name,
		Type:      light.Type,
		Position:  [3]float64{worldMatrix[12], worldMatrix[13], worldMatrix[14]},
		Color:     color,
		Intensity: intensity,
		Range:     lightRange,
	}
	for _, v := range instance.Position {
		if !isFinite(v) {
			return WorldLight{}, fmt.Errorf(
				"gltf: light %d referenced by node %d has a non-finite world position", lightIndex, nodeIndex,
			)
		}
	}

	if light.Type == TypeSpot {
		inner := 0.0
		outer := math.Pi / 4
		if light.Spot != nil {
			inner = light.Spot.InnerConeAngle
			outer = light.Spot.OuterConeAngleOrDefault()
		}
		if !isFinite(inner) || inner < 0 {
			return WorldLight{}, fmt.Errorf(
				"gltf: light %d referenced by node %d has invalid innerConeAngle %v", lightIndex, nodeIndex, inner,
			)
		}
		if !isFinite(outer) || outer <= 0 || outer > math.Pi/2 {
			return WorldLight{}, fmt.Errorf(
				"gltf: light %d referenced by node %d has invalid outerConeAngle %v", lightIndex, nodeIndex, outer,
			)
		}
		if inner >= outer {
			return WorldLight{}, fmt.Errorf(
				"gltf: light %d referenced by node %d has innerConeAngle %v not smaller than outerConeAngle %v",
				lightIndex, nodeIndex, inner, outer,
			)
		}
		instance.InnerConeAngle = inner
		instance.OuterConeAngle = outer
	}

	if light.Type != TypePoint {
		// Local -Z is the emission axis, so it maps to the negated third
		// column of the world rotation.
		direction := [3]float64{-worldRotation[6], -worldRotation[7], -worldRotation[8]}
		length := math.Sqrt(direction[0]*direction[0] + direction[1]*direction[1] + direction[2]*direction[2])
		if !isFinite(length) || length == 0 {
			return WorldLight{}, fmt.Errorf(
				"gltf: light %d referenced by node %d has a degenerate world direction", lightIndex, nodeIndex,
			)
		}
		instance.Direction = [3]float64{
			direction[0] / length,
			direction[1] / length,
			direction[2] / length,
		}
	}
	return instance, nil
}

// localNodeTransform returns the local transformation of node as a
// column-major 4x4 matrix and its rotation as a column-major 3x3 matrix.
//
// The matrix follows the same rules as the glTF encoder: an empty or identity
// Matrix means the node is defined by its translation, rotation and scale.
// The rotation is built without any scale, so directions derived from it are
// not distorted by non-uniform scaling.
func localNodeTransform(node *gltf.Node, index int) ([16]float64, [9]float64, error) {
	if node.Matrix != gltf.DefaultMatrix && node.Matrix != zeroMatrix {
		for _, v := range node.Matrix {
			if !isFinite(v) {
				return gltf.DefaultMatrix, identityMat3, fmt.Errorf(
					"gltf: node %d matrix contains a non-finite value", index,
				)
			}
		}
		rotation, err := rotationFromMatrix(node.Matrix, index)
		if err != nil {
			return gltf.DefaultMatrix, identityMat3, err
		}
		return node.Matrix, rotation, nil
	}

	translation := node.TranslationOrDefault()
	rotation := node.RotationOrDefault()
	scale := node.ScaleOrDefault()
	for _, v := range translation {
		if !isFinite(v) {
			return gltf.DefaultMatrix, identityMat3, fmt.Errorf(
				"gltf: node %d translation contains a non-finite value", index,
			)
		}
	}
	for _, v := range rotation {
		if !isFinite(v) {
			return gltf.DefaultMatrix, identityMat3, fmt.Errorf(
				"gltf: node %d rotation contains a non-finite value", index,
			)
		}
	}
	for _, v := range scale {
		if !isFinite(v) {
			return gltf.DefaultMatrix, identityMat3, fmt.Errorf(
				"gltf: node %d scale contains a non-finite value", index,
			)
		}
	}
	return composeTRS(translation, rotation, scale), quaternionMat3(rotation), nil
}

// rotationFromMatrix extracts the rotation of a local matrix by
// normalizing its basis columns. glTF matrices are decomposable into
// translation, rotation and scale, so the columns are orthogonal and their
// directions are exactly those of the rotation.
func rotationFromMatrix(matrix [16]float64, index int) ([9]float64, error) {
	var rotation [9]float64
	for column := 0; column < 3; column++ {
		x := matrix[column*4]
		y := matrix[column*4+1]
		z := matrix[column*4+2]
		length := math.Sqrt(x*x + y*y + z*z)
		if !isFinite(length) || length == 0 {
			return identityMat3, fmt.Errorf("gltf: node %d matrix has a degenerate rotation basis", index)
		}
		rotation[column*3] = x / length
		rotation[column*3+1] = y / length
		rotation[column*3+2] = z / length
	}
	return rotation, nil
}

// composeTRS builds the local transformation matrix from translation,
// rotation and scale in column-major order, as defined by glTF.
func composeTRS(translation [3]float64, rotation [4]float64, scale [3]float64) [16]float64 {
	x, y, z, w := rotation[0], rotation[1], rotation[2], rotation[3]
	xx, yy, zz := x*x, y*y, z*z
	xy, xz, yz := x*y, x*z, y*z
	wx, wy, wz := w*x, w*y, w*z
	var m [16]float64
	m[0] = scale[0] * (1 - 2*(yy+zz))
	m[1] = scale[0] * 2 * (xy + wz)
	m[2] = scale[0] * 2 * (xz - wy)
	m[4] = scale[1] * 2 * (xy - wz)
	m[5] = scale[1] * (1 - 2*(xx+zz))
	m[6] = scale[1] * 2 * (yz + wx)
	m[8] = scale[2] * 2 * (xz + wy)
	m[9] = scale[2] * 2 * (yz - wx)
	m[10] = scale[2] * (1 - 2*(xx+yy))
	m[12] = translation[0]
	m[13] = translation[1]
	m[14] = translation[2]
	m[15] = 1
	return m
}

// quaternionMat3 converts an (x, y, z, w) quaternion to a column-major
// rotation matrix. The quaternion is not normalized: scaling it only scales
// the matrix, which does not change the directions derived from it.
func quaternionMat3(q [4]float64) [9]float64 {
	x, y, z, w := q[0], q[1], q[2], q[3]
	xx, yy, zz := x*x, y*y, z*z
	xy, xz, yz := x*y, x*z, y*z
	wx, wy, wz := w*x, w*y, w*z
	return [9]float64{
		1 - 2*(yy+zz), 2 * (xy + wz), 2 * (xz - wy),
		2 * (xy - wz), 1 - 2*(xx+zz), 2 * (yz + wx),
		2 * (xz + wy), 2 * (yz - wx), 1 - 2*(xx+yy),
	}
}

// mulMat4 returns a*b for column-major 4x4 matrices.
func mulMat4(a, b [16]float64) [16]float64 {
	var out [16]float64
	for column := 0; column < 4; column++ {
		for row := 0; row < 4; row++ {
			out[column*4+row] =
				a[0*4+row]*b[column*4] +
					a[1*4+row]*b[column*4+1] +
					a[2*4+row]*b[column*4+2] +
					a[3*4+row]*b[column*4+3]
		}
	}
	return out
}

// mulMat3 returns a*b for column-major 3x3 matrices.
func mulMat3(a, b [9]float64) [9]float64 {
	var out [9]float64
	for column := 0; column < 3; column++ {
		for row := 0; row < 3; row++ {
			out[column*3+row] =
				a[0*3+row]*b[column*3] +
					a[1*3+row]*b[column*3+1] +
					a[2*3+row]*b[column*3+2]
		}
	}
	return out
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

func finiteMat3(m [9]float64) bool {
	for _, v := range m {
		if !isFinite(v) {
			return false
		}
	}
	return true
}

var (
	zeroMatrix   = [16]float64{}
	identityMat3 = [9]float64{1, 0, 0, 0, 1, 0, 0, 0, 1}
)
