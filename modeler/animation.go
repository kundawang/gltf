package modeler

import (
	"errors"
	"fmt"
	"math"

	"github.com/qmuntal/gltf"
)

// NodePose is the local translation, rotation and scale of a node at a
// given point in time. Rotation is a unit quaternion in the order
// (x, y, z, w), where w is the scalar.
//
// Components that are not driven by the sampled animation are initialized
// with the node's own values (or their glTF defaults), so a NodePose can
// always be used as a complete TRS transform.
type NodePose struct {
	Translation [3]float64
	Rotation    [4]float64
	Scale       [3]float64
}

// ReadAnimation samples the animation identified by animationIndex at time
// (measured in seconds) and returns the local TRS pose of every node that is
// targeted by one of its channels.
//
// STEP, LINEAR and CUBICSPLINE samplers are supported. Quaternion channels
// are interpolated along the shortest arc and normalized back to a unit
// quaternion. A component that the animation does not target is taken from
// the node itself, falling back to the glTF defaults (zero translation,
// identity rotation, unit scale).
//
// ReadAnimation never modifies the document and holds no state between calls:
// channels targeting the same node, or different animations, cannot interfere
// with each other.
//
// Times outside the keyframe range are clamped to the first or last keyframe.
// A non-finite time, an out of range animation, channel, node or accessor
// index, a malformed accessor or an invalid sampler makes ReadAnimation
// return an error. Channels targeting the "weights" path, or a node-less
// target, are not TRS poses and are ignored.
func ReadAnimation(doc *gltf.Document, animationIndex int, time float64) (map[int]NodePose, error) {
	if doc == nil {
		return nil, errors.New("gltf: document is nil")
	}
	if animationIndex < 0 || animationIndex >= len(doc.Animations) {
		return nil, fmt.Errorf("gltf: animation index %d overflows", animationIndex)
	}
	if math.IsNaN(time) || math.IsInf(time, 0) {
		return nil, errors.New("gltf: animation time must be finite")
	}

	animation := doc.Animations[animationIndex]
	if animation == nil {
		return nil, errors.New("gltf: animation is nil")
	}

	poses := make(map[int]NodePose, len(animation.Channels))
	samplers := make(map[int]*samplerData, len(animation.Samplers))
	for _, channel := range animation.Channels {
		if channel == nil {
			return nil, errors.New("gltf: animation channel is nil")
		}
		// Morph weights and node-less targets are not node TRS poses.
		if channel.Target.Node == nil || channel.Target.Path == gltf.TRSWeights {
			continue
		}
		switch channel.Target.Path {
		case gltf.TRSTranslation, gltf.TRSRotation, gltf.TRSScale:
		default:
			return nil, fmt.Errorf("gltf: unknown animation target path %q", channel.Target.Path)
		}
		nodeIndex := *channel.Target.Node
		if nodeIndex < 0 || nodeIndex >= len(doc.Nodes) {
			return nil, fmt.Errorf("gltf: animation node index %d overflows", nodeIndex)
		}
		if channel.Sampler < 0 || channel.Sampler >= len(animation.Samplers) {
			return nil, fmt.Errorf("gltf: animation sampler index %d overflows", channel.Sampler)
		}

		sampler, ok := samplers[channel.Sampler]
		if !ok {
			var err error
			sampler, err = readSamplerData(doc, animation.Samplers[channel.Sampler])
			if err != nil {
				return nil, err
			}
			samplers[channel.Sampler] = sampler
		}
		expectedComponents := 3
		if channel.Target.Path == gltf.TRSRotation {
			expectedComponents = 4
		}
		if sampler.components != expectedComponents {
			return nil, fmt.Errorf("gltf: animation output accessor of %d components does not target path %q", sampler.components, channel.Target.Path)
		}

		pose, ok := poses[nodeIndex]
		if !ok {
			pose = defaultNodePose(doc.Nodes[nodeIndex])
		}
		value, err := sampler.sample(time, channel.Target.Path)
		if err != nil {
			return nil, err
		}
		switch channel.Target.Path {
		case gltf.TRSTranslation:
			pose.Translation = toVec3(value)
		case gltf.TRSRotation:
			pose.Rotation = normalizeQuaternion(toVec4(value))
		case gltf.TRSScale:
			pose.Scale = toVec3(value)
		}
		poses[nodeIndex] = pose
	}
	return poses, nil
}

// samplerData holds the decoded keyframes of an AnimationSampler.
// values is flat: for non-cubic samplers it holds one component vector per
// keyframe, for CUBICSPLINE it holds in-tangent, value and out-tangent per
// keyframe, in that order.
type samplerData struct {
	interpolation gltf.Interpolation
	times         []float64
	values        []float64
	components    int
}

func readSamplerData(doc *gltf.Document, sampler *gltf.AnimationSampler) (*samplerData, error) {
	if sampler == nil {
		return nil, errors.New("gltf: animation sampler is nil")
	}
	if sampler.Input < 0 || sampler.Input >= len(doc.Accessors) {
		return nil, fmt.Errorf("gltf: animation input accessor index %d overflows", sampler.Input)
	}
	if sampler.Output < 0 || sampler.Output >= len(doc.Accessors) {
		return nil, fmt.Errorf("gltf: animation output accessor index %d overflows", sampler.Output)
	}
	switch sampler.Interpolation {
	case gltf.InterpolationLinear, gltf.InterpolationStep, gltf.InterpolationCubicSpline:
	default:
		return nil, fmt.Errorf("gltf: unknown animation interpolation %v", sampler.Interpolation)
	}

	input := doc.Accessors[sampler.Input]
	output := doc.Accessors[sampler.Output]
	if input == nil || output == nil {
		return nil, errors.New("gltf: animation accessor is nil")
	}
	if input.Type != gltf.AccessorScalar || input.ComponentType != gltf.ComponentFloat {
		return nil, errors.New("gltf: animation input accessor must be a FLOAT SCALAR")
	}
	components := output.Type.Components()
	if components != 3 && components != 4 {
		return nil, fmt.Errorf("gltf: animation output accessor type %v not allowed", output.Type)
	}
	if output.ComponentType != gltf.ComponentFloat && !output.Normalized {
		return nil, errors.New("gltf: integer animation output accessor must be normalized")
	}

	bufPtr := bufPool.Get().(*[]byte)
	defer bufPool.Put(bufPtr)
	timesData, err := ReadAccessor(doc, input, *bufPtr)
	if err != nil {
		return nil, err
	}
	times, err := toFloat64Slice(timesData)
	if err != nil {
		return nil, fmt.Errorf("gltf: animation input accessor: %w", err)
	}
	if len(times) == 0 {
		return nil, errors.New("gltf: animation sampler has no keyframes")
	}
	for i, t := range times {
		if math.IsNaN(t) || math.IsInf(t, 0) {
			return nil, errors.New("gltf: animation keyframe time must be finite")
		}
		if i > 0 && t <= times[i-1] {
			return nil, errors.New("gltf: animation keyframe times must be strictly increasing")
		}
	}

	valuesData, err := ReadAccessor(doc, output, *bufPtr)
	if err != nil {
		return nil, err
	}
	values, err := toFloat64Slice(valuesData)
	if err != nil {
		return nil, fmt.Errorf("gltf: animation output accessor: %w", err)
	}

	framesPerKeyframe := 1
	if sampler.Interpolation == gltf.InterpolationCubicSpline {
		framesPerKeyframe = 3
	}
	if len(values) != len(times)*components*framesPerKeyframe {
		return nil, errors.New("gltf: animation output accessor count does not match the input accessor")
	}
	for _, v := range values {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return nil, errors.New("gltf: animation keyframe value must be finite")
		}
	}

	return &samplerData{
		interpolation: sampler.Interpolation,
		times:         times,
		values:        values,
		components:    components,
	}, nil
}

// sample evaluates the keyframe graph at time. time is clamped to the
// keyframe range. path selects how quaternion outputs are interpolated.
func (s *samplerData) sample(time float64, path gltf.TRSProperty) ([]float64, error) {
	// Clamp the query to the keyframe range.
	if time <= s.times[0] {
		return s.keyframe(0, 1), nil
	}
	last := len(s.times) - 1
	if time >= s.times[last] {
		return s.keyframe(last, 1), nil
	}

	// Locate the surrounding pair of keyframes.
	hi := 1
	for hi < last && s.times[hi] < time {
		hi++
	}
	lo := hi - 1
	// Exact keyframe times always yield the stored value, regardless of
	// the interpolation mode.
	if time == s.times[hi] {
		return s.keyframe(hi, 1), nil
	}
	if time == s.times[lo] {
		return s.keyframe(lo, 1), nil
	}

	alpha := (time - s.times[lo]) / (s.times[hi] - s.times[lo])
	switch s.interpolation {
	case gltf.InterpolationStep:
		return s.keyframe(lo, 1), nil
	case gltf.InterpolationLinear:
		a := s.keyframe(lo, 1)
		b := s.keyframe(hi, 1)
		if path == gltf.TRSRotation {
			return slerp(a, b, alpha), nil
		}
		return lerp(a, b, alpha), nil
	case gltf.InterpolationCubicSpline:
		dt := s.times[hi] - s.times[lo]
		a := s.keyframe(lo, 1)
		outTangent := s.keyframe(lo, 2)
		inTangent := s.keyframe(hi, 0)
		b := s.keyframe(hi, 1)
		return cubicSpline(a, outTangent, inTangent, b, dt, alpha), nil
	default:
		return nil, fmt.Errorf("gltf: unknown animation interpolation %v", s.interpolation)
	}
}

// keyframe returns the frame-th component vector (0: in-tangent, 1: value,
// 2: out-tangent) of keyframe i.
func (s *samplerData) keyframe(i, frame int) []float64 {
	if s.interpolation == gltf.InterpolationCubicSpline {
		start := (3*i + frame) * s.components
		return s.values[start : start+s.components]
	}
	start := i * s.components
	return s.values[start : start+s.components]
}

// defaultNodePose builds a pose from the node's own TRS, applying the glTF
// defaults for empty rotation and scale.
func defaultNodePose(node *gltf.Node) NodePose {
	if node == nil {
		return NodePose{
			Translation: gltf.DefaultTranslation,
			Rotation:    gltf.DefaultRotation,
			Scale:       gltf.DefaultScale,
		}
	}
	return NodePose{
		Translation: node.TranslationOrDefault(),
		Rotation:    node.RotationOrDefault(),
		Scale:       node.ScaleOrDefault(),
	}
}

func lerp(a, b []float64, t float64) []float64 {
	out := make([]float64, len(a))
	for i := range a {
		out[i] = a[i]*(1-t) + b[i]*t
	}
	return out
}

func cubicSpline(p0, outTangent, inTangent, p1 []float64, dt, t float64) []float64 {
	t2 := t * t
	t3 := t2 * t
	out := make([]float64, len(p0))
	for i := range p0 {
		out[i] = (2*t3-3*t2+1)*p0[i] +
			dt*(t3-2*t2+t)*outTangent[i] +
			dt*(-t3+t2)*inTangent[i] +
			(-2*t3+3*t2)*p1[i]
	}
	return out
}

// slerp interpolates two unit quaternions along the shortest arc. The result
// is normalized so a non-unit or degenerate input still yields a unit
// quaternion whenever the input has any length at all.
func slerp(a, b []float64, t float64) []float64 {
	dot := a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
	// Take the shorter arc: negate one quaternion when the angle between
	// them is obtuse.
	if dot < 0 {
		dot = -dot
		b = []float64{-b[0], -b[1], -b[2], -b[3]}
	}
	var out [4]float64
	if dot > 0.9995 {
		// Quasi-parallel quaternions: fall back to normalized linear
		// interpolation to avoid a numerically unstable division.
		for i := range a {
			out[i] = a[i]*(1-t) + b[i]*t
		}
	} else {
		theta0 := math.Acos(math.Max(-1, math.Min(1, dot)))
		theta := theta0 * t
		sinTheta0 := math.Sin(theta0)
		s0 := math.Cos(theta) - dot*math.Sin(theta)/sinTheta0
		s1 := math.Sin(theta) / sinTheta0
		for i := range a {
			out[i] = s0*a[i] + s1*b[i]
		}
	}
	normalized := normalizeQuaternion(out)
	return normalized[:]
}

func normalizeQuaternion(q [4]float64) [4]float64 {
	length := math.Sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3])
	if length == 0 {
		return gltf.DefaultRotation
	}
	for i := range q {
		q[i] /= length
	}
	return q
}

func toVec3(v []float64) [3]float64 {
	return [3]float64{v[0], v[1], v[2]}
}

func toVec4(v []float64) [4]float64 {
	return [4]float64{v[0], v[1], v[2], v[3]}
}

// toFloat64Slice converts the typed slice returned by ReadAccessor into a
// flat slice of float64. Animation input accessors must be FLOAT, while the
// output accessors may also use normalized integer components, so those are
// denormalized here.
func toFloat64Slice(data any) ([]float64, error) {
	switch data := data.(type) {
	case []float32:
		out := make([]float64, len(data))
		for i, v := range data {
			out[i] = float64(v)
		}
		return out, nil
	case [][2]float32:
		out := make([]float64, 0, len(data)*2)
		for _, v := range data {
			out = append(out, float64(v[0]), float64(v[1]))
		}
		return out, nil
	case [][3]float32:
		out := make([]float64, 0, len(data)*3)
		for _, v := range data {
			out = append(out, float64(v[0]), float64(v[1]), float64(v[2]))
		}
		return out, nil
	case [][4]float32:
		out := make([]float64, 0, len(data)*4)
		for _, v := range data {
			out = append(out, float64(v[0]), float64(v[1]), float64(v[2]), float64(v[3]))
		}
		return out, nil
	case [][3]int16:
		out := make([]float64, 0, len(data)*3)
		for _, v := range data {
			out = append(out, float64(gltf.DenormalizeShort(v[0])), float64(gltf.DenormalizeShort(v[1])), float64(gltf.DenormalizeShort(v[2])))
		}
		return out, nil
	case [][3]int8:
		out := make([]float64, 0, len(data)*3)
		for _, v := range data {
			out = append(out, float64(gltf.DenormalizeByte(v[0])), float64(gltf.DenormalizeByte(v[1])), float64(gltf.DenormalizeByte(v[2])))
		}
		return out, nil
	case [][3]uint8:
		out := make([]float64, 0, len(data)*3)
		for _, v := range data {
			out = append(out, float64(gltf.DenormalizeUbyte(v[0])), float64(gltf.DenormalizeUbyte(v[1])), float64(gltf.DenormalizeUbyte(v[2])))
		}
		return out, nil
	case [][4]int16:
		out := make([]float64, 0, len(data)*4)
		for _, v := range data {
			out = append(out, float64(gltf.DenormalizeShort(v[0])), float64(gltf.DenormalizeShort(v[1])), float64(gltf.DenormalizeShort(v[2])), float64(gltf.DenormalizeShort(v[3])))
		}
		return out, nil
	case [][3]uint16:
		out := make([]float64, 0, len(data)*3)
		for _, v := range data {
			out = append(out, float64(gltf.DenormalizeUshort(v[0])), float64(gltf.DenormalizeUshort(v[1])), float64(gltf.DenormalizeUshort(v[2])))
		}
		return out, nil
	case [][4]uint16:
		out := make([]float64, 0, len(data)*4)
		for _, v := range data {
			out = append(out, float64(gltf.DenormalizeUshort(v[0])), float64(gltf.DenormalizeUshort(v[1])), float64(gltf.DenormalizeUshort(v[2])), float64(gltf.DenormalizeUshort(v[3])))
		}
		return out, nil
	case [][4]int8:
		out := make([]float64, 0, len(data)*4)
		for _, v := range data {
			out = append(out, float64(gltf.DenormalizeByte(v[0])), float64(gltf.DenormalizeByte(v[1])), float64(gltf.DenormalizeByte(v[2])), float64(gltf.DenormalizeByte(v[3])))
		}
		return out, nil
	case [][4]uint8:
		out := make([]float64, 0, len(data)*4)
		for _, v := range data {
			out = append(out, float64(gltf.DenormalizeUbyte(v[0])), float64(gltf.DenormalizeUbyte(v[1])), float64(gltf.DenormalizeUbyte(v[2])), float64(gltf.DenormalizeUbyte(v[3])))
		}
		return out, nil
	default:
		return nil, fmt.Errorf("component type %T not allowed", data)
	}
}
