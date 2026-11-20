package modeler_test

import (
	"encoding/binary"
	"math"
	"reflect"
	"testing"

	"github.com/qmuntal/gltf"
	"github.com/qmuntal/gltf/modeler"
)

// animSamplerDef is the description of one AnimationSampler used by
// buildAnimationDoc.
type animSamplerDef struct {
	interpolation gltf.Interpolation
	times         []float32
	// For CUBICSPLINE every keyframe contributes in-tangent, value and
	// out-tangent, in that order. For the other modes it contributes one
	// vector per keyframe.
	values3 [][3]float32
	values4 [][4]float32
}

// animChannelDef is the description of one AnimationChannel.
type animChannelDef struct {
	sampler int
	node    int
	path    gltf.TRSProperty
}

// buildAnimationDoc writes the given samplers and channels into a fresh
// document whose nodes carry non-default TRS values, so tests can verify
// that non-animated components keep the node defaults.
func buildAnimationDoc(t *testing.T, samplers []animSamplerDef, channels []animChannelDef, extraNodes int) *gltf.Document {
	t.Helper()
	doc := gltf.NewDocument()
	doc.Nodes = []*gltf.Node{{
		Name:        "node0",
		Translation: [3]float64{10, 20, 30},
		Rotation:    [4]float64{0, 0, 0, 1},
		Scale:       [3]float64{2, 3, 4},
	}, {
		Name:        "node1",
		Translation: [3]float64{-5, 0, 7},
	}}
	for i := 0; i < extraNodes; i++ {
		doc.Nodes = append(doc.Nodes, &gltf.Node{Name: "extra"})
	}

	anim := &gltf.Animation{Name: "test"}
	for _, s := range samplers {
		input := modeler.WriteAccessor(doc, gltf.TargetArrayBuffer, s.times)
		var output int
		if s.values4 != nil {
			output = modeler.WriteAccessor(doc, gltf.TargetArrayBuffer, s.values4)
		} else {
			output = modeler.WriteAccessor(doc, gltf.TargetArrayBuffer, s.values3)
		}
		anim.Samplers = append(anim.Samplers, &gltf.AnimationSampler{
			Input:         input,
			Output:        output,
			Interpolation: s.interpolation,
		})
	}
	for _, c := range channels {
		anim.Channels = append(anim.Channels, &gltf.AnimationChannel{
			Sampler: c.sampler,
			Target: gltf.AnimationChannelTarget{
				Node: gltf.Index(c.node),
				Path: c.path,
			},
		})
	}
	doc.Animations = []*gltf.Animation{anim}
	doc.Scenes[0].Nodes = []int{0, 1}
	return doc
}

func approxEqual(a, b, tol float64) bool {
	return math.Abs(a-b) <= tol
}

func assertPose(t *testing.T, got modeler.NodePose, want modeler.NodePose) {
	t.Helper()
	const tol = 1e-5
	for i := 0; i < 3; i++ {
		if !approxEqual(got.Translation[i], want.Translation[i], tol) {
			t.Errorf("translation[%d] = %v, want %v", i, got.Translation, want.Translation)
		}
		if !approxEqual(got.Scale[i], want.Scale[i], tol) {
			t.Errorf("scale[%d] = %v, want %v", i, got.Scale, want.Scale)
		}
	}
	for i := 0; i < 4; i++ {
		if !approxEqual(got.Rotation[i], want.Rotation[i], tol) {
			t.Errorf("rotation[%d] = %v, want %v", i, got.Rotation, want.Rotation)
		}
	}
	length := 0.0
	for _, q := range got.Rotation {
		length += q * q
	}
	if !approxEqual(math.Sqrt(length), 1, tol) {
		t.Errorf("rotation %v is not a unit quaternion", got.Rotation)
	}
}

func TestReadAnimationLinearTranslation(t *testing.T) {
	doc := buildAnimationDoc(t,
		[]animSamplerDef{{
			interpolation: gltf.InterpolationLinear,
			times:         []float32{0, 1, 2},
			values3:       [][3]float32{{0, 0, 0}, {10, 20, 30}, {20, 0, -10}},
		}},
		[]animChannelDef{{0, 0, gltf.TRSTranslation}}, 0)

	cases := []struct {
		time float64
		want [3]float64
	}{
		{0, [3]float64{0, 0, 0}},      // start keyframe
		{0.5, [3]float64{5, 10, 15}},  // midpoint between kf0 and kf1
		{1, [3]float64{10, 20, 30}},   // middle keyframe
		{1.5, [3]float64{15, 10, 10}}, // midpoint between kf1 and kf2
		{2, [3]float64{20, 0, -10}},   // end keyframe
		{-10, [3]float64{0, 0, 0}},    // before the range: clamped to the start
		{10, [3]float64{20, 0, -10}},  // after the range: clamped to the end
		{0.25, [3]float64{2.5, 5, 7.5}},
	}
	for _, tc := range cases {
		poses, err := modeler.ReadAnimation(doc, 0, tc.time)
		if err != nil {
			t.Fatalf("time %v: unexpected error %v", tc.time, err)
		}
		pose, ok := poses[0]
		if !ok || len(poses) != 1 {
			t.Fatalf("time %v: expected exactly one pose for node 0, got %v", tc.time, poses)
		}
		// Rotation and scale are not animated: the node defaults survive.
		want := modeler.NodePose{
			Translation: tc.want,
			Rotation:    [4]float64{0, 0, 0, 1},
			Scale:       [3]float64{2, 3, 4},
		}
		assertPose(t, pose, want)
	}
}

func TestReadAnimationStepScale(t *testing.T) {
	doc := buildAnimationDoc(t,
		[]animSamplerDef{{
			interpolation: gltf.InterpolationStep,
			times:         []float32{0, 1},
			values3:       [][3]float32{{1, 1, 1}, {5, 5, 5}},
		}},
		[]animChannelDef{{0, 0, gltf.TRSScale}}, 0)

	cases := []struct {
		time float64
		want [3]float64
	}{
		{0, [3]float64{1, 1, 1}},
		{0.999, [3]float64{1, 1, 1}}, // STEP holds the previous keyframe
		{1, [3]float64{5, 5, 5}},
		{42, [3]float64{5, 5, 5}},  // clamped to the last keyframe
		{-42, [3]float64{1, 1, 1}}, // clamped to the first keyframe
	}
	for _, tc := range cases {
		poses, err := modeler.ReadAnimation(doc, 0, tc.time)
		if err != nil {
			t.Fatalf("time %v: unexpected error %v", tc.time, err)
		}
		want := modeler.NodePose{
			Translation: [3]float64{10, 20, 30}, // node default, not animated
			Rotation:    [4]float64{0, 0, 0, 1},
			Scale:       tc.want,
		}
		assertPose(t, poses[0], want)
	}
}

func TestReadAnimationCubicSplineTranslation(t *testing.T) {
	// Zero tangents turn the cubic Hermite curve into the smoothstep
	// blend of the two values, which has a closed-form expectation.
	doc := buildAnimationDoc(t,
		[]animSamplerDef{{
			interpolation: gltf.InterpolationCubicSpline,
			times:         []float32{0, 2},
			// keyframe 0: in-tangent, value, out-tangent; then keyframe 1.
			values3: [][3]float32{
				{0, 0, 0}, {0, 0, 0}, {0, 0, 0},
				{0, 0, 0}, {10, 0, 0}, {0, 0, 0},
			},
		}},
		[]animChannelDef{{0, 0, gltf.TRSTranslation}}, 0)

	poses, err := modeler.ReadAnimation(doc, 0, 1)
	if err != nil {
		t.Fatal(err)
	}
	// smoothstep(0.5) = 0.5, so the midpoint is the average of both values.
	want := modeler.NodePose{
		Translation: [3]float64{5, 0, 0},
		Rotation:    [4]float64{0, 0, 0, 1},
		Scale:       [3]float64{2, 3, 4},
	}
	assertPose(t, poses[0], want)

	// At a quarter of the interval smoothstep gives 0.15625.
	poses, err = modeler.ReadAnimation(doc, 0, 0.5)
	if err != nil {
		t.Fatal(err)
	}
	if !approxEqual(poses[0].Translation[0], 1.5625, 1e-5) {
		t.Errorf("cubicspline t=0.5 = %v, want 1.5625", poses[0].Translation)
	}

	// Endpoints return the stored values exactly.
	poses, err = modeler.ReadAnimation(doc, 0, 2)
	if err != nil {
		t.Fatal(err)
	}
	if poses[0].Translation != [3]float64{10, 0, 0} {
		t.Errorf("cubicspline endpoint = %v, want [10 0 0]", poses[0].Translation)
	}
}

func TestReadAnimationCubicSplineTangents(t *testing.T) {
	// Hermite curve with p0 = 0, p1 = 10, out-tangent m0 = 2,
	// in-tangent m1 = 4 and dt = 2. At alpha = 0.5 the basis gives
	// 0.125*dt*m0 + 0.125*dt*m1 + 0.5*p1 = 0.5 + 1 + 5 = 6.5.
	doc := buildAnimationDoc(t,
		[]animSamplerDef{{
			interpolation: gltf.InterpolationCubicSpline,
			times:         []float32{0, 2},
			values3: [][3]float32{
				{0, 0, 0}, {0, 0, 0}, {2, 2, 2}, // kf0: in, value, out
				{4, 4, 4}, {10, 10, 10}, {0, 0, 0}, // kf1: in, value, out
			},
		}},
		[]animChannelDef{{0, 0, gltf.TRSTranslation}}, 0)

	poses, err := modeler.ReadAnimation(doc, 0, 1)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range poses[0].Translation {
		if !approxEqual(v, 6.5, 1e-5) {
			t.Errorf("translation[%d] = %v, want 6.5 (full pose %v)", i, v, poses[0].Translation)
		}
	}
}

func TestReadAnimationQuaternionShortestArc(t *testing.T) {
	// 180-degree rotation about Y: the direct blend would take the long
	// way around; slerp must pick the shortest arc and stay unit length.
	doc := buildAnimationDoc(t,
		[]animSamplerDef{{
			interpolation: gltf.InterpolationLinear,
			times:         []float32{0, 1},
			values4:       [][4]float32{{0, 0, 0, 1}, {0, 1, 0, 0}},
		}},
		[]animChannelDef{{0, 0, gltf.TRSRotation}}, 0)

	poses, err := modeler.ReadAnimation(doc, 0, 0.5)
	if err != nil {
		t.Fatal(err)
	}
	want := modeler.NodePose{
		Translation: [3]float64{10, 20, 30},
		Rotation:    [4]float64{0, 0.70710678, 0, 0.70710678},
		Scale:       [3]float64{2, 3, 4},
	}
	assertPose(t, poses[0], want)

	// Exact keyframes return the stored values.
	poses, _ = modeler.ReadAnimation(doc, 0, 0)
	if poses[0].Rotation != [4]float64{0, 0, 0, 1} {
		t.Errorf("start rotation = %v, want identity", poses[0].Rotation)
	}
	poses, _ = modeler.ReadAnimation(doc, 0, 1)
	assertPose(t, poses[0], modeler.NodePose{
		Translation: [3]float64{10, 20, 30},
		Rotation:    [4]float64{0, 1, 0, 0},
		Scale:       [3]float64{2, 3, 4},
	})
}

func TestReadAnimationCubicSplineRotation(t *testing.T) {
	// CUBICSPLINE rotations use per-component Hermite interpolation and
	// are normalized afterwards. With zero tangents the midpoint is the
	// plain component-wise average of both unit quaternions.
	doc := buildAnimationDoc(t,
		[]animSamplerDef{{
			interpolation: gltf.InterpolationCubicSpline,
			times:         []float32{0, 1},
			values4: [][4]float32{
				{0, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 0, 0},
				{0, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 0, 0},
			},
		}},
		[]animChannelDef{{0, 0, gltf.TRSRotation}}, 0)

	poses, err := modeler.ReadAnimation(doc, 0, 0.5)
	if err != nil {
		t.Fatal(err)
	}
	// Average is (0, 0.5, 0, 0.5), whose normalized form is
	// (0, 1/sqrt(2), 0, 1/sqrt(2)).
	want := modeler.NodePose{
		Translation: [3]float64{10, 20, 30},
		Rotation:    [4]float64{0, 0.70710678, 0, 0.70710678},
		Scale:       [3]float64{2, 3, 4},
	}
	assertPose(t, poses[0], want)
}

func TestReadAnimationQuaternionNormalizesNonUnitInput(t *testing.T) {
	// A scaled quaternion must be normalized instead of leaking a
	// non-unit result.
	doc := buildAnimationDoc(t,
		[]animSamplerDef{{
			interpolation: gltf.InterpolationLinear,
			times:         []float32{0, 1},
			values4:       [][4]float32{{0, 0, 0, 2}, {0, 0, 0, 2}},
		}},
		[]animChannelDef{{0, 0, gltf.TRSRotation}}, 0)

	poses, err := modeler.ReadAnimation(doc, 0, 0.5)
	if err != nil {
		t.Fatal(err)
	}
	if poses[0].Rotation != [4]float64{0, 0, 0, 1} {
		t.Errorf("rotation = %v, want normalized identity", poses[0].Rotation)
	}
}

func TestReadAnimationTwoNodesAndDefaults(t *testing.T) {
	// Node 0 animates translation, node 1 (no explicit rotation/scale)
	// animates scale. Both keep their untouched components.
	doc := buildAnimationDoc(t,
		[]animSamplerDef{
			{interpolation: gltf.InterpolationLinear, times: []float32{0, 1}, values3: [][3]float32{{0, 0, 0}, {1, 0, 0}}},
			{interpolation: gltf.InterpolationStep, times: []float32{0, 1}, values3: [][3]float32{{1, 1, 1}, {8, 8, 8}}},
		},
		[]animChannelDef{
			{0, 0, gltf.TRSTranslation},
			{1, 1, gltf.TRSScale},
		}, 0)

	poses, err := modeler.ReadAnimation(doc, 0, 0.5)
	if err != nil {
		t.Fatal(err)
	}
	if len(poses) != 2 {
		t.Fatalf("expected two node poses, got %d", len(poses))
	}
	assertPose(t, poses[0], modeler.NodePose{
		Translation: [3]float64{0.5, 0, 0},
		Rotation:    [4]float64{0, 0, 0, 1},
		Scale:       [3]float64{2, 3, 4},
	})
	assertPose(t, poses[1], modeler.NodePose{
		Translation: [3]float64{-5, 0, 7},   // node 1 own translation
		Rotation:    [4]float64{0, 0, 0, 1}, // glTF default
		Scale:       [3]float64{1, 1, 1},    // STEP holds the first keyframe
	})
}

func TestReadAnimationMultipleChannelsSameNode(t *testing.T) {
	// Two channels drive different components of the same node; the
	// resulting pose merges both animations.
	doc := buildAnimationDoc(t,
		[]animSamplerDef{
			{interpolation: gltf.InterpolationLinear, times: []float32{0, 1}, values3: [][3]float32{{0, 0, 0}, {4, 4, 4}}},
			{interpolation: gltf.InterpolationLinear, times: []float32{0, 1}, values3: [][3]float32{{1, 1, 1}, {3, 3, 3}}},
		},
		[]animChannelDef{
			{0, 0, gltf.TRSTranslation},
			{1, 0, gltf.TRSScale},
		}, 0)

	poses, err := modeler.ReadAnimation(doc, 0, 0.5)
	if err != nil {
		t.Fatal(err)
	}
	assertPose(t, poses[0], modeler.NodePose{
		Translation: [3]float64{2, 2, 2},
		Rotation:    [4]float64{0, 0, 0, 1},
		Scale:       [3]float64{2, 2, 2},
	})
}

func TestReadAnimationWeightsChannelIgnored(t *testing.T) {
	doc := buildAnimationDoc(t,
		[]animSamplerDef{{
			interpolation: gltf.InterpolationLinear,
			times:         []float32{0, 1},
			values3:       [][3]float32{{0, 0, 0}, {1, 1, 1}},
		}},
		[]animChannelDef{{0, 0, gltf.TRSWeights}}, 0)

	poses, err := modeler.ReadAnimation(doc, 0, 0.5)
	if err != nil {
		t.Fatalf("weights channel should be ignored, got error %v", err)
	}
	if len(poses) != 0 {
		t.Errorf("expected no TRS poses for a weights-only animation, got %v", poses)
	}
}

func TestReadAnimationEmptyAnimation(t *testing.T) {
	doc := gltf.NewDocument()
	doc.Nodes = []*gltf.Node{{}}
	doc.Animations = []*gltf.Animation{{Channels: nil, Samplers: nil}}
	poses, err := modeler.ReadAnimation(doc, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(poses) != 0 {
		t.Errorf("expected empty poses, got %v", poses)
	}
}

func TestReadAnimationNormalizedIntegerRotation(t *testing.T) {
	// Normalized SHORT quaternions are a legal animation output.
	doc := gltf.NewDocument()
	doc.Nodes = []*gltf.Node{{}}
	times := modeler.WriteAccessor(doc, gltf.TargetArrayBuffer, []float32{0, 1})
	output := modeler.WriteAccessor(doc, gltf.TargetArrayBuffer, [][4]int16{{0, 0, 0, 32767}, {0, 32767, 0, 0}})
	doc.Accessors[output].Normalized = true
	doc.Animations = []*gltf.Animation{{
		Channels: []*gltf.AnimationChannel{{
			Sampler: 0,
			Target:  gltf.AnimationChannelTarget{Node: gltf.Index(0), Path: gltf.TRSRotation},
		}},
		Samplers: []*gltf.AnimationSampler{{Input: times, Output: output}},
	}}

	poses, err := modeler.ReadAnimation(doc, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	assertPose(t, poses[0], modeler.NodePose{
		Translation: [3]float64{0, 0, 0},
		Rotation:    [4]float64{0, 0, 0, 1},
		Scale:       [3]float64{1, 1, 1},
	})
}

func TestReadAnimationNormalizedUbyteTranslation(t *testing.T) {
	// Normalized UNSIGNED_BYTE VEC3 outputs are denormalized into [-1, 1].
	doc := gltf.NewDocument()
	doc.Nodes = []*gltf.Node{{}}
	times := modeler.WriteAccessor(doc, gltf.TargetArrayBuffer, []float32{0, 1})
	output := modeler.WriteAccessor(doc, gltf.TargetArrayBuffer, [][3]uint8{{0, 0, 0}, {255, 128, 0}})
	doc.Accessors[output].Normalized = true
	doc.Animations = []*gltf.Animation{{
		Channels: []*gltf.AnimationChannel{{
			Sampler: 0,
			Target:  gltf.AnimationChannelTarget{Node: gltf.Index(0), Path: gltf.TRSTranslation},
		}},
		Samplers: []*gltf.AnimationSampler{{Input: times, Output: output}},
	}}

	poses, err := modeler.ReadAnimation(doc, 0, 1)
	if err != nil {
		t.Fatal(err)
	}
	assertPose(t, poses[0], modeler.NodePose{
		Translation: [3]float64{1, 128.0 / 255.0, 0},
		Rotation:    [4]float64{0, 0, 0, 1},
		Scale:       [3]float64{1, 1, 1},
	})
}

func TestReadAnimationErrors(t *testing.T) {
	validDoc := buildAnimationDoc(t,
		[]animSamplerDef{{
			interpolation: gltf.InterpolationLinear,
			times:         []float32{0, 1},
			values3:       [][3]float32{{0, 0, 0}, {1, 1, 1}},
		}},
		[]animChannelDef{{0, 0, gltf.TRSTranslation}}, 0)

	// Hand-crafted accessors for the cases the builder cannot produce.
	nanBytes := make([]byte, 8)
	binary.LittleEndian.PutUint32(nanBytes[0:], math.Float32bits(float32(0)))
	binary.LittleEndian.PutUint32(nanBytes[4:], math.Float32bits(float32(math.NaN())))
	nanDoc := gltf.NewDocument()
	nanDoc.Nodes = []*gltf.Node{{}}
	nanDoc.Buffers = []*gltf.Buffer{{Data: nanBytes, ByteLength: len(nanBytes)}}
	nanDoc.BufferViews = []*gltf.BufferView{{Buffer: 0, ByteLength: len(nanBytes)}}
	nanDoc.Accessors = []*gltf.Accessor{{
		BufferView: gltf.Index(0), ComponentType: gltf.ComponentFloat, Type: gltf.AccessorScalar, Count: 2,
	}, {
		BufferView: gltf.Index(0), ComponentType: gltf.ComponentFloat, Type: gltf.AccessorVec3, Count: 2,
	}}
	nanDoc.Animations = []*gltf.Animation{{
		Channels: []*gltf.AnimationChannel{{
			Sampler: 0,
			Target:  gltf.AnimationChannelTarget{Node: gltf.Index(0), Path: gltf.TRSTranslation},
		}},
		Samplers: []*gltf.AnimationSampler{{Input: 0, Output: 1}},
	}}

	// A document with a broken output accessor (buffer view out of range).
	badAccessorDoc := buildAnimationDoc(t,
		[]animSamplerDef{{
			interpolation: gltf.InterpolationLinear,
			times:         []float32{0, 1},
			values3:       [][3]float32{{0, 0, 0}, {1, 1, 1}},
		}},
		[]animChannelDef{{0, 0, gltf.TRSTranslation}}, 0)
	badAccessorDoc.Accessors[badAccessorDoc.Animations[0].Samplers[0].Output].BufferView = gltf.Index(999)

	tests := []struct {
		name      string
		doc       *gltf.Document
		animation int
		time      float64
		mutate    func(*gltf.Document)
		wantErr   bool
	}{
		{"nil document", nil, 0, 0, nil, true},
		{"animation overflow", validDoc, 7, 0, nil, true},
		{"negative animation", validDoc, -1, 0, nil, true},
		{"NaN time", validDoc, 0, math.NaN(), nil, true},
		{"+Inf time", validDoc, 0, math.Inf(1), nil, true},
		{"-Inf time", validDoc, 0, math.Inf(-1), nil, true},
		{"node overflow", validDoc, 0, 0, func(d *gltf.Document) {
			d.Animations[0].Channels[0].Target.Node = gltf.Index(42)
		}, true},
		{"nil node target", validDoc, 0, 0, func(d *gltf.Document) {
			d.Animations[0].Channels[0].Target.Node = nil
		}, false}, // node-less targets are ignored, not an error
		{"sampler overflow", validDoc, 0, 0, func(d *gltf.Document) {
			d.Animations[0].Channels[0].Sampler = 42
		}, true},
		{"unknown path", validDoc, 0, 0, func(d *gltf.Document) {
			d.Animations[0].Channels[0].Target.Path = "explode"
		}, true},
		{"input accessor overflow", validDoc, 0, 0, func(d *gltf.Document) {
			d.Animations[0].Samplers[0].Input = 99
		}, true},
		{"output accessor overflow", validDoc, 0, 0, func(d *gltf.Document) {
			d.Animations[0].Samplers[0].Output = 99
		}, true},
		{"input not scalar", validDoc, 0, 0, func(d *gltf.Document) {
			d.Accessors[d.Animations[0].Samplers[0].Input].Type = gltf.AccessorVec3
		}, true},
		{"output wrong type", validDoc, 0, 0, func(d *gltf.Document) {
			d.Accessors[d.Animations[0].Samplers[0].Output].Type = gltf.AccessorMat4
		}, true},
		{"rotation channel with vec3 output", validDoc, 0, 0, func(d *gltf.Document) {
			d.Animations[0].Channels[0].Target.Path = gltf.TRSRotation
		}, true},
		{"duplicated keyframe time", validDoc, 0, 0, func(d *gltf.Document) {
			d.Animations[0].Samplers[0] = &gltf.AnimationSampler{
				Input:  modeler.WriteAccessor(d, gltf.TargetArrayBuffer, []float32{1, 1}),
				Output: modeler.WriteAccessor(d, gltf.TargetArrayBuffer, [][3]float32{{0, 0, 0}, {1, 1, 1}}),
			}
		}, true},
		{"reversed keyframe times", validDoc, 0, 0, func(d *gltf.Document) {
			d.Animations[0].Samplers[0] = &gltf.AnimationSampler{
				Input:  modeler.WriteAccessor(d, gltf.TargetArrayBuffer, []float32{1, 0}),
				Output: modeler.WriteAccessor(d, gltf.TargetArrayBuffer, [][3]float32{{0, 0, 0}, {1, 1, 1}}),
			}
		}, true},
		{"empty keyframes", validDoc, 0, 0, func(d *gltf.Document) {
			d.Animations[0].Samplers[0] = &gltf.AnimationSampler{
				Input:  modeler.WriteAccessor(d, gltf.TargetArrayBuffer, []float32{}),
				Output: modeler.WriteAccessor(d, gltf.TargetArrayBuffer, [][3]float32{}),
			}
		}, true},
		{"cubicspline count mismatch", validDoc, 0, 0, func(d *gltf.Document) {
			d.Animations[0].Samplers[0] = &gltf.AnimationSampler{
				Interpolation: gltf.InterpolationCubicSpline,
				Input:         modeler.WriteAccessor(d, gltf.TargetArrayBuffer, []float32{0, 1}),
				// Only one vector per keyframe instead of tangent/value/tangent.
				Output: modeler.WriteAccessor(d, gltf.TargetArrayBuffer, [][3]float32{{0, 0, 0}, {1, 1, 1}}),
			}
		}, true},
		{"non finite keyframe time", nanDoc, 0, 0, nil, true},
		{"malformed output accessor", badAccessorDoc, 0, 0, nil, true},
		{"unnormalized integer output", validDoc, 0, 0, func(d *gltf.Document) {
			output := modeler.WriteAccessor(d, gltf.TargetArrayBuffer, [][4]int16{{0, 0, 0, 32767}, {0, 32767, 0, 0}})
			d.Animations[0].Samplers[0].Output = output
			d.Animations[0].Channels[0].Target.Path = gltf.TRSRotation
		}, true},
		{"nil animation", validDoc, 0, 0, func(d *gltf.Document) {
			d.Animations[0] = nil
		}, true},
		{"nil channel", validDoc, 0, 0, func(d *gltf.Document) {
			d.Animations[0].Channels[0] = nil
		}, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			doc := tt.doc
			if tt.mutate != nil {
				// Mutations must not leak between subtests: work on a
				// fresh copy of the shared valid document.
				if doc == validDoc {
					doc = deepCopyDoc(t, validDoc)
				}
				tt.mutate(doc)
			}
			_, err := modeler.ReadAnimation(doc, tt.animation, tt.time)
			if (err != nil) != tt.wantErr {
				t.Errorf("ReadAnimation() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

// deepCopyDoc round-trips a document through JSON so subtests can mutate it
// without sharing accessor/sampler pointers with the other cases.
func deepCopyDoc(t *testing.T, in *gltf.Document) *gltf.Document {
	t.Helper()
	// A structural copy is enough for these tests: keep the same backing
	// slices by rebuilding the mutable containers manually.
	out := *in
	out.Accessors = append([]*gltf.Accessor(nil), in.Accessors...)
	for i, a := range in.Accessors {
		cp := *a
		out.Accessors[i] = &cp
	}
	out.Animations = append([]*gltf.Animation(nil), in.Animations...)
	for i, a := range in.Animations {
		if a == nil {
			continue
		}
		cp := *a
		cp.Channels = append([]*gltf.AnimationChannel(nil), a.Channels...)
		for j, c := range a.Channels {
			if c == nil {
				continue
			}
			cc := *c
			if c.Target.Node != nil {
				node := *c.Target.Node
				cc.Target.Node = &node
			}
			cp.Channels[j] = &cc
		}
		cp.Samplers = append([]*gltf.AnimationSampler(nil), a.Samplers...)
		for j, s := range a.Samplers {
			if s == nil {
				continue
			}
			ss := *s
			cp.Samplers[j] = &ss
		}
		out.Animations[i] = &cp
	}
	return &out
}

func TestReadAnimationDoesNotModifyDocument(t *testing.T) {
	doc := buildAnimationDoc(t,
		[]animSamplerDef{{
			interpolation: gltf.InterpolationLinear,
			times:         []float32{0, 1, 2},
			values4:       [][4]float32{{0, 0, 0, 1}, {0, 1, 0, 0}, {0, 0, 0, 1}},
		}},
		[]animChannelDef{{0, 0, gltf.TRSRotation}}, 0)

	snapshot := deepCopyDoc(t, doc)
	for _, tm := range []float64{0, 0.3, 1, 1.7, 2, 50, -50} {
		if _, err := modeler.ReadAnimation(doc, 0, tm); err != nil {
			t.Fatal(err)
		}
	}
	if !reflect.DeepEqual(doc, snapshot) {
		t.Errorf("ReadAnimation modified the document")
	}
	// The node's own rotation is untouched as well.
	if doc.Nodes[0].Rotation != [4]float64{0, 0, 0, 1} {
		t.Errorf("node rotation was modified: %v", doc.Nodes[0].Rotation)
	}
}

func TestReadAnimationStateIsolation(t *testing.T) {
	// Two independent animations targeting the same node with different
	// values: sampling one must never affect the other.
	doc := buildAnimationDoc(t,
		[]animSamplerDef{
			{interpolation: gltf.InterpolationLinear, times: []float32{0, 1}, values3: [][3]float32{{0, 0, 0}, {100, 100, 100}}},
			{interpolation: gltf.InterpolationStep, times: []float32{0, 1}, values3: [][3]float32{{7, 7, 7}, {9, 9, 9}}},
		},
		[]animChannelDef{
			{0, 0, gltf.TRSTranslation},
			{1, 0, gltf.TRSScale},
		}, 0)
	// Move the second sampler into a second animation.
	doc.Animations = append(doc.Animations, &gltf.Animation{
		Channels: []*gltf.AnimationChannel{{
			Sampler: 0,
			Target:  gltf.AnimationChannelTarget{Node: gltf.Index(0), Path: gltf.TRSScale},
		}},
		Samplers: []*gltf.AnimationSampler{doc.Animations[0].Samplers[1]},
	})
	doc.Animations[0].Channels = doc.Animations[0].Channels[:1]
	doc.Animations[0].Samplers = doc.Animations[0].Samplers[:1]

	first, err := modeler.ReadAnimation(doc, 0, 0.5)
	if err != nil {
		t.Fatal(err)
	}
	if first[0].Translation != [3]float64{50, 50, 50} {
		t.Errorf("animation 0 translation = %v, want [50 50 50]", first[0].Translation)
	}
	// Animation 0 does not animate scale: node default survives.
	if first[0].Scale != [3]float64{2, 3, 4} {
		t.Errorf("animation 0 scale = %v, want node default", first[0].Scale)
	}

	second, err := modeler.ReadAnimation(doc, 1, 0.5)
	if err != nil {
		t.Fatal(err)
	}
	// STEP before t=1 holds the first keyframe.
	if second[0].Scale != [3]float64{7, 7, 7} {
		t.Errorf("animation 1 scale = %v, want [7 7 7]", second[0].Scale)
	}
	// Animation 1 does not animate translation: node default survives.
	if second[0].Translation != [3]float64{10, 20, 30} {
		t.Errorf("animation 1 translation = %v, want node default", second[0].Translation)
	}

	// Re-sampling the first animation still returns the same result.
	again, err := modeler.ReadAnimation(doc, 0, 0.5)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(again, first) {
		t.Errorf("animation state leaked between calls: %v != %v", again, first)
	}
}
