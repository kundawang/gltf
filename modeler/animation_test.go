package modeler_test

import (
	"bytes"
	"math"
	"testing"

	"github.com/go-test/deep"
	"github.com/qmuntal/gltf"
	"github.com/qmuntal/gltf/modeler"
)

func linearSampler(doc *gltf.Document, times []float32, values any, interp gltf.Interpolation) *gltf.AnimationSampler {
	input := modeler.WriteAccessor(doc, gltf.TargetNone, times)
	output := modeler.WriteAccessor(doc, gltf.TargetNone, values)
	return &gltf.AnimationSampler{Input: input, Output: output, Interpolation: interp}
}

func channel(sampler int, path gltf.TRSProperty, node int) *gltf.AnimationChannel {
	return &gltf.AnimationChannel{
		Sampler: sampler,
		Target:  gltf.AnimationChannelTarget{Node: gltf.Index(node), Path: path},
	}
}

func readSamplerData(t *testing.T, doc *gltf.Document, sampler *gltf.AnimationSampler) ([]float32, any) {
	t.Helper()
	times, err := modeler.ReadAccessor(doc, doc.Accessors[sampler.Input], nil)
	if err != nil {
		t.Fatalf("ReadAccessor(input) error = %v", err)
	}
	values, err := modeler.ReadAccessor(doc, doc.Accessors[sampler.Output], nil)
	if err != nil {
		t.Fatalf("ReadAccessor(output) error = %v", err)
	}
	return times.([]float32), values
}

// TestSimplifyAnimation_Translation_ZeroTolerance covers the requested case:
// t = 0, 1, 2 s with translation x = 0, 1, 2 is exactly linear, so the middle
// keyframe is removed at zero tolerance; with x = 2 at the middle it must be
// retained.
func TestSimplifyAnimation_Translation_ZeroTolerance(t *testing.T) {
	times := []float32{0, 1, 2}

	t.Run("collinear middle keyframe is removed", func(t *testing.T) {
		doc := gltf.NewDocument()
		values := [][3]float32{{0, 0, 0}, {1, 0, 0}, {2, 0, 0}}
		sampler := linearSampler(doc, times, values, gltf.InterpolationLinear)
		doc.Animations = []*gltf.Animation{{
			Samplers: []*gltf.AnimationSampler{sampler},
			Channels: []*gltf.AnimationChannel{channel(0, gltf.TRSTranslation, 0)},
		}}

		result, err := modeler.SimplifyAnimation(doc, 0, modeler.AnimationSimplifyOptions{})
		if err != nil {
			t.Fatalf("SimplifyAnimation() error = %v", err)
		}
		if result.KeyframesRemoved != 1 || result.SamplersSimplified != 1 {
			t.Fatalf("SimplifyAnimation() result = %+v, want 1 removed keyframe", result)
		}
		gotTimes, gotValuesAny := readSamplerData(t, doc, doc.Animations[0].Samplers[0])
		gotValues := gotValuesAny.([][3]float32)
		if len(gotTimes) != 2 || gotTimes[0] != 0 || gotTimes[1] != 2 {
			t.Errorf("retained times = %v, want [0 2]", gotTimes)
		}
		wantValues := [][3]float32{{0, 0, 0}, {2, 0, 0}}
		if diff := deepEqual(gotValues, wantValues); diff != "" {
			t.Errorf("retained values = %v, want %v (%s)", gotValues, wantValues, diff)
		}
	})

	t.Run("deviating middle keyframe is retained", func(t *testing.T) {
		doc := gltf.NewDocument()
		values := [][3]float32{{0, 0, 0}, {2, 0, 0}, {2, 0, 0}}
		sampler := linearSampler(doc, times, values, gltf.InterpolationLinear)
		doc.Animations = []*gltf.Animation{{
			Samplers: []*gltf.AnimationSampler{sampler},
			Channels: []*gltf.AnimationChannel{channel(0, gltf.TRSTranslation, 0)},
		}}

		result, err := modeler.SimplifyAnimation(doc, 0, modeler.AnimationSimplifyOptions{})
		if err != nil {
			t.Fatalf("SimplifyAnimation() error = %v", err)
		}
		if result.KeyframesRemoved != 0 || result.SamplersSimplified != 1 {
			t.Fatalf("SimplifyAnimation() result = %+v, want no removed keyframe", result)
		}
		gotTimes, gotValuesAny := readSamplerData(t, doc, doc.Animations[0].Samplers[0])
		gotValues := gotValuesAny.([][3]float32)
		if len(gotTimes) != 3 {
			t.Fatalf("retained times = %v, want all 3 keyframes", gotTimes)
		}
		wantValues := [][3]float32{{0, 0, 0}, {2, 0, 0}, {2, 0, 0}}
		if diff := deepEqual(gotValues, wantValues); diff != "" {
			t.Errorf("retained values = %v, want %v (%s)", gotValues, wantValues, diff)
		}
	})
}

// TestSimplifyAnimation_Rotation mirrors the translation test with quaternion
// slerp: a midpoint that coincides with the slerp interpolation is removed once
// the angular tolerance covers the float32 quantization error, while a rotated
// middle keyframe is retained unless the tolerance covers its angular error.
func TestSimplifyAnimation_Rotation(t *testing.T) {
	times := []float32{0, 1, 2}
	q0 := [4]float32{0, 0, 0, 1}
	q2 := normalizeQ([4]float32{0, float32(math.Sin(math.Pi / 8)), 0, float32(math.Cos(math.Pi / 8))}) // 45° around Y
	qMidExact := normalizeQ([4]float32{0, float32(math.Sin(math.Pi / 16)), 0, float32(math.Cos(math.Pi / 16))})
	qMidDeviated := normalizeQ([4]float32{0, float32(math.Sin(math.Pi / 4)), 0, float32(math.Cos(math.Pi / 4))}) // 90° around Y

	t.Run("quantized slerp midpoint is retained at zero tolerance", func(t *testing.T) {
		doc := gltf.NewDocument()
		values := [][4]float32{q0, qMidExact, q2}
		sampler := linearSampler(doc, times, values, gltf.InterpolationLinear)
		doc.Animations = []*gltf.Animation{{
			Samplers: []*gltf.AnimationSampler{sampler},
			Channels: []*gltf.AnimationChannel{channel(0, gltf.TRSRotation, 0)},
		}}

		// The midpoint is the analytical slerp midpoint, but float32 storage
		// leaves a sub-milliradian quantization error: zero tolerance must not
		// remove it.
		result, err := modeler.SimplifyAnimation(doc, 0, modeler.AnimationSimplifyOptions{})
		if err != nil {
			t.Fatalf("SimplifyAnimation() error = %v", err)
		}
		if result.KeyframesRemoved != 0 {
			t.Fatalf("SimplifyAnimation() result = %+v, want no removed keyframe at zero tolerance", result)
		}
	})

	t.Run("slerp midpoint is removed when the tolerance covers the quantization error", func(t *testing.T) {
		doc := gltf.NewDocument()
		values := [][4]float32{q0, qMidExact, q2}
		sampler := linearSampler(doc, times, values, gltf.InterpolationLinear)
		doc.Animations = []*gltf.Animation{{
			Samplers: []*gltf.AnimationSampler{sampler},
			Channels: []*gltf.AnimationChannel{channel(0, gltf.TRSRotation, 0)},
		}}

		result, err := modeler.SimplifyAnimation(doc, 0, modeler.AnimationSimplifyOptions{
			RotationTolerance: 1e-3,
		})
		if err != nil {
			t.Fatalf("SimplifyAnimation() error = %v", err)
		}
		if result.KeyframesRemoved != 1 {
			t.Fatalf("SimplifyAnimation() result = %+v, want 1 removed keyframe", result)
		}
		gotTimes, gotValuesAny := readSamplerData(t, doc, doc.Animations[0].Samplers[0])
		gotValues := gotValuesAny.([][4]float32)
		if len(gotTimes) != 2 || gotTimes[0] != 0 || gotTimes[1] != 2 {
			t.Errorf("retained times = %v, want [0 2]", gotTimes)
		}
		wantValues := [][4]float32{q0, q2}
		if diff := deepEqual(gotValues, wantValues); diff != "" {
			t.Errorf("retained values = %v, want %v (%s)", gotValues, wantValues, diff)
		}
	})

	t.Run("deviated midpoint is retained at zero tolerance", func(t *testing.T) {
		doc := gltf.NewDocument()
		values := [][4]float32{q0, qMidDeviated, q2}
		sampler := linearSampler(doc, times, values, gltf.InterpolationLinear)
		doc.Animations = []*gltf.Animation{{
			Samplers: []*gltf.AnimationSampler{sampler},
			Channels: []*gltf.AnimationChannel{channel(0, gltf.TRSRotation, 0)},
		}}

		result, err := modeler.SimplifyAnimation(doc, 0, modeler.AnimationSimplifyOptions{})
		if err != nil {
			t.Fatalf("SimplifyAnimation() error = %v", err)
		}
		if result.KeyframesRemoved != 0 {
			t.Fatalf("SimplifyAnimation() result = %+v, want no removed keyframe", result)
		}
	})

	t.Run("deviated midpoint is removed when the angle tolerance covers it", func(t *testing.T) {
		doc := gltf.NewDocument()
		values := [][4]float32{q0, qMidDeviated, q2}
		sampler := linearSampler(doc, times, values, gltf.InterpolationLinear)
		doc.Animations = []*gltf.Animation{{
			Samplers: []*gltf.AnimationSampler{sampler},
			Channels: []*gltf.AnimationChannel{channel(0, gltf.TRSRotation, 0)},
		}}

		// The interpolated midpoint is the 22.5° quaternion, the actual one is
		// 90°: the angular difference is 67.5°.
		result, err := modeler.SimplifyAnimation(doc, 0, modeler.AnimationSimplifyOptions{
			RotationTolerance: 68 * math.Pi / 180,
		})
		if err != nil {
			t.Fatalf("SimplifyAnimation() error = %v", err)
		}
		if result.KeyframesRemoved != 1 {
			t.Fatalf("SimplifyAnimation() result = %+v, want 1 removed keyframe", result)
		}
	})
}

func TestSimplifyAnimation_Scale(t *testing.T) {
	doc := gltf.NewDocument()
	times := []float32{0, 1, 2, 3}
	values := [][3]float32{{1, 1, 1}, {2, 2, 2}, {3, 3, 3}, {4, 5, 6}}
	sampler := linearSampler(doc, times, values, gltf.InterpolationLinear)
	doc.Animations = []*gltf.Animation{{
		Samplers: []*gltf.AnimationSampler{sampler},
		Channels: []*gltf.AnimationChannel{channel(0, gltf.TRSScale, 0)},
	}}

	result, err := modeler.SimplifyAnimation(doc, 0, modeler.AnimationSimplifyOptions{})
	if err != nil {
		t.Fatalf("SimplifyAnimation() error = %v", err)
	}
	// Point 2 is the worst deviation from the 0-3 chord and must be retained;
	// once 2 is retained, point 1 lies exactly on the 0-2 line and is removed.
	if result.KeyframesRemoved != 1 {
		t.Fatalf("SimplifyAnimation() result = %+v, want 1 removed keyframe", result)
	}
	gotTimes, gotValuesAny := readSamplerData(t, doc, doc.Animations[0].Samplers[0])
	gotValues := gotValuesAny.([][3]float32)
	wantTimes := []float32{0, 2, 3}
	wantValues := [][3]float32{{1, 1, 1}, {3, 3, 3}, {4, 5, 6}}
	if diff := deepEqual(gotTimes, wantTimes); diff != "" {
		t.Errorf("retained times = %v, want %v (%s)", gotTimes, wantTimes, diff)
	}
	if diff := deepEqual(gotValues, wantValues); diff != "" {
		t.Errorf("retained values = %v, want %v (%s)", gotValues, wantValues, diff)
	}
}

func TestSimplifyAnimation_AllCollinear(t *testing.T) {
	buildDoc := func() (*gltf.Document, *gltf.AnimationSampler) {
		doc := gltf.NewDocument()
		times := []float32{0, 1, 2, 3}
		values := [][3]float32{{1, 1, 1}, {2, 2, 2}, {3, 3, 3}, {4, 4, 4}}
		sampler := linearSampler(doc, times, values, gltf.InterpolationLinear)
		doc.Animations = []*gltf.Animation{{
			Samplers: []*gltf.AnimationSampler{sampler},
			Channels: []*gltf.AnimationChannel{channel(0, gltf.TRSScale, 0)},
		}}
		return doc, sampler
	}

	t.Run("zero tolerance keeps quantized interior keyframes", func(t *testing.T) {
		doc, _ := buildDoc()
		result, err := modeler.SimplifyAnimation(doc, 0, modeler.AnimationSimplifyOptions{})
		if err != nil {
			t.Fatalf("SimplifyAnimation() error = %v", err)
		}
		// Point 1 sits at alpha 1/3 and interpolates exactly; point 2 at alpha
		// 2/3 carries a sub-ULP float32 quantization error and must be kept.
		if result.KeyframesRemoved != 1 {
			t.Fatalf("SimplifyAnimation() result = %+v, want 1 removed keyframe", result)
		}
		gotTimes, _ := readSamplerData(t, doc, doc.Animations[0].Samplers[0])
		wantTimes := []float32{0, 2, 3}
		if diff := deepEqual(gotTimes, wantTimes); diff != "" {
			t.Errorf("retained times = %v, want %v (%s)", gotTimes, wantTimes, diff)
		}
	})

	t.Run("small tolerance removes every interior keyframe", func(t *testing.T) {
		doc, _ := buildDoc()
		result, err := modeler.SimplifyAnimation(doc, 0, modeler.AnimationSimplifyOptions{
			ScaleTolerance: 1e-6,
		})
		if err != nil {
			t.Fatalf("SimplifyAnimation() error = %v", err)
		}
		if result.KeyframesRemoved != 2 {
			t.Fatalf("SimplifyAnimation() result = %+v, want 2 removed keyframes", result)
		}
		gotTimes, gotValuesAny := readSamplerData(t, doc, doc.Animations[0].Samplers[0])
		gotValues := gotValuesAny.([][3]float32)
		wantTimes := []float32{0, 3}
		wantValues := [][3]float32{{1, 1, 1}, {4, 4, 4}}
		if diff := deepEqual(gotTimes, wantTimes); diff != "" {
			t.Errorf("retained times = %v, want %v (%s)", gotTimes, wantTimes, diff)
		}
		if diff := deepEqual(gotValues, wantValues); diff != "" {
			t.Errorf("retained values = %v, want %v (%s)", gotValues, wantValues, diff)
		}
	})
}

func TestSimplifyAnimation_PreservesEndpointsAndTolerance(t *testing.T) {
	doc := gltf.NewDocument()
	times := []float32{0, 1, 2}
	// Midpoint deviates by 0.5 on x from the linear interpolation.
	values := [][3]float32{{0, 0, 0}, {1.5, 0, 0}, {2, 0, 0}}
	sampler := linearSampler(doc, times, values, gltf.InterpolationLinear)
	doc.Animations = []*gltf.Animation{{
		Samplers: []*gltf.AnimationSampler{sampler},
		Channels: []*gltf.AnimationChannel{channel(0, gltf.TRSTranslation, 0)},
	}}

	if _, err := modeler.SimplifyAnimation(doc, 0, modeler.AnimationSimplifyOptions{
		PositionTolerance: 0.4,
	}); err != nil {
		t.Fatalf("SimplifyAnimation() error = %v", err)
	}
	gotTimes, _ := readSamplerData(t, doc, doc.Animations[0].Samplers[0])
	if len(gotTimes) != 3 {
		t.Fatalf("retained times = %v, want all 3 keyframes with tolerance below the error", gotTimes)
	}

	if _, err := modeler.SimplifyAnimation(doc, 0, modeler.AnimationSimplifyOptions{
		PositionTolerance: 0.5,
	}); err != nil {
		t.Fatalf("SimplifyAnimation() error = %v", err)
	}
	gotTimes, _ = readSamplerData(t, doc, doc.Animations[0].Samplers[0])
	if len(gotTimes) != 2 || gotTimes[0] != 0 || gotTimes[1] != 2 {
		t.Errorf("retained times = %v, want [0 2] once the tolerance covers the error", gotTimes)
	}
}

func TestSimplifyAnimation_SharedSampler(t *testing.T) {
	t.Run("shared sampler with the same path is processed once", func(t *testing.T) {
		doc := gltf.NewDocument()
		times := []float32{0, 1, 2}
		values := [][3]float32{{0, 0, 0}, {1, 0, 0}, {2, 0, 0}}
		sampler := linearSampler(doc, times, values, gltf.InterpolationLinear)
		doc.Animations = []*gltf.Animation{{
			Samplers: []*gltf.AnimationSampler{sampler},
			Channels: []*gltf.AnimationChannel{
				channel(0, gltf.TRSTranslation, 0),
				channel(0, gltf.TRSTranslation, 1),
			},
		}}

		result, err := modeler.SimplifyAnimation(doc, 0, modeler.AnimationSimplifyOptions{})
		if err != nil {
			t.Fatalf("SimplifyAnimation() error = %v", err)
		}
		if result.SamplersSimplified != 1 || result.KeyframesRemoved != 1 {
			t.Fatalf("SimplifyAnimation() result = %+v, want a single simplified sampler", result)
		}
		if len(result.SkippedSamplers) != 0 {
			t.Errorf("SkippedSamplers = %v, want empty", result.SkippedSamplers)
		}
	})

	t.Run("shared sampler with mixed paths is skipped", func(t *testing.T) {
		doc := gltf.NewDocument()
		times := []float32{0, 1, 2}
		values := [][3]float32{{0, 0, 0}, {1, 0, 0}, {2, 0, 0}}
		sampler := linearSampler(doc, times, values, gltf.InterpolationLinear)
		doc.Animations = []*gltf.Animation{{
			Samplers: []*gltf.AnimationSampler{sampler},
			Channels: []*gltf.AnimationChannel{
				channel(0, gltf.TRSTranslation, 0),
				channel(0, gltf.TRSScale, 1),
			},
		}}
		originalInput := sampler.Input

		result, err := modeler.SimplifyAnimation(doc, 0, modeler.AnimationSimplifyOptions{})
		if err != nil {
			t.Fatalf("SimplifyAnimation() error = %v", err)
		}
		if result.SamplersSimplified != 0 || len(result.SkippedSamplers) != 1 || result.SkippedSamplers[0] != 0 {
			t.Fatalf("SimplifyAnimation() result = %+v, want sampler 0 skipped", result)
		}
		if doc.Animations[0].Samplers[0].Input != originalInput {
			t.Error("the skipped sampler was modified")
		}
	})
}

func TestSimplifyAnimation_StepAndCubicSplineAreSkipped(t *testing.T) {
	doc := gltf.NewDocument()
	times := []float32{0, 1, 2}
	linearValues := [][3]float32{{0, 0, 0}, {1, 0, 0}, {2, 0, 0}}
	// CUBICSPLINE stores 3 VEC3 per keyframe (in/out tangent + value).
	cubicValues := make([][3]float32, 9)
	for i := range cubicValues {
		cubicValues[i] = [3]float32{float32(i), 0, 0}
	}
	stepSampler := linearSampler(doc, times, linearValues, gltf.InterpolationStep)
	cubicSampler := linearSampler(doc, times, cubicValues, gltf.InterpolationCubicSpline)
	originalStepInput, originalCubicInput := stepSampler.Input, cubicSampler.Input
	doc.Animations = []*gltf.Animation{{
		Samplers: []*gltf.AnimationSampler{stepSampler, cubicSampler},
		Channels: []*gltf.AnimationChannel{
			channel(0, gltf.TRSTranslation, 0),
			channel(1, gltf.TRSTranslation, 1),
		},
	}}

	result, err := modeler.SimplifyAnimation(doc, 0, modeler.AnimationSimplifyOptions{})
	if err != nil {
		t.Fatalf("SimplifyAnimation() error = %v", err)
	}
	if result.SamplersSimplified != 0 {
		t.Errorf("SamplersSimplified = %d, want 0", result.SamplersSimplified)
	}
	if len(result.SkippedSamplers) != 2 {
		t.Errorf("SkippedSamplers = %v, want both samplers skipped", result.SkippedSamplers)
	}
	if stepSampler.Input != originalStepInput || cubicSampler.Input != originalCubicInput {
		t.Error("a skipped sampler was modified")
	}
}

func TestSimplifyAnimation_WeightsAreSkipped(t *testing.T) {
	doc := gltf.NewDocument()
	times := []float32{0, 1, 2}
	values := []float32{0, 1, 2}
	sampler := linearSampler(doc, times, values, gltf.InterpolationLinear)
	originalInput := sampler.Input
	doc.Animations = []*gltf.Animation{{
		Samplers: []*gltf.AnimationSampler{sampler},
		Channels: []*gltf.AnimationChannel{channel(0, gltf.TRSWeights, 0)},
	}}

	result, err := modeler.SimplifyAnimation(doc, 0, modeler.AnimationSimplifyOptions{})
	if err != nil {
		t.Fatalf("SimplifyAnimation() error = %v", err)
	}
	if result.SamplersSimplified != 0 || len(result.SkippedSamplers) != 1 {
		t.Fatalf("SimplifyAnimation() result = %+v, want the weights sampler skipped", result)
	}
	if sampler.Input != originalInput {
		t.Error("the skipped sampler was modified")
	}
}

func TestSimplifyAnimation_InvalidTolerance(t *testing.T) {
	doc := gltf.NewDocument()
	times := []float32{0, 1, 2}
	values := [][3]float32{{0, 0, 0}, {1, 0, 0}, {2, 0, 0}}
	sampler := linearSampler(doc, times, values, gltf.InterpolationLinear)
	doc.Animations = []*gltf.Animation{{
		Samplers: []*gltf.AnimationSampler{sampler},
		Channels: []*gltf.AnimationChannel{channel(0, gltf.TRSTranslation, 0)},
	}}
	originalInput := sampler.Input

	for _, opts := range []modeler.AnimationSimplifyOptions{
		{PositionTolerance: -1},
		{ScaleTolerance: -0.1},
		{RotationTolerance: -math.Pi},
		{PositionTolerance: math.NaN()},
		{RotationTolerance: math.Inf(1)},
	} {
		if _, err := modeler.SimplifyAnimation(doc, 0, opts); err == nil {
			t.Errorf("SimplifyAnimation(%+v) error = nil, want an error", opts)
		}
	}
	if sampler.Input != originalInput {
		t.Error("the document was modified after a tolerance validation error")
	}
}

func TestSimplifyAnimation_DuplicatedAndReversedTimes(t *testing.T) {
	for name, badTimes := range map[string][]float32{
		"duplicated times": {0, 1, 1},
		"reversed times":   {0, 2, 1},
	} {
		t.Run(name, func(t *testing.T) {
			doc := gltf.NewDocument()
			values := [][3]float32{{0, 0, 0}, {1, 0, 0}, {2, 0, 0}}
			sampler := linearSampler(doc, badTimes, values, gltf.InterpolationLinear)
			originalInput := sampler.Input
			doc.Animations = []*gltf.Animation{{
				Samplers: []*gltf.AnimationSampler{sampler},
				Channels: []*gltf.AnimationChannel{channel(0, gltf.TRSTranslation, 0)},
			}}

			if _, err := modeler.SimplifyAnimation(doc, 0, modeler.AnimationSimplifyOptions{}); err == nil {
				t.Fatal("SimplifyAnimation() error = nil, want an error")
			}
			if sampler.Input != originalInput {
				t.Error("the document was modified after a time validation error")
			}
		})
	}
}

func TestSimplifyAnimation_NonFiniteValues(t *testing.T) {
	t.Run("non-finite time", func(t *testing.T) {
		doc := gltf.NewDocument()
		values := [][3]float32{{0, 0, 0}, {1, 0, 0}, {2, 0, 0}}
		sampler := linearSampler(doc, []float32{0, 1, float32(math.NaN())}, values, gltf.InterpolationLinear)
		doc.Animations = []*gltf.Animation{{
			Samplers: []*gltf.AnimationSampler{sampler},
			Channels: []*gltf.AnimationChannel{channel(0, gltf.TRSTranslation, 0)},
		}}
		if _, err := modeler.SimplifyAnimation(doc, 0, modeler.AnimationSimplifyOptions{}); err == nil {
			t.Error("SimplifyAnimation() error = nil, want an error")
		}
	})

	t.Run("non-finite translation component", func(t *testing.T) {
		doc := gltf.NewDocument()
		values := [][3]float32{{0, 0, 0}, {1, float32(math.Inf(1)), 0}, {2, 0, 0}}
		sampler := linearSampler(doc, []float32{0, 1, 2}, values, gltf.InterpolationLinear)
		doc.Animations = []*gltf.Animation{{
			Samplers: []*gltf.AnimationSampler{sampler},
			Channels: []*gltf.AnimationChannel{channel(0, gltf.TRSTranslation, 0)},
		}}
		if _, err := modeler.SimplifyAnimation(doc, 0, modeler.AnimationSimplifyOptions{}); err == nil {
			t.Error("SimplifyAnimation() error = nil, want an error")
		}
	})

	t.Run("zero-length quaternion", func(t *testing.T) {
		doc := gltf.NewDocument()
		values := [][4]float32{{0, 0, 0, 1}, {0, 0, 0, 0}, {0, 0, 0, 1}}
		sampler := linearSampler(doc, []float32{0, 1, 2}, values, gltf.InterpolationLinear)
		doc.Animations = []*gltf.Animation{{
			Samplers: []*gltf.AnimationSampler{sampler},
			Channels: []*gltf.AnimationChannel{channel(0, gltf.TRSRotation, 0)},
		}}
		if _, err := modeler.SimplifyAnimation(doc, 0, modeler.AnimationSimplifyOptions{}); err == nil {
			t.Error("SimplifyAnimation() error = nil, want an error")
		}
	})
}

func TestSimplifyAnimation_InvalidDocument(t *testing.T) {
	doc := gltf.NewDocument()
	doc.Animations = []*gltf.Animation{{
		Samplers: []*gltf.AnimationSampler{},
		Channels: []*gltf.AnimationChannel{},
	}}
	for _, index := range []int{-1, 1, 42} {
		if _, err := modeler.SimplifyAnimation(doc, index, modeler.AnimationSimplifyOptions{}); err == nil {
			t.Errorf("SimplifyAnimation(index=%d) error = nil, want an error", index)
		}
	}
	if _, err := modeler.SimplifyAnimation(nil, 0, modeler.AnimationSimplifyOptions{}); err == nil {
		t.Error("SimplifyAnimation(nil) error = nil, want an error")
	}
}

// TestSimplifyAnimation_RoundTrip verifies that the rewritten accessors produce
// a document that can be encoded as GLB and decoded again with the simplified
// animation intact.
func TestSimplifyAnimation_RoundTrip(t *testing.T) {
	doc := gltf.NewDocument()
	times := []float32{0, 1, 2}
	translation := [][3]float32{{0, 0, 0}, {1, 0, 0}, {2, 0, 0}}
	translationSampler := linearSampler(doc, times, translation, gltf.InterpolationLinear)
	rotation := [][4]float32{
		{0, 0, 0, 1},
		normalizeQ([4]float32{0, float32(math.Sin(math.Pi / 16)), 0, float32(math.Cos(math.Pi / 16))}),
		normalizeQ([4]float32{0, float32(math.Sin(math.Pi / 8)), 0, float32(math.Cos(math.Pi / 8))}),
	}
	rotationSampler := linearSampler(doc, times, rotation, gltf.InterpolationLinear)
	doc.Animations = []*gltf.Animation{{
		Samplers: []*gltf.AnimationSampler{translationSampler, rotationSampler},
		Channels: []*gltf.AnimationChannel{
			channel(0, gltf.TRSTranslation, 0),
			channel(1, gltf.TRSRotation, 0),
		},
	}}

	result, err := modeler.SimplifyAnimation(doc, 0, modeler.AnimationSimplifyOptions{
		RotationTolerance: 1e-3,
	})
	if err != nil {
		t.Fatalf("SimplifyAnimation() error = %v", err)
	}
	if result.KeyframesRemoved != 2 {
		t.Fatalf("SimplifyAnimation() result = %+v, want 2 removed keyframes", result)
	}

	var buf bytes.Buffer
	if err := gltf.NewEncoder(&buf).Encode(doc); err != nil {
		t.Fatalf("Encode() error = %v", err)
	}
	decoded := gltf.NewDocument()
	if err := gltf.NewDecoder(&buf).Decode(decoded); err != nil {
		t.Fatalf("Decode() error = %v", err)
	}
	if len(decoded.Animations) != 1 {
		t.Fatalf("decoded animations = %d, want 1", len(decoded.Animations))
	}
	samplers := decoded.Animations[0].Samplers
	if len(samplers) != 2 {
		t.Fatalf("decoded samplers = %d, want 2", len(samplers))
	}
	for i, sampler := range samplers {
		if decoded.Accessors[sampler.Input].Count != 2 {
			t.Errorf("sampler %d input count = %d, want 2", i, decoded.Accessors[sampler.Input].Count)
		}
		if decoded.Accessors[sampler.Output].Count != 2 {
			t.Errorf("sampler %d output count = %d, want 2", i, decoded.Accessors[sampler.Output].Count)
		}
	}

	gotTimes, gotTranslationAny := readSamplerData(t, decoded, samplers[0])
	gotTranslation := gotTranslationAny.([][3]float32)
	wantTimes := []float32{0, 2}
	wantTranslation := [][3]float32{{0, 0, 0}, {2, 0, 0}}
	if diff := deepEqual(gotTimes, wantTimes); diff != "" {
		t.Errorf("decoded times = %v, want %v (%s)", gotTimes, wantTimes, diff)
	}
	if diff := deepEqual(gotTranslation, wantTranslation); diff != "" {
		t.Errorf("decoded translation = %v, want %v (%s)", gotTranslation, wantTranslation, diff)
	}
}

// TestSimplifyAnimation_InterpolationError verifies that after simplification
// every dropped keyframe is approximated within tolerance at its exact time.
func TestSimplifyAnimation_InterpolationError(t *testing.T) {
	doc := gltf.NewDocument()
	times := []float32{0, 1, 2, 3, 4}
	values := [][3]float32{
		{0, 0, 0},
		{0.9, 0.1, 0},
		{2.1, 0, -0.2},
		{3, 0, 0},
		{4, 0, 0},
	}
	sampler := linearSampler(doc, times, values, gltf.InterpolationLinear)
	doc.Animations = []*gltf.Animation{{
		Samplers: []*gltf.AnimationSampler{sampler},
		Channels: []*gltf.AnimationChannel{channel(0, gltf.TRSTranslation, 0)},
	}}
	const tolerance = 0.3

	result, err := modeler.SimplifyAnimation(doc, 0, modeler.AnimationSimplifyOptions{
		PositionTolerance: tolerance,
	})
	if err != nil {
		t.Fatalf("SimplifyAnimation() error = %v", err)
	}
	if result.KeyframesRemoved == 0 {
		t.Fatal("SimplifyAnimation() removed no keyframe, want some reduction")
	}

	gotTimes, gotValuesAny := readSamplerData(t, doc, doc.Animations[0].Samplers[0])
	gotValues := gotValuesAny.([][3]float32)
	for i, originalTime := range times {
		// Find the surrounding retained keyframes.
		j := 0
		for j < len(gotTimes)-1 && gotTimes[j+1] < originalTime {
			j++
		}
		if gotTimes[j] == originalTime {
			continue
		}
		alpha := (originalTime - gotTimes[j]) / (gotTimes[j+1] - gotTimes[j])
		for c := 0; c < 3; c++ {
			interp := gotValues[j][c] + alpha*(gotValues[j+1][c]-gotValues[j][c])
			if err := math.Abs(float64(interp - values[i][c])); err > tolerance {
				t.Errorf("component %d at t=%v: interpolation error %v exceeds tolerance %v", c, originalTime, err, tolerance)
			}
		}
	}
}

// deepEqual returns an empty string when a and b are deeply equal, otherwise a
// human readable description of the differences.
func deepEqual(a, b any) string {
	if diff := deep.Equal(a, b); diff != nil {
		return diff[0]
	}
	return ""
}

func normalizeQ(q [4]float32) [4]float32 {
	var normSq float64
	for _, c := range q {
		normSq += float64(c) * float64(c)
	}
	norm := float32(math.Sqrt(normSq))
	for i := range q {
		q[i] /= norm
	}
	return q
}
