package modeler

import (
	"errors"
	"fmt"
	"math"

	"github.com/qmuntal/gltf"
)

// AnimationSimplifyOptions controls how the keyframes of an animation are reduced.
//
// Only LINEAR samplers targeting the translation, scale or rotation property of
// a node can be simplified: STEP and CUBICSPLINE samplers and samplers targeting
// the weights property are left untouched.
//
// PositionTolerance and ScaleTolerance are Euclidean distances, expressed in the
// same units as the translation and scale components. RotationTolerance is an
// angle expressed in radians: it bounds the angle between the original and the
// interpolated quaternion. A zero tolerance only removes keyframes whose value
// is exactly the value interpolated between the adjacent retained keyframes.
//
// All tolerances must be finite and non-negative, otherwise SimplifyAnimation
// returns an error without modifying the document.
type AnimationSimplifyOptions struct {
	PositionTolerance float64
	ScaleTolerance    float64
	RotationTolerance float64
}

// AnimationSimplifyResult summarizes the work done by SimplifyAnimation.
//
// SamplersSimplified reports the number of samplers whose input and output
// accessors were rewritten and KeyframesRemoved reports the total number of
// keyframes that were dropped from those samplers.
//
// SkippedSamplers reports the indices of the samplers that were not eligible
// for simplification (STEP/CUBICSPLINE interpolation, weights target, shared
// sampler with mixed target paths, ...).
type AnimationSimplifyResult struct {
	SamplersSimplified int
	KeyframesRemoved   int
	SkippedSamplers    []int
}

// SimplifyAnimation reduces the number of keyframes of the animation stored at
// animationIndex while keeping the interpolated animation within the tolerances
// defined by opts.
//
// The first and the last keyframe of every sampler are always retained, so the
// duration of the animation and its boundary poses do not change. A keyframe is
// dropped only when, at its exact time, the pose obtained by linearly
// interpolating between the adjacent retained keyframes does not exceed the
// tolerance associated to the targeted property: Euclidean distance for
// translation and scale, angle in radians for quaternion rotations (the
// quaternion interpolation follows the glTF spec).
//
// Each simplified sampler gets two newly written accessors (input and output);
// the document stays self-contained and can be encoded and decoded again.
// Samplers shared by several channels are processed only once, and only when
// every channel that references them targets the same property.
//
// The document is not modified when an error is returned. SimplifyAnimation
// fails on negative or non-finite tolerances, on out of range animation or
// accessor indices, on unsupported accessor types or component types, on
// non-finite values, on zero-length rotation quaternions and on input times
// that are not strictly increasing (duplicated or reversed times).
func SimplifyAnimation(doc *gltf.Document, animationIndex int, opts AnimationSimplifyOptions) (AnimationSimplifyResult, error) {
	var result AnimationSimplifyResult
	if doc == nil {
		return result, errors.New("gltf: document is nil")
	}
	if animationIndex < 0 || len(doc.Animations) <= animationIndex {
		return result, fmt.Errorf("gltf: animation index %d overflows", animationIndex)
	}
	if opts.PositionTolerance < 0 || opts.ScaleTolerance < 0 || opts.RotationTolerance < 0 {
		return result, errors.New("gltf: animation simplification tolerances must be non-negative")
	}
	if !isFinite(opts.PositionTolerance) || !isFinite(opts.ScaleTolerance) || !isFinite(opts.RotationTolerance) {
		return result, errors.New("gltf: animation simplification tolerances must be finite")
	}

	anim := doc.Animations[animationIndex]
	if anim == nil {
		return result, errors.New("gltf: animation is nil")
	}

	// Collect the target path of every channel that references each sampler.
	pathBySampler := make(map[int][]gltf.TRSProperty)
	for _, ch := range anim.Channels {
		if ch == nil {
			continue
		}
		pathBySampler[ch.Sampler] = append(pathBySampler[ch.Sampler], ch.Target.Path)
	}

	type simplifyJob struct {
		samplerIndex int
		sampler      *gltf.AnimationSampler
		path         gltf.TRSProperty
		tolerance    float64
	}
	jobs := make([]simplifyJob, 0, len(anim.Samplers))
	for samplerIndex, sampler := range anim.Samplers {
		paths := pathBySampler[samplerIndex]
		if len(paths) == 0 {
			result.SkippedSamplers = append(result.SkippedSamplers, samplerIndex)
			continue
		}
		path := paths[0]
		samePath := true
		for _, p := range paths[1:] {
			if p != path {
				samePath = false
				break
			}
		}
		if !samePath || path == gltf.TRSWeights {
			result.SkippedSamplers = append(result.SkippedSamplers, samplerIndex)
			continue
		}
		if sampler == nil || sampler.Interpolation != gltf.InterpolationLinear {
			result.SkippedSamplers = append(result.SkippedSamplers, samplerIndex)
			continue
		}

		var tolerance float64
		switch path {
		case gltf.TRSTranslation:
			tolerance = opts.PositionTolerance
		case gltf.TRSScale:
			tolerance = opts.ScaleTolerance
		case gltf.TRSRotation:
			tolerance = opts.RotationTolerance
		}
		jobs = append(jobs, simplifyJob{samplerIndex: samplerIndex, sampler: sampler, path: path, tolerance: tolerance})
	}

	// Validate every candidate before mutating the document so that a failure
	// leaves the document untouched.
	validated := make([]struct {
		job    simplifyJob
		times  []float32
		values any
	}, len(jobs))
	for i, job := range jobs {
		times, values, err := loadAndValidateSampler(doc, job.sampler, job.path)
		if err != nil {
			return AnimationSimplifyResult{}, fmt.Errorf("gltf: animation sampler %d: %w", job.samplerIndex, err)
		}
		validated[i].job = job
		validated[i].times = times
		validated[i].values = values
	}

	// Rewrite the accessors of the validated samplers.
	for _, v := range validated {
		var keep []int
		switch v.job.path {
		case gltf.TRSTranslation, gltf.TRSScale:
			values := v.values.([][3]float32)
			keep = reduceKeyframes(v.times, values, v.job.tolerance, vec3Error)
			newTimes := make([]float32, len(keep))
			newValues := make([][3]float32, len(keep))
			for j, k := range keep {
				newTimes[j] = v.times[k]
				newValues[j] = values[k]
			}
			inputIndex := WriteAccessor(doc, gltf.TargetNone, newTimes)
			outputIndex := WriteAccessor(doc, gltf.TargetNone, newValues)
			v.job.sampler.Input = inputIndex
			v.job.sampler.Output = outputIndex
		case gltf.TRSRotation:
			values := v.values.([][4]float32)
			keep = reduceKeyframes(v.times, values, v.job.tolerance, quaternionError)
			newTimes := make([]float32, len(keep))
			newValues := make([][4]float32, len(keep))
			for j, k := range keep {
				newTimes[j] = v.times[k]
				newValues[j] = values[k]
			}
			inputIndex := WriteAccessor(doc, gltf.TargetNone, newTimes)
			outputIndex := WriteAccessor(doc, gltf.TargetNone, newValues)
			v.job.sampler.Input = inputIndex
			v.job.sampler.Output = outputIndex
		}
		result.SamplersSimplified++
		result.KeyframesRemoved += len(v.times) - len(keep)
	}
	return result, nil
}

// loadAndValidateSampler reads the input/output accessors of sampler and checks
// that they describe a valid LINEAR keyframe graph for path.
func loadAndValidateSampler(doc *gltf.Document, sampler *gltf.AnimationSampler, path gltf.TRSProperty) ([]float32, any, error) {
	if sampler.Input < 0 || len(doc.Accessors) <= sampler.Input {
		return nil, nil, fmt.Errorf("input accessor index %d overflows", sampler.Input)
	}
	if sampler.Output < 0 || len(doc.Accessors) <= sampler.Output {
		return nil, nil, fmt.Errorf("output accessor index %d overflows", sampler.Output)
	}
	input := doc.Accessors[sampler.Input]
	output := doc.Accessors[sampler.Output]
	if input.ComponentType != gltf.ComponentFloat || input.Type != gltf.AccessorScalar {
		return nil, nil, fmt.Errorf("input accessor must be a FLOAT scalar, got component type %v and type %v", input.ComponentType, input.Type)
	}

	inData, err := ReadAccessor(doc, input, nil)
	if err != nil {
		return nil, nil, err
	}
	times, ok := inData.([]float32)
	if !ok {
		return nil, nil, fmt.Errorf("input accessor resolved to unexpected type %T", inData)
	}
	outData, err := ReadAccessor(doc, output, nil)
	if err != nil {
		return nil, nil, err
	}
	if output.Count != input.Count {
		return nil, nil, fmt.Errorf("input accessor has %d keyframes but output accessor has %d", input.Count, output.Count)
	}

	for i, t := range times {
		if !isFinite(float64(t)) {
			return nil, nil, errors.New("input accessor contains a non-finite time")
		}
		if i > 0 && t <= times[i-1] {
			return nil, nil, errors.New("input times must be strictly increasing: duplicated or reversed times are not supported")
		}
	}

	switch path {
	case gltf.TRSTranslation, gltf.TRSScale:
		if output.ComponentType != gltf.ComponentFloat || output.Type != gltf.AccessorVec3 {
			return nil, nil, fmt.Errorf("%s output accessor must be a FLOAT VEC3, got component type %v and type %v", path, output.ComponentType, output.Type)
		}
		values, ok := outData.([][3]float32)
		if !ok {
			return nil, nil, fmt.Errorf("output accessor resolved to unexpected type %T", outData)
		}
		for _, v := range values {
			for _, c := range v {
				if !isFinite(float64(c)) {
					return nil, nil, errors.New("output accessor contains a non-finite value")
				}
			}
		}
		return times, values, nil
	case gltf.TRSRotation:
		if output.ComponentType != gltf.ComponentFloat || output.Type != gltf.AccessorVec4 {
			return nil, nil, fmt.Errorf("rotation output accessor must be a FLOAT VEC4, got component type %v and type %v", output.ComponentType, output.Type)
		}
		values, ok := outData.([][4]float32)
		if !ok {
			return nil, nil, fmt.Errorf("output accessor resolved to unexpected type %T", outData)
		}
		for _, q := range values {
			normSq := float64(0)
			for _, c := range q {
				if !isFinite(float64(c)) {
					return nil, nil, errors.New("output accessor contains a non-finite value")
				}
				normSq += float64(c) * float64(c)
			}
			if normSq == 0 {
				return nil, nil, errors.New("rotation output accessor contains a zero-length quaternion")
			}
		}
		return times, values, nil
	default:
		return nil, nil, fmt.Errorf("target path %v cannot be simplified", path)
	}
}

// reduceKeyframes returns the indices of the keyframes that must be retained so
// that every dropped keyframe, interpolated at its exact time between the
// adjacent retained keyframes, is within tolerance according to errorAt.
//
// The first and last keyframes are always retained. This is an iterative
// implementation of the Ramer-Douglas-Peucker algorithm generalized to any
// interpolation error metric.
func reduceKeyframes[T any](times []float32, values []T, tolerance float64, errorAt func(T, T, float32, T) float64) []int {
	n := len(times)
	if n <= 2 {
		keep := make([]int, n)
		for i := range keep {
			keep[i] = i
		}
		return keep
	}

	retained := make([]bool, n)
	retained[0] = true
	retained[n-1] = true
	type segment struct{ first, last int }
	stack := []segment{{0, n - 1}}
	for len(stack) > 0 {
		seg := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		first, last := seg.first, seg.last
		dt := times[last] - times[first]
		worstIndex := -1
		worstError := tolerance
		for i := first + 1; i < last; i++ {
			alpha := (times[i] - times[first]) / dt
			if err := errorAt(values[first], values[last], alpha, values[i]); err > worstError {
				worstError = err
				worstIndex = i
			}
		}
		if worstIndex >= 0 {
			retained[worstIndex] = true
			stack = append(stack, segment{first, worstIndex}, segment{worstIndex, last})
		}
	}

	keep := make([]int, 0, n)
	for i, ok := range retained {
		if ok {
			keep = append(keep, i)
		}
	}
	return keep
}

// vec3Error returns the Euclidean distance between actual and the linear
// interpolation of a and b at parameter alpha.
func vec3Error(a, b [3]float32, alpha float32, actual [3]float32) float64 {
	var sum float64
	for i := 0; i < 3; i++ {
		interp := float64(a[i]) + float64(alpha)*(float64(b[i])-float64(a[i]))
		d := interp - float64(actual[i])
		sum += d * d
	}
	return math.Sqrt(sum)
}

// quaternionError returns the angle, in radians, between actual and the
// quaternion obtained by interpolating a and b at parameter alpha following
// the glTF LINEAR rotation interpolation (slerp).
func quaternionError(a, b [4]float32, alpha float32, actual [4]float32) float64 {
	interp := slerp(a, b, alpha)
	ni := normalize4(interp)
	na := normalize4(actual)
	dot := float64(0)
	for i := 0; i < 4; i++ {
		dot += float64(ni[i]) * float64(na[i])
	}
	// Quaternions describe a double cover of the rotation group.
	if dot < 0 {
		dot = -dot
	}
	if dot > 1 {
		dot = 1
	}
	return 2 * math.Acos(dot)
}

// slerp performs spherical linear interpolation between quaternions a and b
// with parameter alpha, as defined by the glTF specification.
func slerp(a, b [4]float32, alpha float32) [4]float32 {
	na := normalize4(a)
	nb := normalize4(b)
	dot := float64(0)
	for i := 0; i < 4; i++ {
		dot += float64(na[i]) * float64(nb[i])
	}
	bPrime := nb
	if dot < 0 {
		dot = -dot
		for i := range bPrime {
			bPrime[i] = -bPrime[i]
		}
	}
	omega := math.Acos(math.Min(dot, 1))
	if omega == 0 {
		return na
	}
	sinOmega := math.Sin(omega)
	s1 := math.Sin((1-float64(alpha))*omega) / sinOmega
	s2 := math.Sin(float64(alpha)*omega) / sinOmega
	var out [4]float32
	for i := 0; i < 4; i++ {
		out[i] = float32(s1*float64(na[i]) + s2*float64(bPrime[i]))
	}
	return out
}

func normalize4(q [4]float32) [4]float32 {
	var normSq float64
	for _, c := range q {
		normSq += float64(c) * float64(c)
	}
	norm := math.Sqrt(normSq)
	var out [4]float32
	if norm == 0 {
		return out
	}
	for i, c := range q {
		out[i] = float32(float64(c) / norm)
	}
	return out
}

func isFinite(v float64) bool {
	return !math.IsNaN(v) && !math.IsInf(v, 0)
}
