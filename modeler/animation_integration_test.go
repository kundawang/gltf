package modeler_test

import (
	"math"
	"testing"

	"github.com/qmuntal/gltf"
	"github.com/qmuntal/gltf/modeler"
)

// TestReadAnimationAnimatedCube samples the official glTF-Sample-Models
// AnimatedCube: a single LINEAR quaternion animation with keyframes at
// t = 0, 1, 2 s rotating 180 degrees around Y and back.
func TestReadAnimationAnimatedCube(t *testing.T) {
	doc, err := gltf.Open("../testdata/AnimatedCube/glTF/AnimatedCube.gltf")
	if err != nil {
		t.Fatal(err)
	}
	if len(doc.Animations) != 1 {
		t.Fatalf("expected one animation, got %d", len(doc.Animations))
	}

	const tol = 1e-5
	cases := []struct {
		name string
		time float64
		want [4]float64
	}{
		{"start keyframe", 0, [4]float64{0, 0, 0, 1}},
		// Shortest arc: instead of rotating 270 degrees the interpolation
		// goes the short way around (90 degrees about -Y).
		{"midpoint first segment", 0.5, [4]float64{0, -0.70710678, 0, 0.70710678}},
		{"middle keyframe", 1, [4]float64{0, 1, 0, 0}},
		{"midpoint second segment", 1.5, [4]float64{0, 0.70710678, 0, 0.70710678}},
		{"end keyframe", 2, [4]float64{0, 0, 0, -1}},
		{"before range is clamped", -1.5, [4]float64{0, 0, 0, 1}},
		{"after range is clamped", 3.5, [4]float64{0, 0, 0, -1}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			poses, err := modeler.ReadAnimation(doc, 0, tc.time)
			if err != nil {
				t.Fatal(err)
			}
			pose, ok := poses[0]
			if !ok || len(poses) != 1 {
				t.Fatalf("expected one pose for node 0, got %v", poses)
			}
			for i := 0; i < 4; i++ {
				if math.Abs(pose.Rotation[i]-tc.want[i]) > tol {
					t.Errorf("rotation = %v, want %v", pose.Rotation, tc.want)
					break
				}
			}
			length := math.Sqrt(
				pose.Rotation[0]*pose.Rotation[0] + pose.Rotation[1]*pose.Rotation[1] +
					pose.Rotation[2]*pose.Rotation[2] + pose.Rotation[3]*pose.Rotation[3])
			if math.Abs(length-1) > tol {
				t.Errorf("rotation %v is not unit length", pose.Rotation)
			}
			// Translation and scale are not animated: glTF defaults apply.
			if pose.Translation != [3]float64{0, 0, 0} {
				t.Errorf("translation = %v, want the glTF default", pose.Translation)
			}
			if pose.Scale != [3]float64{1, 1, 1} {
				t.Errorf("scale = %v, want the glTF default", pose.Scale)
			}
		})
	}

	// Out of range animation index and non-finite time are errors even
	// on a well-formed official model.
	if _, err := modeler.ReadAnimation(doc, 1, 0); err == nil {
		t.Error("expected an error for an out of range animation index")
	}
	if _, err := modeler.ReadAnimation(doc, 0, math.NaN()); err == nil {
		t.Error("expected an error for a NaN time")
	}

	// Sampling leaves the document and the node's own rotation untouched.
	nodeRotation := doc.Nodes[0].Rotation
	if _, err := modeler.ReadAnimation(doc, 0, 0.7); err != nil {
		t.Fatal(err)
	}
	if doc.Nodes[0].Rotation != nodeRotation {
		t.Errorf("node rotation was modified: %v", doc.Nodes[0].Rotation)
	}
}
