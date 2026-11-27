package lightspunctual_test

import (
	"fmt"
	"math"

	"github.com/qmuntal/gltf"
	"github.com/qmuntal/gltf/ext/lightspunctual"
)

func ExampleReadSceneLights() {
	// A parent node rotated 90 degrees around the Y axis, and a child node
	// that instantiates a spot light. The spot emits along its local -Z axis,
	// so after the parent rotation it points to world -X.
	doc := &gltf.Document{
		Extensions: gltf.Extensions{
			lightspunctual.ExtensionName: lightspunctual.Lights{{
				Type: lightspunctual.TypeSpot,
			}},
		},
		Nodes: []*gltf.Node{
			{
				Rotation: [4]float64{0, math.Sqrt2 / 2, 0, math.Sqrt2 / 2},
				Children: []int{1},
			},
			{
				Extensions: gltf.Extensions{
					lightspunctual.ExtensionName: lightspunctual.LightIndex(0),
				},
			},
		},
		Scene:  gltf.Index(0),
		Scenes: []*gltf.Scene{{Nodes: []int{0}}},
	}

	lights, err := lightspunctual.ReadSceneLights(doc, nil)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	light := lights[0]
	fmt.Printf("type=%s direction=(%.0f, %.0f, %.0f) intensity=%.0f outer=%.2f\n",
		light.Type, zeroSnap(light.Direction[0]), zeroSnap(light.Direction[1]), zeroSnap(light.Direction[2]),
		light.Intensity, light.OuterConeAngle)

	// Output:
	// type=spot direction=(-1, 0, 0) intensity=1 outer=0.79
}

// zeroSnap removes the sign of a rounded negative zero so the example output
// is stable across platforms.
func zeroSnap(v float64) float64 {
	if math.Abs(v) < 1e-9 {
		return 0
	}
	return v
}
