package lightspunctual_test

import (
	"encoding/json"
	"math"
	"reflect"
	"strings"
	"testing"

	"github.com/qmuntal/gltf"
	"github.com/qmuntal/gltf/ext/lightspunctual"
)

const lightEpsilon = 1e-9

func quaternionY(angle float64) [4]float64 {
	return [4]float64{0, math.Sin(angle / 2), 0, math.Cos(angle / 2)}
}

func quaternionZ(angle float64) [4]float64 {
	return [4]float64{0, 0, math.Sin(angle / 2), math.Cos(angle / 2)}
}

func assertVec3(t *testing.T, name string, got, want [3]float64) {
	t.Helper()
	for i := range want {
		if math.Abs(got[i]-want[i]) > lightEpsilon {
			t.Errorf("%s = %v, want %v", name, got, want)
			return
		}
	}
}

func lightsDocument(lights lightspunctual.Lights, nodes []*gltf.Node, roots ...int) *gltf.Document {
	return &gltf.Document{
		Extensions: gltf.Extensions{
			lightspunctual.ExtensionName: lights,
		},
		Nodes:  nodes,
		Scene:  gltf.Index(0),
		Scenes: []*gltf.Scene{{Nodes: roots}},
	}
}

func lightNode(lightIndex int) *gltf.Node {
	return &gltf.Node{
		Extensions: gltf.Extensions{
			lightspunctual.ExtensionName: lightspunctual.LightIndex(lightIndex),
		},
	}
}

func TestReadSceneLights_SpotUnderRotatedParent(t *testing.T) {
	// A parent rotated 90 degrees around Y turns the child's local -Z axis
	// into world -X. This is the core case: the light itself only stores
	// node-local data, the world direction must be derived from the hierarchy.
	parent := &gltf.Node{
		Rotation: quaternionY(math.Pi / 2),
		Children: []int{1},
	}
	child := lightNode(0)
	doc := lightsDocument(
		lightspunctual.Lights{{Type: lightspunctual.TypeSpot}},
		[]*gltf.Node{parent, child},
		0,
	)

	lights, err := lightspunctual.ReadSceneLights(doc, nil)
	if err != nil {
		t.Fatalf("ReadSceneLights() error = %v", err)
	}
	if len(lights) != 1 {
		t.Fatalf("got %d lights, want 1", len(lights))
	}
	got := lights[0]
	if got.Node != 1 || got.Light != 0 || got.Type != lightspunctual.TypeSpot {
		t.Errorf("unexpected identity: node=%d light=%d type=%q", got.Node, got.Light, got.Type)
	}
	assertVec3(t, "Direction", got.Direction, [3]float64{-1, 0, 0})
	assertVec3(t, "Position", got.Position, [3]float64{0, 0, 0})
}

func TestReadSceneLights_RotationAxes(t *testing.T) {
	tests := []struct {
		name      string
		rotation  [4]float64
		direction [3]float64
	}{
		{"y90", quaternionY(math.Pi / 2), [3]float64{-1, 0, 0}},
		{"y-90", quaternionY(-math.Pi / 2), [3]float64{1, 0, 0}},
		{"z90", quaternionZ(math.Pi / 2), [3]float64{0, 0, -1}},
		{"x90", [4]float64{math.Sqrt2 / 2, 0, 0, math.Sqrt2 / 2}, [3]float64{0, 1, 0}},
		{"y180", quaternionY(math.Pi), [3]float64{0, 0, 1}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			doc := lightsDocument(
				lightspunctual.Lights{{Type: lightspunctual.TypeDirectional}},
				[]*gltf.Node{{Extensions: lightExt(0), Rotation: tt.rotation}},
				0,
			)
			lights, err := lightspunctual.ReadSceneLights(doc, nil)
			if err != nil {
				t.Fatalf("ReadSceneLights() error = %v", err)
			}
			got := lights[0].Direction
			assertVec3(t, "Direction", [3]float64{
				snapZero(got[0]), snapZero(got[1]), snapZero(got[2]),
			}, tt.direction)
		})
	}
}

func snapZero(v float64) float64 {
	if math.Abs(v) < lightEpsilon {
		return 0
	}
	return v
}

func TestReadSceneLights_NonLightIntermediateNode(t *testing.T) {
	// The rotation lives on an intermediate node that does not reference a
	// light. The child light must still inherit it.
	doc := lightsDocument(
		lightspunctual.Lights{{Type: lightspunctual.TypeSpot}},
		[]*gltf.Node{
			{Children: []int{1}},
			{Rotation: quaternionY(math.Pi / 2), Children: []int{2}},
			{Extensions: lightExt(0), Translation: [3]float64{0, 0, -2}},
		},
		0,
	)

	lights, err := lightspunctual.ReadSceneLights(doc, nil)
	if err != nil {
		t.Fatalf("ReadSceneLights() error = %v", err)
	}
	assertVec3(t, "Direction", [3]float64{lights[0].Direction[0], lights[0].Direction[1], lights[0].Direction[2]}, [3]float64{-1, 0, 0})
	// The child sits at local (0, 0, -2), rotated by the parent to world (-2, 0, 0).
	assertVec3(t, "Position", lights[0].Position, [3]float64{-2, 0, 0})
}

func TestReadSceneLights_TranslatedPointLight(t *testing.T) {
	doc := lightsDocument(
		lightspunctual.Lights{{Type: lightspunctual.TypePoint}},
		[]*gltf.Node{{
			Extensions: gltf.Extensions{
				lightspunctual.ExtensionName: lightspunctual.LightIndex(0),
			},
			Translation: [3]float64{1.5, 2, -3},
		}},
		0,
	)

	lights, err := lightspunctual.ReadSceneLights(doc, doc.Scenes[0])
	if err != nil {
		t.Fatalf("ReadSceneLights() error = %v", err)
	}
	if len(lights) != 1 {
		t.Fatalf("got %d lights, want 1", len(lights))
	}
	assertVec3(t, "Position", lights[0].Position, [3]float64{1.5, 2, -3})
	// Point lights emit in all directions, so there is no meaningful axis.
	assertVec3(t, "Direction", lights[0].Direction, [3]float64{0, 0, 0})
	if lights[0].InnerConeAngle != 0 || lights[0].OuterConeAngle != 0 {
		t.Errorf("point light has cone angles %v/%v, want 0/0",
			lights[0].InnerConeAngle, lights[0].OuterConeAngle)
	}
}

func TestReadSceneLights_Defaults(t *testing.T) {
	doc := lightsDocument(
		lightspunctual.Lights{{Type: lightspunctual.TypeSpot}},
		[]*gltf.Node{lightNode(0)},
		0,
	)

	lights, err := lightspunctual.ReadSceneLights(doc, nil)
	if err != nil {
		t.Fatalf("ReadSceneLights() error = %v", err)
	}
	got := lights[0]
	assertVec3(t, "Color", got.Color, [3]float64{1, 1, 1})
	if got.Intensity != 1 {
		t.Errorf("Intensity = %v, want 1", got.Intensity)
	}
	if !math.IsInf(got.Range, 1) {
		t.Errorf("Range = %v, want +Inf", got.Range)
	}
	if got.InnerConeAngle != 0 {
		t.Errorf("InnerConeAngle = %v, want 0", got.InnerConeAngle)
	}
	if math.Abs(got.OuterConeAngle-math.Pi/4) > lightEpsilon {
		t.Errorf("OuterConeAngle = %v, want Pi/4", got.OuterConeAngle)
	}
	assertVec3(t, "Direction", got.Direction, [3]float64{0, 0, -1})
}

func TestReadSceneLights_ExplicitValues(t *testing.T) {
	light := &lightspunctual.Light{
		Type:      lightspunctual.TypeSpot,
		Name:      "Key",
		Color:     &[3]float64{0.1, 0.2, 0.3},
		Intensity: gltf.Float(7),
		Range:     gltf.Float(12),
		Spot: &lightspunctual.Spot{
			InnerConeAngle: 0.1,
			OuterConeAngle: gltf.Float(0.4),
		},
	}
	doc := lightsDocument(
		lightspunctual.Lights{light},
		[]*gltf.Node{lightNode(0)},
		0,
	)

	lights, err := lightspunctual.ReadSceneLights(doc, nil)
	if err != nil {
		t.Fatalf("ReadSceneLights() error = %v", err)
	}
	got := lights[0]
	if got.Name != "Key" {
		t.Errorf("Name = %q, want Key", got.Name)
	}
	assertVec3(t, "Color", got.Color, [3]float64{0.1, 0.2, 0.3})
	if got.Intensity != 7 {
		t.Errorf("Intensity = %v, want 7", got.Intensity)
	}
	if got.Range != 12 {
		t.Errorf("Range = %v, want 12", got.Range)
	}
	if math.Abs(got.InnerConeAngle-0.1) > lightEpsilon {
		t.Errorf("InnerConeAngle = %v, want 0.1", got.InnerConeAngle)
	}
	if math.Abs(got.OuterConeAngle-0.4) > lightEpsilon {
		t.Errorf("OuterConeAngle = %v, want 0.4", got.OuterConeAngle)
	}
}

func TestReadSceneLights_NonUniformScaleDoesNotSkewDirection(t *testing.T) {
	// The parent is rotated 90 degrees around Y and non-uniformly scaled. A
	// naive transform of (0, 0, -1) by the world matrix would also apply the
	// scale; the light direction must stay a clean world -X.
	parent := &gltf.Node{
		Rotation: quaternionY(math.Pi / 2),
		Scale:    [3]float64{1, 5, 0.25},
		Children: []int{1},
	}
	doc := lightsDocument(
		lightspunctual.Lights{{Type: lightspunctual.TypeSpot}},
		[]*gltf.Node{parent, lightNode(0)},
		0,
	)

	lights, err := lightspunctual.ReadSceneLights(doc, nil)
	if err != nil {
		t.Fatalf("ReadSceneLights() error = %v", err)
	}
	assertVec3(t, "Direction", lights[0].Direction, [3]float64{-1, 0, 0})
	if length := math.Sqrt(
		lights[0].Direction[0]*lights[0].Direction[0] +
			lights[0].Direction[1]*lights[0].Direction[1] +
			lights[0].Direction[2]*lights[0].Direction[2]); math.Abs(length-1) > lightEpsilon {
		t.Errorf("Direction length = %v, want 1", length)
	}
}

func TestReadSceneLights_MatrixTransform(t *testing.T) {
	// A node defined by a matrix has the rotation baked into its columns.
	// Scaling the columns non-uniformly must not affect the direction.
	rotation := [16]float64{0, 0, 1, 0, 0, 1, 0, 0, -1, 0, 0, 0, 3, 4, 5, 1}
	rotation[0] *= 2   // scale X column
	rotation[5] *= 0.5 // scale Y column
	rotation[10] *= 3  // scale Z column
	doc := lightsDocument(
		lightspunctual.Lights{{Type: lightspunctual.TypeDirectional}},
		[]*gltf.Node{{
			Extensions: gltf.Extensions{
				lightspunctual.ExtensionName: lightspunctual.LightIndex(0),
			},
			Matrix: rotation,
		}},
		0,
	)

	lights, err := lightspunctual.ReadSceneLights(doc, nil)
	if err != nil {
		t.Fatalf("ReadSceneLights() error = %v", err)
	}
	assertVec3(t, "Direction", lights[0].Direction, [3]float64{1, 0, 0})
	assertVec3(t, "Position", lights[0].Position, [3]float64{3, 4, 5})
}

func TestReadSceneLights_DepthFirstOrder(t *testing.T) {
	// Root 0 has a child 2 which itself has a child 3. Root 1 is a sibling.
	// Lights must be reported in depth-first order: 0, 2, 3, 1.
	nodes := []*gltf.Node{
		{Children: []int{2}},
		lightNode(0),
		{Children: []int{3}},
		lightNode(0),
	}
	nodes[0].Extensions = gltf.Extensions{lightspunctual.ExtensionName: lightspunctual.LightIndex(0)}
	nodes[2].Extensions = gltf.Extensions{lightspunctual.ExtensionName: lightspunctual.LightIndex(0)}
	doc := lightsDocument(
		lightspunctual.Lights{{Type: lightspunctual.TypeDirectional}},
		nodes,
		0, 1,
	)

	lights, err := lightspunctual.ReadSceneLights(doc, nil)
	if err != nil {
		t.Fatalf("ReadSceneLights() error = %v", err)
	}
	var order []int
	for _, l := range lights {
		order = append(order, l.Node)
	}
	if want := []int{0, 2, 3, 1}; !reflect.DeepEqual(order, want) {
		t.Errorf("light node order = %v, want %v", order, want)
	}
}

func TestReadSceneLights_SceneSelection(t *testing.T) {
	doc := &gltf.Document{
		Extensions: gltf.Extensions{
			lightspunctual.ExtensionName: lightspunctual.Lights{{Type: lightspunctual.TypePoint}},
		},
		Nodes: []*gltf.Node{lightNode(0), lightNode(0)},
		Scene: gltf.Index(0),
		Scenes: []*gltf.Scene{
			{Nodes: []int{0}},
			{Nodes: []int{1}},
		},
	}

	if _, err := lightspunctual.ReadSceneLights(doc, nil); err != nil {
		t.Fatalf("active scene: ReadSceneLights() error = %v", err)
	}
	second := doc.Scenes[1]
	lights, err := lightspunctual.ReadSceneLights(doc, second)
	if err != nil {
		t.Fatalf("explicit scene: ReadSceneLights() error = %v", err)
	}
	if len(lights) != 1 || lights[0].Node != 1 {
		t.Fatalf("explicit scene lights = %+v, want the light on node 1", lights)
	}
}

func TestReadSceneLights_RawDocumentExtension(t *testing.T) {
	// When the extension is not decoded, its raw JSON must still be
	// understood, including the glTF defaults.
	doc := &gltf.Document{
		Extensions: gltf.Extensions{
			lightspunctual.ExtensionName: json.RawMessage(`{"lights":[{"type":"spot","intensity":2}]}`),
		},
		Nodes:  []*gltf.Node{lightNode(0)},
		Scene:  gltf.Index(0),
		Scenes: []*gltf.Scene{{Nodes: []int{0}}},
	}

	lights, err := lightspunctual.ReadSceneLights(doc, nil)
	if err != nil {
		t.Fatalf("ReadSceneLights() error = %v", err)
	}
	got := lights[0]
	if got.Intensity != 2 {
		t.Errorf("Intensity = %v, want 2", got.Intensity)
	}
	assertVec3(t, "Color", got.Color, [3]float64{1, 1, 1})
	if math.Abs(got.OuterConeAngle-math.Pi/4) > lightEpsilon {
		t.Errorf("OuterConeAngle = %v, want Pi/4", got.OuterConeAngle)
	}
}

func TestReadSceneLights_RawNodeExtension(t *testing.T) {
	doc := lightsDocument(
		lightspunctual.Lights{{Type: lightspunctual.TypePoint}},
		[]*gltf.Node{{
			Extensions: gltf.Extensions{
				lightspunctual.ExtensionName: json.RawMessage(`{"light":0}`),
			},
			Translation: [3]float64{4, 0, 0},
		}},
		0,
	)

	lights, err := lightspunctual.ReadSceneLights(doc, nil)
	if err != nil {
		t.Fatalf("ReadSceneLights() error = %v", err)
	}
	assertVec3(t, "Position", lights[0].Position, [3]float64{4, 0, 0})
}

func TestReadSceneLights_DoesNotModifyDocument(t *testing.T) {
	light := &lightspunctual.Light{
		Type:      lightspunctual.TypeSpot,
		Intensity: gltf.Float(3),
		Spot:      &lightspunctual.Spot{OuterConeAngle: gltf.Float(0.5)},
	}
	doc := lightsDocument(
		lightspunctual.Lights{light},
		[]*gltf.Node{{
			Extensions:  gltf.Extensions{lightspunctual.ExtensionName: lightspunctual.LightIndex(0)},
			Translation: [3]float64{1, 2, 3},
			Rotation:    quaternionZ(0.5),
			Scale:       [3]float64{2, 3, 4},
		}},
		0,
	)
	before, err := json.Marshal(doc)
	if err != nil {
		t.Fatal(err)
	}

	if _, err := lightspunctual.ReadSceneLights(doc, nil); err != nil {
		t.Fatalf("ReadSceneLights() error = %v", err)
	}

	after, err := json.Marshal(doc)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(before, after) {
		t.Errorf("document changed after ReadSceneLights:\nbefore %s\nafter  %s", before, after)
	}
}

func TestReadSceneLights_Errors(t *testing.T) {
	spot := &lightspunctual.Light{Type: lightspunctual.TypeSpot}
	point := &lightspunctual.Light{Type: lightspunctual.TypePoint}
	base := func(lights lightspunctual.Lights, nodes []*gltf.Node) *gltf.Document {
		return lightsDocument(lights, nodes, 0)
	}

	tests := []struct {
		name    string
		doc     *gltf.Document
		scene   *gltf.Scene
		wantSub string
	}{
		{
			name: "nil document",
			doc:  nil,
		},
		{
			name:    "bad light index",
			doc:     base(lightspunctual.Lights{spot}, []*gltf.Node{lightNode(4)}),
			wantSub: "light 4",
		},
		{
			name:    "negative light index",
			doc:     base(lightspunctual.Lights{spot}, []*gltf.Node{lightNode(-1)}),
			wantSub: "light -1",
		},
		{
			name: "node references light but document has no lights extension",
			doc: &gltf.Document{
				Nodes:  []*gltf.Node{lightNode(0)},
				Scene:  gltf.Index(0),
				Scenes: []*gltf.Scene{{Nodes: []int{0}}},
			},
			wantSub: "defines 0 lights",
		},
		{
			name: "no active scene",
			doc: &gltf.Document{
				Extensions: gltf.Extensions{lightspunctual.ExtensionName: lightspunctual.Lights{spot}},
				Nodes:      []*gltf.Node{lightNode(0)},
			},
			wantSub: "active scene",
		},
		{
			name: "nil light",
			doc:  base(lightspunctual.Lights{nil}, []*gltf.Node{lightNode(0)}),
		},
		{
			name:    "unknown light type",
			doc:     base(lightspunctual.Lights{{Type: "laser"}}, []*gltf.Node{lightNode(0)}),
			wantSub: "invalid type",
		},
		{
			name: "negative intensity",
			doc: base(lightspunctual.Lights{{
				Type: lightspunctual.TypePoint, Intensity: gltf.Float(-1),
			}}, []*gltf.Node{lightNode(0)}),
			wantSub: "intensity",
		},
		{
			name: "non-finite intensity",
			doc: base(lightspunctual.Lights{{
				Type: lightspunctual.TypePoint, Intensity: gltf.Float(math.Inf(1)),
			}}, []*gltf.Node{lightNode(0)}),
			wantSub: "intensity",
		},
		{
			name: "non-finite color",
			doc: base(lightspunctual.Lights{{
				Type: lightspunctual.TypePoint, Color: &[3]float64{math.NaN(), 0, 0},
			}}, []*gltf.Node{lightNode(0)}),
			wantSub: "color",
		},
		{
			name: "zero range",
			doc: base(lightspunctual.Lights{{
				Type: lightspunctual.TypePoint, Range: gltf.Float(0),
			}}, []*gltf.Node{lightNode(0)}),
			wantSub: "range",
		},
		{
			name: "negative range",
			doc: base(lightspunctual.Lights{{
				Type: lightspunctual.TypePoint, Range: gltf.Float(-2),
			}}, []*gltf.Node{lightNode(0)}),
			wantSub: "range",
		},
		{
			name: "zero outer cone",
			doc: base(lightspunctual.Lights{{
				Type: lightspunctual.TypeSpot, Spot: &lightspunctual.Spot{OuterConeAngle: gltf.Float(0)},
			}}, []*gltf.Node{lightNode(0)}),
			wantSub: "outerConeAngle",
		},
		{
			name: "outer cone beyond half pi",
			doc: base(lightspunctual.Lights{{
				Type: lightspunctual.TypeSpot, Spot: &lightspunctual.Spot{OuterConeAngle: gltf.Float(math.Pi)},
			}}, []*gltf.Node{lightNode(0)}),
			wantSub: "outerConeAngle",
		},
		{
			name: "inner cone beyond outer",
			doc: base(lightspunctual.Lights{{
				Type: lightspunctual.TypeSpot,
				Spot: &lightspunctual.Spot{InnerConeAngle: 1, OuterConeAngle: gltf.Float(0.5)},
			}}, []*gltf.Node{lightNode(0)}),
			wantSub: "innerConeAngle",
		},
		{
			name: "negative inner cone",
			doc: base(lightspunctual.Lights{{
				Type: lightspunctual.TypeSpot, Spot: &lightspunctual.Spot{InnerConeAngle: -0.1},
			}}, []*gltf.Node{lightNode(0)}),
			wantSub: "innerConeAngle",
		},
		{
			name:    "non-finite translation",
			doc:     base(lightspunctual.Lights{point}, []*gltf.Node{{Extensions: lightExt(0), Translation: [3]float64{math.NaN(), 0, 0}}}),
			wantSub: "translation",
		},
		{
			name:    "non-finite rotation",
			doc:     base(lightspunctual.Lights{point}, []*gltf.Node{{Extensions: lightExt(0), Rotation: [4]float64{math.Inf(1), 0, 0, 1}}}),
			wantSub: "rotation",
		},
		{
			name:    "non-finite scale",
			doc:     base(lightspunctual.Lights{point}, []*gltf.Node{{Extensions: lightExt(0), Scale: [3]float64{1, math.Inf(1), 1}}}),
			wantSub: "scale",
		},
		{
			name: "non-finite matrix",
			doc: base(lightspunctual.Lights{point}, []*gltf.Node{{
				Extensions: lightExt(0),
				Matrix:     [16]float64{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, math.Inf(1), 1},
			}}),
			wantSub: "matrix",
		},
		{
			name:    "degenerate matrix basis",
			doc:     base(lightspunctual.Lights{point}, []*gltf.Node{{Extensions: lightExt(0), Matrix: [16]float64{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1}}}),
			wantSub: "degenerate",
		},
		{
			name: "invalid child index",
			doc: base(lightspunctual.Lights{point}, []*gltf.Node{{
				Extensions: lightExt(0), Children: []int{9},
			}}),
			wantSub: "child index 9",
		},
		{
			name: "cycle",
			doc: base(lightspunctual.Lights{point}, []*gltf.Node{
				{Extensions: lightExt(0), Children: []int{1}},
				{Children: []int{0}},
			}),
			wantSub: "cycle",
		},
		{
			name: "multiple parents",
			doc: lightsDocument(lightspunctual.Lights{point}, []*gltf.Node{
				{Children: []int{2}},
				{Children: []int{2}},
				{Extensions: lightExt(0)},
			}, 0, 1),
			wantSub: "more than once",
		},
		{
			name: "invalid root index",
			doc:  base(lightspunctual.Lights{point}, []*gltf.Node{lightNode(0)}),
			// Override the scene to point at an out-of-range root.
			scene:   &gltf.Scene{Nodes: []int{5}},
			wantSub: "root node index 5",
		},
		{
			name:    "wrong node extension shape",
			doc:     base(lightspunctual.Lights{point}, []*gltf.Node{{Extensions: gltf.Extensions{lightspunctual.ExtensionName: "not-an-object"}}}),
			wantSub: "KHR_lights_punctual",
		},
		{
			name:    "node extension without light",
			doc:     base(lightspunctual.Lights{point}, []*gltf.Node{{Extensions: gltf.Extensions{lightspunctual.ExtensionName: map[string]any{"other": 1}}}}),
			wantSub: "light index",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := lightspunctual.ReadSceneLights(tt.doc, tt.scene)
			if tt.doc == nil {
				if err == nil {
					t.Fatal("ReadSceneLights() error = nil, want an error")
				}
				return
			}
			if err == nil {
				t.Fatal("ReadSceneLights() error = nil, want an error")
			}
			if tt.wantSub != "" && !strings.Contains(err.Error(), tt.wantSub) {
				t.Errorf("ReadSceneLights() error = %q, want it to contain %q", err, tt.wantSub)
			}
		})
	}
}

func lightExt(index int) gltf.Extensions {
	return gltf.Extensions{lightspunctual.ExtensionName: lightspunctual.LightIndex(index)}
}
