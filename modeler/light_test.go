package modeler_test

import (
	"reflect"
	"testing"

	"github.com/qmuntal/gltf"
	"github.com/qmuntal/gltf/ext/lightspunctual"
	"github.com/qmuntal/gltf/modeler"
)

// TestReadSceneLights_ForwardsToExtension ensures the modeler convenience
// wrapper exposes the same result as the extension implementation.
func TestReadSceneLights_ForwardsToExtension(t *testing.T) {
	doc := &gltf.Document{
		Extensions: gltf.Extensions{
			lightspunctual.ExtensionName: lightspunctual.Lights{{Type: lightspunctual.TypePoint}},
		},
		Nodes: []*gltf.Node{{
			Extensions:  gltf.Extensions{lightspunctual.ExtensionName: lightspunctual.LightIndex(0)},
			Translation: [3]float64{1, 2, 3},
		}},
		Scene:  gltf.Index(0),
		Scenes: []*gltf.Scene{{Nodes: []int{0}}},
	}

	got, err := modeler.ReadSceneLights(doc, nil)
	if err != nil {
		t.Fatalf("modeler.ReadSceneLights() error = %v", err)
	}
	want, err := lightspunctual.ReadSceneLights(doc, nil)
	if err != nil {
		t.Fatalf("lightspunctual.ReadSceneLights() error = %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("modeler.ReadSceneLights() = %+v, want %+v", got, want)
	}
	if len(got) != 1 || got[0].Position != [3]float64{1, 2, 3} {
		t.Errorf("unexpected world light: %+v", got)
	}
}
