package modeler

import (
	"github.com/qmuntal/gltf"
	"github.com/qmuntal/gltf/ext/lightspunctual"
)

// WorldLight is a KHR_lights_punctual light instance resolved into the world
// space of a scene. See lightspunctual.WorldLight for the field semantics.
type WorldLight = lightspunctual.WorldLight

// ReadSceneLights returns the world-space evaluation of every
// KHR_lights_punctual light instantiated by the nodes of scene, in depth-first
// traversal order.
//
// A node instantiates a light when it defines the KHR_lights_punctual node
// extension with a light index; the light itself is read from the root
// KHR_lights_punctual extension of doc. Position is the world-space origin of
// the light node and Direction is the unit vector a directional or spot light
// emits along (the node's local -Z after applying the hierarchy rotations).
// Point lights have a zero Direction. Only rotations affect Direction, so
// non-uniform scale never distorts it.
//
// Missing light properties use the glTF defaults: white color, intensity 1,
// infinite range, inner cone angle 0 and outer cone angle math.Pi/4. A nil
// scene means the active scene declared by Document.Scene. The document is
// never modified.
//
// An error that names the offending node and, when applicable, light index is
// returned for invalid scene, child, node or light indices, hierarchy cycles,
// nodes with several parents, unknown light types, non-finite lights or
// transforms, out-of-range ranges, intensities or cone angles, and malformed
// KHR_lights_punctual extension values.
//
// ReadSceneLights is equivalent to lightspunctual.ReadSceneLights.
func ReadSceneLights(doc *gltf.Document, scene *gltf.Scene) ([]WorldLight, error) {
	return lightspunctual.ReadSceneLights(doc, scene)
}
