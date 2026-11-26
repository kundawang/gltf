package gltf_test

import (
	"encoding/base64"
	"strings"
	"testing"

	"github.com/qmuntal/gltf"
)

func b64Mat4(m [16]float64) string {
	return base64.StdEncoding.EncodeToString(mat4Float32Bytes(m))
}

// Decoded TRS nodes get Matrix seeded to identity by UnmarshalJSON;
// evaluation must still compose the decoded translation.
func TestJointMatricesDecodedTRSDocument(t *testing.T) {
	raw := `{
		"asset": {"version": "2.0"},
		"scenes": [{"nodes": [0, 1]}],
		"scene": 0,
		"nodes": [
			{"skin": 0, "translation": [2, 0, 0]},
			{"translation": [5, 0, 0]}
		],
		"skins": [{"joints": [1]}]
	}`
	doc := gltf.NewDocument()
	if err := gltf.NewDecoder(strings.NewReader(raw)).Decode(doc); err != nil {
		t.Fatal(err)
	}
	m, err := doc.JointMatrices(doc.Scenes[0], 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	approxTranslation(t, m[0], 3, 0, 0, 1e-9) // no IBM -> -2 + 5
}

// An embedded data-URI inverseBindMatrices accessor is decoded by the glTF
// decoder and consumed by JointMatrices when no matrices are passed.
// A nil scene resolves to Document.Scene.
func TestJointMatricesEmbeddedInverseBindMatrix(t *testing.T) {
	ibm := translationMatrix(-1, 0, 0)
	raw := `{
		"asset": {"version": "2.0"},
		"scenes": [{"nodes": [0, 1]}],
		"scene": 0,
		"nodes": [
			{"skin": 0, "translation": [2, 0, 0]},
			{"translation": [5, 0, 0]}
		],
		"skins": [{"joints": [1], "inverseBindMatrices": 0}],
		"accessors": [{"bufferView": 0, "componentType": 5126, "count": 1, "type": "MAT4"}],
		"bufferViews": [{"buffer": 0}],
		"buffers": [{"byteLength": 64, "uri": "data:application/octet-stream;base64,` + b64Mat4(ibm) + `"}]
	}`
	doc := gltf.NewDocument()
	if err := gltf.NewDecoder(strings.NewReader(raw)).Decode(doc); err != nil {
		t.Fatal(err)
	}
	m, err := doc.JointMatrices(nil, 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	approxTranslation(t, m[0], 2, 0, 0, 1e-6)
}
