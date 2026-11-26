package texturetransform

import (
	"bytes"
	"encoding/json"
	"errors"
	"math"

	"github.com/qmuntal/gltf"
)

var (
	// DefaultScale defines a scaling that does not modify the size of the object.
	DefaultScale = [2]float64{1, 1}
	emptyOffset  = [2]float64{0, 0}
)

const (
	// ExtensionName defines the ExtTextureTransform unique key.
	ExtensionName = "KHR_texture_transform"
)

// Unmarshal decodes the json data into the correct type.
func Unmarshal(data []byte) (any, error) {
	t := new(TextureTranform)
	err := json.Unmarshal(data, t)
	return t, err
}

func init() {
	gltf.RegisterExtension(ExtensionName, Unmarshal)
}

// TextureTranform can be used in textureInfo to pack many low-res texture into a single large texture atlas.
type TextureTranform struct {
	Offset   [2]float64 `json:"offset"`
	Rotation float64    `json:"rotation,omitempty"`
	Scale    [2]float64 `json:"scale"`
	TexCoord *int       `json:"texCoord,omitempty"`
}

// TextureTransform is an alias for TextureTranform with the corrected spelling.
type TextureTransform = TextureTranform

// ScaleOrDefault returns the node scale if it represents a valid scale factor, else return the default one.
func (t *TextureTranform) ScaleOrDefault() [2]float64 {
	return t.Scale
}

// ErrNonFinite is returned when a texture transform contains a non-finite
// (NaN or +-Inf) offset, rotation or scale.
var ErrNonFinite = errors.New("texturetransform: offset, rotation and scale must be finite")

// Apply returns the texture coordinate uv transformed with t following the
// KHR_texture_transform specification:
//
//	uv'(u, v) = offset + R(rotation) * (scale * (u, v))
//
// where R(rotation) is the 2x2 counter-clockwise rotation matrix
// [[cos, -sin], [sin, cos]].
//
// Apply does not mutate uv. It returns ErrNonFinite if any of the offset,
// rotation or scale components is not finite.
func (t *TextureTranform) Apply(uv [2]float32) ([2]float32, error) {
	if !isFinite(t.Offset[0]) || !isFinite(t.Offset[1]) ||
		!isFinite(t.Rotation) ||
		!isFinite(t.Scale[0]) || !isFinite(t.Scale[1]) {
		return [2]float32{}, ErrNonFinite
	}
	u := float64(uv[0]) * t.Scale[0]
	v := float64(uv[1]) * t.Scale[1]
	cos := math.Cos(t.Rotation)
	sin := math.Sin(t.Rotation)
	return [2]float32{
		float32(t.Offset[0] + cos*u - sin*v),
		float32(t.Offset[1] + sin*u + cos*v),
	}, nil
}

func isFinite(v float64) bool {
	return !math.IsNaN(v) && !math.IsInf(v, 0)
}

// UnmarshalJSON unmarshal the pbr with the correct default values.
func (t *TextureTranform) UnmarshalJSON(data []byte) error {
	type alias TextureTranform
	tmp := alias(TextureTranform{Scale: DefaultScale})
	err := json.Unmarshal(data, &tmp)
	if err == nil {
		*t = TextureTranform(tmp)
	}
	return err
}

// MarshalJSON marshal the pbr with the correct default values.
func (t *TextureTranform) MarshalJSON() ([]byte, error) {
	type alias TextureTranform
	out, err := json.Marshal(&struct{ *alias }{alias: (*alias)(t)})
	if err == nil {
		if t.Scale == DefaultScale {
			out = removeProperty([]byte(`"scale":[1,1]`), out)
		}
		if t.Offset == emptyOffset {
			out = removeProperty([]byte(`"offset":[0,0]`), out)
		}
		out = sanitizeJSON(out)
	}
	return out, err
}

func removeProperty(str []byte, b []byte) []byte {
	b = bytes.Replace(b, str, []byte(""), 1)
	return bytes.Replace(b, []byte(`,,`), []byte(","), 1)
}

func sanitizeJSON(b []byte) []byte {
	b = bytes.Replace(b, []byte(`{,`), []byte("{"), 1)
	return bytes.Replace(b, []byte(`,}`), []byte("}"), 1)
}
