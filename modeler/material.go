package modeler

import (
	"errors"
	"fmt"
	"math"

	"github.com/qmuntal/gltf"
	"github.com/qmuntal/gltf/ext/texturetransform"
)

// MaterialTextureSlot identifies one of the texture slots that a glTF material
// can expose.
type MaterialTextureSlot uint8

const (
	// MaterialBaseColorTexture is the base color texture slot of
	// pbrMetallicRoughness.
	MaterialBaseColorTexture MaterialTextureSlot = iota
	// MaterialMetallicRoughnessTexture is the metallic-roughness texture slot
	// of pbrMetallicRoughness.
	MaterialMetallicRoughnessTexture
	// MaterialNormalTexture is the normal texture slot.
	MaterialNormalTexture
	// MaterialOcclusionTexture is the occlusion texture slot.
	MaterialOcclusionTexture
	// MaterialEmissiveTexture is the emissive texture slot.
	MaterialEmissiveTexture
)

var (
	errTextureSlotMissing = errors.New("gltf: material does not define the requested texture slot")
	errNoUVSet            = errors.New("gltf: primitive does not define the requested TEXCOORD set")
	errNonFiniteUV        = errors.New("gltf: texture coordinate accessor contains a non-finite component")
)

// textureInfo is the subset of a texture slot that is shared by every
// material texture type.
type textureInfo struct {
	texCoord   int
	extensions gltf.Extensions
}

func textureSlotInfo(mat *gltf.Material, slot MaterialTextureSlot) (textureInfo, error) {
	switch slot {
	case MaterialBaseColorTexture:
		if mat.PBRMetallicRoughness == nil || mat.PBRMetallicRoughness.BaseColorTexture == nil {
			return textureInfo{}, errTextureSlotMissing
		}
		ti := mat.PBRMetallicRoughness.BaseColorTexture
		return textureInfo{texCoord: ti.TexCoord, extensions: ti.Extensions}, nil
	case MaterialMetallicRoughnessTexture:
		if mat.PBRMetallicRoughness == nil || mat.PBRMetallicRoughness.MetallicRoughnessTexture == nil {
			return textureInfo{}, errTextureSlotMissing
		}
		ti := mat.PBRMetallicRoughness.MetallicRoughnessTexture
		return textureInfo{texCoord: ti.TexCoord, extensions: ti.Extensions}, nil
	case MaterialNormalTexture:
		if mat.NormalTexture == nil {
			return textureInfo{}, errTextureSlotMissing
		}
		return textureInfo{texCoord: mat.NormalTexture.TexCoord, extensions: mat.NormalTexture.Extensions}, nil
	case MaterialOcclusionTexture:
		if mat.OcclusionTexture == nil {
			return textureInfo{}, errTextureSlotMissing
		}
		return textureInfo{texCoord: mat.OcclusionTexture.TexCoord, extensions: mat.OcclusionTexture.Extensions}, nil
	case MaterialEmissiveTexture:
		if mat.EmissiveTexture == nil {
			return textureInfo{}, errTextureSlotMissing
		}
		return textureInfo{texCoord: mat.EmissiveTexture.TexCoord, extensions: mat.EmissiveTexture.Extensions}, nil
	default:
		return textureInfo{}, fmt.Errorf("gltf: unknown material texture slot %d", slot)
	}
}

// ReadMaterialTextureCoord returns the per-vertex texture coordinates that the
// given texture slot of primitive's material would sample.
//
// The TEXCOORD set is selected with the texCoord property of the texture
// slot, unless a KHR_texture_transform extension overrides it with its own
// texCoord property. When the slot uses KHR_texture_transform, the returned
// coordinates are computed from the source UVs applying the extension's
// offset, rotation and scale:
//
//	uv' = offset + R(rotation) * (scale * uv)
//
// Without the extension the source coordinates are returned unchanged.
//
// Sparse, interleaved and buffer-less accessors are supported through
// ReadTextureCoord, as are normalized integer component types and the
// different TEXCOORD sets. The document and its shared vertex data are never
// modified: the result is always allocated in buffer (or in a new slice when
// buffer is too small).
//
// An error is returned, among other malformed-document cases, when the
// material or the requested texture slot is missing, when the primitive does
// not define the selected TEXCOORD set, when the UV accessor has an invalid
// component type or accessor type, or when the transform parameters or the
// source coordinates are not finite.
func ReadMaterialTextureCoord(doc *gltf.Document, primitive *gltf.Primitive, material *gltf.Material, slot MaterialTextureSlot, buffer [][2]float32) ([][2]float32, error) {
	if primitive == nil {
		return nil, errors.New("gltf: primitive is nil")
	}
	if material == nil {
		return nil, errors.New("gltf: material is nil")
	}
	info, err := textureSlotInfo(material, slot)
	if err != nil {
		return nil, err
	}

	transform, hasTransform, err := parseTextureTransform(info.extensions)
	if err != nil {
		return nil, err
	}
	if hasTransform && transform.TexCoord != nil {
		info.texCoord = *transform.TexCoord
	}

	attribute := fmt.Sprintf("TEXCOORD_%d", info.texCoord)
	accessorIndex, ok := primitive.Attributes[attribute]
	if !ok {
		return nil, fmt.Errorf("%w: %s", errNoUVSet, attribute)
	}
	if len(doc.Accessors) <= accessorIndex {
		return nil, fmt.Errorf("gltf: accessor index overflows: %d", accessorIndex)
	}

	uvs, err := ReadTextureCoord(doc, doc.Accessors[accessorIndex], buffer)
	if err != nil {
		return nil, err
	}
	for _, uv := range uvs {
		if math.IsNaN(float64(uv[0])) || math.IsInf(float64(uv[0]), 0) ||
			math.IsNaN(float64(uv[1])) || math.IsInf(float64(uv[1]), 0) {
			return nil, errNonFiniteUV
		}
	}

	if !hasTransform {
		return uvs, nil
	}
	for i, uv := range uvs {
		uvs[i], err = transform.Apply(uv)
		if err != nil {
			return nil, err
		}
	}
	return uvs, nil
}

// parseTextureTransform extracts a registered KHR_texture_transform extension
// from the texture slot extensions. hasTransform is false when the slot does
// not use the extension.
func parseTextureTransform(extensions gltf.Extensions) (transform *texturetransform.TextureTranform, hasTransform bool, err error) {
	if extensions == nil {
		return nil, false, nil
	}
	ext, ok := extensions[texturetransform.ExtensionName]
	if !ok || ext == nil {
		return nil, false, nil
	}
	transform, ok = ext.(*texturetransform.TextureTranform)
	if !ok {
		return nil, true, fmt.Errorf("gltf: %s extension has an unexpected payload type %T", texturetransform.ExtensionName, ext)
	}
	return transform, true, nil
}
