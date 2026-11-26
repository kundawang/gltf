package modeler

import (
	"errors"
	"fmt"
	"math"

	"github.com/qmuntal/gltf"
)

// This file implements optional, area-weighted smooth normal generation for
// TRIANGLES primitives. Low-poly glTF assets frequently store only POSITION
// and indices; a preview renderer needs vertex normals to shade them.
//
// GenerateNormals never mutates the document it is given. It works on a
// structurally copied document that shares the entities it does not touch
// (materials, textures, other meshes and primitives, ...) and appends a new
// NORMAL accessor only to the selected primitive.

// Sentinel errors returned by GenerateNormals and ComputeSmoothNormals.
// Callers can test them with errors.Is.
var (
	// ErrNilDocument is returned when a nil document is given.
	ErrNilDocument = errors.New("gltf: document is nil")

	// ErrPrimitiveNotFound is returned when the mesh or primitive selector
	// does not reference an existing, non-nil mesh or primitive.
	ErrPrimitiveNotFound = errors.New("gltf: primitive not found")

	// ErrMissingPosition is returned when the primitive has no POSITION
	// attribute or the attribute does not reference a valid accessor.
	ErrMissingPosition = errors.New("gltf: primitive has no valid POSITION attribute")

	// ErrNotTriangles is returned when normals are requested for a primitive
	// whose mode is not TRIANGLES. Strip and fan topologies need to be
	// expanded first, and points/lines have no surface normal.
	ErrNotTriangles = errors.New("gltf: normals can only be generated for TRIANGLES primitives")

	// ErrNormalsExist is returned when the primitive already has a NORMAL
	// attribute and overwriting was not explicitly requested. Use
	// OverwriteNormals to regenerate it.
	ErrNormalsExist = errors.New("gltf: primitive already has a NORMAL attribute")

	// ErrIndexOutOfRange is returned when an index references a vertex that
	// does not exist in the POSITION accessor.
	ErrIndexOutOfRange = errors.New("gltf: primitive index out of range")

	// ErrInvalidIndexCount is returned when the number of indices (or
	// positions, for a non-indexed primitive) is not a multiple of three.
	ErrInvalidIndexCount = errors.New("gltf: triangle index count is not a multiple of three")

	// ErrNonFinitePosition is returned when a position is NaN or infinite,
	// whether or not a triangle references it.
	ErrNonFinitePosition = errors.New("gltf: primitive position is not finite")

	// ErrNoValidFaces is returned when the primitive contains no
	// non-degenerate triangle, so no normal direction can be derived.
	ErrNoValidFaces = errors.New("gltf: primitive has no non-degenerate triangles")
)

// fallbackNormal is assigned to vertices that are not referenced by any
// non-degenerate triangle (isolated vertices, or vertices used only by
// degenerate faces). glTF requires NORMAL values to be unit length, so a
// deterministic unit vector is used instead of a zero vector.
var fallbackNormal = [3]float32{0, 0, 1}

// NormalOption customizes GenerateNormals.
type NormalOption func(*normalConfig)

type normalConfig struct {
	overwrite bool
}

// OverwriteNormals makes GenerateNormals replace an existing NORMAL
// attribute instead of failing with ErrNormalsExist. The previous NORMAL
// accessor and its buffer views are left in the returned document, as other
// primitives or documents may still reference them.
func OverwriteNormals() NormalOption {
	return func(c *normalConfig) { c.overwrite = true }
}

// GenerateNormals returns a copy of doc in which the TRIANGLES primitive
// selected by meshIndex and primitiveIndex has a newly written NORMAL
// accessor with area-weighted smooth vertex normals. It also returns the
// index of that accessor in the returned document.
//
// POSITION and indices (when present) are read with ReadPosition and
// ReadIndices, so sparse, interleaved, and non-interleaved accessors are all
// supported. Vertices shared through the index buffer receive one accumulated
// normal; non-indexed primitives have no shared vertices and therefore get one
// face normal per triangle corner. Triangles are accumulated without
// normalizing the per-face cross product, which makes each face contribute in
// proportion to its area. The final vertex normals are normalized to unit
// length.
//
// The winding is the glTF/right-handed convention: for a counter-clockwise
// triangle in the XY plane, cross(b-a, c-a) is +Z. Reversing the winding yields
// -Z.
//
// Degenerate triangles (zero-area, collinear, or repeating an index) are
// skipped. Vertices that end up without any contribution get a deterministic
// unit +Z normal, which also covers contributions that cancel out exactly.
// A primitive with no non-degenerate triangle at all returns ErrNoValidFaces.
//
// The position, index, texture coordinate and all other accessors, the
// material, the primitive mode and every other primitive are preserved. doc
// itself is not modified: the returned document is a structural copy that
// only differs by the appended accessor, buffer view, buffer bytes and NORMAL
// attribute. Unchanged entities are shared with doc, so mutating them through
// the returned document would be visible in doc.
//
// An existing NORMAL attribute makes GenerateNormals return ErrNormalsExist
// unless OverwriteNormals is passed. Malformed input is reported instead of
// panicking: out-of-range indices return ErrIndexOutOfRange, non-finite
// positions return ErrNonFinitePosition, and an index or position count that
// is not a multiple of three returns ErrInvalidIndexCount.
func GenerateNormals(doc *gltf.Document, meshIndex, primitiveIndex int, opts ...NormalOption) (*gltf.Document, int, error) {
	var cfg normalConfig
	for _, opt := range opts {
		if opt != nil {
			opt(&cfg)
		}
	}

	if doc == nil {
		return nil, 0, ErrNilDocument
	}
	if meshIndex < 0 || meshIndex >= len(doc.Meshes) || doc.Meshes[meshIndex] == nil {
		return nil, 0, fmt.Errorf("%w: mesh %d", ErrPrimitiveNotFound, meshIndex)
	}
	mesh := doc.Meshes[meshIndex]
	if primitiveIndex < 0 || primitiveIndex >= len(mesh.Primitives) || mesh.Primitives[primitiveIndex] == nil {
		return nil, 0, fmt.Errorf("%w: mesh %d primitive %d", ErrPrimitiveNotFound, meshIndex, primitiveIndex)
	}
	primitive := mesh.Primitives[primitiveIndex]
	if primitive.Mode != gltf.PrimitiveTriangles {
		return nil, 0, fmt.Errorf("%w: got %s", ErrNotTriangles, primitive.Mode)
	}

	positionIndex, ok := primitive.Attributes[gltf.POSITION]
	if !ok || positionIndex < 0 || positionIndex >= len(doc.Accessors) || doc.Accessors[positionIndex] == nil {
		return nil, 0, fmt.Errorf("%w: accessor %d", ErrMissingPosition, positionIndex)
	}
	_, hasNormals := primitive.Attributes[gltf.NORMAL]
	if hasNormals && !cfg.overwrite {
		return nil, 0, ErrNormalsExist
	}

	positions, err := ReadPosition(doc, doc.Accessors[positionIndex], nil)
	if err != nil {
		return nil, 0, err
	}

	var indices []uint32
	if primitive.Indices != nil {
		indexAccessor := *primitive.Indices
		if indexAccessor < 0 || indexAccessor >= len(doc.Accessors) || doc.Accessors[indexAccessor] == nil {
			return nil, 0, fmt.Errorf("gltf: indices accessor %d overflows", indexAccessor)
		}
		indices, err = ReadIndices(doc, doc.Accessors[indexAccessor], nil)
		if err != nil {
			return nil, 0, err
		}
	}

	normals, err := ComputeSmoothNormals(positions, indices)
	if err != nil {
		return nil, 0, err
	}

	out := cloneForNormals(doc, meshIndex, primitiveIndex)
	normalAccessor := WriteNormal(out, normals)
	out.Meshes[meshIndex].Primitives[primitiveIndex].Attributes[gltf.NORMAL] = normalAccessor
	return out, normalAccessor, nil
}

// GenerateMeshNormals is an alias for GenerateNormals.
func GenerateMeshNormals(doc *gltf.Document, meshIndex, primitiveIndex int, opts ...NormalOption) (*gltf.Document, int, error) {
	return GenerateNormals(doc, meshIndex, primitiveIndex, opts...)
}

// ComputeSmoothNormals computes area-weighted smooth vertex normals for the
// given triangles and returns one unit-length normal per position.
//
// positions holds the vertex positions. If indices is nil the positions are
// treated as a flat, non-indexed triangle list; otherwise each group of three
// indices selects a triangle. In both cases the count must be a multiple of
// three, every index must be in range, and every position must be finite,
// including positions that no triangle references.
//
// The contribution of a triangle is its unnormalized cross product
// cross(p1-p0, p2-p0), whose magnitude is twice the triangle area. Contributions
// are summed per vertex and the result is normalized, so larger adjacent
// triangles weigh more. Degenerate triangles (exactly zero cross product, for
// example collinear vertices) contribute nothing.
//
// Vertices not reached by any non-degenerate triangle get a unit +Z normal.
// If no non-degenerate triangle exists, ErrNoValidFaces is returned.
func ComputeSmoothNormals(positions [][3]float32, indices []uint32) ([][3]float32, error) {
	if len(positions) == 0 {
		return nil, ErrNoValidFaces
	}
	// Reject non-finite coordinates up front, including vertices that no
	// triangle references: a NaN or infinite POSITION is malformed data and
	// silently ignoring it would make the result depend on which triangles
	// happen to reference which vertex.
	for i, p := range positions {
		if !isFiniteVec3(p) {
			return nil, fmt.Errorf("%w: vertex %d position %v", ErrNonFinitePosition, i, p)
		}
	}

	vertexCount := len(positions)
	indexed := indices != nil
	triangleCount := 0
	if indexed {
		if len(indices)%3 != 0 {
			return nil, fmt.Errorf("%w: %d indices", ErrInvalidIndexCount, len(indices))
		}
		triangleCount = len(indices) / 3
	} else {
		if len(positions)%3 != 0 {
			return nil, fmt.Errorf("%w: %d positions", ErrInvalidIndexCount, len(positions))
		}
		triangleCount = len(positions) / 3
	}

	// Accumulate in float64 to keep large meshes and unequal areas accurate.
	sums := make([][3]float64, vertexCount)
	validTriangles := 0
	for t := 0; t < triangleCount; t++ {
		var a, b, c int
		if indexed {
			ia, ib, ic := indices[3*t], indices[3*t+1], indices[3*t+2]
			if ia >= uint32(vertexCount) || ib >= uint32(vertexCount) || ic >= uint32(vertexCount) {
				return nil, fmt.Errorf("%w: triangle %d references (%d, %d, %d) of %d vertices",
					ErrIndexOutOfRange, t, ia, ib, ic, vertexCount)
			}
			a, b, c = int(ia), int(ib), int(ic)
		} else {
			a, b, c = 3*t, 3*t+1, 3*t+2
		}

		p0, p1, p2 := positions[a], positions[b], positions[c]

		// Twice the signed area vector. Its length encodes the area, so
		// accumulating it without normalizing gives area weighting for free.
		ux := float64(p1[0]) - float64(p0[0])
		uy := float64(p1[1]) - float64(p0[1])
		uz := float64(p1[2]) - float64(p0[2])
		vx := float64(p2[0]) - float64(p0[0])
		vy := float64(p2[1]) - float64(p0[1])
		vz := float64(p2[2]) - float64(p0[2])
		nx := uy*vz - uz*vy
		ny := uz*vx - ux*vz
		nz := ux*vy - uy*vx

		// Zero cross product: the three vertices are equal or collinear.
		if nx == 0 && ny == 0 && nz == 0 {
			continue
		}
		sums[a][0] += nx
		sums[a][1] += ny
		sums[a][2] += nz
		sums[b][0] += nx
		sums[b][1] += ny
		sums[b][2] += nz
		sums[c][0] += nx
		sums[c][1] += ny
		sums[c][2] += nz
		validTriangles++
	}
	if validTriangles == 0 {
		return nil, ErrNoValidFaces
	}

	normals := make([][3]float32, vertexCount)
	for i, sum := range sums {
		length := math.Sqrt(sum[0]*sum[0] + sum[1]*sum[1] + sum[2]*sum[2])
		if length == 0 || math.IsNaN(length) || math.IsInf(length, 0) {
			normals[i] = fallbackNormal
			continue
		}
		normals[i] = [3]float32{
			float32(sum[0] / length),
			float32(sum[1] / length),
			float32(sum[2] / length),
		}
	}
	return normals, nil
}

func isFiniteVec3(v [3]float32) bool {
	for _, f := range v {
		f64 := float64(f)
		if math.IsNaN(f64) || math.IsInf(f64, 0) {
			return false
		}
	}
	return true
}

// cloneForNormals returns a document that GenerateNormals can append to without
// touching doc. Slices that are appended to are copied, as are the buffer
// structs (and the data of the last buffer, where the new bytes are written)
// and the selected mesh, primitive and attribute map. Everything else is shared
// because GenerateNormals never mutates it.
func cloneForNormals(doc *gltf.Document, meshIndex, primitiveIndex int) *gltf.Document {
	out := *doc

	out.Accessors = append([]*gltf.Accessor(nil), doc.Accessors...)
	out.BufferViews = append([]*gltf.BufferView(nil), doc.BufferViews...)

	out.Buffers = make([]*gltf.Buffer, len(doc.Buffers))
	for i, buffer := range doc.Buffers {
		if buffer == nil {
			continue
		}
		cp := *buffer
		if i == len(doc.Buffers)-1 {
			// WriteAccessor appends to the last buffer; copy its data so
			// appending cannot write into doc's backing array.
			cp.Data = append([]byte(nil), buffer.Data...)
		}
		out.Buffers[i] = &cp
	}
	if len(out.Buffers) > 0 && out.Buffers[len(out.Buffers)-1] == nil {
		out.Buffers[len(out.Buffers)-1] = new(gltf.Buffer)
	}

	out.Meshes = append([]*gltf.Mesh(nil), doc.Meshes...)
	mesh := *doc.Meshes[meshIndex]
	mesh.Primitives = append([]*gltf.Primitive(nil), doc.Meshes[meshIndex].Primitives...)
	primitive := *doc.Meshes[meshIndex].Primitives[primitiveIndex]
	primitive.Attributes = make(gltf.PrimitiveAttributes, len(doc.Meshes[meshIndex].Primitives[primitiveIndex].Attributes)+1)
	for name, accessor := range doc.Meshes[meshIndex].Primitives[primitiveIndex].Attributes {
		primitive.Attributes[name] = accessor
	}
	mesh.Primitives[primitiveIndex] = &primitive
	out.Meshes[meshIndex] = &mesh

	return &out
}
