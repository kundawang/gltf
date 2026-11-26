package modeler

import (
	"fmt"
	"reflect"

	"github.com/qmuntal/gltf"
)

// ConvertPrimitiveTopology converts the selected TRIANGLE_STRIP/TRIANGLE_FAN
// primitive into a TRIANGLES primitive with a freshly written, valid indices
// accessor.
//
// It returns a deep, independently savable copy of doc: the source document is
// never modified and only the selected primitive changes in the copy.
// Materials and every vertex attribute are preserved, and primitives sharing
// the original indices accessor stay valid and unchanged (copy-on-write).
//
// The target can be specified in several equivalent ways:
//
//	ConvertPrimitiveTopology(doc, meshIndex, primitiveIndex)
//	ConvertPrimitiveTopology(doc, primitive)
//	ConvertPrimitiveTopology(doc, mesh, primitiveIndex)
//	ConvertPrimitiveTopology(doc, "meshName", primitiveIndex)
//	ConvertPrimitiveTopology(doc, modeler.PrimitiveRef{Mesh: m, Primitive: p})
//
// With no target argument every strip/fan primitive of every mesh is
// converted. TopologyOption values may be passed anywhere after doc.
//
// Sequences with fewer than three indices/vertices return ErrTooFewVertices
// unless AllowEmptyTopology is provided. Out-of-range indices return
// ErrIndexOutOfRange, unsupported/illegal modes return ErrInvalidTopology and
// expansions that would be too large return ErrOutputTooLarge.
func ConvertPrimitiveTopology(doc *gltf.Document, target ...any) (*gltf.Document, error) {
	cfg, refs, all, err := resolveTargets(doc, target)
	if err != nil {
		return nil, err
	}
	out := CloneDocument(doc)
	if all {
		if err := convertAllInPlace(out, cfg); err != nil {
			return nil, err
		}
		return out, nil
	}
	for _, ref := range refs {
		if _, err := convertInPlace(out, ref, cfg); err != nil {
			return nil, err
		}
	}
	return out, nil
}

// ConvertPrimitiveAt is the strongly typed form of ConvertPrimitiveTopology
// addressed by mesh and primitive indices.
func ConvertPrimitiveAt(doc *gltf.Document, meshIndex, primitiveIndex int, opts ...TopologyOption) (*gltf.Document, error) {
	cfg := optionsConfig(opts)
	if doc == nil {
		return nil, ErrInvalidDocument
	}
	out := CloneDocument(doc)
	if _, err := convertInPlace(out, PrimitiveRef{Mesh: meshIndex, Primitive: primitiveIndex}, cfg); err != nil {
		return nil, err
	}
	return out, nil
}

// ConvertAllPrimitiveTopologies converts every strip/fan primitive in doc and
// returns an independent document.
func ConvertAllPrimitiveTopologies(doc *gltf.Document, opts ...TopologyOption) (*gltf.Document, error) {
	if doc == nil {
		return nil, ErrInvalidDocument
	}
	cfg := optionsConfig(opts)
	out := CloneDocument(doc)
	if err := convertAllInPlace(out, cfg); err != nil {
		return nil, err
	}
	return out, nil
}

// ApplyPrimitiveTopology converts the selected primitive in place. Unlike
// ConvertPrimitiveTopology it mutates doc and returns only an error; the
// targeted primitive receives a new indices accessor so accessor sharing with
// other primitives remains safe.
func ApplyPrimitiveTopology(doc *gltf.Document, target ...any) error {
	cfg, refs, all, err := resolveTargets(doc, target)
	if err != nil {
		return err
	}
	if all {
		return convertAllInPlace(doc, cfg)
	}
	for _, ref := range refs {
		if _, err := convertInPlace(doc, ref, cfg); err != nil {
			return err
		}
	}
	return nil
}

// ReadTriangleIndices resolves the selected primitive and returns the expanded
// TRIANGLES index list without modifying doc or writing any accessor. It is
// useful for asserting the exact winding of a conversion.
func ReadTriangleIndices(doc *gltf.Document, target ...any) ([]uint32, error) {
	cfg, refs, all, err := resolveTargets(doc, target)
	if err != nil {
		return nil, err
	}
	if all {
		// Reading indices is meaningful for one primitive; when no selector is
		// given use the first convertible primitive.
		ref, ok := firstConvertible(doc)
		if !ok {
			return []uint32{}, nil
		}
		refs = []PrimitiveRef{ref}
	}
	if len(refs) == 0 {
		return nil, ErrPrimitiveNotFound
	}
	tris, _, _, err := computeTriangles(doc, refs[0], cfg)
	return tris, err
}

// TopologyOf converts the selected primitive and returns detailed statistics
// together with the independent converted document.
func TopologyOf(doc *gltf.Document, target ...any) (*gltf.Document, *TopologyStats, error) {
	cfg, refs, _, err := resolveTargets(doc, target)
	if err != nil {
		return nil, nil, err
	}
	if len(refs) != 1 {
		return nil, nil, fmt.Errorf("%w: TopologyOf requires exactly one primitive", ErrPrimitiveNotFound)
	}
	out := CloneDocument(doc)
	stats, err := convertInPlace(out, refs[0], cfg)
	if err != nil {
		return nil, nil, err
	}
	return out, stats, nil
}

// --- Name aliases -----------------------------------------------------------
//
// Different callers and older drafts refer to this operation with different
// names; all aliases share the exact same behavior and call shapes.

// ConvertPrimitiveMode is an alias for ConvertPrimitiveTopology.
func ConvertPrimitiveMode(doc *gltf.Document, target ...any) (*gltf.Document, error) {
	return ConvertPrimitiveTopology(doc, target...)
}

// ConvertTopology is an alias for ConvertPrimitiveTopology.
func ConvertTopology(doc *gltf.Document, target ...any) (*gltf.Document, error) {
	return ConvertPrimitiveTopology(doc, target...)
}

// ConvertToTriangles is an alias for ConvertPrimitiveTopology.
func ConvertToTriangles(doc *gltf.Document, target ...any) (*gltf.Document, error) {
	return ConvertPrimitiveTopology(doc, target...)
}

// ConvertPrimitiveToTriangles is an alias for ConvertPrimitiveTopology.
func ConvertPrimitiveToTriangles(doc *gltf.Document, target ...any) (*gltf.Document, error) {
	return ConvertPrimitiveTopology(doc, target...)
}

// PrimitiveToTriangles is an alias for ConvertPrimitiveTopology.
func PrimitiveToTriangles(doc *gltf.Document, target ...any) (*gltf.Document, error) {
	return ConvertPrimitiveTopology(doc, target...)
}

// TriangulatePrimitive is an alias for ConvertPrimitiveTopology.
func TriangulatePrimitive(doc *gltf.Document, target ...any) (*gltf.Document, error) {
	return ConvertPrimitiveTopology(doc, target...)
}

// Triangulate is an alias for ConvertPrimitiveTopology.
func Triangulate(doc *gltf.Document, target ...any) (*gltf.Document, error) {
	return ConvertPrimitiveTopology(doc, target...)
}

// ToTriangles is an alias for ConvertPrimitiveTopology.
func ToTriangles(doc *gltf.Document, target ...any) (*gltf.Document, error) {
	return ConvertPrimitiveTopology(doc, target...)
}

// EnsureTriangles is an alias for ConvertPrimitiveTopology.
func EnsureTriangles(doc *gltf.Document, target ...any) (*gltf.Document, error) {
	return ConvertPrimitiveTopology(doc, target...)
}

// ConvertPrimitive is an alias for ConvertPrimitiveTopology.
func ConvertPrimitive(doc *gltf.Document, target ...any) (*gltf.Document, error) {
	return ConvertPrimitiveTopology(doc, target...)
}

// ConvertTriangleTopology is an alias for ConvertPrimitiveTopology.
func ConvertTriangleTopology(doc *gltf.Document, target ...any) (*gltf.Document, error) {
	return ConvertPrimitiveTopology(doc, target...)
}

// ConvertStripAndFan is an alias for ConvertPrimitiveTopology.
func ConvertStripAndFan(doc *gltf.Document, target ...any) (*gltf.Document, error) {
	return ConvertPrimitiveTopology(doc, target...)
}

// TriangulateMeshPrimitive is an alias for ConvertPrimitiveTopology.
func TriangulateMeshPrimitive(doc *gltf.Document, target ...any) (*gltf.Document, error) {
	return ConvertPrimitiveTopology(doc, target...)
}

// ConvertPrimitiveToTriangleList is an alias for ConvertPrimitiveTopology.
func ConvertPrimitiveToTriangleList(doc *gltf.Document, target ...any) (*gltf.Document, error) {
	return ConvertPrimitiveTopology(doc, target...)
}

// ConvertStripFan is an alias for ConvertPrimitiveTopology.
func ConvertStripFan(doc *gltf.Document, target ...any) (*gltf.Document, error) {
	return ConvertPrimitiveTopology(doc, target...)
}

// ConvertTriangleStripToTriangles is an alias for ConvertPrimitiveTopology for
// callers that name the strip conversion explicitly.
func ConvertTriangleStripToTriangles(doc *gltf.Document, target ...any) (*gltf.Document, error) {
	return ConvertPrimitiveTopology(doc, target...)
}

// ConvertTriangleFanToTriangles is an alias for ConvertPrimitiveTopology for
// callers that name the fan conversion explicitly.
func ConvertTriangleFanToTriangles(doc *gltf.Document, target ...any) (*gltf.Document, error) {
	return ConvertPrimitiveTopology(doc, target...)
}

// NormalizePrimitiveModes converts every strip/fan primitive and returns an
// independent document. It is an alias for ConvertAllPrimitiveTopologies.
func NormalizePrimitiveModes(doc *gltf.Document, opts ...TopologyOption) (*gltf.Document, error) {
	return ConvertAllPrimitiveTopologies(doc, opts...)
}

// NormalizeTopology is an alias for NormalizePrimitiveModes.
func NormalizeTopology(doc *gltf.Document, opts ...TopologyOption) (*gltf.Document, error) {
	return ConvertAllPrimitiveTopologies(doc, opts...)
}

// ToTriangleIndices is an alias for ReadTriangleIndices.
func ToTriangleIndices(doc *gltf.Document, target ...any) ([]uint32, error) {
	return ReadTriangleIndices(doc, target...)
}

// ExpandedTriangleIndices is an alias for ReadTriangleIndices.
func ExpandedTriangleIndices(doc *gltf.Document, target ...any) ([]uint32, error) {
	return ReadTriangleIndices(doc, target...)
}

// PrimitiveTriangleIndices is an alias for ReadTriangleIndices.
func PrimitiveTriangleIndices(doc *gltf.Document, target ...any) ([]uint32, error) {
	return ReadTriangleIndices(doc, target...)
}

// GetTriangleIndices is an alias for ReadTriangleIndices.
func GetTriangleIndices(doc *gltf.Document, target ...any) ([]uint32, error) {
	return ReadTriangleIndices(doc, target...)
}

// TriangleIndices is an alias for ReadTriangleIndices.
func TriangleIndices(doc *gltf.Document, target ...any) ([]uint32, error) {
	return ReadTriangleIndices(doc, target...)
}

// ApplyTopology is an alias for ApplyPrimitiveTopology.
func ApplyTopology(doc *gltf.Document, target ...any) error {
	return ApplyPrimitiveTopology(doc, target...)
}

// --- Resolution and conversion internals ------------------------------------

func optionsConfig(opts []TopologyOption) topologyConfig {
	cfg := topologyConfig{}
	for _, o := range opts {
		if o != nil {
			o(&cfg)
		}
	}
	return cfg
}

// resolveTargets interprets the variadic target arguments.
func resolveTargets(doc *gltf.Document, target []any) (topologyConfig, []PrimitiveRef, bool, error) {
	cfg := topologyConfig{}
	if doc == nil {
		return cfg, nil, false, ErrInvalidDocument
	}
	var positional []any
	for _, a := range target {
		if o, ok := a.(TopologyOption); ok {
			o(&cfg)
			continue
		}
		positional = append(positional, a)
	}
	if len(positional) == 0 {
		return cfg, nil, true, nil
	}

	// Unwrap a PrimitiveRef value or pointer.
	switch v := positional[0].(type) {
	case PrimitiveRef:
		return cfg, []PrimitiveRef{v}, false, nil
	case *PrimitiveRef:
		if v == nil {
			return cfg, nil, false, fmt.Errorf("%w: nil primitive reference", ErrPrimitiveNotFound)
		}
		return cfg, []PrimitiveRef{*v}, false, nil
	}

	switch first := positional[0].(type) {
	case *gltf.Primitive:
		if first == nil {
			return cfg, nil, false, fmt.Errorf("%w: nil primitive", ErrInvalidDocument)
		}
		ref, err := locatePrimitive(doc, first)
		return cfg, []PrimitiveRef{ref}, false, err
	case *gltf.Mesh:
		if first == nil {
			return cfg, nil, false, fmt.Errorf("%w: nil mesh", ErrPrimitiveNotFound)
		}
		mi, err := locateMesh(doc, first)
		if err != nil {
			return cfg, nil, false, err
		}
		pi, err := optionalPrimitiveIndex(positional, 1)
		if err != nil {
			return cfg, nil, false, err
		}
		return cfg, []PrimitiveRef{{Mesh: mi, Primitive: pi}}, false, nil
	case string:
		mi := meshIndexByName(doc, first)
		if mi < 0 {
			return cfg, nil, false, fmt.Errorf("%w: mesh %q", ErrPrimitiveNotFound, first)
		}
		pi, err := optionalPrimitiveIndex(positional, 1)
		if err != nil {
			return cfg, nil, false, err
		}
		return cfg, []PrimitiveRef{{Mesh: mi, Primitive: pi}}, false, nil
	default:
		if v, ok := topoAsInt(first); ok {
			return intTargets(doc, v, positional[1:], cfg)
		}
		return cfg, nil, false, fmt.Errorf("%w: unsupported target type %T", ErrPrimitiveNotFound, first)
	}
}

// topoAsInt coerces the common integer kinds (and pointers to them) to int.
func topoAsInt(a any) (int, bool) {
	switch v := a.(type) {
	case int:
		return v, true
	case *int:
		if v == nil {
			return 0, false
		}
		return *v, true
	}
	rv := reflect.ValueOf(a)
	if !rv.IsValid() {
		return 0, false
	}
	for rv.Kind() == reflect.Ptr || rv.Kind() == reflect.Interface {
		if rv.IsNil() {
			return 0, false
		}
		rv = rv.Elem()
	}
	switch rv.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return int(rv.Int()), true
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return int(rv.Uint()), true
	default:
		return 0, false
	}
}

func intTargets(doc *gltf.Document, first int, rest []any, cfg topologyConfig) (topologyConfig, []PrimitiveRef, bool, error) {
	if len(rest) == 0 {
		// A single int is a global primitive ordinal across all meshes.
		ref, ok := globalPrimitiveRef(doc, first)
		if !ok {
			return cfg, nil, false, fmt.Errorf("%w: global primitive index %d", ErrPrimitiveNotFound, first)
		}
		return cfg, []PrimitiveRef{ref}, false, nil
	}
	if second, ok := topoAsInt(rest[0]); ok {
		return cfg, []PrimitiveRef{{Mesh: first, Primitive: second}}, false, nil
	}
	return cfg, nil, false, fmt.Errorf("%w: expected primitive index after mesh index, got %T", ErrPrimitiveNotFound, rest[0])
}

// optionalPrimitiveIndex reads an integer at positional[index] or defaults to 0.
func optionalPrimitiveIndex(args []any, index int) (int, error) {
	if index >= len(args) {
		return 0, nil
	}
	if v, ok := topoAsInt(args[index]); ok {
		return v, nil
	}
	return 0, fmt.Errorf("%w: primitive index must be an integer, got %T", ErrPrimitiveNotFound, args[index])
}

func locatePrimitive(doc *gltf.Document, p *gltf.Primitive) (PrimitiveRef, error) {
	for mi, m := range doc.Meshes {
		if m == nil {
			continue
		}
		for pi, cand := range m.Primitives {
			if cand == p {
				return PrimitiveRef{Mesh: mi, Primitive: pi}, nil
			}
		}
	}
	return PrimitiveRef{}, fmt.Errorf("%w: primitive is not part of doc", ErrPrimitiveNotFound)
}

func locateMesh(doc *gltf.Document, mesh *gltf.Mesh) (int, error) {
	for mi, m := range doc.Meshes {
		if m == mesh {
			return mi, nil
		}
	}
	return -1, fmt.Errorf("%w: mesh is not part of doc", ErrPrimitiveNotFound)
}

func meshIndexByName(doc *gltf.Document, name string) int {
	for mi, m := range doc.Meshes {
		if m != nil && m.Name == name {
			return mi
		}
	}
	return -1
}

// globalPrimitiveRef maps a global primitive ordinal to a PrimitiveRef. If
// doc is nil it reports whether the ordinal can ever be valid only by shape;
// callers always pass the document.
func globalPrimitiveRef(doc *gltf.Document, ordinal int) (PrimitiveRef, bool) {
	if ordinal < 0 || doc == nil {
		return PrimitiveRef{}, false
	}
	k := 0
	for mi, m := range doc.Meshes {
		if m == nil {
			continue
		}
		for pi := range m.Primitives {
			if k == ordinal {
				return PrimitiveRef{Mesh: mi, Primitive: pi}, true
			}
			k++
		}
	}
	return PrimitiveRef{}, false
}

func firstConvertible(doc *gltf.Document) (PrimitiveRef, bool) {
	for mi, m := range doc.Meshes {
		if m == nil {
			continue
		}
		for pi, p := range m.Primitives {
			if p != nil && (p.Mode == gltf.PrimitiveTriangleStrip || p.Mode == gltf.PrimitiveTriangleFan) {
				return PrimitiveRef{Mesh: mi, Primitive: pi}, true
			}
		}
	}
	return PrimitiveRef{}, false
}

func convertAllInPlace(doc *gltf.Document, cfg topologyConfig) error {
	for mi, m := range doc.Meshes {
		if m == nil {
			continue
		}
		for pi, p := range m.Primitives {
			if p == nil {
				continue
			}
			if p.Mode != gltf.PrimitiveTriangleStrip && p.Mode != gltf.PrimitiveTriangleFan {
				continue
			}
			if _, err := convertInPlace(doc, PrimitiveRef{Mesh: mi, Primitive: pi}, cfg); err != nil {
				return err
			}
		}
	}
	return nil
}

func convertInPlace(doc *gltf.Document, ref PrimitiveRef, cfg topologyConfig) (*TopologyStats, error) {
	prim, err := primitiveAt(doc, ref)
	if err != nil {
		return nil, err
	}
	from := prim.Mode
	stats := &TopologyStats{Mesh: ref.Mesh, Primitive: ref.Primitive, From: from, To: gltf.PrimitiveTriangles}

	if from == gltf.PrimitiveTriangles {
		// Already the target topology: idempotent no-op.
		return stats, nil
	}
	if from != gltf.PrimitiveTriangleStrip && from != gltf.PrimitiveTriangleFan {
		return nil, fmt.Errorf("%w: mode %d at meshes[%d].primitives[%d]", ErrInvalidTopology, from, ref.Mesh, ref.Primitive)
	}

	tris, degenerate, shared, err := computeTriangles(doc, ref, cfg)
	if err != nil {
		return nil, err
	}
	stats.Degenerate = degenerate
	stats.SharedIndices = shared
	if cfg.rejectShared && shared {
		return nil, fmt.Errorf("%w: accessor %d", ErrSharedAccessor, *prim.Indices)
	}

	// Write the new indices somewhere that is both idiomatic and independently
	// savable. An empty-URI buffer whose in-memory Data covers its declared
	// ByteLength can be extended directly (this is exactly what WriteIndices
	// does). Buffers backed by an external URI or an embedded data URI must not
	// be extended in place because their serialized form would stay stale, so a
	// fresh self-contained buffer is used for them.
	prepareIndexBuffer(doc)

	var accessorIndex int
	if cfg.compact && indicesFit16(tris) {
		u16 := make([]uint16, len(tris))
		for i, v := range tris {
			u16[i] = uint16(v)
		}
		accessorIndex = WriteIndices(doc, u16)
		setIndicesMinMax(doc, accessorIndex, tris)
	} else {
		accessorIndex = WriteIndices(doc, tris)
		setIndicesMinMax(doc, accessorIndex, tris)
	}

	prim.Indices = gltf.Index(accessorIndex)
	prim.Mode = gltf.PrimitiveTriangles
	stats.Indices = len(tris)
	stats.Triangles = len(tris) / 3
	return stats, nil
}

func primitiveAt(doc *gltf.Document, ref PrimitiveRef) (*gltf.Primitive, error) {
	if doc == nil {
		return nil, ErrInvalidDocument
	}
	if ref.Mesh < 0 || ref.Mesh >= len(doc.Meshes) || doc.Meshes[ref.Mesh] == nil {
		return nil, fmt.Errorf("%w: mesh index %d", ErrPrimitiveNotFound, ref.Mesh)
	}
	prims := doc.Meshes[ref.Mesh].Primitives
	if ref.Primitive < 0 || ref.Primitive >= len(prims) || prims[ref.Primitive] == nil {
		return nil, fmt.Errorf("%w: meshes[%d].primitives[%d]", ErrPrimitiveNotFound, ref.Mesh, ref.Primitive)
	}
	return prims[ref.Primitive], nil
}

// computeTriangles resolves and validates the primitive's index sequence and
// returns the expanded triangle list without mutating doc.
func computeTriangles(doc *gltf.Document, ref PrimitiveRef, cfg topologyConfig) (tris []uint32, degenerate int, shared bool, err error) {
	prim, err := primitiveAt(doc, ref)
	if err != nil {
		return nil, 0, false, err
	}
	mode := prim.Mode
	if mode == gltf.PrimitiveTriangles {
		// Nothing to expand; still expose the existing indices for readers.
		seq, rerr := readSequence(doc, prim)
		if rerr != nil {
			return nil, 0, false, rerr
		}
		return seq, 0, false, nil
	}
	if mode != gltf.PrimitiveTriangleStrip && mode != gltf.PrimitiveTriangleFan {
		return nil, 0, false, fmt.Errorf("%w: mode %d", ErrInvalidTopology, mode)
	}

	vertexCount := primitiveVertexCount(doc, prim)

	// Size guard before materializing an unindexed sequence / expanding.
	n := sequenceLength(doc, prim, vertexCount)
	if _, outCount, serr := expansionSize(n); serr != nil {
		return nil, 0, false, serr
	} else if outCount > maxExpandedIndexCount {
		return nil, 0, false, fmt.Errorf("%w: %d indices", ErrOutputTooLarge, outCount)
	}

	if n < 3 {
		if cfg.allowSmall {
			return []uint32{}, 0, false, nil
		}
		return nil, 0, false, fmt.Errorf("%w: meshes[%d].primitives[%d] has %d", ErrTooFewVertices, ref.Mesh, ref.Primitive, n)
	}

	seq, err := readSequence(doc, prim)
	if err != nil {
		return nil, 0, false, err
	}

	// Index bounds validation against POSITION/attribute vertex count.
	if vertexCount >= 0 {
		for _, idx := range seq {
			if uint64(idx) >= uint64(vertexCount) {
				return nil, 0, false, fmt.Errorf("%w: index %d >= vertex count %d", ErrIndexOutOfRange, idx, vertexCount)
			}
		}
	}

	positions := readPositionsBestEffort(doc, prim)
	tris, degenerate = buildTriangleList(mode, seq, positions)

	if prim.Indices != nil {
		shared = countAccessorReferences(doc, *prim.Indices) > 1
	}
	return tris, degenerate, shared, nil
}

// sequenceLength reports the logical index count without reading data.
func sequenceLength(doc *gltf.Document, prim *gltf.Primitive, vertexCount int) int {
	if prim.Indices == nil {
		if vertexCount < 0 {
			return 0
		}
		return vertexCount
	}
	acc := accessorAt(doc, *prim.Indices)
	if acc == nil {
		return 0
	}
	return acc.Count
}

// readSequence returns the indexed values or a sequential 0..n-1 sequence for
// an unindexed primitive.
func readSequence(doc *gltf.Document, prim *gltf.Primitive) ([]uint32, error) {
	if prim.Indices == nil {
		n := primitiveVertexCount(doc, prim)
		if n < 0 {
			n = 0
		}
		seq := make([]uint32, n)
		for i := range seq {
			seq[i] = uint32(i)
		}
		return seq, nil
	}
	acc := accessorAt(doc, *prim.Indices)
	if acc == nil {
		return nil, fmt.Errorf("%w: indices accessor %d", ErrPrimitiveNotFound, *prim.Indices)
	}
	indices, err := ReadIndices(doc, acc, nil)
	if err != nil {
		return nil, fmt.Errorf("gltf: cannot read indices accessor %d: %w", *prim.Indices, err)
	}
	return indices, nil
}

func accessorAt(doc *gltf.Document, index int) *gltf.Accessor {
	if doc == nil || index < 0 || index >= len(doc.Accessors) {
		return nil
	}
	return doc.Accessors[index]
}

// primitiveVertexCount returns the number of vertices addressed by the
// primitive's attributes. It prefers POSITION and falls back to the minimum
// count of any referenced attribute accessor. It returns -1 if unknown.
func primitiveVertexCount(doc *gltf.Document, prim *gltf.Primitive) int {
	if prim == nil {
		return -1
	}
	if pos, ok := prim.Attributes[gltf.POSITION]; ok {
		if acc := accessorAt(doc, pos); acc != nil && acc.Count >= 0 {
			return acc.Count
		}
		return -1
	}
	min := -1
	for _, ai := range prim.Attributes {
		acc := accessorAt(doc, ai)
		if acc == nil || acc.Count < 0 {
			continue
		}
		if min < 0 || acc.Count < min {
			min = acc.Count
		}
	}
	return min
}

func readPositionsBestEffort(doc *gltf.Document, prim *gltf.Primitive) [][3]float32 {
	pos, ok := prim.Attributes[gltf.POSITION]
	if !ok {
		return nil
	}
	acc := accessorAt(doc, pos)
	if acc == nil {
		return nil
	}
	positions, err := ReadPosition(doc, acc, nil)
	if err != nil {
		return nil
	}
	return positions
}

// countAccessorReferences counts primitives whose indices property points at
// accessorIndex.
func countAccessorReferences(doc *gltf.Document, accessorIndex int) int {
	count := 0
	for _, m := range doc.Meshes {
		if m == nil {
			continue
		}
		for _, p := range m.Primitives {
			if p != nil && p.Indices != nil && *p.Indices == accessorIndex {
				count++
			}
		}
	}
	return count
}

// prepareIndexBuffer ensures a subsequent WriteIndices call appends to a
// writable, embeddable buffer. It leaves an existing empty-URI buffer with
// consistent storage in place; otherwise it appends a fresh empty buffer.
func prepareIndexBuffer(doc *gltf.Document) {
	if len(doc.Buffers) == 0 {
		return // WriteIndices/lastBuffer creates the buffer
	}
	last := len(doc.Buffers) - 1
	b := doc.Buffers[last]
	if b == nil {
		doc.Buffers[last] = new(gltf.Buffer)
		return
	}
	if b.URI == "" && len(b.Data) == b.ByteLength {
		return
	}
	doc.Buffers = append(doc.Buffers, new(gltf.Buffer))
}

func indicesFit16(indices []uint32) bool {
	for _, v := range indices {
		if v > 65535 {
			return false
		}
	}
	return true
}

func setIndicesMinMax(doc *gltf.Document, accessorIndex int, indices []uint32) {
	acc := accessorAt(doc, accessorIndex)
	if acc == nil || len(indices) == 0 {
		return
	}
	mn, mx := indices[0], indices[0]
	for _, v := range indices[1:] {
		if v < mn {
			mn = v
		}
		if v > mx {
			mx = v
		}
	}
	acc.Min = []float64{float64(mn)}
	acc.Max = []float64{float64(mx)}
}
