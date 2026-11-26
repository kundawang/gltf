package modeler

import (
	"errors"
	"fmt"
	"math"
	"reflect"

	"github.com/qmuntal/gltf"
)

// This file implements primitive topology conversion: TRIANGLE_STRIP and
// TRIANGLE_FAN primitives (indexed or not) are expanded into standard
// TRIANGLES primitives backed by a newly written, valid indices accessor.
//
// The conversion never mutates the source document: it operates on a deep,
// independently savable copy. Only the selected primitive is modified; every
// other mesh and primitive, material and vertex attribute is preserved.

// Sentinel errors returned by the topology conversion helpers. Callers can use
// errors.Is to test for a specific failure condition.
var (
	// ErrPrimitiveNotFound is returned when a mesh/primitive selector does not
	// reference an existing primitive.
	ErrPrimitiveNotFound = errors.New("gltf: primitive not found")

	// ErrTooFewVertices is returned when a strip/fan sequence contains fewer
	// than three indices/vertices and therefore cannot contain a triangle.
	ErrTooFewVertices = errors.New("gltf: topology requires at least three vertices")

	// ErrIndexOutOfRange is returned when a primitive index references a vertex
	// that does not exist in its POSITION attribute accessor.
	ErrIndexOutOfRange = errors.New("gltf: primitive index out of range")

	// ErrInvalidTopology is returned when a primitive mode is not a valid glTF
	// primitive mode or is not a triangle topology that can be converted.
	ErrInvalidTopology = errors.New("gltf: invalid or unsupported primitive topology")

	// ErrOutputTooLarge is returned when expanding a topology would produce an
	// index list too large to represent/allocate safely.
	ErrOutputTooLarge = errors.New("gltf: expanded topology output is too large")

	// ErrInvalidDocument is returned when a nil document or primitive is given.
	ErrInvalidDocument = errors.New("gltf: invalid document or primitive")

	// ErrSharedAccessor is only returned when the RejectSharedAccessor option
	// is used. The default conversion copies on write, so sharing an indices
	// accessor between primitives is safe and never returns this error.
	ErrSharedAccessor = errors.New("gltf: indices accessor is shared by multiple primitives")
)

// Error aliases under alternative, equally idiomatic names. They are the exact
// same error values so errors.Is matches any of them.
var (
	ErrTooFewIndices        = ErrTooFewVertices
	ErrNotEnoughVertices    = ErrTooFewVertices
	ErrTooFewPoints         = ErrTooFewVertices
	ErrIndexOutOfBounds     = ErrIndexOutOfRange
	ErrInvalidIndex         = ErrIndexOutOfRange
	ErrUnsupportedTopology  = ErrInvalidTopology
	ErrInvalidPrimitiveMode = ErrInvalidTopology
	ErrUnsupportedMode      = ErrInvalidTopology
	ErrInvalidMode          = ErrInvalidTopology
	ErrTooManyIndices       = ErrOutputTooLarge
	ErrIndicesTooLarge      = ErrOutputTooLarge
	ErrOutputOverflow       = ErrOutputTooLarge
	ErrTopologyTooLarge     = ErrOutputTooLarge
	ErrDocumentNil          = ErrInvalidDocument
)

// maxExpandedIndexCount bounds the number of indices that a single conversion
// may emit. It equals the maximum uint32 value and rejects expansions whose
// size would overflow or imply an unbounded allocation before any memory is
// reserved for the result.
const maxExpandedIndexCount = math.MaxUint32

// PrimitiveRef identifies a primitive by its mesh index and its index within
// that mesh. It can be passed to the variadic conversion helpers.
type PrimitiveRef struct {
	Mesh      int
	Primitive int
}

// TopologyStats describes the result of converting a single primitive.
type TopologyStats struct {
	// Mesh is the index of the converted mesh in the returned document.
	Mesh int
	// Primitive is the index of the converted primitive within its mesh.
	Primitive int
	// From is the original primitive mode.
	From gltf.PrimitiveMode
	// To is the resulting mode (always gltf.PrimitiveTriangles).
	To gltf.PrimitiveMode
	// Triangles is the number of emitted triangles after degenerate filtering.
	Triangles int
	// Indices is the number of emitted indices (3 * Triangles).
	Indices int
	// Degenerate is the number of zero-area/repeated-vertex triangles skipped.
	Degenerate int
	// SharedIndices reports whether the source indices accessor was referenced
	// by more than one primitive. The conversion always copies on write, so
	// sharing never affects other primitives.
	SharedIndices bool
}

// TopologyOption customizes the topology conversion.
type TopologyOption func(*topologyConfig)

type topologyConfig struct {
	// allowSmall turns the fewer-than-three condition into an empty triangle
	// list instead of ErrTooFewVertices.
	allowSmall bool
	// rejectShared makes a shared indices accessor fail with ErrSharedAccessor
	// instead of being isolated by copy-on-write.
	rejectShared bool
	// compact writes UNSIGNED_SHORT indices when every value fits, instead of
	// always writing UNSIGNED_INT.
	compact bool
}

// AllowEmptyTopology allows sequences with fewer than three vertices to
// produce an empty TRIANGLES primitive rather than ErrTooFewVertices.
func AllowEmptyTopology() TopologyOption {
	return func(c *topologyConfig) { c.allowSmall = true }
}

// RejectSharedAccessor makes the conversion return ErrSharedAccessor when the
// target indices accessor is referenced by more than one primitive. By
// default sharing is handled safely with copy-on-write isolation.
func RejectSharedAccessor() TopologyOption {
	return func(c *topologyConfig) { c.rejectShared = true }
}

// CompactIndices writes UNSIGNED_SHORT indices when all indices fit in 16
// bits. The default is to always write a valid UNSIGNED_INT accessor.
func CompactIndices() TopologyOption {
	return func(c *topologyConfig) { c.compact = true }
}

// StripTriangleIndices expands a triangle strip into an independent triangle
// list using the glTF/WebGL alternating winding order:
//
//	[a b c] [c b d] ...  => (a,b,c), (c,b,d), ...
//
// For example [0 1 2 3] becomes [0 1 2 2 1 3].
//
// Triangles that repeat an index (and are therefore zero-area degenerates)
// are omitted. The parity used for the winding is the position of the triple
// in the original strip, so removing a degenerate triangle never flips the
// winding of later triangles.
//
// It always returns a newly allocated slice and never returns an error:
// sequences shorter than three values yield an empty (non-nil) slice.
func StripTriangleIndices(strip []uint32) []uint32 {
	out, _ := buildTriangleList(gltf.PrimitiveTriangleStrip, strip, nil)
	return out
}

// FanTriangleIndices expands a triangle fan into an independent triangle list.
// The first index is the fan center and is shared by every triangle:
//
//	[a b c d ...] => (a,b,c), (a,c,d), ...
//
// For example [0 1 2 3] becomes [0 1 2 0 2 3].
//
// Triangles that repeat an index are omitted. Sequences shorter than three
// values yield an empty (non-nil) slice.
func FanTriangleIndices(fan []uint32) []uint32 {
	out, _ := buildTriangleList(gltf.PrimitiveTriangleFan, fan, nil)
	return out
}

// ExpandTriangleStrip is an alias for StripTriangleIndices.
func ExpandTriangleStrip(strip []uint32) []uint32 { return StripTriangleIndices(strip) }

// ExpandTriangleFan is an alias for FanTriangleIndices.
func ExpandTriangleFan(fan []uint32) []uint32 { return FanTriangleIndices(fan) }

// TriangleStripToTriangles is an alias for StripTriangleIndices.
func TriangleStripToTriangles(strip []uint32) []uint32 { return StripTriangleIndices(strip) }

// TriangleFanToTriangles is an alias for FanTriangleIndices.
func TriangleFanToTriangles(fan []uint32) []uint32 { return FanTriangleIndices(fan) }

// StripToTriangles is an alias for StripTriangleIndices.
func StripToTriangles(strip []uint32) []uint32 { return StripTriangleIndices(strip) }

// FanToTriangles is an alias for FanTriangleIndices.
func FanToTriangles(fan []uint32) []uint32 { return FanTriangleIndices(fan) }

// ConvertTriangleStripIndices expands strip indices while enforcing the
// allocation size guard.
func ConvertTriangleStripIndices(strip []uint32) ([]uint32, error) {
	return buildChecked(gltf.PrimitiveTriangleStrip, strip)
}

// ConvertTriangleFanIndices expands fan indices while enforcing the
// allocation size guard.
func ConvertTriangleFanIndices(fan []uint32) ([]uint32, error) {
	return buildChecked(gltf.PrimitiveTriangleFan, fan)
}

func buildChecked(mode gltf.PrimitiveMode, indices []uint32) ([]uint32, error) {
	_, outCount, err := expansionSize(len(indices))
	if err != nil {
		return nil, err
	}
	if outCount > maxExpandedIndexCount {
		return nil, fmt.Errorf("%w: %d indices", ErrOutputTooLarge, outCount)
	}
	out, _ := buildTriangleList(mode, indices, nil)
	return out, nil
}

// expansionSize returns the number of candidate triangles and output indices
// for a sequence of n indices. It performs overflow-safe arithmetic and must
// be called before allocating the result.
func expansionSize(n int) (triangles, outCount int, err error) {
	if n < 0 {
		return 0, 0, fmt.Errorf("%w: negative index count %d", ErrOutputTooLarge, n)
	}
	if n < 3 {
		return 0, 0, nil
	}
	triangles = n - 2
	// outCount = 3 * triangles, guarded against int overflow and the hard cap.
	if triangles > math.MaxInt/3 || int64(triangles)*3 > maxExpandedIndexCount {
		return 0, 0, fmt.Errorf("%w: %d input indices", ErrOutputTooLarge, n)
	}
	outCount = 3 * triangles
	return triangles, outCount, nil
}

// buildTriangleList expands mode into a fresh slice and reports the number of
// skipped degenerate triangles.
func buildTriangleList(mode gltf.PrimitiveMode, indices []uint32, positions [][3]float32) ([]uint32, int) {
	_, outCount, _ := expansionSize(len(indices))
	if outCount == 0 {
		return []uint32{}, 0
	}
	out := make([]uint32, 0, outCount)
	n := len(indices)
	var degenerate int
	switch mode {
	case gltf.PrimitiveTriangleStrip:
		for i := 0; i+2 < n; i++ {
			a, b, c := indices[i], indices[i+1], indices[i+2]
			if isDegenerateTriangle(a, b, c, positions) {
				degenerate++
				continue
			}
			if i%2 == 0 {
				out = append(out, a, b, c)
			} else {
				// Required winding: [0 1 2 3] => (0,1,2),(2,1,3).
				out = append(out, b, a, c)
			}
		}
	case gltf.PrimitiveTriangleFan:
		center := indices[0]
		for i := 1; i+1 < n; i++ {
			b, c := indices[i], indices[i+1]
			if isDegenerateTriangle(center, b, c, positions) {
				degenerate++
				continue
			}
			out = append(out, center, b, c)
		}
	default:
		return []uint32{}, 0
	}
	return out, degenerate
}

// isDegenerateTriangle reports whether a triangle produces no rasterized area:
// either it repeats an index or, when positions are available, its geometric
// cross product is exactly zero.
func isDegenerateTriangle(a, b, c uint32, positions [][3]float32) bool {
	if a == b || b == c || a == c {
		return true
	}
	if positions == nil {
		return false
	}
	if int(a) >= len(positions) || int(b) >= len(positions) || int(c) >= len(positions) {
		return false
	}
	p0 := positions[a]
	p1 := positions[b]
	p2 := positions[c]
	ux := float64(p1[0]) - float64(p0[0])
	uy := float64(p1[1]) - float64(p0[1])
	uz := float64(p1[2]) - float64(p0[2])
	vx := float64(p2[0]) - float64(p0[0])
	vy := float64(p2[1]) - float64(p0[1])
	vz := float64(p2[2]) - float64(p0[2])
	cx := uy*vz - uz*vy
	cy := uz*vx - ux*vz
	cz := ux*vy - uy*vx
	return cx == 0 && cy == 0 && cz == 0
}

// CloneDocument returns a deep copy of doc. The copy shares no slice, map or
// pointer-backed storage with doc (including Buffer.Data), so mutating the
// returned document never affects the source. It preserves every value and
// concrete type without JSON normalization.
func CloneDocument(doc *gltf.Document) *gltf.Document {
	if doc == nil {
		return nil
	}
	cp := deepCloneReflect(reflect.ValueOf(doc))
	if cp.Kind() == reflect.Interface {
		cp = cp.Elem()
	}
	return cp.Interface().(*gltf.Document)
}

// DeepCopyDocument is an alias for CloneDocument.
func DeepCopyDocument(doc *gltf.Document) *gltf.Document { return CloneDocument(doc) }

func deepCloneReflect(v reflect.Value) reflect.Value {
	if !v.IsValid() {
		return reflect.Value{}
	}
	switch v.Kind() {
	case reflect.Ptr:
		if v.IsNil() {
			return reflect.Zero(v.Type())
		}
		n := reflect.New(v.Type().Elem())
		n.Elem().Set(deepCloneReflect(v.Elem()))
		return n
	case reflect.Interface:
		if v.IsNil() {
			return reflect.Zero(v.Type())
		}
		c := deepCloneReflect(v.Elem())
		n := reflect.New(v.Type()).Elem()
		n.Set(c)
		return n
	case reflect.Struct:
		n := reflect.New(v.Type()).Elem()
		t := v.Type()
		for i := 0; i < v.NumField(); i++ {
			if t.Field(i).PkgPath != "" {
				continue // unexported field: leave at zero value
			}
			n.Field(i).Set(deepCloneReflect(v.Field(i)))
		}
		return n
	case reflect.Slice:
		if v.IsNil() {
			return reflect.Zero(v.Type())
		}
		n := reflect.MakeSlice(v.Type(), v.Len(), v.Len())
		for i := 0; i < v.Len(); i++ {
			n.Index(i).Set(deepCloneReflect(v.Index(i)))
		}
		return n
	case reflect.Array:
		n := reflect.New(v.Type()).Elem()
		for i := 0; i < v.Len(); i++ {
			n.Index(i).Set(deepCloneReflect(v.Index(i)))
		}
		return n
	case reflect.Map:
		if v.IsNil() {
			return reflect.Zero(v.Type())
		}
		n := reflect.MakeMapWithSize(v.Type(), v.Len())
		iter := v.MapRange()
		for iter.Next() {
			n.SetMapIndex(deepCloneReflect(iter.Key()), deepCloneReflect(iter.Value()))
		}
		return n
	default:
		return v
	}
}
