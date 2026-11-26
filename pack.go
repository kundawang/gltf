package gltf

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"math"
	"net/url"
	"path"
	"strings"
)

// glbAlignment is the mandatory alignment of GLB chunks and resource starts
// inside the BIN chunk.
const glbAlignment = 4

// Image MIME types allowed in a core glTF 2.0 GLB.
const (
	packMIMEPNG  = "image/png"
	packMIMEJPEG = "image/jpeg"
)

// Packer reads the relative resources referenced by a glTF document from a
// controlled file system and packs them into a single self-contained GLB.
//
// All relative buffer and image URIs are resolved through Fsys. Embedded
// data URIs are decoded and moved into the BIN chunk as well, so the produced
// GLB references no external resource. The source Document and the files in
// Fsys are never modified.
//
// Use NewPacker to build a Packer.
type Packer struct {
	// Fsys is the controlled file system used to resolve relative URIs.
	// It may be nil when the document has no external URIs.
	Fsys fs.FS
}

// NewPacker returns a Packer that resolves relative URIs through fsys.
func NewPacker(fsys fs.FS) *Packer {
	return &Packer{Fsys: fsys}
}

// Pack is shorthand for PackGLB.
func Pack(doc *Document, fsys fs.FS) ([]byte, error) {
	return PackGLB(doc, fsys)
}

// PackGLB reads every external buffer and image referenced by doc from fsys
// and returns a self-contained GLB byte slice. It is the simplest packaging
// entry point and wraps a Packer backed by fsys.
//
// Relative resource URIs are resolved through fsys; pass nil when doc only
// uses embedded data URIs, bufferView images or in-memory Buffer.Data.
//
// doc and the files exposed by fsys are never modified. See Packer for the
// full rewriting and validation rules.
func PackGLB(doc *Document, fsys fs.FS) ([]byte, error) {
	return NewPacker(fsys).Pack(doc)
}

// PackBinary packs doc into a self-contained GLB and writes it to w. It is the
// streaming counterpart of PackGLB.
func PackBinary(w io.Writer, doc *Document, fsys fs.FS) error {
	return NewPacker(fsys).Encode(w, doc)
}

// Pack reads every external buffer and image referenced by doc from p.Fsys
// and returns a self-contained GLB.
func (p *Packer) Pack(doc *Document) ([]byte, error) {
	var buf bytes.Buffer
	if err := p.Encode(&buf, doc); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// Encode packs doc and writes the resulting GLB to w.
func (p *Packer) Encode(w io.Writer, doc *Document) error {
	packed, bin, err := p.packDocument(doc)
	if err != nil {
		return err
	}
	return writePackedGLB(w, packed, bin)
}

// PackDocument resolves all resources and returns the rewritten,
// self-contained document. Its sole buffer has an empty URI and carries the
// merged BIN bytes in Buffer.Data; encode it with Encoder.AsBinary or inspect
// its retargeted bufferViews and images. doc itself is not modified.
func (p *Packer) PackDocument(doc *Document) (*Document, error) {
	packed, _, err := p.packDocument(doc)
	return packed, err
}

// binBuilder accumulates the BIN chunk while tracking overflow-safe offsets.
type binBuilder struct {
	data bytes.Buffer
	size int
}

// add appends b at a 4-byte aligned offset and returns that offset.
func (b *binBuilder) add(resource []byte) (int, error) {
	start, err := align4(b.size)
	if err != nil {
		return 0, err
	}
	if pad := start - b.size; pad > 0 {
		b.data.Write(make([]byte, pad))
	}
	if start > math.MaxInt-len(resource) {
		return 0, errors.New("gltf: pack: BIN chunk size overflow")
	}
	offset := start
	b.data.Write(resource)
	b.size = start + len(resource)
	return offset, nil
}

func align4(n int) (int, error) {
	if n < 0 {
		return 0, errors.New("gltf: pack: negative size")
	}
	if n > math.MaxInt-(glbAlignment-1) {
		return 0, errors.New("gltf: pack: alignment size overflow")
	}
	return (n + glbAlignment - 1) &^ (glbAlignment - 1), nil
}

type packSession struct {
	fsys fs.FS
	bin  binBuilder
	// baseOffset maps each source buffer index to its BIN base offset.
	baseOffset []int
	// bufferBytes holds the validated bytes of each source buffer.
	bufferBytes [][]byte
}

// packDocument resolves all resources and returns a rewritten document whose
// only buffer is the in-memory BIN buffer (URI empty), together with the BIN
// chunk bytes. The receiver document is copied before rewriting and is never
// modified.
//
// Rewriting rules:
//
//   - Every buffer is appended, in index order, to the BIN chunk at a
//     4-byte aligned offset. Each existing BufferView is retargeted to the
//     single BIN buffer and its ByteOffset is shifted by the owning buffer
//     base offset. BufferView indices are preserved.
//   - Every image referenced by URI (relative file or embedded data URI) is
//     appended to the BIN chunk at a 4-byte aligned offset and gains a new
//     BufferView appended to Document.BufferViews; the image's URI is cleared
//     and its MimeType is normalized. Images already referenced by
//     BufferView keep their (retargeted) view.
//   - Embedded data URIs are decoded and consolidated into the BIN chunk.
//   - An empty buffer (empty URI, no data and byteLength 0) contributes no
//     bytes. An image with neither URI nor BufferView is left untouched.
func (p *Packer) packDocument(doc *Document) (*Document, []byte, error) {
	if doc == nil {
		return nil, nil, errors.New("gltf: pack: nil document")
	}
	s := &packSession{
		fsys:        p.Fsys,
		baseOffset:  make([]int, len(doc.Buffers)),
		bufferBytes: make([][]byte, len(doc.Buffers)),
	}

	// 1. Resolve every buffer and lay it down in the BIN chunk.
	for i, buffer := range doc.Buffers {
		if buffer == nil {
			return nil, nil, fmt.Errorf("gltf: pack: buffers[%d] is nil", i)
		}
		data, err := s.readBuffer(buffer)
		if err != nil {
			return nil, nil, fmt.Errorf("gltf: pack: buffers[%d] (%q): %w", i, buffer.URI, err)
		}
		if buffer.ByteLength < 0 {
			return nil, nil, fmt.Errorf("gltf: pack: buffers[%d]: negative byteLength %d", i, buffer.ByteLength)
		}
		if len(data) != buffer.ByteLength {
			return nil, nil, fmt.Errorf("gltf: pack: buffers[%d]: length mismatch: byteLength=%d but resource has %d bytes", i, buffer.ByteLength, len(data))
		}
		offset, err := s.bin.add(data)
		if err != nil {
			return nil, nil, err
		}
		s.baseOffset[i] = offset
		s.bufferBytes[i] = data
	}

	// 2. Clone the document and the slices that get rewritten so the caller
	// document stays untouched.
	out := *doc
	if doc.BufferViews != nil {
		out.BufferViews = make([]*BufferView, len(doc.BufferViews))
		for i, bv := range doc.BufferViews {
			if bv == nil {
				return nil, nil, fmt.Errorf("gltf: pack: bufferViews[%d] is nil", i)
			}
			bvCopy := *bv
			out.BufferViews[i] = &bvCopy
		}
	}
	if doc.Images != nil {
		out.Images = make([]*Image, len(doc.Images))
		for i, img := range doc.Images {
			if img == nil {
				return nil, nil, fmt.Errorf("gltf: pack: images[%d] is nil", i)
			}
			imgCopy := *img
			out.Images[i] = &imgCopy
		}
	}

	// 3. Validate and retarget every BufferView. BufferView indices stay
	// stable; only the buffer index and offset change.
	for i, bv := range out.BufferViews {
		if bv.Buffer < 0 || bv.Buffer >= len(doc.Buffers) {
			return nil, nil, fmt.Errorf("gltf: pack: bufferViews[%d]: buffer index %d out of range", i, bv.Buffer)
		}
		if bv.ByteOffset < 0 || bv.ByteLength < 0 {
			return nil, nil, fmt.Errorf("gltf: pack: bufferViews[%d]: negative offset/length", i)
		}
		data := s.bufferBytes[bv.Buffer]
		end, overflow := addOverflow(bv.ByteOffset, bv.ByteLength)
		if overflow || end > len(data) {
			return nil, nil, fmt.Errorf("gltf: pack: bufferViews[%d]: view [%d, %d) out of buffer %d bounds (length %d)",
				i, bv.ByteOffset, end, bv.Buffer, len(data))
		}
		newOffset, overflow := addOverflow(s.baseOffset[bv.Buffer], bv.ByteOffset)
		if overflow {
			return nil, nil, fmt.Errorf("gltf: pack: bufferViews[%d]: offset overflow", i)
		}
		if int64(newOffset) > math.MaxUint32 {
			// A GLB cannot address offsets outside the uint32 range.
			return nil, nil, fmt.Errorf("gltf: pack: bufferViews[%d]: packed offset %d too large", i, newOffset)
		}
		bv.Buffer = 0
		bv.ByteOffset = newOffset
	}

	// 4. Pack URI-based images and normalize bufferView-based images.
	for i, img := range out.Images {
		switch {
		case img.BufferView != nil:
			if *img.BufferView < 0 || *img.BufferView >= len(out.BufferViews) {
				return nil, nil, fmt.Errorf("gltf: pack: images[%d]: bufferView index %d out of range", i, *img.BufferView)
			}
			bv := out.BufferViews[*img.BufferView]
			viewEnd, overflow := addOverflow(bv.ByteOffset, bv.ByteLength)
			if overflow || viewEnd > s.bin.size {
				return nil, nil, fmt.Errorf("gltf: pack: images[%d]: bufferView %d is out of the BIN chunk", i, *img.BufferView)
			}
			viewBytes := s.bin.data.Bytes()[bv.ByteOffset:viewEnd]
			mime, err := resolveImageMIME(img.MimeType, viewBytes)
			if err != nil {
				return nil, nil, fmt.Errorf("gltf: pack: images[%d]: %w", i, err)
			}
			img.MimeType = mime
		case img.URI != "":
			data, err := s.readImage(img)
			if err != nil {
				return nil, nil, fmt.Errorf("gltf: pack: images[%d] (%q): %w", i, img.URI, err)
			}
			mime, err := resolveImageMIME(img.MimeType, data)
			if err != nil {
				return nil, nil, fmt.Errorf("gltf: pack: images[%d] (%q): %w", i, img.URI, err)
			}
			offset, err := s.bin.add(data)
			if err != nil {
				return nil, nil, err
			}
			viewIndex := len(out.BufferViews)
			out.BufferViews = append(out.BufferViews, &BufferView{
				Buffer:     0,
				ByteOffset: offset,
				ByteLength: len(data),
			})
			img.URI = ""
			img.BufferView = Index(viewIndex)
			img.MimeType = mime
		default:
			// Empty image resource: neither uri nor bufferView. Keep it as a
			// content-less entry rather than inventing bytes.
		}
	}

	// 5. Replace all source buffers with the single merged BIN buffer.
	binBytes := s.bin.data.Bytes()
	if s.bin.size == 0 {
		// No resource carries bytes. Emit a GLB without a buffers array so it
		// stays re-readable by decoders that reject byteLength 0 buffers.
		out.Buffers = nil
	} else {
		out.Buffers = []*Buffer{{
			ByteLength: s.bin.size,
			Data:       binBytes,
		}}
	}
	return &out, binBytes, nil
}

// readBuffer resolves a buffer's bytes from its data URI, in-memory Data or
// the controlled file system.
func (s *packSession) readBuffer(buffer *Buffer) ([]byte, error) {
	if buffer.IsEmbeddedResource() {
		data, err := buffer.marshalData()
		if err != nil {
			return nil, fmt.Errorf("invalid embedded data URI: %w", err)
		}
		return data, nil
	}
	if buffer.URI == "" {
		// GLB BIN buffer or caller-provided in-memory buffer.
		return buffer.Data, nil
	}
	path, err := resolveRelativePath(buffer.URI)
	if err != nil {
		return nil, err
	}
	if s.fsys == nil {
		// Support documents whose bytes were already loaded in memory (for
		// example via Open): use Data only when it exactly matches the
		// declared length.
		if buffer.ByteLength > 0 && len(buffer.Data) == buffer.ByteLength {
			return buffer.Data, nil
		}
		return nil, fmt.Errorf("%w: %q", fs.ErrNotExist, path)
	}
	data, err := fs.ReadFile(s.fsys, path)
	if err != nil {
		return nil, err
	}
	return data, nil
}

// readImage resolves the bytes of a URI-based image.
func (s *packSession) readImage(img *Image) ([]byte, error) {
	if img.IsEmbeddedResource() {
		data, err := img.MarshalData()
		if err != nil {
			return nil, fmt.Errorf("invalid embedded data URI: %w", err)
		}
		return data, nil
	}
	ref, err := resolveRelativePath(img.URI)
	if err != nil {
		return nil, err
	}
	if s.fsys == nil {
		return nil, fmt.Errorf("%w: %q", fs.ErrNotExist, ref)
	}
	return fs.ReadFile(s.fsys, ref)
}

// resolveRelativePath maps a glTF relative URI to a validated slash-separated
// fs.FS path. It rejects absolute paths, URI schemes (including Windows drive
// letters and UNC roots) and any reference that escapes the file system root.
func resolveRelativePath(uri string) (string, error) {
	if uri == "" {
		return "", errors.New("gltf: pack: empty URI")
	}
	normalized := strings.ReplaceAll(uri, "\\", "/")
	u, err := url.Parse(normalized)
	if err != nil {
		return "", fmt.Errorf("gltf: pack: invalid URI %q: %w", uri, err)
	}
	if u.Scheme != "" {
		return "", fmt.Errorf("gltf: pack: unsupported absolute URI scheme %q in %q", u.Scheme, uri)
	}
	if u.Host != "" {
		return "", fmt.Errorf("gltf: pack: absolute/UNC URI %q is not allowed", uri)
	}
	p := u.Path
	if p == "" {
		// Tolerate URIs that only carry a query/fragment? They identify no
		// readable resource.
		return "", fmt.Errorf("gltf: pack: URI %q does not name a file", uri)
	}
	if strings.HasPrefix(p, "/") {
		return "", fmt.Errorf("gltf: pack: absolute path %q is not allowed", uri)
	}
	// url.Parse already percent-decodes u.Path, so traversal sequences hidden
	// in percent-encoding are visible here.
	p = path.Clean(p)
	if p == ".." || strings.HasPrefix(p, "../") {
		return "", fmt.Errorf("gltf: pack: path %q escapes the controlled file system", uri)
	}
	if !fs.ValidPath(p) {
		return "", fmt.Errorf("gltf: pack: invalid relative path %q", uri)
	}
	return p, nil
}

// resolveImageMIME validates a declared image MIME type against the actual
// content and returns the MIME type to store in the packed image.
func resolveImageMIME(declared string, data []byte) (string, error) {
	sniffed := sniffImageMIME(data)
	if declared != "" {
		if declared != packMIMEPNG && declared != packMIMEJPEG {
			return "", fmt.Errorf("unsupported image mimeType %q: GLB supports %q and %q", declared, packMIMEPNG, packMIMEJPEG)
		}
		if sniffed == "" {
			return "", fmt.Errorf("declared image mimeType %q does not match recognizable image content", declared)
		}
		if sniffed != declared {
			return "", fmt.Errorf("wrong image mimeType: declared %q but content is %q", declared, sniffed)
		}
		return declared, nil
	}
	if sniffed == "" {
		return "", errors.New("cannot determine image mimeType from content and no mimeType declared")
	}
	return sniffed, nil
}

// sniffImageMIME detects PNG or JPEG bytes by their magic signatures.
func sniffImageMIME(data []byte) string {
	if len(data) >= 8 && bytes.Equal(data[:8], []byte{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'}) {
		return packMIMEPNG
	}
	if len(data) >= 3 && data[0] == 0xff && data[1] == 0xd8 && data[2] == 0xff {
		return packMIMEJPEG
	}
	return ""
}

func addOverflow(a, b int) (int, bool) {
	if b < 0 || a < 0 || a > math.MaxInt-b {
		return 0, true
	}
	return a + b, false
}

// writePackedGLB serializes packed as a GLB with a single aligned BIN chunk.
// Unlike Encoder.Encode it performs all chunk-length overflow checks before
// writing anything.
func writePackedGLB(w io.Writer, packed *Document, bin []byte) error {
	e := &Encoder{AsBinary: true}
	jsonText, err := e.marshalJSONDoc(packed)
	if err != nil {
		return err
	}
	jsonPadded, err := paddedLen(len(jsonText))
	if err != nil {
		return fmt.Errorf("gltf: pack: JSON chunk %w", err)
	}
	var binPadded int
	hasBIN := len(packed.Buffers) > 0
	if hasBIN {
		if packed.Buffers[0] == nil {
			return errors.New("gltf: pack: nil packed buffer")
		}
		if packed.Buffers[0].ByteLength != len(bin) {
			return fmt.Errorf("gltf: pack: internal BIN length mismatch: byteLength=%d bin=%d", packed.Buffers[0].ByteLength, len(bin))
		}
		binPadded, err = paddedLen(len(bin))
		if err != nil {
			return fmt.Errorf("gltf: pack: BIN chunk %w", err)
		}
	}
	total := uint64(12) + uint64(8) + uint64(jsonPadded) // header + JSON chunk header + JSON
	if hasBIN {
		total += uint64(8) + uint64(binPadded)
	}
	if total > math.MaxUint32 {
		return errors.New("gltf: pack: GLB total length exceeds uint32")
	}
	if int64(jsonPadded) > math.MaxUint32 || (hasBIN && int64(binPadded) > math.MaxUint32) {
		return errors.New("gltf: pack: chunk length exceeds uint32")
	}

	header := glbHeader{
		Magic:      glbHeaderMagic,
		Version:    2,
		Length:     uint32(total),
		JSONHeader: chunkHeader{Length: uint32(jsonPadded), Type: glbChunkJSON},
	}
	if err := binary.Write(w, binary.LittleEndian, &header); err != nil {
		return err
	}
	if _, err := w.Write(jsonText); err != nil {
		return err
	}
	if _, err := w.Write(bytes.Repeat([]byte{' '}, jsonPadded-len(jsonText))); err != nil {
		return err
	}
	if hasBIN {
		binHeader := chunkHeader{Length: uint32(binPadded), Type: glbChunkBIN}
		if err := binary.Write(w, binary.LittleEndian, &binHeader); err != nil {
			return err
		}
		if _, err := w.Write(bin); err != nil {
			return err
		}
		if pad := binPadded - len(bin); pad > 0 {
			if _, err := w.Write(make([]byte, pad)); err != nil {
				return err
			}
		}
	}
	return nil
}

func paddedLen(n int) (int, error) {
	padded, err := align4(n)
	if err != nil {
		return 0, err
	}
	return padded, nil
}
