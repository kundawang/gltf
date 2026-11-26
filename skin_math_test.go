package gltf

import (
	"math"
	"testing"
)

func mat4Near(a, b [16]float64, eps float64) bool {
	for i := 0; i < 16; i++ {
		if math.Abs(a[i]-b[i]) > eps {
			return false
		}
	}
	return true
}

func TestMulMat4(t *testing.T) {
	// T(2) * T(-1) = T(1).
	a := translationMat4([3]float64{2, 0, 0})
	b := translationMat4([3]float64{-1, 0, 0})
	got := mulMat4(a, b)
	want := translationMat4([3]float64{1, 0, 0})
	if !mat4Near(got, want, 1e-12) {
		t.Fatalf("T2*T-1 = %v, want %v", got, want)
	}

	// Identity is neutral on both sides.
	m := scaleMat4([3]float64{2, 3, 4})
	if got := mulMat4(DefaultMatrix, m); !mat4Near(got, m, 0) {
		t.Fatalf("I*m = %v, want %v", got, m)
	}
	if got := mulMat4(m, DefaultMatrix); !mat4Near(got, m, 0) {
		t.Fatalf("m*I = %v, want %v", got, m)
	}
}

func TestInverseMat4RoundTrip(t *testing.T) {
	transforms := [][16]float64{
		translationMat4([3]float64{2, -3.5, 10}),
		mulMat4(translationMat4([3]float64{1, 2, 3}),
			mulMat4(quaternionMat4([4]float64{0, 0, math.Sqrt2 / 2, math.Sqrt2 / 2}),
				scaleMat4([3]float64{2, 0.5, 3}))),
		scaleMat4([3]float64{1e-8, 1, 1}),
	}
	for i, m := range transforms {
		inv, ok := inverseMat4(m)
		if !ok {
			t.Fatalf("transform %d unexpectedly singular", i)
		}
		if got := mulMat4(m, inv); !mat4Near(got, DefaultMatrix, 1e-8) {
			t.Fatalf("m*inverse(m) != I for transform %d: %v", i, got)
		}
		if got := mulMat4(inv, m); !mat4Near(got, DefaultMatrix, 1e-8) {
			t.Fatalf("inverse(m)*m != I for transform %d: %v", i, got)
		}
	}
}

func TestInverseMat4Singular(t *testing.T) {
	zeroColumn := DefaultMatrix
	zeroColumn[0] = 1
	zeroColumn[5] = 0
	zeroColumn[10] = 0
	if _, ok := inverseMat4(zeroColumn); ok {
		t.Fatal("expected zero-scale matrix to be singular")
	}
	allZero := emptyMatrix
	if _, ok := inverseMat4(allZero); ok {
		t.Fatal("expected zero matrix to be singular")
	}
}

func TestQuaternionMat4NinetyDegreesZ(t *testing.T) {
	r := quaternionMat4([4]float64{0, 0, math.Sqrt2 / 2, math.Sqrt2 / 2})
	want := [16]float64{
		0, 1, 0, 0,
		-1, 0, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
	}
	if !mat4Near(r, want, 1e-12) {
		t.Fatalf("Rz90 = %v, want %v", r, want)
	}
}
