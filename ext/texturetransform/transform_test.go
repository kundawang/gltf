package texturetransform_test

import (
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/qmuntal/gltf"
	"github.com/qmuntal/gltf/ext/texturetransform"
)

func TestTextureTranform_ScaleOrDefault(t *testing.T) {
	tests := []struct {
		name string
		t    *texturetransform.TextureTranform
		want [2]float64
	}{
		{"default", &texturetransform.TextureTranform{Scale: texturetransform.DefaultScale}, texturetransform.DefaultScale},
		{"zeros", &texturetransform.TextureTranform{Scale: [2]float64{0, 0}}, [2]float64{0, 0}},
		{"other", &texturetransform.TextureTranform{Scale: [2]float64{1, 2}}, [2]float64{1, 2}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.t.ScaleOrDefault(); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("TextureTranform.ScaleOrDefault() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestTextureTranform_UnmarshalJSON(t *testing.T) {
	type args struct {
		data []byte
	}
	tests := []struct {
		name    string
		t       *texturetransform.TextureTranform
		args    args
		want    *texturetransform.TextureTranform
		wantErr bool
	}{
		{"default", new(texturetransform.TextureTranform), args{[]byte("{}")}, &texturetransform.TextureTranform{Scale: texturetransform.DefaultScale}, false},
		{"nodefault", new(texturetransform.TextureTranform), args{[]byte(`{"offset": [0.1,0.2],"rotation":1.57,"scale":[1, -1],"texCoord":2}`)}, &texturetransform.TextureTranform{
			Offset: [2]float64{0.1, 0.2}, Rotation: 1.57, Scale: [2]float64{1, -1}, TexCoord: gltf.Index(2),
		}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.t.UnmarshalJSON(tt.args.data); (err != nil) != tt.wantErr {
				t.Errorf("TextureTranform.UnmarshalJSON() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(tt.t, tt.want) {
				t.Errorf("TextureTranform.UnmarshalJSON() = %v, want %v", tt.t, tt.want)
			}
		})
	}
}

func TestTextureTranform_MarshalJSON(t *testing.T) {
	tests := []struct {
		name    string
		t       *texturetransform.TextureTranform
		want    []byte
		wantErr bool
	}{
		{"default", &texturetransform.TextureTranform{Scale: texturetransform.DefaultScale}, []byte(`{}`), false},
		{"empty", &texturetransform.TextureTranform{}, []byte(`{"scale":[0,0]}`), false},
		{"nodefault", &texturetransform.TextureTranform{Offset: [2]float64{0.1, 0.2}, Rotation: 1.57, Scale: [2]float64{1, -1}, TexCoord: gltf.Index(2)}, []byte(`{"offset":[0.1,0.2],"rotation":1.57,"scale":[1,-1],"texCoord":2}`), false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tt.t.MarshalJSON()
			if (err != nil) != tt.wantErr {
				t.Errorf("TextureTranform.MarshalJSON() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("TextureTranform.MarshalJSON() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestUnmarshal(t *testing.T) {
	type args struct {
		data []byte
	}
	tests := []struct {
		name    string
		args    args
		want    any
		wantErr bool
	}{
		{"base", args{[]byte("{}")}, &texturetransform.TextureTranform{Scale: texturetransform.DefaultScale}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := texturetransform.Unmarshal(tt.args.data)
			if (err != nil) != tt.wantErr {
				t.Errorf("Unmarshal() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Unmarshal() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestTextureTranform_Apply(t *testing.T) {
	tests := []struct {
		name    string
		t       *texturetransform.TextureTranform
		uv      [2]float32
		want    [2]float32
		wantErr bool
	}{
		{
			name: "no-transform",
			t:    &texturetransform.TextureTranform{Scale: texturetransform.DefaultScale},
			uv:   [2]float32{0.5, 0.25},
			want: [2]float32{0.5, 0.25},
		},
		{
			name: "offset-only",
			t: &texturetransform.TextureTranform{
				Offset: [2]float64{0.1, 0.2}, Scale: texturetransform.DefaultScale,
			},
			uv:   [2]float32{1, 0},
			want: [2]float32{1.1, 0.2},
		},
		{
			name: "scale-only",
			t: &texturetransform.TextureTranform{
				Scale: [2]float64{2, 3},
			},
			uv:   [2]float32{1, 1},
			want: [2]float32{2, 3},
		},
		{
			name: "rotation-90deg",
			t: &texturetransform.TextureTranform{
				Rotation: math.Pi / 2, Scale: texturetransform.DefaultScale,
			},
			uv:   [2]float32{1, 0},
			want: [2]float32{float32(math.Cos(math.Pi / 2)), 1},
		},
		{
			name: "offset-rotation-scale",
			t: &texturetransform.TextureTranform{
				Offset: [2]float64{0.1, 0.2}, Rotation: math.Pi / 2, Scale: [2]float64{2, 3},
			},
			uv:   [2]float32{1, 0},
			want: [2]float32{0.1, 2.2},
		},
		{
			name: "rotation-180deg",
			t: &texturetransform.TextureTranform{
				Rotation: math.Pi, Scale: texturetransform.DefaultScale,
			},
			uv:   [2]float32{1, 1},
			want: [2]float32{-1, -1},
		},
		{
			name: "non-finite-offset",
			t: &texturetransform.TextureTranform{
				Offset: [2]float64{math.NaN(), 0}, Scale: texturetransform.DefaultScale,
			},
			uv:      [2]float32{1, 0},
			wantErr: true,
		},
		{
			name: "non-finite-rotation",
			t: &texturetransform.TextureTranform{
				Rotation: math.Inf(1), Scale: texturetransform.DefaultScale,
			},
			uv:      [2]float32{1, 0},
			wantErr: true,
		},
		{
			name: "non-finite-scale",
			t: &texturetransform.TextureTranform{
				Scale: [2]float64{1, math.Inf(-1)},
			},
			uv:      [2]float32{1, 0},
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			uvCopy := tt.uv
			got, err := tt.t.Apply(tt.uv)
			if (err != nil) != tt.wantErr {
				t.Errorf("TextureTranform.Apply() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if tt.wantErr {
				if !errors.Is(err, texturetransform.ErrNonFinite) {
					t.Errorf("TextureTranform.Apply() error = %v, want ErrNonFinite", err)
				}
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("TextureTranform.Apply() = %v, want %v", got, tt.want)
			}
			if tt.uv != uvCopy {
				t.Errorf("TextureTranform.Apply() mutated the input uv: got %v, want %v", tt.uv, uvCopy)
			}
		})
	}
}
