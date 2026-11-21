// SPDX-License-Identifier: MIT
// Build this file inside the pinned Ollama mlxrunner/model/inkling2 package.
package inkling2

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/model"
)

// Access is restricted to the single worker in the capture-only command.
// Each map lives within one MLX scope and is saved after Forward completes.
var oracleCapture map[string]*mlx.Array
var oracleVisit int
var oracleEmbedding bool

func oraclePut(name string, a *mlx.Array) {
	if oracleVisit != 0 {
		if oracleVisit == -1 {
			switch name {
			case "trace.embedding_rms", "trace.attn_norm_0", "trace.attn_norm_1", "trace.attn_norm_2", "trace.q_raw", "trace.key_raw", "trace.value_raw":
			default:
				return
			}
		}
		oracleCapture[fmt.Sprintf("%d.%s", max(oracleVisit, 0), name)] = a
	}
}

func oracleKV(a *mlx.Array) *mlx.Array {
	return a.Transpose(0, 2, 1, 3).Reshape(a.Dim(2), a.Dim(1)*a.Dim(3))
}

func oracleSetVisit(layer int, state residualStream, previous []*mlx.Array, messages *mlx.Array, kv cache.Layer, b *batch.Batch) {
	oracleVisit = layer
	if layer == 10 {
		oracleCapture["partial"] = state.partial
		oracleCapture["blocks"] = mlx.Stack(state.blocks, -2)
		oracleCapture["previous"] = mlx.Stack(previous, -2)
	}
	oraclePut("messages", messages)
	if _, first := oracleCapture["start"]; !first {
		return
	}
	capacity := 4352
	if layer == 10 {
		capacity = 1024
	}
	offset := int(b.SeqOffsets[0])
	for i, a := range kv.State()[:2] {
		n := min(capacity, a.Dim(2))
		values := oracleKV(a.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(a.Dim(2)-n, a.Dim(2)), mlx.Slice()))
		if n < capacity {
			values = mlx.PadConstant(values, []int{0}, []int{0}, []int{capacity - n})
		} else if split := n - offset%capacity; split != n {
			values = mlx.Concatenate([]*mlx.Array{
				values.Slice(mlx.Slice(split, n)),
				values.Slice(mlx.Slice(0, split)),
			}, 0)
		}
		oraclePut([]string{"keys", "values"}[i], values)
	}
}

// CaptureOracle uses the exact loaded target, tokenizer and cache classes.
// It feeds explicit singleton teacher tokens after a chunked prompt prefix.
func CaptureOracle(m model.Model, inputPath, outputDir string) error {
	target, ok := m.(*targetModel)
	if !ok {
		return fmt.Errorf("oracle capture requires an Inkling 2 target")
	}
	if outputDir == "" {
		return fmt.Errorf("INKLING_ORACLE_OUTPUT is required")
	}
	data, err := os.ReadFile(inputPath)
	if err != nil {
		return err
	}
	var input struct {
		Prompt    string
		Start     int
		Chunks    []int
		Embedding bool
	}
	if err := json.Unmarshal(data, &input); err != nil {
		return err
	}
	ids := target.tok.Encode(input.Prompt, false)
	if input.Start < 4 || len(ids) < input.Start {
		return fmt.Errorf("invalid oracle prefix extent")
	}
	if len(ids) < input.Start+4 {
		// When using an exactly filled prompt, use its last four IDs again.
		ids = append(ids[:input.Start], ids[input.Start-4:input.Start]...)
	}
	if len(input.Chunks) == 0 {
		return fmt.Errorf("set explicit prefill chunk lengths")
	}
	oracleEmbedding = input.Embedding
	defer func() { oracleEmbedding = false }()
	tokenHash := sha256.New()
	if err := binary.Write(tokenHash, binary.LittleEndian, ids[:input.Start+4]); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "Oracle sequence SHA-256 %x\n", tokenHash.Sum(nil))
	caches := target.NewCaches()
	defer func() {
		for _, c := range caches {
			c.Free()
		}
	}()
	layers := cache.Layers(caches)
	step := func(offset, n int, path string) (err error) {
		mlx.Scoped(func() {
			defer func() {
				oracleCapture = nil
				oracleVisit = 0
			}()
			if path != "" {
				oracleCapture = make(map[string]*mlx.Array)
				if offset == input.Start {
					oracleCapture["start"] = mlx.FromValues([]uint32{uint32(offset)}, 1)
				}
			}
			b := &batch.Batch{
				InputIDs:   mlx.FromValues(ids[offset:offset+n], 1, n),
				SeqOffsets: []int32{int32(offset)}, SeqQueryLens: []int32{int32(n)},
			}
			var state []*mlx.Array
			if path == "" {
				mlx.Scoped(func() {
					target.Forward(b, layers)
				})
			} else {
				hidden, _ := target.Forward(b, layers)
				state = append(state, hidden)
			}
			for _, c := range caches {
				state = append(state, c.State()...)
			}
			mlx.Eval(state...)
			if path != "" {
				err = mlx.SaveSafetensors(path, oracleCapture)
			}
		})
		mlx.ClearCache()
		return err
	}
	offset := 0
	for _, n := range input.Chunks {
		if n <= 0 || offset+n > input.Start {
			return fmt.Errorf("invalid oracle prefill chunk extent")
		}
		if err := step(offset, n, ""); err != nil {
			return err
		}
		offset += n
		fmt.Fprintf(os.Stderr, "Oracle prefill %d/%d\n", offset, input.Start)
	}
	if offset != input.Start {
		return fmt.Errorf("prefill chunks must exactly cover the oracle prefix")
	}
	for row := range 4 {
		path := filepath.Join(outputDir, fmt.Sprintf("row-%d.safetensors", row))
		if err := step(offset+row, 1, path); err != nil {
			return err
		}
	}
	return nil
}
