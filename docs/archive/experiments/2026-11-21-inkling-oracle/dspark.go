// SPDX-License-Identifier: MIT
// Build this file inside the pinned Ollama mlxrunner/model/inkling2 package.
package inkling2

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/model"
)

// CaptureDSpark uses the original loaded draft and a small pinned-runtime
// prefix. The output tensors are private.
func CaptureDSpark(m model.Model, configPath, draftPath, inputPath, output string) error {
	target, ok := m.(*targetModel)
	if !ok || configPath == "" || draftPath == "" || output == "" {
		return fmt.Errorf("DSpark oracle requires an Inkling target, draft and private output path")
	}
	config, err := os.ReadFile(configPath)
	if err != nil {
		return err
	}
	var cfg struct {
		Draft draftConfig `json:"mtp_config"`
	}
	if err := json.Unmarshal(config, &cfg); err != nil {
		return err
	}
	if cfg.Draft.Block != 8 || cfg.Draft.NumLayers != 3 {
		return fmt.Errorf("DSpark oracle requires the original draft geometry")
	}
	data, err := os.ReadFile(inputPath)
	if err != nil {
		return err
	}
	var input struct {
		Prompt string
		Start  int
	}
	if err := json.Unmarshal(data, &input); err != nil {
		return err
	}
	ids := target.tok.Encode(input.Prompt, false)
	if input.Start < 4 || input.Start > 64 || input.Start > len(ids) {
		return fmt.Errorf("DSpark oracle prefix must contain 4 to 64 rows")
	}
	ids = ids[:input.Start]
	row := func(offset, count int) *batch.Batch {
		return &batch.Batch{SeqOffsets: []int32{int32(offset)}, SeqQueryLens: []int32{int32(count)}}
	}
	free := func(states []cache.Cache) {
		for _, state := range states {
			state.Free()
		}
	}
	mlx.Scoped(func() {
		d := &draft{config: cfg.Draft, target: target}
		weights := make(map[string]*mlx.Array)
		for key, value := range mlx.Load(draftPath) {
			weights[key] = value
		}
		if err = d.LoadWeights(weights); err != nil {
			return
		}
		targetCache := target.NewCaches()
		defer free(targetCache)
		prefix := row(0, len(ids))
		prefix.InputIDs = mlx.FromValues(ids, 1, len(ids))
		hidden, taps := target.Forward(prefix, cache.Layers(targetCache))
		last := hidden.Slice(mlx.Slice(), mlx.Slice(len(ids)-1, len(ids)), mlx.Slice())
		anchors := []*mlx.Array{target.Unembed(last).Argmax(-1, false).Reshape(1).AsType(mlx.DTypeInt32)}
		for _, id := range ids[:3] {
			anchors = append(anchors, mlx.FromValues([]int32{id}, 1))
		}
		proposals := make([]*mlx.Array, 0, len(anchors))
		hiddenRows := make([]*mlx.Array, 0, len(anchors))
		latentRows := make([]*mlx.Array, 0, len(anchors))
		var feature *mlx.Array
		for _, anchor := range anchors {
			state := d.NewCaches()
			defer free(state)
			probe := row(0, d.config.Block)
			probe.Hidden = taps
			masks := mlx.Zeros(mlx.DTypeInt32, 1, d.config.Block-1)
			probe.InputIDs = anchor.ExpandDims(1).Concatenate(1, masks)
			h, _ := d.Forward(probe, nil, cache.Layers(state))
			h = h.Slice(mlx.Slice(), mlx.Slice(1, d.config.Block), mlx.Slice())
			logits := d.Unembed(h).Slice(mlx.Slice(), mlx.Slice(0, 1), mlx.Slice())
			h = h.Slice(mlx.Slice(), mlx.Slice(0, 1), mlx.Slice())
			hidden := h.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, d.config.Width)).Reshape(1, d.config.Width)
			feature = h.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(d.config.Width, 2*d.config.Width)).Reshape(1, d.config.Width)
			zero := mlx.Zeros(mlx.DTypeBFloat16, 1, d.config.StateWidth)
			input := mlx.Concatenate([]*mlx.Array{zero, d.Previous.Forward(anchor), hidden, feature}, -1)
			latent := d.Recurrent.Forward(input).Slice(mlx.Slice(), mlx.Slice(2*d.config.StateWidth, 3*d.config.StateWidth)).Tanh()
			hiddenRows = append(hiddenRows, hidden)
			latentRows = append(latentRows, latent)
			proposal, _, _ := d.Select(h, logits, anchor, func(ids, scores *mlx.Array) (*mlx.Array, *mlx.Array) {
				return ids.TakeAlongAxis(scores.Argmax(-1, true), -1).Squeeze(-1), scores
			})
			proposals = append(proposals, proposal.Reshape(1))
		}
		err = mlx.SaveSafetensors(output, map[string]*mlx.Array{
			"tokens":             prefix.InputIDs.Reshape(len(ids)).AsType(mlx.DTypeUint32),
			"taps":               taps,
			"anchors":            mlx.Concatenate(anchors, 0).AsType(mlx.DTypeUint32),
			"proposals":          mlx.Concatenate(proposals, 0).AsType(mlx.DTypeUint32),
			"draft_hidden":       mlx.Concatenate(hiddenRows, 0),
			"draft_latent":       mlx.Concatenate(latentRows, 0),
			"draft_feature":      feature.Reshape(d.config.Width),
			"folded_checks":      d.Folded.Weight.TakeAxis(mlx.Concatenate(anchors, 0), 0),
			"folded_bias_checks": d.Folded.Bias.TakeAxis(mlx.Concatenate(anchors, 0), 0),
		})
	})
	return err
}
