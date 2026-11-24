// SPDX-License-Identifier: MIT
// This validation-only shim builds inside the pinned Ollama mlxrunner package.
package mlxrunner

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/model"
)

func captureInklingVision(loaded model.Model, images, output string) error {
	media, ok := loaded.(model.MediaModel)
	if !ok {
		return fmt.Errorf("model has no media path")
	}
	entries, err := os.ReadDir(images)
	if err != nil {
		return fmt.Errorf("cannot enumerate image fixtures")
	}
	for _, entry := range entries {
		if entry.IsDir() || !(strings.HasSuffix(entry.Name(), ".png") || strings.HasSuffix(entry.Name(), ".jpg")) {
			continue
		}
		encoded, err := os.ReadFile(filepath.Join(images, entry.Name()))
		if err != nil {
			return fmt.Errorf("cannot read image fixture")
		}
		request, err := media.PrepareMedia([]model.Segment{{Kind: "image", Data: encoded}})
		if err != nil || len(request.Items) != 1 {
			return fmt.Errorf("oracle media preparation failed")
		}
		item := &request.Items[0]
		mlx.Scoped(func() {
			pixels := mlx.FromValues(item.MediaData, item.Dims...)
			features := media.EncodeMedia(item, pixels)
			mlx.Eval(features)
			err = mlx.SaveSafetensors(filepath.Join(output, entry.Name()+".safetensors"), map[string]*mlx.Array{
				"features": features,
			})
		})
		if err != nil {
			return fmt.Errorf("cannot write private vision oracle")
		}
	}
	return nil
}
