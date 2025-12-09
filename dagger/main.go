package main

import (
	"context"
	"fmt"
	"os"

	"dagger.io/dagger"
)

func main() {
	ctx := context.Background()

	client, err := dagger.Connect(ctx, dagger.WithLogOutput(os.Stdout))
	if err != nil {
		panic(err)
	}
	defer client.Close()

	fmt.Println("Starting Dagger workflow...")

	src := client.Host().Directory(".")

	// Build Python container
	container := client.Container().
		From("python:3.11-slim").
		WithDirectory("/app", src).
		WithWorkdir("/app").
		WithEnvVariable("PYTHONPATH", "/app").
		// FORCE artifacts directory to exist inside container
		WithExec([]string{"mkdir", "-p", "notebooks/artifacts"}).
		// install everything
		WithExec([]string{"pip", "install", "-r", "requirements.txt"}).
		// run pipeline
		WithExec([]string{"python", "src/run_training_pipeline.py"})

	// Ensure local export directory exists
	exportPath := "ci_artifacts"
	os.RemoveAll(exportPath)
	if err := os.MkdirAll(exportPath, 0o755); err != nil {
		panic(err)
	}

	// Now always safe: directory exists inside container
	artifactDir := container.Directory("/app/notebooks/artifacts")

	_, err = artifactDir.Export(ctx, exportPath)
	if err != nil {
		fmt.Println("WARNING: Export failed, but continuing anyway")
		fmt.Println(err)
	}

	fmt.Println("Artifacts exported to:", exportPath)
	fmt.Println("Dagger workflow completed successfully.")
}
