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

	// Load entire repo
	src := client.Host().Directory(".")

	// Build container
	container := client.Container().
		From("python:3.11-slim").
		WithDirectory("/app", src).
		WithWorkdir("/app").
		WithEnvVariable("PYTHONPATH", "/app").
		WithExec([]string{"pip", "install", "-r", "requirements.txt"}).
		WithExec([]string{"mkdir", "-p", "notebooks/artifacts"}).
		WithExec([]string{"python", "src/run_training_pipeline.py"})

	// Materialize directory object
	artifactDir := container.Directory("/app/notebooks/artifacts")

	// Export to a FOLDER IN REPOSITORY (CI-safe)
	exportPath := "ci_artifacts"

	os.RemoveAll(exportPath) // avoid conflicts in CI
	err = os.MkdirAll(exportPath, 0o755)
	if err != nil {
		panic(err)
	}

	_, err = artifactDir.Export(ctx, exportPath)
	if err != nil {
		panic(err)
	}

	fmt.Println("Artifacts exported to:", exportPath)
	fmt.Println("Dagger workflow completed successfully.")
}
