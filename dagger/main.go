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

	// Load full repository WITHOUT excluding anything
	src := client.Host().Directory(".")

	// Container setup
	container := client.Container().
		From("python:3.11-slim").
		WithDirectory("/app", src).
		WithWorkdir("/app").
		WithEnvVariable("PYTHONPATH", "/app").
		WithExec([]string{"pip", "install", "-r", "requirements.txt"}).
		WithExec([]string{"mkdir", "-p", "notebooks/artifacts"}).
		WithExec([]string{"python", "src/run_training_pipeline.py"})

	// Export artifacts to /tmp (CI-friendly)
	exportPath := "/tmp/model_artifacts"
	_, err = container.Directory("/app/notebooks/artifacts").Export(ctx, exportPath)
	if err != nil {
		panic(err)
	}

	fmt.Println("Artifacts exported to:", exportPath)
	fmt.Println("Dagger workflow completed successfully.")
}
