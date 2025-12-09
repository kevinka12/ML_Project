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

	// Load repo
	src := client.Host().Directory(".", dagger.HostDirectoryOpts{})

	// Build container
	container := client.Container().
		From("python:3.11-slim").
		WithDirectory("/app", src).
		WithWorkdir("/app").
		WithEnvVariable("PYTHONPATH", "/app").
		WithExec([]string{"mkdir", "-p", "notebooks/artifacts"}).
		WithExec([]string{"pip", "install", "-r", "requirements.txt"}).
		WithExec([]string{"python", "src/run_training_pipeline.py"})

	// Export to a NEW, non-existing folder
	_, err = container.Directory("/app/notebooks/artifacts").Export(ctx, "./exported_artifacts")
	if err != nil {
		panic(err)
	}

	fmt.Println("Dagger workflow complete.")
}
