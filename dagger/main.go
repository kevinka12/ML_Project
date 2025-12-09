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

	// Load project folder (excluding caches)
	src := client.Host().Directory(".", dagger.HostDirectoryOpts{
		Exclude: []string{"__pycache__", "mlruns"},
	})

	// Build & run container
	container := client.Container().
		From("python:3.11-slim").
		WithDirectory("/app", src).
		WithWorkdir("/app").
		WithEnvVariable("PYTHONPATH", "/app").
		WithExec([]string{"pip", "install", "-r", "requirements.txt"}).
		WithExec([]string{"python", "src/run_training_pipeline.py"})

	// Export artifacts folder to ci_artifacts/
	_, err = container.Directory("/app/notebooks/artifacts").Export(ctx, "ci_artifacts")
	if err != nil {
		panic(err)
	}

	fmt.Println("Dagger workflow completed successfully.")
}
