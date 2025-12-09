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

	// Load full repo
	src := client.Host().Directory(".", dagger.HostDirectoryOpts{})

	// Build training container
	container := client.Container().
		From("python:3.11-slim").
		WithDirectory("/app", src).
		WithWorkdir("/app").
		WithEnvVariable("PYTHONPATH", "/app").
		WithExec([]string{"mkdir", "-p", "notebooks/artifacts"}).
		WithExec([]string{"pip", "install", "-r", "requirements.txt"}).
		WithExec([]string{"python", "src/run_training_pipeline.py"})

	// Export artifacts to /tmp inside the GitHub runner
	_, err = container.Directory("/app/notebooks/artifacts").Export(ctx, "/tmp/model_artifacts")
	if err != nil {
		panic(err)
	}

	fmt.Println("Dagger workflow completed successfully.")
}
