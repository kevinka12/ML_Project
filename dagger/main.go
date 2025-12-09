package main

import (
	"context"
	"fmt"
	"os"

	"dagger.io/dagger"
)

func main() {
	ctx := context.Background()

	// Connect to Dagger engine
	client, err := dagger.Connect(ctx, dagger.WithLogOutput(os.Stdout))
	if err != nil {
		panic(err)
	}
	defer client.Close()

	fmt.Println("Starting Dagger workflow...")

	// Load local project directory
	src := client.Host().Directory(".")

	// Create container for training pipeline
	container := client.Container().
		From("python:3.11-slim").
		WithDirectory("/app", src).
		WithWorkdir("/app").
		WithEnvVariable("PYTHONPATH", "/app").
		// Fetch raw dataset (DVC remote) so the training script can read it
		WithExec([]string{
			"sh", "-c",
			"mkdir -p notebooks/artifacts && " +
				"curl -L https://raw.githubusercontent.com/Jeppe-T-K/itu-sdse-project-data/refs/heads/main/raw_data.csv " +
				"-o notebooks/artifacts/raw_data.csv",
		}).
		WithExec([]string{"pip", "install", "-r", "requirements.txt"}).
		WithExec([]string{"python", "src/run_training_pipeline.py"})

	// Export artifacts to host
	_, err = container.
		Directory("artifacts").
		Export(ctx, "notebooks/artifacts")

	if err != nil {
		panic(err)
	}

	fmt.Println("Dagger workflow completed successfully.")
}
