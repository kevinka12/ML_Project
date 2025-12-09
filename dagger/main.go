package main

import (
	"context"
	"fmt"
	"os"

	"dagger.io/dagger"
)

func main() {
	ctx := context.Background()

	// Connect to Dagger
	client, err := dagger.Connect(ctx, dagger.WithLogOutput(os.Stdout))
	if err != nil {
		panic(err)
	}
	defer client.Close()

	fmt.Println("Starting Dagger workflow...")

	// Load entire repo – NO EXCLUDES!
	src := client.Host().Directory(".")

	// Build container
	container := client.Container().
		From("python:3.11-slim").
		WithDirectory("/app", src).
		WithWorkdir("/app").
		WithExec([]string{"pip", "install", "-r", "requirements.txt"}).
		WithExec([]string{"mkdir", "-p", "notebooks/artifacts"}). // <--- ADD THIS
		WithExec([]string{"python", "src/run_training_pipeline.py"})

	// Export artifacts after training
	_, err = container.Directory("/app/notebooks/artifacts").Export(ctx, "ci_artifacts")
	if err != nil {
		panic(err)
	}

	fmt.Println("Dagger workflow completed successfully.")
}
