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

	// Load entire repo
	src := client.Host().Directory(".")

	// Create container
	container := client.Container().
		From("python:3.11-slim").
		WithDirectory("/app", src).
		WithWorkdir("/app/notebooks").
		WithEnvVariable("PYTHONPATH", "/app").
		WithExec([]string{"pip", "install", "-r", "../requirements.txt"}). 
		WithExec([]string{"python", "../src/run_training_pipeline.py"}) 

	// Debug logging to see the artifacts
	container = container.WithExec([]string{"sh", "-c", "echo '---- Listing notebooks ----' && ls -R /app/notebooks"})
	container = container.WithExec([]string{"sh", "-c", "echo '---- Listing artifacts ----' && ls -R /app/notebooks/artifacts || echo 'artifacts folder DOES NOT EXIST'"})

	// Export artifacts from container to host
	_, err = container.
		Directory("/app/artifacts").
		Export(ctx, "./notebooks/artifacts")
	if err != nil {
		panic(err)
	}

	fmt.Println("Dagger workflow completed successfully.")
}
