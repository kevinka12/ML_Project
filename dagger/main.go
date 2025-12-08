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
		WithWorkdir("/app/notebooks"). // ← IMPORTANT: DVC ROOT
		WithEnvVariable("PYTHONPATH", "/app").
		WithExec([]string{"pip", "install", "-r", "../requirements.txt"}). // pipeline reqs
		WithExec([]string{"pip", "install", "dvc[s3]"}). // install DVC engine
		WithExec([]string{"dvc", "pull"}).               // let DVC restore raw_data.csv
		WithExec([]string{"python", "../src/run_training_pipeline.py"}) // run pipeline from root

	// Export artifacts
	// DEBUG: list directory contents inside container
	container = container.WithExec([]string{"sh", "-c", "echo '---- Listing notebooks ----' && ls -R /app/notebooks"})
	container = container.WithExec([]string{"sh", "-c", "echo '---- Listing artifacts (expected) ----' && ls -R /app/notebooks/artifacts || echo 'artifacts folder DOES NOT EXIST'"})

	_, err = container.
		Directory("notebooks/artifacts").           
		Export(ctx, "./notebooks/artifacts") 
	if err != nil {
		panic(err)
	}

	fmt.Println("Dagger workflow completed successfully.")
}
