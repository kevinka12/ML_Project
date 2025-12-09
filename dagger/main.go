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

    // Load repository but exclude heavy artifacts folder
    src := client.Host().Directory(".", dagger.HostDirectoryOpts{
        Exclude: []string{"notebooks/artifacts"},
    })

    container := client.Container().
        From("python:3.11-slim").
        WithDirectory("/app", src).
        WithWorkdir("/app"). // IMPORTANT FIX
        WithEnvVariable("PYTHONPATH", "/app").
        WithExec([]string{"pip", "install", "-r", "requirements.txt"}).
        WithExec([]string{"python", "src/run_training_pipeline.py"})

    // Export artifacts back to host
    _, err = container.Directory("/app/notebooks/artifacts").Export(ctx, "notebooks/artifacts")
    if err != nil {
        panic(err)
    }

    fmt.Println("Dagger workflow completed successfully.")
}
