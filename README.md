# Project overview
## Who is this for
This repository contains the exam project for the course Data Science in Production: MLOps and Software Engineering. 

This project is intended for the teaching staff and validators reviewing the exam submission.

## What does this do
This repository refactors a previously developed  jupyter notebook from a prior project and is a re-written example of what an actual production model includes. 
The pipeline trains a model in a containerized environment using a Dagger workflow.

In this project the Dagger workflow is executed inside a GitHub Actions runner to ensure the reproducibility. The workflow installs all the required dependencies, and executes the training process and will produce a trained model artifact.

## Repository structure 

.
├── README.md
├── dagger
│   └── main.go
├── docs
│   ├── diagrams.excalidraw
│   └── project-architecture.png
├── go.mod
├── go.sum
├── notebooks
│   ├── artifacts
│   ├── main.ipynb
│   └── model_inference.py
├── requirements.txt
└── src
    ├──__init__.py
    ├── data      
    ├── features
    ├── models                                    
    └── run_training_pipeline.py         #main pipeline orchestrator

The repository follows a standard MLOps project structure as well as a cookiecutter template.
The root directory contains configuration files (go.mod, go.sum, requirements.txt) and documentation (README.md). The src/ directory contains the modularized Python codebase, organized into data/ (data loading and preprocessing), features/ (feature engineering), and models/ (model training and evaluation), with the main pipeline orchestrator in run_training_pipeline.py. 
The dagger/ directory contains the Go implementation of the containerized Dagger workflow (main.go).  .github/workflows/ defines the GitHub Actions CI/CD pipeline (train_and_validate.yml). 
The notebooks/ directory preserves the original Jupyter notebook (main.ipynb) for reference. The docs/ directory includes project architecture diagrams and visual documentation. 

## Dagger workflow
By using Dagger, we’re executing the model training inside a containerized environment. The implementation of the Dagger workflow is inside: dagger/main.go.

The Dagger workflow is performing the following steps:

It builds a Python 3.11 container.
Mounts the project source code into the container.
Fetches the raw dataset. 
Installing all the Python dependencies.
Runs the training pipeline.
Exports generated artifacts back to the host environment. 

## Github Actions
As mentioned above, the automation is handled through GitHub Actions. The training workflow is defined in the following file:
.github/workflows/train_and_validate.yml

## How do you run it
The project is structured around a Dagger workflow that executes the Dagger training pipeline, collects the trained model artifact, and lastly it uploads the model. 

 After the training part, the uploaded model artifact is being downloaded in a separate job and then it’s being validated by using the provided model-validator GitHub Action. This done to ensure the following:

That the model artifact is produced correctly.
And the model passes the required inference tests.

## Output
The changes are being pushed to the main branch, or by triggering the workflow manually, which we did many times to test the workflow. The the GitHub Actions will do the following:

Run the Dagger workflow
Uploading the trained model
And validate the model automatically 

The output of this project is a trained machine learning model. The output was a Logistic Regression model. The automated training pipeline is producing and uploading the model as a GitHub Actions artifact called “model”. 
