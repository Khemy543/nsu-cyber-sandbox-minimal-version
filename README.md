# NSU Cyber Sandbox Minimal

This repository is a small, containerized sandbox for experimenting with a simple property-risk classification workflow. It combines:

- a `Jupyter` notebook environment for exploratory analysis and model training
- a `MinIO` service for local S3-compatible object storage
- a sample dataset of troubled property inspection records

The current project centers on the notebook [`notebooks/simple-classifier.ipynb`](/Users/gideon/code/nsu-cyber-sambox-minimal/notebooks/simple-classifier.ipynb), which loads inspection data, cleans it, and trains a decision tree classifier against the `Analysis Rating` label.

## What is in this repo

- [`data/troubled-properties.csv`](/Users/gideon/code/nsu-cyber-sambox-minimal/data/troubled-properties.csv): primary dataset with 677 rows and 14 columns
- [`notebooks/simple-classifier.ipynb`](/Users/gideon/code/nsu-cyber-sambox-minimal/notebooks/simple-classifier.ipynb): notebook for loading data, basic preprocessing, visualization, and decision tree training
- [`docker/docker-compose.yml`](/Users/gideon/code/nsu-cyber-sambox-minimal/docker/docker-compose.yml): local services for MinIO and Jupyter
- [`minio-data/`](/Users/gideon/code/nsu-cyber-sambox-minimal/minio-data): persisted MinIO storage mounted into the `storage` container
- [`configs/`](/Users/gideon/code/nsu-cyber-sambox-minimal/configs), [`src/`](/Users/gideon/code/nsu-cyber-sambox-minimal/src), and the nested README files: currently placeholders for future expansion

## Dataset summary

The sample dataset contains property inspection and condition signals such as:

- `City`
- `Zip Code`
- `Inspection Frequency`
- `Compliant`
- `Unit Count`
- `Units Inspected`
- `Average Violations Per Unit`
- `Severity Index`
- `No Violations Observed`
- `Infested Units Percentage`
- `Units with Mold`
- `Analysis Rating`

Current label distribution in [`data/troubled-properties.csv`](/Users/gideon/code/nsu-cyber-sambox-minimal/data/troubled-properties.csv):

- `compliant`: 417
- `at-risk`: 114
- `troubled`: 92
- `TBD`: 54

The notebook filters out rows where `Analysis Rating == TBD` before training.

## Quick start

### Prerequisites

- Docker
- Docker Compose

### Start the environment

From the repository root:

```bash
docker compose -f docker/docker-compose.yml up
```

This starts two services:

- `storage`: MinIO S3-compatible storage
- `model-trainer`: Jupyter SciPy notebook environment

### Service URLs

- Jupyter Notebook: `http://localhost:8888`
- Jupyter token: `admin`
- MinIO API: `http://localhost:9100`
- MinIO Console: `http://localhost:9101`
- MinIO username: `admin`
- MinIO password: `password123`

## How the notebook works

The notebook currently does the following:

1. Connects to MinIO using `boto3` at `http://storage:9000`
2. Ensures a bucket named `datasets` exists
3. Optionally uploads the local CSV into MinIO
4. Loads the troubled-properties data into pandas
5. Drops duplicates and fills selected numeric nulls with column means
6. Removes records labeled `TBD`
7. Visualizes class and feature relationships
8. Encodes labels and trains a `DecisionTreeClassifier`

Inside the `model-trainer` container, the mounted paths are:

- `/home/jovyan/work`: notebook files from [`notebooks/`](/Users/gideon/code/nsu-cyber-sambox-minimal/notebooks)
- `/home/jovyan/data`: local datasets from [`data/`](/Users/gideon/code/nsu-cyber-sambox-minimal/data)

If you need the CSV available in MinIO for the first time, the notebook already includes a commented `s3.upload_file(...)` example that targets `/home/jovyan/data/troubled-properties.csv`.

## Repository layout

```text
.
├── README.md
├── configs/
├── data/
│   ├── README.md
│   └── troubled-properties.csv
├── docker/
│   ├── README.md
│   └── docker-compose.yml
├── minio-data/
├── notebooks/
│   ├── README.md
│   └── simple-classifier.ipynb
└── src/
```

## Notes

- The top-level workflow is notebook-first; there is not yet a packaged training script or application entrypoint in `src/`.
- The checked-in `minio-data/` directory stores local MinIO state and may change as containers run.
- Several nested README files are still placeholders and can be expanded later if the project grows beyond this minimal sandbox.
