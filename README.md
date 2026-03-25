# NSU Cyber Sandbox Minimal

This repository is a small, containerized sandbox for testing data workflows in a local environment. It combines:

- a `Jupyter` notebook environment for experimentation
- a `MinIO` service for local S3-compatible object storage
- mounted project data for testing

This is a minimal test setup intended for trying out notebooks, data movement, storage integration, and future model experiments.

## What is in this repo

- [`data/`](/Users/gideon/code/nsu-cyber-sambox-minimal/data): local data directory for sandbox inputs
- [`notebooks/simple-classifier.ipynb`](/Users/gideon/code/nsu-cyber-sambox-minimal/notebooks/simple-classifier.ipynb): example notebook for testing the environment
- [`docker/docker-compose.yml`](/Users/gideon/code/nsu-cyber-sambox-minimal/docker/docker-compose.yml): local services for MinIO and Jupyter
- [`minio-data/`](/Users/gideon/code/nsu-cyber-sambox-minimal/minio-data): persisted MinIO storage mounted into the `storage` container
- [`configs/`](/Users/gideon/code/nsu-cyber-sambox-minimal/configs), [`src/`](/Users/gideon/code/nsu-cyber-sambox-minimal/src), and the nested README files: currently placeholders for future expansion

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
- Jupyter token: configured in [`docker/docker-compose.yml`](/Users/gideon/code/nsu-cyber-sambox-minimal/docker/docker-compose.yml)
- MinIO API: `http://localhost:9100`
- MinIO Console: `http://localhost:9101`
- MinIO credentials: configured in [`docker/docker-compose.yml`](/Users/gideon/code/nsu-cyber-sambox-minimal/docker/docker-compose.yml)

## Working in the sandbox

The current notebook setup is meant to support general experimentation. You can use it to:

1. open notebooks in Jupyter
2. access local data from the mounted `data/` directory
3. connect to MinIO for S3-compatible storage tests
4. prototype analysis and future model workflows

Inside the `model-trainer` container, the mounted paths are:

- `/home/jovyan/work`: notebook files from [`notebooks/`](/Users/gideon/code/nsu-cyber-sambox-minimal/notebooks)
- `/home/jovyan/data`: local datasets from [`data/`](/Users/gideon/code/nsu-cyber-sambox-minimal/data)

If you want to test object storage flows, you can upload files from `/home/jovyan/data` into MinIO from inside the notebook environment.

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

- This repository is intentionally minimal and currently serves as a test sandbox.
- The top-level workflow is notebook-first; there is not yet a packaged application entrypoint in `src/`.
- The checked-in `minio-data/` directory stores local MinIO state and may change as containers run.
- Several nested README files are still placeholders and can be expanded later if the project grows beyond this minimal sandbox.
