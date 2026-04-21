.PHONY: help preprocess train evaluate explain full ids-pipeline check-raw

COMPOSE_FILE ?= docker/docker-compose.yml
SERVICE ?= intrusion-detection-system
RUN_IDS = docker compose -f $(COMPOSE_FILE) run --rm $(SERVICE) python /app/src/main.py

# Container-visible paths
RAW_TRAIN_PATH ?= /app/data/intrusion-detection-system/raw/KDDTrain+.txt
RAW_TEST_PATH ?= /app/data/intrusion-detection-system/raw/KDDTest+.txt
TRAIN_PATH ?= /app/data/intrusion-detection-system/nsl-kdd/train.csv
TEST_PATH ?= /app/data/intrusion-detection-system/nsl-kdd/test.csv
RESULTS_DIR ?= /app/results/intrusion-detection-system
ARTIFACTS_DIR ?= /app/results/intrusion-detection-system/artifacts

# Host paths used for local preflight checks
LOCAL_RAW_TRAIN ?= data/intrusion-detection-system/raw/KDDTrain+.txt
LOCAL_RAW_TEST ?= data/intrusion-detection-system/raw/KDDTest+.txt

# Shared options
DOWNLOAD_FROM_MINIO ?= false

# Preprocess options
PREPROCESS_SELECTION_METHOD ?= info_gain
PREPROCESS_TOP_K ?= 15
PREPROCESS_TARGET_COLUMN ?= binary_attack
PREPROCESS_SAVE_SCALER ?= true
PREPROCESS_SAVE_CLASS_NAMES ?= true

# Train/eval/explain options
USE_SMOTE ?= false
USE_CLASS_WEIGHTS ?= true
EXPLAIN_SAMPLES ?= 5

help:
	@echo "IDS Make Targets:"
	@echo "  make preprocess     # Raw NSL-KDD txt -> preprocessed train/test CSV"
	@echo "  make train          # Train model using preprocessed train/test CSV"
	@echo "  make evaluate       # Evaluate using saved model artifacts"
	@echo "  make explain        # Run explainability using saved model artifacts"
	@echo "  make full           # Single-command full mode from app (train+eval+explain)"
	@echo "  make ids-pipeline   # Sequential: preprocess -> train -> evaluate -> explain"
	@echo ""
	@echo "Common overrides:"
	@echo "  DOWNLOAD_FROM_MINIO=true|false"
	@echo "  PREPROCESS_SELECTION_METHOD=none|kbest|info_gain"
	@echo "  PREPROCESS_TOP_K=15"
	@echo "  PREPROCESS_TARGET_COLUMN=label|binary_attack"

check-raw:
	@if [ "$(DOWNLOAD_FROM_MINIO)" != "true" ]; then \
		test -f "$(LOCAL_RAW_TRAIN)" || (echo "Missing raw train file: $(LOCAL_RAW_TRAIN)" && exit 1); \
		test -f "$(LOCAL_RAW_TEST)" || (echo "Missing raw test file: $(LOCAL_RAW_TEST)" && exit 1); \
	fi

preprocess: check-raw
	$(RUN_IDS) \
	  --mode preprocess \
	  --download-from-minio $(DOWNLOAD_FROM_MINIO) \
	  --raw-train-path $(RAW_TRAIN_PATH) \
	  --raw-test-path $(RAW_TEST_PATH) \
	  --train-path $(TRAIN_PATH) \
	  --test-path $(TEST_PATH) \
	  --results-dir $(RESULTS_DIR) \
	  --artifacts-dir $(ARTIFACTS_DIR) \
	  --preprocess-selection-method $(PREPROCESS_SELECTION_METHOD) \
	  --preprocess-top-k $(PREPROCESS_TOP_K) \
	  --preprocess-target-column $(PREPROCESS_TARGET_COLUMN) \
	  --preprocess-save-scaler $(PREPROCESS_SAVE_SCALER) \
	  --preprocess-save-class-names $(PREPROCESS_SAVE_CLASS_NAMES)

train:
	$(RUN_IDS) \
	  --mode train \
	  --download-from-minio $(DOWNLOAD_FROM_MINIO) \
	  --train-path $(TRAIN_PATH) \
	  --test-path $(TEST_PATH) \
	  --results-dir $(RESULTS_DIR) \
	  --artifacts-dir $(ARTIFACTS_DIR) \
	  --use-smote $(USE_SMOTE) \
	  --use-class-weights $(USE_CLASS_WEIGHTS)

evaluate:
	$(RUN_IDS) \
	  --mode evaluate \
	  --download-from-minio $(DOWNLOAD_FROM_MINIO) \
	  --train-path $(TRAIN_PATH) \
	  --test-path $(TEST_PATH) \
	  --results-dir $(RESULTS_DIR) \
	  --artifacts-dir $(ARTIFACTS_DIR)

explain:
	$(RUN_IDS) \
	  --mode explain \
	  --download-from-minio $(DOWNLOAD_FROM_MINIO) \
	  --train-path $(TRAIN_PATH) \
	  --test-path $(TEST_PATH) \
	  --results-dir $(RESULTS_DIR) \
	  --artifacts-dir $(ARTIFACTS_DIR) \
	  --explain-samples $(EXPLAIN_SAMPLES)

full:
	$(RUN_IDS) \
	  --mode full \
	  --download-from-minio $(DOWNLOAD_FROM_MINIO) \
	  --train-path $(TRAIN_PATH) \
	  --test-path $(TEST_PATH) \
	  --results-dir $(RESULTS_DIR) \
	  --artifacts-dir $(ARTIFACTS_DIR)

ids-pipeline: preprocess train evaluate explain
	@echo "Completed IDS pipeline: preprocess -> train -> evaluate -> explain"
