from __future__ import annotations

from pathlib import Path

import boto3
from botocore.exceptions import ClientError, EndpointConnectionError, NoCredentialsError


def _build_client(endpoint_url: str, access_key: str, secret_key: str):
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )


def _candidate_keys(primary_key: str, fallback_keys: list[str]) -> list[str]:
    keys = [primary_key]
    for item in fallback_keys:
        if item and item not in keys:
            keys.append(item)
    return keys


def ensure_dataset_file(
    file_path: Path,
    download_from_minio: bool,
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    minio_bucket: str,
    minio_key: str,
    fallback_keys: list[str],
) -> Path:
    if file_path.exists():
        return file_path

    if not download_from_minio:
        raise FileNotFoundError(
            f"Dataset file not found: {file_path}. "
            "Enable --download-from-minio or place the file locally."
        )

    file_path.parent.mkdir(parents=True, exist_ok=True)
    client = _build_client(minio_endpoint, minio_access_key, minio_secret_key)
    tried = []
    for key in _candidate_keys(minio_key, fallback_keys):
        tried.append(key)
        try:
            client.download_file(minio_bucket, key, str(file_path))
            print(f"Downloaded s3://{minio_bucket}/{key} -> {file_path}", flush=True)
            return file_path
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code", "")
            if error_code not in {"404", "NoSuchKey", "NoSuchBucket"}:
                raise RuntimeError(
                    f"MinIO download failed for s3://{minio_bucket}/{key}: {error_code}"
                ) from exc
        except (EndpointConnectionError, NoCredentialsError) as exc:
            raise RuntimeError(
                f"Unable to connect to MinIO endpoint {minio_endpoint}: {exc}"
            ) from exc

    raise FileNotFoundError(
        "Could not locate dataset in MinIO. "
        f"Bucket={minio_bucket}, tried keys={tried}, destination={file_path}"
    )
