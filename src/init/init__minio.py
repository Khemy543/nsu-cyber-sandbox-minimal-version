import time
import boto3

BUCKETS = ["datasets", "models"]

# wait for MinIO to be ready
time.sleep(5)

s3 = boto3.client(
    "s3",
    endpoint_url="http://storage:9000",
    aws_access_key_id="admin",
    aws_secret_access_key="password123",
)

existing_buckets = [b["Name"] for b in s3.list_buckets()["Buckets"]]

for bucket in BUCKETS:
    if bucket not in existing_buckets:
        s3.create_bucket(Bucket=bucket)
        print(f"Created bucket: {bucket}")
    else:
        print(f"Bucket already exists: {bucket}")

print("MinIO initialization complete.")