#!/usr/bin/env bash

set -euo pipefail

BUCKET="speech-model-data"
OBJECT_KEY="model.onnx"
PROFILE="personal"

echo "Creating temporary policy file..."

cat > /tmp/s3_public_model_policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "PublicReadOnlyModelOnnx",
      "Effect": "Allow",
      "Principal": "*",
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::${BUCKET}/${OBJECT_KEY}"
    }
  ]
}
EOF

echo "Disabling only public bucket policy block (keeping ACL blocks enabled)..."

aws s3api put-public-access-block \
  --bucket "${BUCKET}" \
  --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=false,RestrictPublicBuckets=false \
  --profile "${PROFILE}"

echo "Applying bucket policy..."

aws s3api put-bucket-policy \
  --bucket "${BUCKET}" \
  --policy file:///tmp/s3_public_model_policy.json \
  --profile "${PROFILE}"

echo "Applying CORS configuration..."

aws s3api put-bucket-cors \
  --bucket "${BUCKET}" \
  --cors-configuration '{
    "CORSRules": [
      {
        "AllowedOrigins": ["*"],
        "AllowedMethods": ["GET", "HEAD"],
        "AllowedHeaders": ["*"],
        "MaxAgeSeconds": 300
      }
    ]
  }' \
  --profile "${PROFILE}"

echo "Done."