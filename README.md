# Speech Model

## Getting data

### Download from S3
```bash
aws s3 cp s3://speech-model-data/data.tar.gz data/ --profile personal
```

### Extract
```bash
tar -xzf model/data/data.tar.gz 
```


## Training

```bash
# Install
make install

# Train
make train NAME="train-name" NOTE="some description"

# Test
make test
```

## Release model
```bash
aws s3 cp model/checkpoints/model-int8.onnx s3://speech-model-data/model.onnx --profile personal
```

# Run frontend

```bash
cd frontend
npm install
npm run dev
```