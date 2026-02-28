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
gh release create v0.0.1 'model/checkpoints/model-int8.onnx#model.onnx' -R Siyer2/speech-model
```

# Run frontend

```bash
cd frontend
npm install
npm run dev
```