# How to reproduce

## build environment

```sh
docker compose up
```

## onnx is ok

```sh
python test-onnx.py
```

## trtexec produces nan and inf

```sh
trtexec --onnx=repro.onnx --fp16 --dumpOutput
```

## convert model without InstanceNormalization

```sh
# generate "repro_noIN.onnx"
python reconstruct_IN.py
```

## test reconstructed onnx has same result of original

```sh
python test-onnx-noIN.py
```

## trtexec by reconstructed not produces nan and inf

```sh
trtexec --onnx=repro_noIN.onnx --fp16 --dumpOutput
```
