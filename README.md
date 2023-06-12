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
