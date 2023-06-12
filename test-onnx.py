import onnxruntime
import numpy

session = onnxruntime.InferenceSession("repro.onnx", providers=["CPUExecutionProvider"])
result = session.run(
    ["/down_blocks.0/resnets.0/norm1/InstanceNormalization_output_0"],
    {
        "/down_blocks.0/resnets.0/norm1/Reshape_output_0": numpy.random.randn(
            2, 32, 64, 64
        ).astype(numpy.float16)
    },
)
print(
    {
        "isnan": numpy.any(numpy.isnan(result[0])),
        "isinf": numpy.any(numpy.isinf(result[0])),
    }
)
