import onnx_graphsurgeon
import onnx
import numpy

model = onnx.load("repro.onnx")
gs_graph = onnx_graphsurgeon.import_onnx(model)

for i in reversed(range(len(gs_graph.nodes))):
    if gs_graph.nodes[i].op == "InstanceNormalization":
        node = gs_graph.nodes.pop(i)

        ### y = scale * (x - mean) / sqrt(variance + epsilon) + B ###
        # by: https://github.com/onnx/onnx/blob/main/docs/Operators.md#instancenormalization

        input = node.inputs[0]
        scale = node.inputs[1]
        b = node.inputs[2]
        output = node.outputs[0]

        input.outputs = []
        scale.outputs = []
        b.outputs = []
        output.inputs = []

        dims_input = len(input.shape)

        epsilon = onnx_graphsurgeon.Constant(f"{node.name}/Constant_0", numpy.array(node.attrs["epsilon"], dtype=numpy.float16))
        axis = onnx_graphsurgeon.Constant(f"{node.name}/Constant_1", numpy.array(range(2, dims_input), dtype=numpy.int64))
        dim_ones = onnx_graphsurgeon.Constant(f"{node.name}/Constant_2", numpy.array((-1,) + (1,) * (dims_input - 2), dtype=numpy.int64))
        power_2 = onnx_graphsurgeon.Constant(f"{node.name}/Constant_3", numpy.array(2, dtype=numpy.float16))

        reduce_mean_0_output = onnx_graphsurgeon.Variable(f"{node.name}/ReduceMean_0_output_0")
        gs_graph.nodes.append(onnx_graphsurgeon.Node(
            "ReduceMean",
            f"{node.name}/ReduceMaen_0",
            attrs={"axes": axis.values, "keepdims":1},
            inputs=[input],
            outputs=[reduce_mean_0_output]
        ))

        sub_0_output = onnx_graphsurgeon.Variable(f"{node.name}/Sub_0_output_0")
        gs_graph.nodes.append(onnx_graphsurgeon.Node(
            "Sub",
            f"{node.name}/Sub_0",
            inputs=[input, reduce_mean_0_output],
            outputs=[sub_0_output]
        ))

        pow_0_output = onnx_graphsurgeon.Variable(f"{node.name}/Pow_0_output_0")
        gs_graph.nodes.append(onnx_graphsurgeon.Node(
            "Pow",
            f"{node.name}/Pow_0",
            inputs=[sub_0_output, power_2],
            outputs=[pow_0_output]
        ))

        reduce_mean_1_output = onnx_graphsurgeon.Variable(f"{node.name}/ReduceMain_1_output_0")
        gs_graph.nodes.append(onnx_graphsurgeon.Node(
            "ReduceMean",
            f"{node.name}/ReduceMean_1",
            attrs={"axes": axis.values, "keepdims":1},
            inputs=[pow_0_output],
            outputs=[reduce_mean_1_output]
        ))

        add_0_output = onnx_graphsurgeon.Variable(f"{node.name}/Add_0_output_0")
        gs_graph.nodes.append(onnx_graphsurgeon.Node(
            "Add",
            f"{node.name}/Add_0",
            inputs=[reduce_mean_1_output, epsilon],
            outputs=[add_0_output]
        ))

        sqrt_0_output = onnx_graphsurgeon.Variable(f"{node.name}/Sqrt_0_output_0")
        gs_graph.nodes.append(onnx_graphsurgeon.Node(
            "Sqrt",
            f"{node.name}/Sqrt_0",
            inputs=[add_0_output],
            outputs=[sqrt_0_output]
        ))

        reshape_0_output = onnx_graphsurgeon.Variable(f"{node.name}/Reshape_0_output_0")
        gs_graph.nodes.append(onnx_graphsurgeon.Node(
            "Reshape",
            f"{node.name}/Reshape_0",
            inputs=[scale, dim_ones],
            outputs=[reshape_0_output]
        ))

        reshape_1_output = onnx_graphsurgeon.Variable(f"{node.name}/Reshape_1_output_0")
        gs_graph.nodes.append(onnx_graphsurgeon.Node(
            "Reshape",
            f"{node.name}/Reshape_1",
            inputs=[b, dim_ones],
            outputs=[reshape_1_output]
        ))

        mul_0_output = onnx_graphsurgeon.Variable(f"Mul_0_output_0")
        gs_graph.nodes.append(onnx_graphsurgeon.Node(
            "Mul",
            f"{node.name}/Mul_0",
            inputs=[reshape_0_output, sub_0_output],
            outputs=[mul_0_output]
        ))

        div_0_output = onnx_graphsurgeon.Variable(f"Div_0_output_0")
        gs_graph.nodes.append(onnx_graphsurgeon.Node(
            "Div",
            f"{node.name}/Div_0",
            inputs=[mul_0_output, sqrt_0_output],
            outputs=[div_0_output]
        ))

        gs_graph.nodes.append(onnx_graphsurgeon.Node(
            "Add",
            f"{node.name}/Add_1",
            inputs=[div_0_output, reshape_1_output],
            outputs=[output]
        ))

gs_graph.cleanup()

new_onnx_model = onnx_graphsurgeon.export_onnx(gs_graph)
onnx.save(new_onnx_model, "repro_noIN.onnx")
