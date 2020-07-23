import onnx

def change_input_dim(model, new_batch_size):
    sym_batch_dim = new_batch_size
    
    inputs = model.graph.input
    for input in inputs:
        dim1 = input.type.tensor_type.shape.dim[0]
        dim1.dim_param = sym_batch_dim

def apply(transform, infile, outfile, new_batch_size):
    model = onnx.load(infile)
    transform(model, new_batch_size)
    onnx.save(model, outfile)

origin_onnx_path = "./model.onnx"
new_batch_size = "16"
new_onnx_path = "./model" + new_batch_size + ".onnx"
apply(change_input_dim, origin_onnx_path, new_onnx_path, new_batch_size)
