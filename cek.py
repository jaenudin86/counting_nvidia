import tensorrt as trt

engine_path = "best.engine"

# Logger untuk TensorRT
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Buka engine
with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

# Print info input & output
print("=== TensorRT Engine Info ===")
for idx in range(engine.num_bindings):
    name = engine.get_binding_name(idx)
    dtype = engine.get_binding_dtype(idx)
    shape = engine.get_binding_shape(idx)
    io_type = "Input" if engine.binding_is_input(idx) else "Output"
    print(f"{io_type} {idx}: {name}, dtype={dtype}, shape={shape}")
