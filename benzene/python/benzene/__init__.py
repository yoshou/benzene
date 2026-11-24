"""The host side of benzene: the weight streams and numpy references of the DSP kernels (kernels), the model file
(model_file), running benzene on the board (board), the quantization methods the models share (quantization), and per
model its quantization recipes and their accuracy evaluation, export to the model file, device input bundles and checks
of the device (models).

Run modules with benzene/python on the module path, e.g. from the repository root:
  PYTHONPATH=benzene/python python -m benzene.models.vggt.evaluate table
"""
