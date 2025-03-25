import torch
import os
import onnx
from onnxruntime.training import artifacts
from utils import get_device, test_onnx


# with dynamo-based onnx exporter, only for test
def save_onnx_inferencing(model, opt, onnxfile_name):
    x = torch.randn((opt.batch_size, opt.max_sequence_length, opt.word_vector_size), requires_grad=False)
    if opt.device == "gpu":
        device = get_device()
        x = x.to(device)
    onnxfile = torch.onnx.dynamo_export(model, x)
    onnxfile.save(onnxfile_name)
    test_onnx(onnxfile_name)

# with Torchscript-based onnx exporter, for test and ondevice training, even for inference
def save_onnx_training(model, opt, onnxfile_name):
    x = torch.randn((opt.batch_size, opt.max_sequence_length, opt.word_vector_size), requires_grad=True)
    if opt.device == "gpu":
        device = get_device()
        x = x.to(device)
    torch.onnx.export(
        model,
        x,
        onnxfile_name,
        input_names = ['input'],
        output_names = ['output'],
        opset_version=12,
        dynamic_axes={'input' : {0 : 'batch_size'}, 
                      'output' : {0 : 'batch_size'}}
    )
    test_onnx(onnxfile_name)
    

def save_generated_artifacts_runtime(opt):
    save_artifacts_path = opt.artifacts_folder+"/"+opt.name
    if not os.path.exists(save_artifacts_path):
        os.makedirs(save_artifacts_path)
        print(f"Directory '{save_artifacts_path}' created")
    else:
        print(f"Directory '{save_artifacts_path}' already exists")

    model = onnx.load(opt.onnxfile_train_name)
    requires_grad = [para.name for para in model.graph.initializer]
    artifacts.generate_artifacts(
        model,
        requires_grad=requires_grad,
        loss=artifacts.LossType.CrossEntropyLoss,
        optimizer=artifacts.OptimType.AdamW,
        artifact_directory=save_artifacts_path
    )
    print("Artifacts generation for onnx resuming training and ondevice training is done")
