import onnx
from onnx2pytorch import ConvertModel
from utils import grab_loader, validate
from torch.nn import CrossEntropyLoss
from torch.cuda import is_available

def onnx2pytorchModel(filename):
    # Load the ONNX model and convert
    onnx_model = onnx.load(filename)
    pytorch_model = ConvertModel(onnx_model)
    return pytorch_model

def onnx2pytorch_test(opt, data_binary, label_binary, onnxfile):
    print(f"ONNX inference in PyTorch with {onnxfile} : ", end=" ")
    model = onnx2pytorchModel(onnxfile)
    infer_loader = grab_loader(data_binary, label_binary, opt)
    criterion = CrossEntropyLoss()
    if opt.device == "gpu":
        if is_available():
            model = model.cuda()
            criterion = criterion.cuda()
    test_loss, test_acc = validate(infer_loader, model, criterion, opt)