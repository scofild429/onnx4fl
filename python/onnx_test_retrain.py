import onnxruntime
import time
from custom_classes import AverageMeter
import numpy as np
import onnxruntime
import torch
from utils import grab_loader, get_model_ondevice_onnxruntime
from serialization import save_onnx_inferencing, save_onnx_training
from plots import plotAccLoss
import numpy
import onnx
import evaluate


def onnx_test_cpu(opt, data_binary, label_binary, onnxfile_infer_name):
    ort_session = onnxruntime.InferenceSession(onnxfile_infer_name, providers=['CPUExecutionProvider'])
    infer_loader = grab_loader(data_binary, label_binary, opt)
    criterion = torch.nn.CrossEntropyLoss()
    losses = AverageMeter()
    correction = 0
    count = 0
    start = time.time()
    for idx, (sentences, labels) in enumerate(infer_loader):
        if len(sentences) == opt.batch_size:
            ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(sentences.numpy())
            result = ort_session.run([ort_session.get_outputs()[0].name], {ort_session.get_inputs()[0].name: ortvalue})
            loss = criterion(torch.Tensor(result[0]), labels)
            losses.update(loss)
            if labels.argmax(dim=1) == torch.Tensor(result[0]).argmax(dim=1):
                correction += 1
            count += 1
    print(f"{onnxfile_infer_name}: ONNX inferencing on {opt.device}  has accuracy of {float(correction/count)}, and loss of {losses.avg}, time:  {time.time() - start}")



def onnx_test_gpu(opt, data_binary, label_binary, onnxfile_train_name):
    ort_session = onnxruntime.InferenceSession(onnxfile_train_name, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    io_binding = ort_session.io_binding()
    infer_loader = grab_loader(data_binary, label_binary, opt)
    criterion = torch.nn.CrossEntropyLoss()
    losses = AverageMeter()
    correction = 0
    count = 0
    start = time.time()
    for idx, (sentences, labels) in enumerate(infer_loader):
        if len(sentences) == opt.batch_size:
            io_binding.bind_cpu_input(ort_session.get_inputs()[0].name, sentences.numpy())
            io_binding.bind_output(ort_session.get_outputs()[0].name)
            ort_session.run_with_iobinding(io_binding)
            output = io_binding.copy_outputs_to_cpu()[0]
            output = torch.Tensor(output)
            loss = criterion(output, labels)
            if labels.argmax(dim=1) == output.argmax(dim=1):
                correction += 1
            count += 1
            losses.update(loss)
    print(f"With data: {data_binary}, ONNX: {onnxfile_train_name} inferencing on {opt.device} has accuracy of {float(correction/count)}, and loss of {losses.avg}, time:  {time.time() - start}")


def onnx_retraining_ondevice(opt, data_binary, label_binary):
    if opt.device == "gpu":
        model, optimizer = get_model_ondevice_onnxruntime(opt, "cuda")
    elif opt.device == "cpu":
        model, optimizer = get_model_ondevice_onnxruntime(opt, "cpu")
    train_loader = grab_loader(data_binary, label_binary, opt)
    valid_loader = grab_loader(opt.valid_data_binary, opt.valid_label_binary, opt)
    print(f"Model on device is done on {opt.device}, start to training with {len(train_loader)} instances and validate with {len(valid_loader)} instances !")
    for epoch in range(1, opt.epochs + 1):
        train_losses = AverageMeter()
        val_losses = AverageMeter()
        model.train()
        for idx, (sentences, labels) in enumerate(train_loader):
            if len(sentences) == opt.batch_size:
                data = sentences.numpy()
                labels = numpy.array(labels.argmax(dim=1), dtype=numpy.int64)
                loss = model(data, labels)
                train_losses.update(loss)
                optimizer.step()
                model.lazy_reset_grad()
        print('Epoch {} -- Training: Loss {:.5}, '.format(epoch, train_losses.avg), end='')
        model.eval()
        for idx, (sentences, labels) in enumerate(valid_loader):
            if len(sentences) == opt.batch_size:
                data = sentences.numpy()
                label = numpy.array(labels.argmax(dim=1), dtype=numpy.int64)
                loss = model(data, label)
                val_losses.update(loss)
        print('Validation: Loss {:.5} .'.format(val_losses.avg))

    print(f"Valid loss array is {val_losses.avg}")
    if data_binary == opt.train_data_binary:
        model.export_model_for_inferencing(opt.onnxfile_resuming_infer_name, ["output"])
    elif data_binary == opt.ondevice_data_binary:
        model.export_model_for_inferencing(opt.onnxfile_ondevice_infer_name, ["output"])
    else:
        print("Retraining must be ondevice  or resuming training")


def onnx_inferencing_ondevice(opt, data_binary, label_binary, onnxfile_infer_name):
    start = time.time()
    if opt.device == "gpu":
        ort_session = onnxruntime.InferenceSession(onnxfile_infer_name, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    elif opt.device == "cpu":
        ort_session = onnxruntime.InferenceSession(onnxfile_infer_name, providers=['CPUExecutionProvider'])
    else:
        print("device has to be set correct as string, CPU, GPU")
        return
    infer_loader = grab_loader(data_binary, label_binary, opt)
    losses = AverageMeter()
    correction = 0
    count = 0
    for idx, (sentences, labels) in enumerate(infer_loader):
        input_name = "input"
        output_name = "output"
        output = ort_session.run([output_name], {input_name: sentences.numpy()})
        #        print(labels.argmax(dim=1)[0], torch.Tensor(output[0][0]).argmax(axis=0), get_pred(output[0]))
        if labels.argmax(dim=1)[0] == torch.Tensor(output[0][0]).argmax(axis=0):
            count += 1
    print(f"ONNX inference accuracy is {count/len(infer_loader)} for {len(infer_loader)} instances with {time.time() - start}s !")
            
def get_pred(logits):
    return np.argmax(logits, axis=1)
