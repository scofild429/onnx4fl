from utils import set_optimizer, set_model, grab_loader
from serialization import save_onnx_inferencing, save_onnx_training
from onnxruntime.training import ORTModule
from plots import plotAccLoss
from utils import train, validate
import torch
import time


def model_exploring_gpu(opt):
    model, criterion = set_model(opt)
    train_loader = grab_loader(opt.train_data_binary, opt.train_label_binary, opt)
    valid_loader = grab_loader(opt.valid_data_binary, opt.valid_label_binary, opt)
    if train_loader is None or valid_loader is None:
        return
    optimizer = set_optimizer(opt, model)
    print("Model prepration is done in GPU, start to explore its parameters!")
    for epoch in range(1, opt.epochs + 1):
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, opt)
        val_loss, val_acc = validate(valid_loader, model, criterion, opt)

def model_training_cpu(opt):
    model, criterion = set_model(opt)
    train_loader = grab_loader(opt.train_data_binary, opt.train_label_binary, opt)
    valid_loader = grab_loader(opt.valid_data_binary, opt.valid_label_binary, opt)
    if train_loader is None or valid_loader is None:
        return
    optimizer = set_optimizer(opt, model)
    print("Model prepration is done in CPU, start to training!")
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    for epoch in range(1, opt.epochs + 1):
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, opt)
        val_loss, val_acc = validate(valid_loader, model, criterion, opt)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

    save_onnx_training(model, opt)
    save_onnx_inferencing(model, opt)
    plotAccLoss(opt, train_losses, val_losses,  train_accs, val_accs,  yscale='linear')
    
    torch.save(model.state_dict(), opt.pytorch_state_dict)
    torch.save(model, opt.pytorch_entire_model)
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, opt.pytorch_check_points)


def model_training_gpu(opt):
    start = time.time()
    model, criterion = set_model(opt)
    train_loader = grab_loader(opt.train_data_binary, opt.train_label_binary, opt)
    valid_loader = grab_loader(opt.valid_data_binary, opt.valid_label_binary, opt)
    if train_loader is None or valid_loader is None:
        return
    optimizer = set_optimizer(opt, model)
    print("Model prepration is done in GPU, start to training!")
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    for epoch in range(1, opt.epochs + 1):
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, opt)
        val_loss, val_acc = validate(valid_loader, model, criterion, opt)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
    print(f"Model training with GPU takes {time.time() - start} s<<<<<<<<<<<<<<<<<<<<<<<<<<")
    save_onnx_inferencing(model, opt, opt.onnxfile_infer_name)
    save_onnx_training(model, opt, opt.onnxfile_train_name)
    plotAccLoss(opt, train_losses, val_losses,  train_accs, val_accs,  yscale='linear')
    torch.save(model.state_dict(), opt.pytorch_state_dict)
    torch.save(model, opt.pytorch_entire_model)
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, opt.pytorch_check_points)


def model_training_ort(opt):
    opt.device = "gpu"
    model, criterion = set_model(opt)
    model = ORTModule(model)
    train_loader = grab_loader(opt.train_data_binary, opt.train_label_binary, opt)
    valid_loader = grab_loader(opt.valid_data_binary, opt.valid_label_binary, opt)
    optimizer = set_optimizer(opt, model)
    print(f"Model prepration is done {opt.device} with in runtime(ort), start to training!")

    train_losses, val_losses, train_accs, val_accs = [],[],[],[]
    for epoch in range(1, opt.epochs + 1):
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, opt)
        val_loss, val_acc = validate(valid_loader, model, criterion, opt)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

    plotAccLoss(opt, train_losses, val_losses,  train_accs, val_accs,  yscale='linear')
    torch.save(model.state_dict(), opt.pytorch_state_dict)
    torch.save(model, opt.pytorch_entire_model)
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, opt.pytorch_check_points)
