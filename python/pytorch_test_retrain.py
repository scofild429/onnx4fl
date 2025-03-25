from utils import set_model, validate, grab_loader, set_optimizer
from custom_classes import AverageMeter
from utils import train, validate
from encoder import Encoder
import torch
import time


def torch_test_state_dict(opt, state_dict):
    model, criterion = set_model(opt)
    model.load_state_dict(torch.load(state_dict))
    infer_loader = grab_loader(opt.infer_data_binary, opt.infer_label_binary, opt)
    print(f"state_dict: Model testing is started in {opt.device}!")
    test_loss, test_acc = validate(infer_loader, model, criterion, opt)
    #    print(f"state_dict: test_loss and test_acc is {test_loss}, {test_acc}")
    
    
def torch_test_entire_model(opt, entire_model):
    model = torch.load(entire_model)
    criterion = torch.nn.CrossEntropyLoss()
    if opt.device == "gpu":
        if torch.cuda.is_available():
            model = model.cuda()
            criterion = criterion.cuda()
    if opt.device == "cpu":
        pass
    
    infer_loader = grab_loader(opt.infer_data_binary, opt.infer_label_binary, opt)
    print(f"entrie_mode: Model testing is started in {opt.device}!")
    test_loss, test_acc = validate(infer_loader, model, criterion, opt)
    #    print(f"entrie_mode: test_loss and test_acc is {test_loss}, {test_acc}")
    

def torch_test_check_point(opt, check_points):
    model, criterion = set_model(opt)
    optimizer = set_optimizer(opt, model)
    infer_loader = grab_loader(opt.infer_data_binary, opt.infer_label_binary, opt)
    checkpoint = torch.load(check_points)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"check points: Model testing is started in {opt.device}!")
    test_loss, test_acc = validate(infer_loader, model, criterion, opt)
    #    print(f"check points: test_loss and test_acc is {test_loss}, {test_acc}")

def torch_resuming_check_point_training(opt):
    model, criterion = set_model(opt)
    optimizer = set_optimizer(opt, model)
    checkpoint = torch.load(opt.pytorch_check_points)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    resuming_train_loader = grab_loader(opt.train_data_binary, opt.train_label_binary, opt)
    valid_loader = grab_loader(opt.valid_data_binary, opt.valid_label_binary, opt)
    print(f"check points: Model resuming training is started in {opt.device}!")

    start_time = time.time()
    for epoch in range(1, opt.epochs + 1):
        model.train()
        train_loss, train_acc = train(resuming_train_loader, model, criterion, optimizer, epoch, opt)
        model.eval()
        val_loss, val_acc = validate(valid_loader, model, criterion, opt)
    retrain_time = time.time()
    print(f"Pytorch retraining takes time {retrain_time - start_time}!")
    
    torch.save(model.state_dict(), opt.pytorch_state_dict_resuming_train)
    torch.save(model, opt.pytorch_entire_model_resuming_train)
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, opt.pytorch_check_points_resuming_train)


def torch_ondevice_check_point_training(opt):
    model, criterion = set_model(opt)
    optimizer = set_optimizer(opt, model)
    checkpoint = torch.load(opt.pytorch_check_points)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    ondevice_train_loader = grab_loader(opt.ondevice_data_binary, opt.ondevice_label_binary, opt)
    valid_loader = grab_loader(opt.valid_data_binary, opt.valid_label_binary, opt)
    print(f"check points: Model resuming training is started in {opt.device}!")

    start_time = time.time()
    for epoch in range(1, opt.epochs + 1):
        model.train()
        train_loss, train_acc = train(ondevice_train_loader, model, criterion, optimizer, epoch, opt)
        model.eval()
        val_loss, val_acc = validate(valid_loader, model, criterion, opt)
    ondevice_train_time = time.time()
    print(f"Pytorch ondevice training takes time {ondevice_train_time - start_time} s!")
    
    torch.save(model.state_dict(), opt.pytorch_state_dict_ondevice_train)
    torch.save(model, opt.pytorch_entire_model_ondevice_train)
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, opt.pytorch_check_points_ondevice_train)
