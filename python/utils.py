from __future__ import print_function
import torch
import torch.optim as optim
import onnx
import struct
from encoder import Encoder
from torch.utils.data import DataLoader
from custom_classes import CustomDataset
import torch.backends.cudnn as cudnn
from custom_classes import AverageMeter
from onnxruntime.training.api import CheckpointState, Module, Optimizer
import numpy as np
from tokenization import data_clean
from onnxruntime import SessionOptions
import time
import os
import json
import shutil
import re
import random



def get_useable_data(opt):
    count = 0
    for file in os.listdir(opt.raw_data_folder):
        if file.endswith(".json"):
            path = os.path.join(os.getcwd(), os.path.join(opt.raw_data_folder,file))
            with open(path) as json_data:
                d = json.load(json_data)
                responsible = d["Ticket"]["Responsible"]
                owner = d["Ticket"]["Owner"]
                if responsible == owner and responsible != "niemand" and owner != "niemand":
                        shutil.copy(path, os.path.join(os.getcwd(), opt.data_folder))
                else:
                    count += 1
                    continue
    print(f"{count} files are excluded")

    
def extract_data(opt):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    texts = []
    question = []
    for file in os.listdir(opt.data_folder):
        if file.endswith(".json"):
            path = os.path.join(os.getcwd(), os.path.join(opt.data_folder,file))
            with open(path) as json_data:
                d = json.load(json_data)
                handler = d["Ticket"]["Responsible"]
                body_len = len(d["Ticket"]["Article"])
                datas = dict()
                datas[handler] = []
                conversation = ''
                question = ''
                for j in range(body_len):
                    text = d["Ticket"]["Article"][j]["Body"].split("---------------------------------------------------------")[0]
                    text = text.split("\n")
                    for i in range(len(text)):
                        if not text[i].startswith(">"):
                            conversation += str(text[i])
                    if j == 0:
                        question = conversation
                datas[handler].append(conversation)
                datas[handler].append(question)
                texts.append(datas)
    name_dict = {}
    name_counter = 1
    if opt.extreme_conversation_cut != 0 :
        for i in range(len(texts)):      
            for key, value in texts[i].items():
                texts_without_urls = url_pattern.sub("URL", value[0])
                question_without_urls = url_pattern.sub("URL", value[1])
            cleanedC, name_dict, name_counter = data_clean(texts_without_urls, name_dict, name_counter)
            texts[i][key][0] = cleanedC[:opt.extreme_conversation_cut]
            cleanedQ, name_dict, name_counter = data_clean(question_without_urls, name_dict, name_counter)
            texts[i][key][1] = cleanedQ[:opt.extreme_conversation_cut]
    else:
        for i in range(len(texts)):
            for key, value in texts[i].items():
                texts_without_urls = url_pattern.sub("URL", value[0])
                question_without_urls = url_pattern.sub("URL", value[1])
            texts[i][key][0] = data_clean(texts_without_urls)
            texts[i][key][1] = data_clean(question_without_urls)
    if opt.shuffle:
        random.Random(opt.seed).shuffle(texts)

    print(f"the number of names is {name_counter}\n")
    if name_counter != 1:
        with open(opt.name_file, "w") as f:
            json.dump(name_dict, f, indent=4)
    return texts


    
def get_device(cuda_preference=True):
    print('cuda available:', torch.cuda.is_available(),
          '; cudnn available:', torch.backends.cudnn.is_available(),
          '; num devices:', torch.cuda.device_count()
          )

    use_cuda = False if not cuda_preference else torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    device_name = torch.cuda.get_device_name(device) if use_cuda else 'cpu'
    print('Using device', device_name)
    return device


def get_model_ondevice_onnxruntime(opt, device):
    #    state = CheckpointState.load_checkpoint(opt.artifacts_folder + "/checkpoint")
    if opt.shuffle == "True":
        opt.name = f"C{opt.extreme_conversation_cut}_D{opt.device}_P{opt.training_set_percentage}_E{opt.evaluation}_BS{str(opt.batch_size).zfill(3)}_Head{opt.num_heads}_L{opt.num_layers}_EP20_R{opt.learning_rate}_Shuffle"
    else:
        opt.name = f"C{opt.extreme_conversation_cut}_D{opt.device}_P{opt.training_set_percentage}_E{opt.evaluation}_BS{str(opt.batch_size).zfill(3)}_Head{opt.num_heads}_L{opt.num_layers}_EP20_R{opt.learning_rate}_Unshuffle"

    path = opt.artifacts_folder + f"/{opt.name}"
    state = CheckpointState.load_checkpoint(path + "/checkpoint")
    
    session_options = SessionOptions()
    session_options.use_deterministic_compute = True
    model = Module(path + "/training_model.onnx",
                   state,
                   path + "/eval_model.onnx",
                   device=device,
                   session_options=session_options)
    optimizer = Optimizer(path + "/optimizer_model.onnx", model)
    optimizer.set_learning_rate(0.0000001)
    opt.learning_rate = 0.0000001
    print(f"*******Learning rate for ONNX training is: {optimizer.get_learning_rate()}*******")
    return model, optimizer



def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer


def set_model(opt):
    # if opt.phase == "training":
    #     # only use DDP for training phase
    #     local_rank = int(os.environ.get("LOCAL_RANK", 0))
    #     torch.cuda.set_device(local_rank)
    #     # Initialize model and wrap with DDP
    #     model = Encoder(opt).to(local_rank)
    #     model = DDP(model, device_ids=[local_rank])
    # else:
    model = Encoder(opt)
    criterion = torch.nn.CrossEntropyLoss()
    if opt.device == "gpu" and torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
    return model, criterion


def grab_loader(dataname, labelname, opt):
    with open(dataname, 'rb') as f:
        fromfiledata = f.read()
        fromfiledata = list(struct.unpack("<%df" % (len(fromfiledata)/(4)), fromfiledata))
    train = torch.Tensor(fromfiledata)
    train = train.reshape(-1, opt.max_sequence_length, opt.word_vector_size)

    with open(labelname, 'rb') as f:
        fromfiledata = f.read()
        fromfiledata = list(struct.unpack("<%df" % (len(fromfiledata)/(4)), fromfiledata))
    label = torch.Tensor(fromfiledata)
    label = label.reshape(-1, opt.n_cls)
    if len(train) != len(label):
        print(f"{dataname} has length: {len(train)} but {labelname} has lenght: {len(label)}")
        return None
    else:
        return DataLoader(CustomDataset(train, label), batch_size=opt.batch_size)


def test_onnx(onnxfile_name):
    onnx_model = onnx.load(onnxfile_name)
    onnx.checker.check_model(onnx_model)


def train(dataloader, model, criterion, optimizer, epoch, opt):
    model.train()
    losses = AverageMeter()
    accuracy = AverageMeter()

    if opt.device != "gpu" or not torch.cuda.is_available():
        print("Model training is only supported on GPUs")
        return None, None

    # # Multiple GPUs inits
    # local_rank = int(os.environ.get("LOCAL_RANK", 0))
    # torch.cuda.set_device(local_rank)

    for idx, (sentences, labels) in enumerate(dataloader):
        sentences = sentences.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        output = model(sentences)
        loss = criterion(output, labels)

        losses.update(loss.item(), opt.batch_size)
        accuracy.update(float(sum(labels.argmax(dim=1) == output.argmax(dim=1))/opt.batch_size))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print('Epoch {} -- Training: Loss {:.5} and Acc {:.3f},'.format(epoch, losses.avg, accuracy.avg), end='')
    return losses.avg, accuracy.avg


def validate(val_loader, model, criterion, opt):
    start = time.time()
    model.eval()
    losses = AverageMeter()
    accuracy = AverageMeter()

    with torch.no_grad():
        for idx, (sentences, labels) in enumerate(val_loader):
            if opt.device == "gpu":
                sentences = sentences.float().cuda()
                labels = labels.cuda()
            output = model(sentences)
            loss = criterion(output, labels)
            losses.update(loss.item(), opt.batch_size)
            acc = float(sum(labels.argmax(dim=1) == output.argmax(dim=1))/opt.batch_size)
            accuracy.update(acc)
    tt = time.time() - start
    print(f'Validation: Loss {losses.avg:.5} and Acc {accuracy.avg:.3f}, with time {tt:.2f} seconds')
    return losses.avg, accuracy.avg

def gauss(x, H, A, x0, sigma):
    return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


