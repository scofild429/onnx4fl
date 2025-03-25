from sklearn.preprocessing import OneHotEncoder
from gensim.models import Word2Vec
from plots import plot_data_length_stats
from utils import extract_data
import os
import json
import struct
import numpy as np
import random


def data_aligning(opt):
    texts = extract_data(opt)
    #    max_sequence_length =  plot_data_length_stats(texts, opt)
    max_sequence_length =  280
    opt.max_sequence_length = max_sequence_length
    
    mydata = [list(texts[i].values())[0][j] for i in range(len(texts)) for j in range(2)]
    mydata_len = len(mydata)
    for i in range(mydata_len):
        data_len = len(mydata[i].split(" "))
        if data_len < opt.max_sequence_length:
            for j in range(data_len, opt.max_sequence_length):
                mydata[i] += " SPC"

    for i in range(mydata_len):
        mydata[i] = mydata[i].split(" ")[:opt.max_sequence_length]
        
    model = Word2Vec(mydata, vector_size=opt.word_vector_size, window=5, min_count=1, sg=1, workers=8, epochs=5)
    mydata = [model.wv[mydata[i]] for i in range(mydata_len)]
    model.save(opt.w2v_model_path)
    
    mydata_len /= 2
    for iter, percent in zip(range(3), ["40", "60", "90"]):         #"40", "60", "90"
        train_data_len = int(mydata_len*opt.train_ratio[iter])
        valid_data_len = int(mydata_len*opt.valid_ratio[iter])
        test_data_len = int(mydata_len*opt.test_ratio[iter])
        infer_data_len = int(mydata_len*opt.infer_ratio[iter])
        ondevice_data_len = int(mydata_len*opt.ondevice_ratio[iter])
        dataset_lens = [train_data_len, valid_data_len, test_data_len, infer_data_len, ondevice_data_len]
        for eval in ["question", "conversation"]:
            dataset_binary = [
                opt.binary_data_folder+'/'+percent+'_'+eval+'_Data_host_train',
                opt.binary_data_folder+'/'+percent+'_'+eval+'_Data_host_valid',
                opt.binary_data_folder+'/'+percent+'_'+eval+'_Data_host_test',
                opt.binary_data_folder+'/'+percent+'_'+eval+'_Data_onnx_infer',
                opt.binary_data_folder+'/'+percent+'_'+eval+'_Data_ondevice_train'
                ]
            startindex = 0
            endindex = 0
            for dataset_length, binary_name in zip(dataset_lens, dataset_binary):
                startindex = endindex
                endindex = dataset_length - 1 + endindex
                with open(binary_name, "wb") as f:
                    if eval == "conversation":   ##                    if opt.evaluation == "conversation":
                        for i in range(startindex, endindex):
                            instance = [mydata[2*i][j][k] for j in range(opt.max_sequence_length) for k in range(opt.word_vector_size)]
                            f.write(struct.pack("<%df" % len(instance), *instance))
                    else:
                        for i in range(startindex, endindex):
                            if binary_name.split("_")[-1] == "train":
                                instance = [mydata[2*i][j][k] for j in range(opt.max_sequence_length) for k in range(opt.word_vector_size)]
                            else:
                                instance = [mydata[2*i+1][j][k] for j in range(opt.max_sequence_length) for k in range(opt.word_vector_size)]
                            f.write(struct.pack("<%df" % len(instance), *instance))
    return mydata_len


def label_aligning(opt):
    target_class = []
    for file in os.listdir(opt.data_folder):
        if file.endswith(".json"):
            path = os.path.join(os.getcwd(), os.path.join(opt.data_folder,file))
            with open(path) as json_data:
                d = json.load(json_data)
                target_class.append(d["Ticket"]["Responsible"])

    target_set = list(set(target_class))
    target_set_len = len(target_set)
    target_set_dic = {}
    for i in range(target_set_len):
        target_set_dic[target_set[i]] = i
    target_set_Nr = np.array([i for i in range(target_set_len)]).reshape(-1, 1)
    onehot_encoder = OneHotEncoder(sparse_output=False)
    onehot = onehot_encoder.fit_transform(target_set_Nr)

    targets = []
    for item in target_class:
        label = dict()
        label[item] = onehot[target_set_dic[item]].tolist()
        targets.append(label)

    if opt.shuffle:
        random.Random(opt.seed).shuffle(targets)
        
    targets_len = len(targets)
    for iter, percent in zip(range(3), ["40", "60", "90"]):    # "40", "60", "90"
        train_label_len = int(targets_len*opt.train_ratio[iter])
        valid_label_len = int(targets_len*opt.valid_ratio[iter])
        test_label_len = int(targets_len*opt.test_ratio[iter])
        infer_label_len = int(targets_len*opt.infer_ratio[iter])
        ondevice_label_len = int(targets_len*opt.ondevice_ratio[iter])
        labelset_lens = [train_label_len, valid_label_len, test_label_len, infer_label_len, ondevice_label_len]
        for eval in ["question", "conversation"]:
            labelset_binary = [
                opt.binary_data_folder+'/'+percent+'_'+eval+'_Label_host_train',
                opt.binary_data_folder+'/'+percent+'_'+eval+'_Label_host_valid',
                opt.binary_data_folder+'/'+percent+'_'+eval+'_Label_host_test',
                opt.binary_data_folder+'/'+percent+'_'+eval+'_Label_onnx_infer',
                opt.binary_data_folder+'/'+percent+'_'+eval+'_Label_ondevice_train'
                ]
            startindex = 0
            endindex = 0
            for labelset_length, binary_name in zip(labelset_lens, labelset_binary):
                startindex = endindex
                endindex = labelset_length - 1 + endindex
                with open(binary_name, "wb") as f:
                    for i in range(startindex, endindex):
                        for key, value in targets[i].items():
                            f.write(struct.pack("<%df" % len(value), *value))

    return targets_len, target_set_len,
