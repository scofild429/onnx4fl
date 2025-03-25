import numpy as np
import torch
from config import parse_option
from utils import get_useable_data
from utils import extract_data
from config import data_config
from plots import plot_data_length_stats
from training import model_training_cpu, model_exploring_gpu
from training import model_training_gpu, model_training_ort
from serialization import save_generated_artifacts_runtime
from onnx_test_retrain import onnx_test_gpu, onnx_test_cpu
from onnx_test_retrain import onnx_inferencing_ondevice, onnx_retraining_ondevice
from pytorch_test_retrain import torch_test_state_dict, torch_test_entire_model, torch_test_check_point
from pytorch_test_retrain import  torch_resuming_check_point_training, torch_ondevice_check_point_training
from onnx4pytorch import onnx2pytorch_test
from onnx4tensorflow import onnx2tf_test
import os
import time


def main():
    opt = parse_option()
    if opt == None:
        return

    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    start = time.time()
    print(f"{time.ctime()} {opt.phase} starts......")
    match opt.phase:
        case "cleaning":
            get_useable_data(opt)
            texts = extract_data(opt)
            plot_data_length_stats(texts, opt)
        case "vectoring":
            if not data_config(opt):
                return
        case "exploring":
            model_exploring_gpu(opt)
        case "training":
#               dist.init_process_group(backend='nccl')
            if opt.device == "cpu":
                model_training_cpu(opt)
            elif opt.device == "gpu":
                model_training_gpu(opt)
                save_generated_artifacts_runtime(opt)
            elif opt.device == "ort":
                os.system("export MKL_SERVICE_FORCE_INTEL=1")
                os.system("python -m torch_ort.configure")
                start = time.time()
                model_training_ort(opt)
            else:
                print("device has to be set correct as string, CPU, GPU, ORT")
                return
#               dist.destroy_process_group()
        case "pytorch_test":
            ### pytorch test with state_dict
            torch_test_state_dict(opt, opt.pytorch_state_dict)
            torch_test_state_dict_time = time.time()
            print(f">>>Pytorch test with: state_dict takes {torch_test_state_dict_time - start} s<<<")
            ### pytorch test with entire model
            torch_test_entire_model(opt, opt.pytorch_entire_model)
            torch_test_entire_model_time = time.time()
            print(f">>>Pytorch test with entire_model takes {torch_test_entire_model_time - torch_test_state_dict_time} s<<<")
            ### pytorch test with checkpoint            
            torch_test_check_point(opt, opt.pytorch_check_points)
            torch_test_check_point_time = time.time()
            print(f">>>Pytorch test with check_point takes {torch_test_check_point_time - torch_test_entire_model_time } s<<<")
        case "pytorch_resuming_training_inference":
            torch_resuming_check_point_training(opt)
            torch_resuming_training_time = time.time()
            ### pytorch test with state_dict after resuming training
            torch_test_state_dict(opt, opt.pytorch_state_dict_resuming_train)
            torch_resuming_state_dict_time = time.time()
            print(f">>>After resuming training: Pytorch test with state_dict takes {torch_resuming_state_dict_time - torch_resuming_training_time} s<<<")
            ### pytorch test with whole_model after resuming training
            torch_test_entire_model(opt, opt.pytorch_entire_model_resuming_train)
            torch_resuming_entire_model_time = time.time()
            print(f">>>After resuming training: Pytorch test with entire_model takes {torch_resuming_entire_model_time - torch_resuming_state_dict_time} s<<<")
            ### pytorch test with check_point after resuming training
            torch_test_check_point(opt, opt.pytorch_check_points_resuming_train)
            torch_resuming_check_point_time = time.time()
            print(f">>>After resuming training: Pytorch test with check_point takes {torch_resuming_check_point_time - torch_resuming_entire_model_time } s<<<<<<<<<")
        case "pytorch_ondevice_training_inference":
            torch_ondevice_check_point_training(opt)
            torch_ondevice_training_time = time.time()
            ### pytorch test with state_dict after ondevice training
            torch_test_state_dict(opt, opt.pytorch_state_dict_ondevice_train)
            torch_ondevice_state_dict_time = time.time()
            print(f">>>After ondevice training: Pytorch test with state_dict takes {torch_ondevice_state_dict_time - torch_ondevice_training_time} s<<<")
            ### pytorch test with whole_model after ondevice training
            torch_test_entire_model(opt, opt.pytorch_entire_model_ondevice_train)
            torch_ondevice_entire_model_time = time.time()
            print(f">>>After ondevice training: Pytorch test with entire_model takes {torch_ondevice_entire_model_time - torch_ondevice_state_dict_time} s<<<")
            ### pytorch test with check_point after ondevice training
            torch_test_check_point(opt, opt.pytorch_check_points_ondevice_train)
            torch_ondevice_check_point_time = time.time()
            print(f">>>After ondevice training: Pytorch test with check_point takes {torch_ondevice_check_point_time - torch_ondevice_entire_model_time } s<<<")
        case "onnx_test":   
            onnx2pytorch_test(opt, opt.test_data_binary, opt.test_label_binary, opt.onnxfile_train_name)
            onnx2tf_test(opt, opt.test_data_binary, opt.test_label_binary, opt.onnxfile_train_name)
            if opt.device == "cpu":
                onnx_test_cpu(opt, opt.test_data_binary, opt.test_label_binary, opt.onnxfile_infer_name)
                onnx_test_cpu(opt, opt.test_data_binary, opt.test_label_binary, opt.onnxfile_train_name)
            elif opt.device == "gpu":
                onnx_test_gpu(opt, opt.test_data_binary, opt.test_label_binary, opt.onnxfile_infer_name)
                onnx_test_gpu(opt, opt.test_data_binary, opt.test_label_binary, opt.onnxfile_train_name)
            else:
                print("device has to be set correct as string, CPU, GPU")
                return
        case "onnx_infer":
            onnx2pytorch_test(opt, opt.infer_data_binary, opt.infer_label_binary, opt.onnxfile_train_name)
            onnx2tf_test(opt, opt.infer_data_binary, opt.infer_label_binary, opt.onnxfile_train_name)
            if opt.device == "cpu":
                onnx_test_cpu(opt, opt.infer_data_binary, opt.infer_label_binary, opt.onnxfile_infer_name)
                onnx_test_cpu(opt, opt.infer_data_binary, opt.infer_label_binary, opt.onnxfile_train_name)
            elif opt.device == "gpu":
                onnx_test_gpu(opt, opt.infer_data_binary, opt.infer_label_binary, opt.onnxfile_infer_name)
                onnx_test_gpu(opt, opt.infer_data_binary, opt.infer_label_binary, opt.onnxfile_train_name)
            else:
                print("device has to be set correct as string, CPU, GPU")
                return                    
        case "onnx_generated_artifacts":
            save_generated_artifacts_runtime(opt)
        case "onnx_resuming_training_inference":
            onnx_retraining_ondevice(opt, opt.train_data_binary, opt.train_label_binary)
            onnx_inferencing_ondevice(opt, opt.infer_data_binary, opt.infer_label_binary, opt.onnxfile_resuming_infer_name)
        case "onnx_ondevice_training_inference":
            onnx_retraining_ondevice(opt, opt.ondevice_data_binary, opt.ondevice_label_binary)
            onnx_inferencing_ondevice(opt, opt.infer_data_binary, opt.infer_label_binary, opt.onnxfile_ondevice_infer_name)
        case "onnx_resuming_inference":
            onnx_inferencing_ondevice(opt, opt.infer_data_binary, opt.infer_label_binary, opt.onnxfile_resuming_infer_name)
        case "onnx_ondevice_inference":
            onnx_inferencing_ondevice(opt, opt.infer_data_binary, opt.infer_label_binary, opt.onnxfile_ondevice_infer_name)
        case _:
            print("phase has to be set correct as string, options: cleaning/vectering/training/inferencing/artifacts_generating/ondevice_training")
            return
    print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>{opt.name} takes {time.time() - start} s<<<<<<<<<<<<<<<<<<<<<<<<<<")
if __name__ == '__main__':    
    main()
