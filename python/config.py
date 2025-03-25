import argparse
from aligning import data_aligning, label_aligning


def parse_option():
    parser = argparse.ArgumentParser('Training supervised Attention Model')
    
    parser.add_argument('--epochs', type=int, default=0, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--learning_rate', type=float, default=0.0000005, help='learning rate')
    parser.add_argument('--num_heads', type=int, default=1, help='The number of multi self-attention layer')
    parser.add_argument('--num_layers', type=int, default=2, help='The number of layers')
    parser.add_argument('--seed', type=int, default=40, help='random seed')
    parser.add_argument('--shuffle', type=str, default='', help='If shuffle data for vectoring')
    
    parser.add_argument('--word_vector_size', type=int, default=512, help='The size of word vector')
    parser.add_argument('--n_cls', type=int, default=144, help='The number of classification')
    parser.add_argument('--max_sequence_length', type=int, default=280, help='The maximum allowed words number of each instance')
    parser.add_argument('--instance_Nr', type=int, default=0, help='The langth of the whole dataset')
    

    parser.add_argument('--train_data_binary', type=str, default='', help='training dataset')
    parser.add_argument('--train_label_binary', type=str, default='', help='training labelset')
    parser.add_argument('--train_ratio', type=float, default=0.0, help='The ratio of train dataset')


    parser.add_argument('--valid_data_binary', type=str, default='', help='validation dataset')
    parser.add_argument('--valid_label_binary', type=str, default='', help='validation labelet')
    parser.add_argument('--valid_ratio', type=float, default=0.0, help='The ratio of validation dataset')
    

    parser.add_argument('--test_data_binary', type=str, default='', help='test dataset')
    parser.add_argument('--test_label_binary', type=str, default='', help='test labelset')
    parser.add_argument('--test_ratio', type=float, default=0.0, help='The ratio of test dataset')

    parser.add_argument('--infer_data_binary', type=str, default='', help='inferencing dataset')
    parser.add_argument('--infer_label_binary', type=str, default='', help='inferencing labelset')
    parser.add_argument('--infer_ratio', type=float, default=0.0, help='The ratio of inferencing dataset')
    

    parser.add_argument('--ondevice_data_binary', type=str, default='', help='ondevice training dataset')
    parser.add_argument('--ondevice_label_binary', type=str, default='', help='ondevice labelset')
    parser.add_argument('--ondevice_ratio', type=float, default=0.0, help='The ratio of on-device training dataset')
    
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--device', type=str, default='cpu', help='using CPU or GPU')

    # optimization
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    # encoder configuation
    parser.add_argument('--ffn_hidden', type=int, default=2048, help='The size of hidden layer in encoder')
    parser.add_argument('--drop_prob', type=float, default=0.1, help='The drop probability')
    parser.add_argument('--min_words_conversation', type=int, default=5, help='The minimal required length of each conversation')

    # word2vecModel
    parser.add_argument('--w2v_model_path', type=str, default="", help='The drop probability')
    parser.add_argument('--extreme_conversation_cut', type=int, default=0, help='The max allowed number of words in conversation')
    parser.add_argument('--training_set_percentage', type=str, default="60", help='How many percentage data are used for training')
    parser.add_argument('--evaluation', type=str, default="question", help='Using the whole conversation or only with the question for evaluation')
    parser.add_argument('--phase', type=str, default="", help='which phase want to be executed, training, inference or ondevice')

    parser.add_argument('--raw_data_folder', type=str, default="", help='Where original data saved')
    parser.add_argument('--data_folder', type=str, default="", help='Where useable data saved')
    parser.add_argument('--binary_data_folder', type=str, default="", help='Where binary data saved')
    parser.add_argument('--onnx_folder', type=str, default="", help='Where onnx file saved')
    parser.add_argument('--artifacts_folder', type=str, default="", help='Where generated onnx artifacts saved')
    parser.add_argument('--words_count_cover_confidence_level', type=float, default=0.95, help='Confidence level of the cover of the conversation words count')
    parser.add_argument('--name_file', type=str, default="", help='Where the names saved')

    parser.add_argument('--pytorch_state_dict', type=str, default="../../torchmodel/state-dict.pt", help='Save the state dict of pytroch')
    parser.add_argument('--pytorch_entire_model', type=str, default="../../torchmodel/entire-model.pt", help='Save the entire model of pytroch')
    parser.add_argument('--pytorch_check_points', type=str, default="../../torchmodel/check-points", help='Save the check points of pytroch model')
    
    opt = parser.parse_args()
    opt.train_ratio =    [0.4, 0.6, 0.9]
    opt.valid_ratio =    [0.1, 0.1, 0.1]
    opt.test_ratio =     [0.0, 0.1, 0.0]
    opt.infer_ratio =    [0.1, 0.1, 0.0]
    opt.ondevice_ratio = [0.4, 0.1, 0.0]
    # opt.train_ratio =    [0.6, 0.9]
    # opt.valid_ratio =    [0.1, 0.1]
    # opt.test_ratio =     [0.1, 0.0]
    # opt.infer_ratio =    [0.1, 0.0]
    # opt.ondevice_ratio = [0.1, 0.0]
    opt.train_data_binary =   opt.binary_data_folder+'/'+opt.training_set_percentage+'_'+opt.evaluation+'_Data_host_train'
    opt.train_label_binary =  opt.binary_data_folder+'/'+opt.training_set_percentage+'_'+opt.evaluation+'_Label_host_train'    
    opt.valid_data_binary =   opt.binary_data_folder+'/'+opt.training_set_percentage+'_'+opt.evaluation+'_Data_host_valid'    
    opt.valid_label_binary =  opt.binary_data_folder+'/'+opt.training_set_percentage+'_'+opt.evaluation+'_Label_host_valid'    
    opt.test_data_binary =    opt.binary_data_folder+'/'+opt.training_set_percentage+'_'+opt.evaluation+'_Data_host_test'     
    opt.test_label_binary =   opt.binary_data_folder+'/'+opt.training_set_percentage+'_'+opt.evaluation+'_Label_host_test'     
    opt.infer_data_binary =   opt.binary_data_folder+'/'+opt.training_set_percentage+'_'+opt.evaluation+'_Data_onnx_infer'    
    opt.infer_label_binary =  opt.binary_data_folder+'/'+opt.training_set_percentage+'_'+opt.evaluation+'_Label_onnx_infer'    
    opt.ondevice_data_binary= opt.binary_data_folder+'/'+opt.training_set_percentage+'_'+opt.evaluation+'_Data_ondevice_train'
    opt.ondevice_label_binary=opt.binary_data_folder+'/'+opt.training_set_percentage+'_'+opt.evaluation+'_Label_ondevice_train'

    if opt.shuffle == "True":
        opt.name = f"C{opt.extreme_conversation_cut}_D{opt.device}_P{opt.training_set_percentage}_E{opt.evaluation}_BS{str(opt.batch_size).zfill(3)}_Head{opt.num_heads}_L{opt.num_layers}_EP{opt.epochs}_R{opt.learning_rate}_Shuffle"
    elif opt.shuffle == "False":
        opt.name = f"C{opt.extreme_conversation_cut}_D{opt.device}_P{opt.training_set_percentage}_E{opt.evaluation}_BS{str(opt.batch_size).zfill(3)}_Head{opt.num_heads}_L{opt.num_layers}_EP{opt.epochs}_R{opt.learning_rate}_Unshuffle"
    else:
        print("Please specify  True of False for shuffle")

    print("{0:20} : {1}".format("Phase",                 opt.phase))
    print("{0:20} : {1}".format("Epochs",                opt.epochs))
    print("{0:20} : {1}".format("Learning Rate",         opt.learning_rate))
    print("{0:20} : {1}".format("Batch Size",            opt.batch_size))
    print("{0:20} : {1}".format("Classes Nr.",           opt.n_cls))
    print("{0:20} : {1}".format("multihead Nr.",         opt.num_heads))
    print("{0:20} : {1}".format("Encoder replication",   opt.num_layers))
    print("{0:20} : {1}".format("Instance Nr.",          opt.instance_Nr))
    print("{0:20} : {1}".format("Sequence Length",       opt.max_sequence_length))
    print("{0:20} : {1}".format("Vector Size",           opt.word_vector_size))
    print("{0:20} : {1}".format("Evaluation",            opt.evaluation))
    print("{0:20} : {1}".format("Training percentage",   opt.training_set_percentage))
    print("{0:20} : {1}".format("Device",                opt.device))
    print("{0:20} : {1}".format("Seed",                  opt.seed))
    print("{0:20} : {1}".format("data folder",           opt.data_folder))    
    print("{0:20} : {1}".format("binary data folder",    opt.binary_data_folder))
    print("{0:20} : {1}".format("train_data_binary",     opt.train_data_binary))
    print("{0:20} : {1}".format("train_label_binary",    opt.train_label_binary))
    print("{0:20} : {1}".format("valid_data_binary",     opt.valid_data_binary))
    print("{0:20} : {1}".format("valid_label_binary",    opt.valid_label_binary))
    print("{0:20} : {1}".format("test_data_binary",      opt.test_data_binary))
    print("{0:20} : {1}".format("test_label_binary",     opt.test_label_binary))
    print("{0:20} : {1}".format("infer_data_binary",     opt.infer_data_binary))
    print("{0:20} : {1}".format("infer_label_binary",    opt.infer_label_binary))
    print("{0:20} : {1}".format("ondevice_data_binay",   opt.ondevice_data_binary))
    print("{0:20} : {1}".format("ondevice_label_binary", opt.ondevice_label_binary))


    opt.pytorch_state_dict += opt.device
    opt.pytorch_entire_model += opt.device
    opt.pytorch_check_points += opt.device

    opt.pytorch_state_dict_resuming_train = opt.pytorch_state_dict + "_resuming_train_" + opt.device
    opt.pytorch_entire_model_resuming_train = opt.pytorch_entire_model + "_resuming_train_" + opt.device
    opt.pytorch_check_points_resuming_train = opt.pytorch_check_points + "_resuming_train_" + opt.device

    opt.pytorch_state_dict_ondevice_train = opt.pytorch_state_dict + "_ondevice_train_" + opt.device
    opt.pytorch_entire_model_ondevice_train = opt.pytorch_entire_model + "_ondevice_train_" + opt.device
    opt.pytorch_check_points_ondevice_train = opt.pytorch_check_points + "_ondevice_train_" + opt.device

    opt.onnxfile_infer_name = opt.onnx_folder+ f"/{opt.name}_Onnx_Infer.onnx"
    opt.onnxfile_train_name = opt.onnx_folder+ f"/{opt.name}_Onnx_Train.onnx"
    opt.onnxfile_resuming_infer_name = opt.onnx_folder+ f"/{opt.name}_RE{opt.epochs}_Onnx_Infer_Resume.onnx"
    opt.onnxfile_ondevice_infer_name = opt.onnx_folder+ f"/{opt.name}_RE{opt.epochs}_Onnx_Infer_Ondevice.onnx"

    return opt

def data_config(opt):
    data_Nr = data_aligning(opt)
    label_Nr, label_num = label_aligning(opt)
    if data_Nr == label_Nr:
        print(f"Data prepration is done, {int(data_Nr)}-data instances with {label_num} classification, maximum allowed sequence is {opt.max_sequence_length}")
        return True
    else:
        print(f"Data prepration is failed, {int(data_Nr)}-data /{int(label_Nr)}-label instances with {label_num} classification, maximum allowed sequence is {opt.max_sequence_length}")
        return False
