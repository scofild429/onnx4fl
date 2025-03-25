python main.py                                           \
       --instance_Nr                             15578   \
       --max_sequence_length                       280   \
       --n_cls                                     144   \
       --phase                               onnx_infer   \
       --device                                    gpu   \
       --epochs                                    200   \
       --training_set_percentage                    60   \
       --evaluation                       conversation   \
       --num_workers                                 8   \
       --batch_size                                  1   \
       --learning_rate                        0.000001   \
       --num_heads                                   1   \
       --num_layers                                  4   \
       --extreme_conversation_cut                 5000   \
       --seed                                       40   \
       --onnx_folder                         ../../onnx  \
       --artifacts_folder               ../../artifacts  \
       --w2v_model_path         ../../artifacts/w2vmodel \
       --shuffle                                    True \
       --raw_data_folder        /mnt/lustre-emmy-hdd/usr/u11656/silin_onnx/data_raw           \
       --data_folder            /mnt/lustre-emmy-hdd/usr/u11656/silin_onnx/data_singlehander  \
       --binary_data_folder      ../../../gwdg_tickets/mark_name_data_binary_shuffle          \
       --name_file               ../../../gwdg_tickets/namesdict.json                         \


       #       --n_cls                   number of agents
       #       --device                  following options for which device is used
       #                                     gpu
       #                                     cpu
       #                                     ort
       #       --training_set_percentage following options for the amount of training set
       #                                     40
       #                                     60
       #                                     90
       #       --evaluation              following options for evaluation option
       #                                     question
       #                                     conversation	     
       #       --phase                   following options for which step to run
       #                                     cleaning
       #                                     vectoring
       #                                     exploring       
       #                                     training
       #                                     onnx_generated_artifacts
       #                                     pytorch_test
       #                                     pytorch_resuming_training_inference
       #                                     tpytorch_ondevice_training_inference
       #                                     onnx_test
       #                                     onnx_infer
       #                                     onnx_resuming_training_inference
       #                                     onnx_ondevice_training_inference
       #       --num_workers             number of workers		    
       #       --epochs                  epochs				    
       #       --batch_size              batch size					   
       #       --learning_rate           learning rate			   
       #       --num_heads               number of multihead		     
       #       --num_layers              number of replication of layers	   
       #       --max_sequence_length     The maximum allowed words number of each instance   
       #       --binary_data_folde       following options for where to save
       #                                     ../../data_binary_unshuffle  
       #                                     ../../data_binary_shuffle  
       #       --shuffle                 following options for where to save
       #                                     False
       #                                     True
