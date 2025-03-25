import tensorflow as tf
import struct
import time
import numpy as np
import onnx
from onnx_tf.backend import prepare


def onnx2tfModel(filename):
    # Load the ONNX model and convert
    onnx_model = onnx.load(filename)
    tf_model = prepare(onnx_model)
    return tf_model


def grab_loader_tensor(dataname, labelname, opt):
    # Read and unpack binary data for train
    with open(dataname, 'rb') as f:
        fromfiledata = f.read()
        fromfiledata = struct.unpack(f"<{len(fromfiledata)//4}f", fromfiledata)
    train = tf.convert_to_tensor(fromfiledata, dtype=tf.float32)
    train = tf.reshape(train, [-1, opt.max_sequence_length, opt.word_vector_size])

    # Read and unpack binary data for labels
    with open(labelname, 'rb') as f:
        fromfiledata = f.read()
        fromfiledata = struct.unpack(f"<{len(fromfiledata)//4}f", fromfiledata)
    label = tf.convert_to_tensor(fromfiledata, dtype=tf.float32)
    label = tf.reshape(label, [-1, opt.n_cls])

    if train.shape[0] != label.shape[0]:
        print(f"{dataname} has length: {train.shape[0]} but {labelname} has length: {label.shape[0]}")
        return None
    else:
        dataset = tf.data.Dataset.from_tensor_slices((train, label)).batch(opt.batch_size)
        return dataset


def validate_tensor(val_loader, model, loss_fn, opt):
    start = time.time()
    losses = tf.keras.metrics.Mean()
    accuracy = tf.keras.metrics.CategoricalAccuracy()

    for sentences, labels in val_loader:
        start_time = time.time()

        # Transfer to GPU if available
        if opt.device == "gpu":
            sentences = tf.cast(sentences, tf.float32)
            labels = tf.cast(labels, tf.float32)

        # Perform forward pass and calculate loss
        predictions = model.run(sentences)
        predictions = tf.squeeze(predictions, axis=1)
        loss = loss_fn(labels, predictions)
        
        losses.update_state(loss)
        accuracy.update_state(labels, predictions)

    tt = time.time() - start
    print(f'Validation: Loss {losses.result().numpy():.5} and Acc {accuracy.result().numpy():.3f} with {tt:.2f} seconds.')
    return losses.result().numpy(), accuracy.result().numpy()


def onnx2tf_test(opt, data_binary, label_binary, onnxfile):
    print(f"ONNX inference in Tensorflow with {onnxfile} :", end = " ")
    model = onnx2tfModel(onnxfile)
    infer_loader = grab_loader_tensor(data_binary, label_binary, opt)
    criterion = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    test_loss, test_acc = validate_tensor(infer_loader, model, criterion, opt)

