const ort = require('onnxruntime-node');
const fs = require('fs');
require('dotenv').config();


function softmax(arr) {
    const expValues = arr.map(x => Math.exp(x));
    const sumExpValues = expValues.reduce((a, b) => a + b, 0);
    return expValues.map(value => value / sumExpValues);
}


async function main() {
  const start = new Date();
  
  const instance_number =                  Number(process.env.instance_number);
  const instance_words_length =            Number(process.env.instance_words_length);
  const instance_words_dim =               Number(process.env.instance_words_dim);
  const target_class_number =              Number(process.env.target_class_number);
  const onnx_dynamo_inference_input =      process.env.onnx_dynamo_inference_input ;
  const onnx_dynamo_inference_output =     process.env.onnx_dynamo_inference_output;
  const onnx_torchscript_inference_input = process.env.onnx_torchscript_inference_input;
  const onnx_torchscript_inference_output =process.env.onnx_torchscript_inference_output;
  const ondevice_inference_input =         process.env.ondevice_inference_input ;
  const ondevice_inference_output =        process.env.ondevice_inference_output ;
  const onnx_dynamo_inference_path =       process.env.onnx_dynamo_inference_path;
  const onnx_torchscript_inference_path =  process.env.onnx_torchscript_inference_path ;

  const onnx_c20resuming_inference_path =  process.env.onnx_c20resuming_inference_path;
  const onnx_c40resuming_inference_path =  process.env.onnx_c40resuming_inference_path;
  const onnx_c20ondevice_inference_path =  process.env.onnx_c20ondevice_inference_path;
  const onnx_c40ondevice_inference_path =  process.env.onnx_c40ondevice_inference_path;
  const onnx_q20resuming_inference_path =  process.env.onnx_q20resuming_inference_path;
  const onnx_q40resuming_inference_path =  process.env.onnx_q40resuming_inference_path;
  const onnx_q20ondevice_inference_path =  process.env.onnx_q20ondevice_inference_path;
  const onnx_q40ondevice_inference_path =  process.env.onnx_q40ondevice_inference_path;
    
  const ondevice_inference_path =          process.env.ondevice_inference_path ;
  const data_name =                        process.env.data_name;
  const label_name =                       process.env.label_name ;
  let   data_path =                        process.env.data_path;
  let   label_path =                       process.env.label_path;
  
  let instance_length = instance_words_length*instance_words_dim;
  let onnx_inference_path = "";
  let onnx_inference_input_name = "";
  let onnx_inference_output_name = "";
  
  if (process.argv.length != 5) {
    console.log("Please speicfy your argument as following order: \n\
                 node \n\
                 . \n\
                 ONNX type: from 1 to 10 \n\
                 Percentage of trainset: 40 or 60\n\
                 Evaluation: question or conversation.\n"
	       );
    process.exit(1);
  }

  if (process.argv[2] == "1") {
    console.log("You are using Dynamo ONNX for inference with Nodejs");
    onnx_inference_path = onnx_dynamo_inference_path;
    onnx_inference_input_name = onnx_dynamo_inference_input;
    onnx_inference_output_name = onnx_dynamo_inference_output;
  }else if (process.argv[2] == "2") {
    console.log("You are using Torchscript ONNX for inference with Nodejs");
    onnx_inference_path = onnx_torchscript_inference_path;
    onnx_inference_input_name = onnx_torchscript_inference_input;
    onnx_inference_output_name = onnx_torchscript_inference_output;
  }else if (process.argv[2] == "3") {
    console.log("You are doing inference with 20 epochs resuming trained ONNX for conversation inference with Nodejs");
    onnx_inference_path = onnx_c20resuming_inference_path;
    onnx_inference_input_name = ondevice_inference_input;
    onnx_inference_output_name = ondevice_inference_output;
  }else if (process.argv[2] == "4") {
    console.log("You are doing inference with 40 epochs resuming trained ONNX for conversation inference with Nodejs");
    onnx_inference_path = onnx_c40resuming_inference_path;
    onnx_inference_input_name = ondevice_inference_input;
    onnx_inference_output_name = ondevice_inference_output;
  }else if (process.argv[2] == "5") {
    console.log("You are doing inference with 20 epochs ondevice trained ONNX for conversation inference with Nodejs");
    onnx_inference_path = onnx_c20ondevice_inference_path;
    onnx_inference_input_name = ondevice_inference_input;
    onnx_inference_output_name = ondevice_inference_output;
  }else if (process.argv[2] == "6") {
    console.log("You are doing inference with 40 epochs ondevice trained ONNX for conversation inference with Nodejs");
    onnx_inference_path = onnx_c40ondevice_inference_path;
    onnx_inference_input_name = ondevice_inference_input;
    onnx_inference_output_name = ondevice_inference_output;
  }else if (process.argv[2] == "7") {
    console.log("You are doing inference with 20 epochs resuming trained ONNX for question inference with Nodejs");
    onnx_inference_path = onnx_q20resuming_inference_path;
    onnx_inference_input_name = ondevice_inference_input;
    onnx_inference_output_name = ondevice_inference_output;
  }else if (process.argv[2] == "8") {
    console.log("You are doing inference with 40 epochs resuming trained ONNX for question inference with Nodejs");
    onnx_inference_path = onnx_q40resuming_inference_path;
    onnx_inference_input_name = ondevice_inference_input;
    onnx_inference_output_name = ondevice_inference_output;
  }else if (process.argv[2] == "9") {
    console.log("You are doing inference with 20 epochs ondevice trained ONNX for question inference with Nodejs");
    onnx_inference_path = onnx_q20ondevice_inference_path;
    onnx_inference_input_name = ondevice_inference_input;
    onnx_inference_output_name = ondevice_inference_output;
  }else if (process.argv[2] == "10") {
    console.log("You are doing inference with 40 epochs ondevice trained ONNX for question inference with Nodejs");
    onnx_inference_path = onnx_q40ondevice_inference_path;
    onnx_inference_input_name = ondevice_inference_input;
    onnx_inference_output_name = ondevice_inference_output;
  }else{
    console.log("Only from 1 to 10.");
    process.exit(1);
  }

  if (fs.existsSync(onnx_inference_path)) {
    console.log('File exists: ', onnx_inference_path);
  } else {
    console.log('File does not exist: ', onnx_inference_path);
  }
    
  if (!((process.argv[3] == "40") || (process.argv[3] == "60"))) {
    console.log("Percentage of trainset can only be specified by 40 or 60");
    process.exit(1);
  }
  if (!((process.argv[4] == "question") || (process.argv[4] == "conversation"))) {
    console.log("Evaluation can only be specified by question or conversation");
    process.exit(1);
  }

  data_path += process.argv[3];
  data_path += "_";
  data_path += process.argv[4];
  data_path += "_";
  data_path += data_name;
  if (fs.existsSync(data_path)) {
    console.log('File exists: ', data_path);
  } else {
    console.log('File does not exist: ', data_path);
  }
  
  label_path += process.argv[3];
  label_path += "_";
  label_path += process.argv[4];
  label_path += "_";
  label_path += label_name;
  if (fs.existsSync(label_path)) {
    console.log('File exists: ', label_path);
  } else {
    console.log('File does not exist: ', label_path);
  }
  
  try {
    const session = await ort.InferenceSession.create(onnx_inference_path);
    const buffer = fs.readFileSync(data_path);
    const floatArrayLength = buffer.length / Float32Array.BYTES_PER_ELEMENT;
    const floatArray = new Float32Array(buffer.buffer, buffer.byteOffset, floatArrayLength);

    let predict = [];
    for (let i = 0; i<instance_number; i++){
      const arr = new Float32Array(instance_length);
      for (let j = 0; j<instance_length; j++){
	arr[j] = floatArray[j+i*instance_length];
      }
      const myinstance = new ort.Tensor('float32', arr, [1, instance_words_length, instance_words_dim]);
      const feeds = {[onnx_inference_input_name]: myinstance};
      const results = await session.run(feeds);
      const  dataC = results[onnx_inference_output_name].cpuData;
      
      let resuls = [];
      Object.keys(dataC).forEach(key => {
	resuls.push(dataC[key]);
      });
      resuls = softmax(resuls);
      let maxVal = Math.max(...resuls);
      let index = resuls.indexOf(maxVal); 
      predict[i] = index;

    }
    const label_buffer = fs.readFileSync(label_path);
    const label_floatArrayLength = label_buffer.length / Float32Array.BYTES_PER_ELEMENT;
    const label_floatArray = new Float32Array(label_buffer.buffer, label_buffer.byteOffset, label_floatArrayLength);
    
    let labels = [];
    for (let i = 0; i<instance_length; i++){
      const arr = new Float32Array(target_class_number);
      for (let j = 0; j<target_class_number; j++){
	arr[j] = label_floatArray[j+i*target_class_number];
      }
      for (let j = 0; j<target_class_number; j++){
	if (arr[j] == 1.0) {
	  labels.push(j);
	}
      }
    }

    let corrected = 0;
    for (let i = 0; i<instance_number; i++){
      if (predict[i] == labels[i]) {
	corrected += 1;
      }
    }

    let accuracy = corrected/instance_number;

    const end = new Date();
    const timeDiff = (end - start)/1000;
    console.log("Ondevice inference arruracy is: ", accuracy, "by using ", timeDiff, "seconds");
    
  } catch (e) {
    console.error(`failed to inference ONNX model: ${e}.`);
  }
}

main();
