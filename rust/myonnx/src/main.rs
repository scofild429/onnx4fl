use byteorder::{LittleEndian, ReadBytesExt};
use onnxruntime::{
    environment::Environment, ndarray::Array, tensor::OrtOwnedTensor, GraphOptimizationLevel,
    LoggingLevel,
};
use std::fs::File;
use std::io::Read;
type Error = Box<dyn std::error::Error>;
mod config;
use std::env;
use std::path::Path;
use std::time::Instant;

fn main() {
    let start = Instant::now();
    let config = match config::load_or_initialize() {
        Ok(v) => v,
        Err(err) => {
            match err {
                config::ConfigError::IoError(err) => {
                    eprintln!("An error occurred while loading the config: {err}");
                }
                config::ConfigError::InvalidConfig(err) => {
                    eprintln!("An error occurred while parsing the config:");
                    eprintln!("{err}");
                }
            }
            return;
        }
    };
    // println!("Configration is {:#?}", config);

    let args: Vec<String> = env::args().collect();
    let mut onnx_inference_path = String::from("");
    if args.len() != 4 {
        println!(
            "Please speicfy your argument as following order: \n\
             ONNX type: 1 for Dynamo, 2 for Torchscript, 3 for ondevice trained\n\
             Percentage of trainset: 40 or 60\n\
             Evaluation: question or conversation.\n"
        );
        return;
    }

    if args[1] == "1" {
        println!("You are using Dynamo ONNX for inference with Rust");
        onnx_inference_path = config.onnx_dynamo_inference_path.clone();
    } else if args[1] == "2" {
        println!("You are using Torchscript ONNX for inference with Rust");
        onnx_inference_path = config.onnx_torchscript_inference_path.clone();
    } else if args[1] == "3" {
        println!("You are doing inference with 20 epochs resuming trained ONNX for conversation inference with Rust");
        onnx_inference_path = config.onnx_c20resuming_inference_path.clone();
    } else if args[1] == "4" {
        println!("You are doing inference with 40 epochs resuming trained ONNX for conversation inference with Rust");
        onnx_inference_path = config.onnx_c40resuming_inference_path.clone();
    } else if args[1] == "5" {
        println!("You are doing inference with 20 epochs ondevice trained ONNX for conversation inference with Rust");
        onnx_inference_path = config.onnx_c20ondevice_inference_path.clone();
    } else if args[1] == "6" {
        println!("You are doing inference with 40 epochs ondevice trained ONNX for conversation inference with Rust");
        onnx_inference_path = config.onnx_c40ondevice_inference_path.clone();
    } else if args[1] == "7" {
        println!("You are doing inference with 20 epochs resuming trained ONNX for question inference with Rust");
        onnx_inference_path = config.onnx_q20resuming_inference_path.clone();
    } else if args[1] == "8" {
        println!("You are doing inference with 40 epochs resuming trained ONNX for question inference with Rust");
        onnx_inference_path = config.onnx_q40resuming_inference_path.clone();
    } else if args[1] == "9" {
        println!("You are doing inference with 20 epochs ondevice trained ONNX for question inference with Rust");
        onnx_inference_path = config.onnx_q20ondevice_inference_path.clone();
    } else if args[1] == "10" {
        println!("You are doing inference with 40 epochs ondevice trained ONNX for question inference with Rust");
        onnx_inference_path = config.onnx_q40ondevice_inference_path.clone();
    } else {
        println!("Only 1 to 10 are allowed for different onnx file!");
        return;
    }
    let check_inference_path = Path::new(&onnx_inference_path);
    if check_inference_path.exists() {
        println!("File exists :{}", onnx_inference_path);
    } else {
	println!("File does not exists: {}", onnx_inference_path);
    }


    if !((args[2] == "40") || (args[2] == "60")) {
        println!("Percentage of trainset can only be specified by 40 or 60");
        return;
    }
    if !((args[3] == "question") || (args[3] == "conversation")) {
        println!("Evaluation can only be specified by question or conversation");
        return;
    }

    let mut data_path = String::from("");
    data_path.push_str(&config.data_path);
    data_path.push_str(&args[2]);
    data_path.push_str("_");
    data_path.push_str(&args[3]);
    data_path.push_str("_");
    data_path.push_str(&config.data_name);
    let check_data_path = Path::new(&data_path);
    if check_data_path.exists() {
        println!("File exists :{}", data_path);
    } else {
        println!("File does not exists: {}", data_path);
    }

    let mut label_path = String::from("");
    label_path.push_str(&config.label_path);
    label_path.push_str(&args[2]);
    label_path.push_str("_");
    label_path.push_str(&args[3]);
    label_path.push_str("_");
    label_path.push_str(&config.label_name);
    let check_label_path = Path::new(&label_path);
    if check_label_path.exists() {
        println!("File exists :{}", label_path);
    } else {
        println!("File does not exists: {}", label_path);
    }

    if let Err(e) = run(&config, &onnx_inference_path, &data_path, &label_path) {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }

    let duration = start.elapsed();
    println!(
        "Ondevice inference with Rust take: {} seconds",
        duration.as_secs_f64()
    );
}

fn run(
    myconfig: &config::AppConfig,
    onnx_inference_path: &String,
    data_path: &String,
    label_path: &String,
) -> Result<(), Error> {
    let environment = Environment::builder()
        .with_name(&myconfig.environment_name)
        .with_log_level(LoggingLevel::Info)
        .build()?;

    let mut session = environment
        .new_session_builder()?
        .with_optimization_level(GraphOptimizationLevel::Basic)?
        .with_number_threads(1)?
        .with_model_from_file(onnx_inference_path)?;

    // let input0_shape: Vec<usize> = session.inputs[0].dimensions().map(|d| d.unwrap()).collect();
    // let output0_shape: Vec<usize> = session.outputs[0]
    //     .dimensions()
    //     .map(|d| d.unwrap())
    //     .collect();
    // println!("{:#?}", input0_shape);
    // println!("{:#?}", output0_shape);

    let mut file = File::open(data_path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    let mut cursor = &buffer[..];
    let mut predict: Vec<i32> = vec![i32::default(); myconfig.instance_number];
    for index_instance in 0..myconfig.instance_number {
        let mut array = Array::zeros((
            1,
            myconfig.instance_words_length,
            myconfig.instance_words_dim,
        ));
        for i in 0..myconfig.instance_words_length {
            for j in 0..myconfig.instance_words_dim {
                let number = cursor.read_f32::<LittleEndian>();
                if let Ok(value) = number {
                    array[[0, i, j]] = value;
                }
            }
        }
        let input_tensor_values = vec![array];
        let outputs: Vec<OrtOwnedTensor<f32, _>> = session.run(input_tensor_values)?;

        let mut index = 0;
        let mut ma = 0.0;
        for k in 0..myconfig.target_class_number {
            //[[0, k]]: 0 stands for the unique dimension, k is the iteration index
            let tmp = outputs[0].softmax(ndarray::Axis(1))[[0, k]];
            if tmp > ma {
                ma = tmp;
                index = k;
            }
        }
        predict[index_instance] = index as i32;
    }

    let mut label_file = File::open(label_path)?;
    let mut label_buffer = Vec::new();
    label_file.read_to_end(&mut label_buffer)?;
    let mut label_cursor = &label_buffer[..];
    let mut labels_array = Array::zeros((myconfig.instance_number, myconfig.target_class_number));
    for i in 0..=myconfig.instance_number {
        for j in 0..myconfig.target_class_number {
            let number = label_cursor.read_f32::<LittleEndian>();
            if let Ok(value) = number {
                labels_array[[i, j]] = value;
            }
        }
    }

    let mut labels: Vec<i32> = vec![i32::default(); myconfig.instance_number];
    for i in 0..myconfig.instance_number {
        for j in 0..myconfig.target_class_number {
            if labels_array[[i, j]] == 1.0 {
                labels[i] = j as i32;
            }
        }
    }
    let mut correct = 0;
    for i in 0..myconfig.instance_number {
        if predict[i] == labels[i] {
            correct += 1;
        }
    }
    let accuracy: f32 = correct as f32 / myconfig.instance_number as f32;
    println!("The accuracy is {:.3}", accuracy);

    Ok(())
}
