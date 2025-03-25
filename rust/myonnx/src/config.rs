use serde::{Deserialize, Serialize};
use std::path::Path;
use std::{fs, io};

pub enum ConfigError {
    IoError(io::Error),
    InvalidConfig(toml::de::Error),
}

impl From<io::Error> for ConfigError {
    fn from(value: io::Error) -> Self {
        Self::IoError(value)
    }
}

impl From<toml::de::Error> for ConfigError {
    fn from(value: toml::de::Error) -> Self {
        Self::InvalidConfig(value)
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AppConfig {
    pub epochs: usize,
    pub instance_number: usize,
    pub instance_words_length: usize,
    pub instance_words_dim: usize,
    pub target_class_number: usize,
    pub data_path: String,
    pub label_path: String,
    pub data_name: String,
    pub label_name: String,
    pub onnx_dynamo_inference_path: String,
    pub onnx_dynamo_inference_input: String,
    pub onnx_dynamo_inference_output: String,
    pub onnx_torchscript_inference_path: String,
    pub onnx_torchscript_inference_input: String,
    pub onnx_torchscript_inference_output: String,
    pub ondevice_inference_path: String,
    pub ondevice_inference_input: String,
    pub ondevice_inference_output: String,
    pub onnx_c20resuming_inference_path: String,
    pub onnx_c40resuming_inference_path: String,
    pub onnx_c20ondevice_inference_path: String,
    pub onnx_c40ondevice_inference_path: String,
    pub onnx_q20resuming_inference_path: String,
    pub onnx_q40resuming_inference_path: String,
    pub onnx_q20ondevice_inference_path: String,
    pub onnx_q40ondevice_inference_path: String,
    pub environment_name: String,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            epochs: 2,
            instance_number: 1556,
            instance_words_length: 280,
            instance_words_dim: 512,
            target_class_number: 144,
            data_path: "".to_string(),
            label_path: "".to_string(),
            data_name: "".to_string(),
            label_name: "".to_string(),
            onnx_dynamo_inference_path: "".to_string(),
            onnx_dynamo_inference_input: "".to_string(),
            onnx_dynamo_inference_output: "".to_string(),
            onnx_torchscript_inference_path: "".to_string(),
            onnx_torchscript_inference_input: "".to_string(),
            onnx_torchscript_inference_output: "".to_string(),
            ondevice_inference_path: "".to_string(),
            ondevice_inference_input: "".to_string(),
            ondevice_inference_output: "".to_string(),
            onnx_c20resuming_inference_path: "".to_string(),
            onnx_c40resuming_inference_path: "".to_string(),
            onnx_c20ondevice_inference_path: "".to_string(),
            onnx_c40ondevice_inference_path: "".to_string(),
            onnx_q20resuming_inference_path: "".to_string(),
            onnx_q40resuming_inference_path: "".to_string(),
            onnx_q20ondevice_inference_path: "".to_string(),
            onnx_q40ondevice_inference_path: "".to_string(),
            environment_name: "".to_string(),
        }
    }
}

pub fn load_or_initialize() -> Result<AppConfig, ConfigError> {
    let config_path = Path::new("Config.toml");
    if config_path.exists() {
        let content = fs::read_to_string(config_path)?;
        let config = toml::from_str(&content)?;
        return Ok(config);
    } else {
        print!("No configuration file is found!");
    }

    let config = AppConfig::default();
    let toml = toml::to_string(&config).unwrap();

    fs::write(config_path, toml)?;
    Ok(config)
}
