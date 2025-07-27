// This entire module will only be compiled when the "native" feature is enabled.
#![cfg(feature = "native")]

use serde::{Deserialize, Serialize};
use rand::Rng;
use std::ops::Add;
use tch;
use std::io::Write;
use tempfile::NamedTempFile;
use anyhow;

fn tanh(x: f32) -> f32 {
    x.tanh()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer {
    weights: Vec<Vec<f32>>,
    biases: Vec<f32>,
}

impl Layer {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let weights = (0..output_size)
            .map(|_| (0..input_size).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();
        let biases = (0..output_size).map(|_| rng.gen_range(-1.0..1.0)).collect();
        Self { weights, biases }
    }

    fn forward(&self, inputs: &[f32]) -> Vec<f32> {
        self.weights.iter().zip(&self.biases).map(|(neuron_weights, bias)| {
            let output = neuron_weights.iter().zip(inputs)
                .map(|(weight, input)| weight * input)
                .sum::<f32>().add(bias);
            tanh(output)
        }).collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralNetwork {
    layers: Vec<Layer>,
}

impl NeuralNetwork {
    pub fn new(layer_sizes: &[usize]) -> Self {
        let layers = layer_sizes.windows(2).map(|sizes| Layer::new(sizes[0], sizes[1])).collect();
        Self { layers }
    }

    pub fn forward(&self, inputs: &[f32]) -> Vec<f32> {
        self.layers.iter().fold(inputs.to_vec(), |acc, layer| layer.forward(&acc))
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, anyhow::Error> {
        let mut vs = tch::nn::VarStore::new(tch::Device::Cpu);
        
        let mut temp_file = NamedTempFile::new()?;
        temp_file.write_all(bytes)?;
        
        vs.load(temp_file.path())?;
        
        println!("Successfully loaded model VarStore from memory (NOTE: weight extraction is a placeholder).");
        
        // Placeholder: return a new network.
        let policy_size = 50;
        let input_size = 583;
        let hidden_size = 256;
        let value_size = 1;
        Ok(NeuralNetwork::new(&[input_size, hidden_size, hidden_size, policy_size + value_size]))
    }
}
