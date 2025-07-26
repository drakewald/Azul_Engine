use serde::{Deserialize, Serialize};
use rand::Rng;
use std::ops::{Add};

// Activation function (hyperbolic tangent)
fn tanh(x: f32) -> f32 {
    x.tanh()
}

// Represents a single layer in the neural network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer {
    weights: Vec<Vec<f32>>,
    biases: Vec<f32>,
}

impl Layer {
    // Creates a new layer with random weights and biases
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let weights = (0..output_size)
            .map(|_| {
                (0..input_size)
                    .map(|_| rng.gen_range(-1.0..1.0))
                    .collect()
            })
            .collect();
        let biases = (0..output_size).map(|_| rng.gen_range(-1.0..1.0)).collect();

        Self { weights, biases }
    }

    // Performs the forward pass for this layer
    fn forward(&self, inputs: &[f32]) -> Vec<f32> {
        self.weights
            .iter()
            .zip(&self.biases)
            .map(|(neuron_weights, bias)| {
                let output = neuron_weights
                    .iter()
                    .zip(inputs)
                    .map(|(weight, input)| weight * input)
                    .sum::<f32>()
                    .add(bias);
                tanh(output)
            })
            .collect()
    }
}

// Represents the complete neural network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralNetwork {
    layers: Vec<Layer>,
}

impl NeuralNetwork {
    // Creates a new neural network with a specified architecture
    pub fn new(layer_sizes: &[usize]) -> Self {
        let layers = layer_sizes
            .windows(2)
            .map(|sizes| Layer::new(sizes[0], sizes[1]))
            .collect();
        Self { layers }
    }

    // Performs a full forward pass through the entire network
    pub fn forward(&self, inputs: &[f32]) -> Vec<f32> {
        self.layers
            .iter()
            .fold(inputs.to_vec(), |acc, layer| layer.forward(&acc))
    }
}
