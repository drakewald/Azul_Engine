/// Quick forward-pass timing benchmark for TacticianNet.
///
/// Measures single and batched inference at varying num_res_blocks (0–3)
/// to isolate whether the slowdown is from extra linear layers or LayerNorm.
/// Also compares single vs batch-of-8 to see if MCTS batching helps.
///
/// Run with:
///   cargo run --release --features="native" --bin bench_nn

use azul_engine::ai::nn::{TacticianConfig, TacticianNet, AzulNetwork, INPUT_SIZE};
use std::time::Instant;

const WARMUP: usize = 50;
const ITERS: usize = 1_000;
const BATCH_SIZE: usize = 8;

fn time_single(net: &TacticianNet, input: &[f32]) -> f64 {
    // warmup
    for _ in 0..WARMUP {
        let _ = net.forward(input);
    }
    let t = Instant::now();
    for _ in 0..ITERS {
        let _ = net.forward(input);
    }
    t.elapsed().as_secs_f64() / ITERS as f64 * 1_000_000.0 // µs per call
}

fn time_batch(net: &TacticianNet, inputs: &[Vec<f32>]) -> f64 {
    // warmup
    for _ in 0..WARMUP {
        let _ = net.forward_batch(inputs);
    }
    let t = Instant::now();
    for _ in 0..ITERS {
        let _ = net.forward_batch(inputs);
    }
    t.elapsed().as_secs_f64() / ITERS as f64 * 1_000_000.0 // µs per call
}

fn main() {
    let input: Vec<f32> = vec![0.5_f32; INPUT_SIZE];
    let batch: Vec<Vec<f32>> = vec![input.clone(); BATCH_SIZE];

    println!(
        "{:<12} {:>14} {:>18} {:>18}",
        "res_blocks", "params (approx)", "single (µs)", "batch-8 (µs)"
    );
    println!("{}", "-".repeat(66));

    for num_res_blocks in 0..=3 {
        let cfg = TacticianConfig {
            num_res_blocks,
            ..TacticianConfig::default()
        };

        // Rough parameter count: stem + blocks + bottleneck + heads
        let w = cfg.trunk_width;
        let m = cfg.mid_dim;
        let v = cfg.value_hidden_dim;
        let stem_params = cfg.input_dim * w + w;                        // fc + bias
        let block_params = num_res_blocks * (2 * (w * w + w));          // 2 fc per block
        let ln_params    = num_res_blocks * (2 * 2 * w);                // 2 LN per block (scale+bias)
        let stem_ln_params = 2 * w;
        let bottleneck_params = w * m + m;
        let policy_params = m * cfg.policy_dim + cfg.policy_dim;
        let value_params  = m * v + v + v * 1 + 1;
        let total = stem_params + stem_ln_params + block_params + ln_params
                  + bottleneck_params + policy_params + value_params;

        let net = TacticianNet::new_random(cfg);
        let single_us = time_single(&net, &input);
        let batch_us  = time_batch(&net, &batch);

        println!(
            "{:<12} {:>14} {:>17.1} {:>17.1}",
            num_res_blocks, total, single_us, batch_us
        );
    }

    println!();
    println!("Note: 'single' = one state at a time (MCTS leaf eval).");
    println!("      'batch-8' = 8 states at once (MCTS virtual-loss batch).");
    println!("      Divide µs by batch size to get per-state cost in batch mode.");
}
