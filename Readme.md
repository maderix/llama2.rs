# Llama2 Transformer Model in Rust

This repository contains the Rust implementation of the Llama2 Transformer model, focusing on performance and correctness. The implementation covers model creation, tokenization, and operations such as matrix multiplication and softmax, essential for the transformer's forward pass.

## Features

- Fast matrix multiplication with quantized tensors.
- Efficient softmax implementation.
- Custom tokenizer compatible with pre-trained models.
- Memory-efficient operation utilizing memory mapping.

## Getting Started

To get started with the Llama2 Transformer in Rust, clone the repository and build the project using Cargo.

### Prerequisites

- Rust programming language
- Cargo package manager
- llama2 model converted to .bin format using https://github.com/karpathy/llama2.c/tree/master?tab=readme-ov-file#metas-llama-2-models

### Installation

1. Clone the repository:

```bash
git clone https://github.com/your-github-username/llama2-rs.git
cd llama2-rs
cargo build --release
./target/release/llama2_rs llama2.bin -n 100 -m "Once upon a time"


