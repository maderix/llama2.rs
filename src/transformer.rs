use llama2_rs::read_checkpoint;
use llama2_rs::{matmul, rmsnorm, softmax};
use llama2_rs::{tokenize_input, Config, RunState, Sampler, TransformerWeights, GS};
use memmap2::Mmap;
use std::fs::File;
use std::io::{self, BufReader, Read, Write};
use std::path::Path;
use std::time::Instant;
use tokenizers::{Encoding, Tokenizer};

/// The entire transformer model, encapsulating configuration, weights, and runtime state.
pub struct Transformer {
    pub config: Config,
    pub weights: TransformerWeights,
    pub state: RunState,
    pub data: Option<Mmap>,
}

impl Transformer {
    /// Create a new Transformer from the given checkpoint file.
    pub fn new<P: AsRef<Path>>(checkpoint_path: P) -> io::Result<Self> {
        let (config, weights, mmap) = read_checkpoint(checkpoint_path.as_ref())?;

        // Initialize the RunState based on the configuration
        let state = RunState::new(&config);

        Ok(Transformer {
            config,
            weights,
            state,
            data: Some(mmap),
        })
    }
    fn rope_relative_positional_encoding(&mut self, pos: usize, kv_dim: usize, head_size: usize) {
        let dim = self.config.dim;

        for i in (0..dim).step_by(2) {
            let head_dim = i % head_size;
            let freq = 1.0f32 / (10000.0f32).powf(head_dim as f32 / head_size as f32);
            let val = pos as f32 * freq;
            let (fcr, fci) = (val.cos(), val.sin());

            let rotn = if i < kv_dim { 2 } else { 1 };
            for v in 0..rotn {
                let vec = if v == 0 {
                    &mut self.state.q
                } else {
                    &mut self.state.k
                };
                let (v0, v1) = (vec[i], vec[i + 1]);
                vec[i] = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }
    }
    fn compute_attention(
        &mut self,
        layer: usize,
        pos: usize,
        kv_dim: usize,
        kv_mul: usize,
        head_size: usize,
    ) {
        let seq_len = self.config.seq_len;
        let kv_cache_offset = layer * seq_len * kv_dim;

        // Iterate over all heads for multi-head attention
        for h in 0..self.config.n_heads {
            let head_offset = h * head_size;
            let q = &self.state.q[head_offset..head_offset + head_size];
            let att = &mut self.state.att[h * seq_len..(h + 1) * seq_len];

            // Compute attention scores
            for t in 0..=pos {
                let k = &self.state.key_cache
                    [kv_cache_offset + t * kv_dim + (h / kv_mul) * head_size..];
                let score = q.iter().zip(k).map(|(&qi, &ki)| qi * ki).sum::<f32>()
                    / (head_size as f32).sqrt();
                att[t] = score;
            }
            // Apply softmax to attention scores slicing attention to pos
            softmax(&mut att[..pos+1]);
            // Compute weighted sum of values
            let xb = &mut self.state.xb[head_offset..head_offset + head_size];
            xb.fill(0.0); // equivalent to memset(xb, 0, head_size * sizeof(float));
            for t in 0..=pos {
                let v = &self.state.value_cache
                    [kv_cache_offset + t * kv_dim + (h / kv_mul) * head_size..];
                let a = att[t];
                for i in 0..head_size {
                    xb[i] += a * v[i];
                }
            }
        }
        //print min max att
    }
    fn ffn_operations(&mut self, layer: usize, hidden_dim: usize, dim: usize) {
        let layer_offset = layer * dim;

        // RMS normalization
        rmsnorm(
            &mut self.state.xb,
            self.state.x.as_mut_slice(),
            &self.weights.rms_ffn_weight[layer_offset..layer_offset + dim],
        );
        // First matrix multiplication (w1 * xb)
        self.state.xq.quantize(&self.state.xb, dim);
        matmul(
            &mut self.state.hb,
            &mut self.state.xq,
            &mut self.weights.w1,
            dim,
            hidden_dim,
            layer,
            GS,
        );

        // Second matrix multiplication (w3 * xb), store in hb2
        matmul(
            &mut self.state.hb2,
            &mut self.state.xq,
            &mut self.weights.w3,
            dim,
            hidden_dim,
            layer,
            GS,
        );

        // SwiGLU non-linearity
        for i in 0..hidden_dim {
            let val = self.state.hb[i];
            // silu(x) = x * sigmoid(x)
            let sigmoid = 1.0 / (1.0 + (-val).exp());
            self.state.hb[i] = val * sigmoid * self.state.hb2[i];
        }

        // Final matrix multiplication (w2 * hb)
        self.state.hq.quantize(&self.state.hb, hidden_dim);
        matmul(
            &mut self.state.xb,
            &mut self.state.hq,
            &mut self.weights.w2,
            hidden_dim,
            dim,
            layer,
            GS,
        );

        // Add residual connection
        for i in 0..dim {
            self.state.x[i] += self.state.xb[i];
        }
    }
    pub fn forward(&mut self, token: u32, pos: usize) -> &[f32] {
        let dim = self.config.dim;
        let kv_dim = (self.config.dim * self.config.n_kv_heads) / self.config.n_heads;
        let kv_mul = self.config.n_heads / self.config.n_kv_heads;
        let hidden_dim = self.config.hidden_dim;
        let head_size = dim / self.config.n_heads;

        // Copy the token embedding into x
        let token_usize = token as usize;
        self.state.x.copy_from_slice(
            &self.weights.token_embedding_table[token_usize * dim..(token as usize + 1) * dim],
        );
        //dump x to a file
        let mut file = File::create("x.txt").unwrap();
        for i in 0..dim {
            writeln!(file, "{}", self.state.x[i]).unwrap();
        }


        // Forward all the layers
        for l in 0..self.config.n_layers {
            // Attention RMSNorm
            rmsnorm(
                &mut self.state.xb,
                &self.state.x,
                &self.weights.rms_att_weight[l * dim..(l + 1) * dim],
            );
           // QKV MatMuls for this position
            self.state.xq.quantize(&self.state.xb, dim);
            matmul(
                &mut self.state.q,
                &mut self.state.xq,
                &mut self.weights.wq,
                dim,
                dim,
                l,
                GS,
            );

            // For wk
            //self.state.xq.quantize(&self.state.xb, dim); // You can reuse self.state.xq if the input is the same
            matmul(
                &mut self.state.k,
                &mut self.state.xq,
                &mut self.weights.wk,
                dim,
                kv_dim, // Assuming kv_dim is (config.dim * config.n_kv_heads) / config.n_heads
                l,
                GS,
            );
            // For wv
            //self.state.xq.quantize(&self.state.xb, dim); // Reuse self.state.xq again
            matmul(
                &mut self.state.v,
                &mut self.state.xq,
                &mut self.weights.wv,
                dim,
                kv_dim,
                l,
                GS,
            );
            // RoPE relative positional encoding
            self.rope_relative_positional_encoding(pos, kv_dim, head_size);

            let kv_cache_offset = l * self.config.seq_len * kv_dim; // Layer offset in the kv cache
            let key_cache_row_offset = kv_cache_offset + pos * kv_dim; // Position offset for keys
            let value_cache_row_offset = kv_cache_offset + pos * kv_dim; // Position offset for values, same as keys in this simplified approach
            self.state.key_cache[key_cache_row_offset..key_cache_row_offset + kv_dim]
                .copy_from_slice(&self.state.k);

            // Copy v to the value cache
            self.state.value_cache[value_cache_row_offset..value_cache_row_offset + kv_dim]
                .copy_from_slice(&self.state.v);

            // Compute attention
            self.compute_attention(l, pos, kv_dim, kv_mul, head_size);
            // Final matrix multiplication and residual connection for attention
            self.state.xq.quantize(&self.state.xb, dim);
            matmul(
                &mut self.state.xb2,
                &mut self.state.xq,
                &mut self.weights.wo,
                dim,
                dim,
                l,
                GS,
            );
            for i in 0..dim {
                self.state.x[i] += self.state.xb2[i];
            }
            // FFN operations including SwiGLU non-linearity
            self.ffn_operations(l, hidden_dim, dim);
        }

        // Final RMSNorm
        let x_final = self.state.x.clone();
        rmsnorm(&mut self.state.x, &x_final, &self.weights.rms_final_weight);
        // Classifier into logits
        self.state.xq.quantize(&self.state.x, dim);
        //lets call this for all layers while accumulating the logits
        matmul(
            &mut self.state.logits,
            &mut self.state.xq,
            &mut self.weights.wcls,
            dim,
            self.config.vocab_size,
            0,
            GS,
        );
        &self.state.logits
    }
    pub fn generate(
        &mut self,
        tokenizer: &Tokenizer,
        sampler: &mut Sampler,
        prompt: &str,
        steps: usize,
    ) {
        let mut prompt_tokens = tokenize_input(tokenizer, prompt).unwrap();
        //add bos token
        prompt_tokens.insert(0, 1);
        let mut pos = 0;
        let mut token = prompt_tokens[pos];
        let start = Instant::now();

        while pos < steps {
            let logits = self.forward(token, pos);
            let next = if pos < prompt_tokens.len() - 1 {
                prompt_tokens[pos + 1]
            } else {
                sampler.sample(logits) as u32
            };
            pos += 1;

            if next == 1 {
                // Assuming 1 is the BOS token
                break;
            }

            let piece = tokenizer.decode(&[token], false);
            match piece {
                Ok(piece) => {
                    print!("{} ", piece);
                    io::stdout().flush().unwrap();
                }
                Err(e) => {
                    eprintln!("Error decoding: {}", e);
                }
            }
            token = next;
        }
        println!();

        if pos > 1 {
            let elapsed = start.elapsed();
            eprintln!(
                "achieved tok/s: {}",
                (pos - 1) as f64 / elapsed.as_secs_f64()
            );
        }
    }
    pub fn chat(
        &mut self,
        tokenizer: &Tokenizer,
        sampler: &mut Sampler,
        cli_user_prompt: Option<&str>,
        cli_system_prompt: Option<&str>,
        steps: usize,
    ) {
        let mut user_turn = true;
        let mut pos = 0;
        let mut token: u32 = 0;

        while pos < steps {
            let prompt = if user_turn {
                let system_prompt = cli_system_prompt.map_or_else(
                    || {
                        println!("Enter system prompt (optional): ");
                        let mut input = String::new();
                        io::stdin()
                            .read_line(&mut input)
                            .expect("Failed to read line");
                        input.trim().to_string()
                    },
                    |prompt| prompt.to_string(),
                );

                let user_prompt = if pos == 0 && cli_user_prompt.is_some() {
                    cli_user_prompt.unwrap().to_string()
                } else {
                    println!("User: ");
                    let mut input = String::new();
                    io::stdin().read_line(&mut input).unwrap();
                    input.trim().to_string()
                };

                if pos == 0 && !system_prompt.is_empty() {
                    format!(
                        "[INST] <<SYS>>\n{}\n<</SYS>>\n\n{} [/INST]",
                        system_prompt, user_prompt
                    )
                } else {
                    format!("[INST] {} [/INST]", user_prompt)
                }
            } else {
                "".to_string()
            };

            let prompt_tokens = if !prompt.is_empty() {
                tokenizer.encode(prompt, false)
            } else {
                //Return the default prompt token
                Ok(Encoding::default())
            };

            if user_turn {
                print!("Assistant: ");
                io::stdout().flush().unwrap();
                user_turn = false;
            }
            if let Ok(prompt_tokens) = prompt_tokens {
                if pos < prompt_tokens.get_tokens().len() {
                    token = prompt_tokens.get_ids()[pos];
                }
            }

            let logits = self.forward(token, pos);
            let next = sampler.sample(logits);
            pos += 1;

            if token != 2 && next != 2 {
                let tokens = [token, next as u32];
                match tokenizer.decode(&tokens, false) {
                    Ok(piece) => {
                        print!("{}", piece);
                        io::stdout().flush().unwrap();
                    }
                    Err(e) => {
                        eprintln!("Error decoding: {}", e);
                    }
                }
            }

            if next == 2 {
                println!("\n");
                user_turn = true;
            }

            token = next as u32;
        }
        println!();
    }
}
