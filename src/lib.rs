use std::{arch::x86_64, cmp::{min, Ordering}, fs::File, io::{self, BufReader, Read}, path::Path};
use std::f32::{self};
use rayon::prelude::*;
use memmap2::{Mmap, MmapOptions};
use rand::{rngs::StdRng, Rng, SeedableRng};
use tokenizers::Tokenizer;
pub mod optimfn;
pub const GS: usize = 64; 
/// Represents a quantized tensor with quantized values and scaling factors.
#[derive(Clone)]
#[repr(align(32))]
pub struct QuantizedTensor {
    pub q: Vec<i8>,
    pub s: Vec<f32>,
}

impl QuantizedTensor {
    /// Create a new QuantizedTensor with specified sizes.
    pub fn new(size: usize) -> Self {
        QuantizedTensor {
            q: vec![0; size],
            s: vec![0.0; size / GS],
        }
    }

    /// Dequantize the tensor.
    pub fn dequantize(&self, n: usize) -> Vec<f32> {
        let num_groups = n / GS;
        let mut x = vec![0.0; n];

        for group in 0..num_groups {
            let start = group * GS;
            let end = start + GS;
            let group_values = &self.q[start..end];
            let scale = self.s[group];

            for (i, &val) in group_values.iter().enumerate() {
                x[start + i] = val as f32 * scale;
            }
        }

        x
    }

    /// Quantize the tensor.
    pub fn quantize(&mut self, x: &[f32], n: usize) {
        let num_groups = n / GS;
        let q_max = 127.0f32;

        for group in 0..num_groups {
            let start = group * GS;
            let end = start + GS;
            let group_values = &x[start..end];

            let wmax = group_values
                .iter()
                .fold(0.0f32, |max, &val| max.max(val.abs()));
            let scale = wmax / q_max;
            self.s[group] = scale;

            for (i, &val) in group_values.iter().enumerate() {
                let quant_value = val / scale;
                self.q[start + i] = quant_value.round() as i8;
            }
        }
    }
}
/// Configuration parameters for the transformer model.
#[repr(C, packed)]
pub struct Config {
    pub dim: usize,
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub vocab_size: usize,
    pub seq_len: usize,
}
impl Config {
    fn from_bytes(bytes: &[u8]) -> Result<Self, &'static str> {

        let dim = i32::from_ne_bytes(bytes[0..4].try_into().unwrap()) as usize;
        println!("dim: {}", dim);
        let hidden_dim = i32::from_ne_bytes(bytes[4..8].try_into().unwrap()) as usize;
        println!("hidden_dim: {}", hidden_dim);
        let n_layers = i32::from_ne_bytes(bytes[8..12].try_into().unwrap()) as usize;
        println!("n_layers: {}", n_layers);
        let n_heads = i32::from_ne_bytes(bytes[12..16].try_into().unwrap()) as usize;
        println!("n_heads: {}", n_heads);
        let n_kv_heads = i32::from_ne_bytes(bytes[16..20].try_into().unwrap()) as usize;
        println!("n_kv_heads: {}", n_kv_heads);
        let vocab_size = i32::from_ne_bytes(bytes[20..24].try_into().unwrap()) as usize;
        println!("vocab_size: {}", vocab_size);
        let seq_len = i32::from_ne_bytes(bytes[24..28].try_into().unwrap()) as usize;
        println!("max_seq_len: {}", seq_len);

        Ok(Config {
            dim,
            hidden_dim,
            n_layers,
            n_heads,
            n_kv_heads,
            vocab_size,
            seq_len,
        })
    }
}

pub struct RunState {
    pub x: Vec<f32>,
    pub xb: Vec<f32>,
    pub xb2: Vec<f32>,
    pub hb: Vec<f32>,
    pub hb2: Vec<f32>,
    pub xq: QuantizedTensor,
    pub hq: QuantizedTensor,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub att: Vec<f32>,
    pub logits: Vec<f32>,
    pub key_cache: Vec<f32>,
    pub value_cache: Vec<f32>,
}

impl RunState {
    /// Initialize a new RunState based on the provided configuration.
    pub fn new(config: &Config) -> RunState {
        let kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;

        RunState {
            x: vec![0.0; config.dim],
            xb: vec![0.0; config.dim],
            xb2: vec![0.0; config.dim],
            hb: vec![0.0; config.hidden_dim],
            hb2: vec![0.0; config.hidden_dim],
            xq: QuantizedTensor::new(config.dim),
            hq: QuantizedTensor::new(config.hidden_dim),
            q: vec![0.0; config.dim],
            k: vec![0.0; kv_dim],
            v: vec![0.0; kv_dim],
            att: vec![0.0; config.n_heads * config.seq_len],
            logits: vec![0.0; config.vocab_size],
            key_cache: vec![0.0; config.n_layers * config.seq_len * kv_dim],
            value_cache: vec![0.0; config.n_layers * config.seq_len * kv_dim],
        }
    }
}
pub struct TransformerWeights {
    pub q_tokens: QuantizedTensor,
    pub token_embedding_table: Vec<f32>,
    pub rms_att_weight: Vec<f32>,
    pub rms_ffn_weight: Vec<f32>,
    pub wq: Vec<QuantizedTensor>,
    pub wk: Vec<QuantizedTensor>,
    pub wv: Vec<QuantizedTensor>,
    pub wo: Vec<QuantizedTensor>,
    pub w1: Vec<QuantizedTensor>,
    pub w2: Vec<QuantizedTensor>,
    pub w3: Vec<QuantizedTensor>,
    pub rms_final_weight: Vec<f32>,
    pub wcls: Vec<QuantizedTensor>,
    // Other fields...
}

impl TransformerWeights {
    fn memory_map_weights(config: &Config, data: &[u8], shared_classifier: bool) -> Self {
        let head_size = config.dim / config.n_heads;
        let mut offset = 0;

        // Read RMS normalization weights
        let rms_att_weight = Self::read_floats(&data[offset..], config.n_layers * config.dim);
        offset += config.n_layers * config.dim * std::mem::size_of::<f32>();
        let rms_ffn_weight = Self::read_floats(&data[offset..], config.n_layers * config.dim);
        offset += config.n_layers * config.dim * std::mem::size_of::<f32>();
        let rms_final_weight = Self::read_floats(&data[offset..], config.dim);
        offset += config.dim * std::mem::size_of::<f32>();
        //print min max rms att weight
        println!("min rms_att_weight: {}, max rms_att_weight: {}", rms_att_weight.iter().cloned().fold(f32::INFINITY, f32::min), rms_att_weight.iter().cloned().fold(f32::NEG_INFINITY, f32::max));

        // Read quantized weights
        let q_tokens =
            QuantizedTensor::from_data(&data, 1, config.vocab_size * config.dim, &mut offset)
                .pop()
                .expect("Failed to create q_tokens");
        let token_embedding_table = q_tokens.dequantize(config.vocab_size * config.dim);
        //print min max token embedding table
        println!("min token_embedding_table: {}, max token_embedding_table: {}", token_embedding_table.iter().cloned().fold(f32::INFINITY, f32::min), token_embedding_table.iter().cloned().fold(f32::NEG_INFINITY, f32::max));        
        let wq = QuantizedTensor::from_data(
            &data,
            config.n_layers,
            config.dim * config.n_heads * head_size,
            &mut offset,
        );
        //print min max wq

        println!("offset: {}", offset);
        /*offset += config.n_layers
            * config.dim
            * (config.n_heads * head_size)
            * std::mem::size_of::<QuantizedTensor>(); // Adjust size accordingly
            */

        let wk = QuantizedTensor::from_data(
            &data,
            config.n_layers,
            config.n_kv_heads * head_size * config.dim,
            &mut offset,
        );
        println!("offset: {}", offset);
        /*offset += config.n_layers
            * config.dim
            * (config.n_kv_heads * head_size)
            * std::mem::size_of::<QuantizedTensor>(); // Adjust size accordingly
        */
        let wv = QuantizedTensor::from_data(
            &data,
            config.n_layers,
            config.n_kv_heads * head_size * config.dim,
            &mut offset,
        );

        /*/
        offset += config.n_layers
            * config.dim
            * (config.n_kv_heads * head_size)
            * std::mem::size_of::<QuantizedTensor>(); // Adjust size accordingly
        */
        let wo = QuantizedTensor::from_data(
            &data,
            config.n_layers,
            config.n_heads * head_size * config.dim,
            &mut offset,
        );
        println!("offset: {}", offset);
        /*
        offset += config.n_layers
            * (config.n_heads * head_size)
            * config.dim
            * std::mem::size_of::<QuantizedTensor>(); // Adjust size accordingly
        */
        let w1 = QuantizedTensor::from_data(
            &data,
            config.n_layers,
            config.dim * config.hidden_dim,
            &mut offset,
        );
        /*
        offset += config.n_layers
            * config.dim
            * config.hidden_dim
            * std::mem::size_of::<QuantizedTensor>(); // Adjust size accordingly
        */
        println!("offset: {}", offset);
        let w2 = QuantizedTensor::from_data(
            &data,
            config.n_layers,
            config.hidden_dim * config.dim,
            &mut offset,
        );
        /*offset += config.n_layers
            * config.hidden_dim
            * config.dim
            * std::mem::size_of::<QuantizedTensor>(); // Adjust size accordingly
        */
        println!("offset: {}", offset);
        let w3 = QuantizedTensor::from_data(
            &data,
            config.n_layers,
            config.hidden_dim * config.dim,
            &mut offset,
        );
        /*offset += config.n_layers
            * config.dim
            * config.hidden_dim
            * std::mem::size_of::<QuantizedTensor>(); // Adjust size accordingly
        */
        println!("offset: {}", offset);
        let wcls = if shared_classifier {
            vec![q_tokens.clone()]
        } else {
            QuantizedTensor::from_data(
                &data,
                1,
                config.dim*config.vocab_size,
                &mut offset,
            )
        };
        //offset += config.dim * config.vocab_size * std::mem::size_of::<QuantizedTensor>(); // Adjust size accordingly

        TransformerWeights {
            rms_att_weight,
            rms_ffn_weight,
            rms_final_weight,
            q_tokens,
            token_embedding_table,
            wq,
            wk,
            wv,
            wo,
            w1,
            w2,
            w3,
            wcls,
        }
    }

    fn read_floats(data: &[u8], count: usize) -> Vec<f32> {
        data.chunks_exact(std::mem::size_of::<f32>())
            .take(count)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect()
    }
    // Function to find the min and max dequantized values across a vector of QuantizedTensor
    fn print_min_max_quantized(&self) {
        //wq
        let min_quantized = self.wq.iter()
            .flat_map(|tensor| tensor.q.iter())
            .min()
            .cloned()
            .unwrap_or(i8::MIN); // Default to i8::MIN if no values are found

        let max_quantized = self.wq.iter()
            .flat_map(|tensor| tensor.q.iter())
            .max()
            .cloned()
            .unwrap_or(i8::MAX); // Default to i8::MAX if no values are found
        println!("min_quantized wq: {}, max_quantized wq: {}", min_quantized, max_quantized);

        //wk
        let min_quantized = self.wk.iter()
            .flat_map(|tensor| tensor.q.iter())
            .min()
            .cloned()
            .unwrap_or(i8::MIN); // Default to i8::MIN if no values are found
        let max_quantized = self.wk.iter()
            .flat_map(|tensor| tensor.q.iter())
            .max()
            .cloned()
            .unwrap_or(i8::MAX); // Default to i8::MAX if no values are found
        println!("min_quantized wk: {}, max_quantized wk: {}", min_quantized, max_quantized);
        //wv
        let min_quantized = self.wv.iter()
            .flat_map(|tensor| tensor.q.iter())
            .min()
            .cloned()
            .unwrap_or(i8::MIN); // Default to i8::MIN if no values are found
        let max_quantized = self.wv.iter()
            .flat_map(|tensor| tensor.q.iter())
            .max()
            .cloned()
            .unwrap_or(i8::MAX); // Default to i8::MAX if no values are found
        println!("min_quantized wv: {}, max_quantized wv: {}", min_quantized, max_quantized);
    }
}


// Functions like `init_quantized_tensors` are typically not required in Rust,
// as vector and struct initialization is more straightforward and safe.

impl QuantizedTensor {
    fn from_data(
        data: &[u8],
        num_tensors: usize,
        size_each: usize,
        offset: &mut usize,
    ) -> Vec<Self> {
        let mut tensors = Vec::with_capacity(num_tensors);
        for _ in 0..num_tensors {
            // Map quantized int8 values
            let q_slice = &data[*offset..*offset + size_each];
            let q: Vec<i8> = q_slice.iter().map(|&x| x as i8).collect();
            *offset += size_each;
    
            let num_groups = size_each / GS;
        let mut s = Vec::with_capacity(num_groups);

        for i in 0..num_groups {
            let start = *offset + i * std::mem::size_of::<f32>();
            let end = start + std::mem::size_of::<f32>();

            // Handle the case where the slice does not contain enough bytes to create a f32
            if end <= data.len() {
                let chunk = &data[start..end];

                // Convert the 4-byte slice into a 4-byte array
                let chunk_array: [u8; 4] = match chunk.try_into() {
                    Ok(arr) => arr,
                    Err(e) => {
                        eprintln!("Failed to convert chunk to array: {:?}", e);
                        break;
                    }
                };

                // Convert the 4-byte array into a f32
                let scale = f32::from_le_bytes(chunk_array);

                // Add the scale to the vector
                s.push(scale);
            } else {
                eprintln!("Scale data slice is out of bounds");
                break;
            }
        }

            *offset += num_groups * std::mem::size_of::<f32>();
    
            tensors.push(QuantizedTensor { q, s });
        }
    
        tensors
    }
}

const MAGIC_NUMBER: u32 = 0x616b3432; // "ak42" in ASCII
const EXPECTED_VERSION: i32 = 2;
const HEADER_SIZE: usize = 256;

pub fn read_checkpoint<P: AsRef<Path>>(
    checkpoint_path: P,
) -> io::Result<(Config, TransformerWeights, Mmap)> {
    let file = File::open(checkpoint_path.as_ref())?;
    let mut reader = BufReader::new(file);

    // Read and check the magic number
    let mut magic_number_buf = [0; 4];
    reader.read_exact(&mut magic_number_buf)?;
    let magic_number = u32::from_le_bytes(magic_number_buf);

    if magic_number != MAGIC_NUMBER {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Bad magic number",
        ));
    }

    // Read and check the version number
    let mut version_buf = [0; 4];
    reader.read_exact(&mut version_buf)?;
    let version = i32::from_le_bytes(version_buf);
    println!("Version: {}", version);
    if version != EXPECTED_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Bad version number",
        ));
    }

    // Read Config : we need to read 28 bytes into config which is 56 bytes
    const PACKED_DATA_SIZE: usize = 7 * 4;
    let mut config_data = [0; PACKED_DATA_SIZE];
    reader.read_exact(&mut config_data)?;
    let config = Config::from_bytes(&config_data).unwrap();

    // Read shared_classifier flag
    let mut shared_classifier_buf = [0; 1];
    reader.read_exact(&mut shared_classifier_buf)?;
    let shared_classifier = shared_classifier_buf[0] != 0;

    //Read group size as an integer
    let mut group_size_buf = [0; 4];
    reader.read_exact(&mut group_size_buf)?;
    let group_size = i32::from_le_bytes(group_size_buf);
    println!("Group size: {}", group_size);

    /*
    pad = 256 - out_file.tell() # pad rest with zeros; tell returns current pos
    assert pad >= 0
    out_file.write(b'\0' * pad)
     */
    //skip pad bytes
    let mut pad_buf = [0; 256];
    reader.read_exact(&mut pad_buf)?;

    // Memory map the weights
    let mmap = unsafe { MmapOptions::new().map(&File::open(checkpoint_path)?)? };
    let weights_data = &mmap[HEADER_SIZE..];
    let weights = TransformerWeights::memory_map_weights(&config, weights_data, shared_classifier);

    Ok((config, weights, mmap))
}

pub fn build_tokenizer(
    tokenizer_path: &str,
) -> Result<Tokenizer, Box<dyn std::error::Error + Send + Sync>> {
    // Load a pre-trained BPE model
    let tokenizer = Tokenizer::from_file(tokenizer_path)?;
    Ok(tokenizer)
}

pub fn tokenize_input(
    tokenizer: &Tokenizer,
    input: &str,
) -> Result<Vec<u32>, Box<dyn std::error::Error + Send + Sync>> {
    let output = tokenizer.encode(input, false)?;
    //println!("input ids: {:?}", output.get_ids());
    Ok(output.get_ids().to_vec())
}

pub struct ProbIndex {
    prob: f32,
    index: usize,
}

pub struct Sampler {
    vocab_size: usize,
    probindex: Vec<ProbIndex>,
    temperature: f32,
    topp: f32,
    rng: StdRng,
}

impl Sampler {
    pub fn new(vocab_size: usize, temperature: f32, topp: f32, seed: u64) -> Self {
        Sampler {
            vocab_size,
            probindex: Vec::with_capacity(vocab_size),
            temperature,
            topp,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    pub fn sample_argmax(&self, probabilities: &[f32]) -> usize {
        probabilities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    pub fn sample_multinomial(&mut self, probabilities: &[f32]) -> usize {
        let mut cdf: f32 = 0.0;
        let coin: f32 = self.rng.gen();
        for (i, &prob) in probabilities.iter().enumerate() {
            cdf += prob;
            if coin < cdf {
                return i;
            }
        }
        probabilities.len() - 1
    }

    pub fn sample_topp(&mut self, probabilities: &[f32]) -> usize {
        let cutoff = (1.0 - self.topp) / (probabilities.len() - 1) as f32;
        self.probindex.clear();
        self.probindex.extend(
            probabilities
                .iter()
                .enumerate()
                .filter_map(|(index, &prob)| {
                    if prob >= cutoff {
                        Some(ProbIndex { prob, index })
                    } else {
                        None
                    }
                }),
        );
        self.probindex
            .sort_unstable_by(|a, b| b.prob.partial_cmp(&a.prob).unwrap_or(Ordering::Equal));

        let cumulative_prob: f32 = self
            .probindex
            .iter()
            .take_while(|x| x.prob < self.topp)
            .map(|x| x.prob)
            .sum();
        let coin: f32 = self.rng.gen::<f32>() * cumulative_prob;
        let mut accumulator = 0.0;
        for probindex in &self.probindex {
            accumulator += probindex.prob;
            if accumulator > coin {
                return probindex.index;
            }
        }
        self.probindex.last().map(|x| x.index).unwrap_or(0)
    }

    pub fn sample(&mut self, logits: &[f32]) -> usize {
        let probabilities = if self.temperature != 0.0 {
            logits
                .iter()
                .map(|&logit| (logit / self.temperature).exp())
                .collect::<Vec<f32>>()
        } else {
            logits.to_vec()
        };

        let sum_probs: f32 = probabilities.iter().sum();
        let normalized_probs: Vec<f32> = probabilities
            .into_iter()
            .map(|prob| prob / sum_probs)
            .collect();

        if self.topp > 0.0 && self.topp < 1.0 {
            self.sample_topp(&normalized_probs)
        } else {
            self.sample_multinomial(&normalized_probs)
        }
    }
}

pub fn rmsnorm(o: &mut [f32], x: &[f32], weight: &[f32]) {
    assert_eq!(o.len(), x.len());
    assert_eq!(x.len(), weight.len());

    // Calculate sum of squares
    let ss: f32 = x.iter().map(|&xi| xi * xi).sum::<f32>() / x.len() as f32;
    let ss_normalized = 1.0 / (ss + 1e-5).sqrt();

    // Normalize and scale
    for (oi, (&xi, &wi)) in o.iter_mut().zip(x.iter().zip(weight.iter())) {
        *oi = wi * (ss_normalized * xi);
    }
}

pub fn softmax(x: &mut [f32]) {
    let max_val = x.iter().cloned().fold(f32::MIN, f32::max);

    let sum: f32 = x
        .iter_mut()
        .map(|xi| {
            *xi = (*xi - max_val).exp();
            *xi
        })
        .sum();

    x.iter_mut().for_each(|xi| *xi /= sum);
}


pub fn matmul(
    xout: &mut Vec<f32>,
    x: &QuantizedTensor,
    ws: &[QuantizedTensor],
    n: usize,
    d: usize,
    layer_index: usize,
    gs: usize,
) {
    let w = &ws[layer_index];
    assert_eq!(w.q.len(), d * n); // Ensure dimensions match
    assert_eq!(xout.len(), d);

    xout.par_iter_mut().enumerate().for_each(|(i, val)| {
        let mut local_val = 0.0f32;
        let mut ival = 0i32;
        let in_base = i * n;

        for j in (0..n).step_by(gs) {
            for k in 0..gs {
                let idx_x = j + k;
                let idx_w = in_base + j + k;

                if idx_x < x.q.len() && idx_w < w.q.len() {
                    ival += x.q[idx_x] as i32 * w.q[idx_w] as i32;
                }
            }

            let x_scale_idx = min(j / gs, x.s.len() - 1);
            let w_scale_idx = min((in_base + j) / gs, w.s.len() - 1);

            local_val += (ival as f32) * x.s[x_scale_idx] * w.s[w_scale_idx];
            ival = 0;
        }

        *val = local_val;
    });
}


