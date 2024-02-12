
use std::env;
use std::time::{SystemTime, UNIX_EPOCH};
mod transformer;
use llama2_rs::{build_tokenizer, Sampler};
use transformer::Transformer;

fn time_in_ms() -> u128 {
    let start = SystemTime::now();
    let since_the_epoch = start
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards");
    since_the_epoch.as_millis()
}

fn error_usage() {
    eprintln!("Usage:   llama2_rs <checkpoint> [options]");
    eprintln!("Example: llama2_rs model.bin -n 256 -i \"Once upon a time\"");
    eprintln!("Options:");
    eprintln!("  -t <float>  temperature in [0,inf], default 1.0");
    eprintln!("  -p <float>  p value in top-p (nucleus) sampling in [0,1], default 0.9");
    eprintln!("  -s <int>    random seed, default time(NULL)");
    eprintln!("  -n <int>    number of steps to run for, default 256. 0 = max_seq_len");
    eprintln!("  -i <string> input prompt");
    eprintln!("  -z <string> optional path to custom tokenizer");
    eprintln!("  -m <string> mode: generate|chat, default: generate");
    eprintln!("  -y <string> (optional) system prompt in chat mode");
    std::process::exit(1); // Exits with a status code of 1 to indicate error
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        error_usage();
        return;
    }

    let checkpoint_path = &args[1];
    let mut tokenizer_path = "tokenizer.json".to_string();
    let mut temperature = 1.0;
    let mut topp = 0.9;
    let mut steps = 256;
    let mut prompt: Option<&str> = None;
    let mut rng_seed = 0;
    let mut mode = "generate".to_string();
    let mut system_prompt: Option<&str> = None;

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "-t" => temperature = args[i + 1].parse().unwrap_or(1.0),
            "-p" => topp = args[i + 1].parse().unwrap_or(0.9),
            "-s" => rng_seed = args[i + 1].parse().unwrap_or(0),
            "-n" => steps = args[i + 1].parse().unwrap_or(256),
            "-i" => prompt = Some(&args[i + 1]),
            "-z" => tokenizer_path = args[i + 1].clone(),
            "-m" => mode = args[i + 1].clone(),
            "-y" => system_prompt = Some(&args[i + 1]),
            _ => error_usage(),
        }
        i += 2;
    }

    let mut transformer = Transformer::new(checkpoint_path).unwrap();
    let tokenizer = build_tokenizer(&tokenizer_path).unwrap();
    let mut sampler = Sampler::new(transformer.config.vocab_size, temperature, topp, rng_seed);

    match mode.as_str() {
        "generate" => transformer.generate(&tokenizer, &mut sampler, prompt.unwrap_or(""), steps),
        "chat" => transformer.chat(&tokenizer, &mut sampler, prompt, system_prompt, steps),
        _ => {
            eprintln!("unknown mode: {}", mode);
            error_usage();
        }
    }
}
