use tokenizers::tokenizer::{Tokenizer, Result};

fn main() -> Result<()> {
    let tokenizer_path = "./tokenizer.json"; // Adjust the path as needed

    // Load the tokenizer
    let tokenizer = Tokenizer::from_file(tokenizer_path)?;

    // Example: Encode a string
    let encoding = tokenizer.encode("Hello, world!", false)?;

    println!("Encoded tokens: {:?}", encoding.get_tokens());
    println!("Token IDs: {:?}", encoding.get_ids());
    //Decode the token ids
    let decoded = tokenizer.decode(encoding.get_ids(), false);
    println!("Decoded string: {:?}", decoded);

    Ok(())
}
