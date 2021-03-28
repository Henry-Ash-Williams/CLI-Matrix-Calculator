use std::error::Error;

mod tokenizer;
use tokenizer::tokenize;

use matrix::*;

fn main() -> Result<(), Box<dyn Error>> {
    let mut raw = "1 / det([64,-97,39;-88,84,-24;-4,11,-37])";
    println!("{:#?}", tokenize(&mut raw));
    Ok(())
}
