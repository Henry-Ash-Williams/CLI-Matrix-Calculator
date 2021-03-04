use std::io::{self, Read};

use evaluator::tokenize;
use matrix::*;

mod evaluator;

fn main() -> Result<(), io::Error> {
    /* loop {
        let mut input = String::new();
        let mut stdin = io::stdin();
        print!("> ");
        stdin.read_to_string(&mut input)?;
    } */
    tokenize("(1/10)*[1,2,3;4,5,6;7,8,9]");
    Ok(())
}
