use std::error::Error;
use std::io::{self, Read}; 

mod evaluator;
use evaluator::*;

use matrix::*;

fn main() -> Result<(), Box<dyn Error>> {
    loop {
        let mut tokens: Vec<evaluator::Token>; 
        let mut buffer = String::new();
        let mut stdin = io::stdin();
        stdin.read_line(&mut buffer)?; 
        let mut tokens = evaluator::tokenize(&mut buffer); 
        println!("{:?}", tokens); 
        let mut tokens = evaluator::evaluate_unary(&mut tokens); 
        println!("{:?}", tokens); 
        let mut tokens = evaluator::strip_unmatched_scope_operators(&mut tokens); 
        println!("{:?}", tokens); 
    }
    Ok(())

    /* let mut raw = "1 / det([64,-97,39;-88,84,-24;-4,11,-37])";
    let mut tokens = tokenize(&mut raw);
    println!("Tokens: {:#?}", tokens);
    let mut evaled = evaluate_unary(&mut tokens);
    println!("After unary evaluation: {:#?}", evaled); */

}
