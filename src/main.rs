use std::error::Error;

use matrix::*;

fn main() -> Result<(), Box<dyn Error>> {
    let a: Matrix<f64> = matrix![[1.0, 2.0], [3.0, 4.0]];

    println!("A = \n{}", a);
    println!("cof(A) = \n{}", a.cof());

    Ok(())
}
