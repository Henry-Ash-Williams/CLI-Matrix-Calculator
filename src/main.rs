use std::error::Error;

use matrix::*;

fn main() -> Result<(), Box<dyn Error>> {
    let a: Matrix<f64> = matrix![[1.0, 9.0, 1.0], [0.0, 7.0, 5.0], [1.0, 6.0, 6.0]];

    println!("A = \n{}", a);
    println!("cof(A) = {}", a.det());

    Ok(())
}
