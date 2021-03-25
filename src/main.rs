use std::error::Error;

use matrix::*;

fn main() -> Result<(), Box<dyn Error>> {
    let a: Matrix<f64> = matrix![[5, 7, 7, 7], [10, 1, 8, 2], [8, 9, 8, 8], [10, 10, 4, 1]];

    println!("A = \n{}", a);
    println!("det(A) = {}", a.det());

    Ok(())
}
