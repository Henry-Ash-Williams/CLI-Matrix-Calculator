use std::error::Error;

use matrix::*;

fn main() -> Result<(), Box<dyn Error>> {
    let a: Matrix<f64> = matrix![[9.0, 10.0, 3.0], [10.0, 7.0, 6.0], [2.0, 1.0, 10.0]];

    println!("A = \n{}", a);
    println!("inv(A) =\n{}", a.inv());

    println!("A * inv(A) =\n{}", a.clone() * a.inv());

    Ok(())
}
