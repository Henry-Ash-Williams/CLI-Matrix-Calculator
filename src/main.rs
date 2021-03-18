use std::error::Error;

use matrix::*;

fn main() -> Result<(), Box<dyn Error>> {
    let a: Matrix<i32> = Matrix::from_string("[1,2,3;4,5,6;7,8,9]");
    let b: Matrix<i32> = Matrix::from_string("[1;2;3]");

    println!("A = \n{}", a);
    println!("B = \n{}", b);

    match a.mul_checked(b) {
        Ok(m) => println!("A * B =\n{}", m),
        Err(_) => println!("Cannot multiply due to invalid shape"),
    };

    Ok(())
}
