use matrix::*;
// use matrix::traits::*;

fn main() {
    let a: matrix::Matrix<i32, 2, 3> = matrix::Matrix::from_array([[1, 2, 3], [4, 5, 6]]);
    println!("A:\n{}", a.transpose());
    // println!("A^T:\n{}", a.transpose());
}
