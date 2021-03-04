#![allow(dead_code, unused_variables, unused_imports)]

extern crate prettytable;


use std::{convert::TryInto, ops::Div};
use std::fmt;
use std::fmt::{Display, Formatter};
// use std::iter::{IntoIterator, Iterator};
use std::marker::PhantomData;
use std::ops::{Add, AddAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};

pub mod iter;
pub mod traits;
// pub mod matrix_slice;

pub use traits::*;
// pub use matrix_slice::*;

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone)]
pub struct Matrix<'a, T> {
    data: Vec<Vec<T>>,
    shape: Shape,
    _phantom: PhantomData<&'a T>,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub struct Shape {
    m: usize,
    n: usize
}

impl Shape {
    pub fn new(m: usize, n: usize) -> Self {
        Self {
            m, 
            n,
        }
    }
}

impl<'a, T> Matrix<'a, T> {
    pub fn identity(shape: usize) -> Self 
    where 
        T: From<i32> + Copy 
    {
        let mut matrix_data: Vec<Vec<T>> = Vec::new();
        for i in 0 .. shape {
            matrix_data[i][i] = 1.into();
        }
        Self::from_array(matrix_data)
    }

    
    pub fn get_minor(&self, idx: (usize, usize)) -> T
    where
        T: Copy + From<i32> + Eq + Display,
    {
        let mut minor: Matrix<'a, T> = Matrix::new((self.shape.m - 1, self.shape.n - 1));

        for i in self.data.iter().enumerate() {
            for j in i.1.iter().enumerate() {
                if i.0 != idx.0 && j.0 != idx.1 {
                    minor[(i.0, j.0)] = self[(i.0, j.0)];
                }
            }
        }
        // println!("{}", minor);  
        minor.det()
    }

    pub fn new(shape: (usize, usize)) -> Self
    where
        T: From<i32> + Copy,
    {
        Self {
            data: Vec::new(),
            shape: Shape {m: shape.0, n: shape.1},
            _phantom: PhantomData,
        }
    }

    pub fn from_array(array: Vec<Vec<T>>) -> Self 
    where 
        T: Copy 
    {
        for i in array.iter().enumerate().next() {
            if i.1.len() != array[i.0 - 1].len() {
                panic!("TODO")
            }
        }
        Self {
            // data: array,
            shape: Shape {m: array.len(), n: array[0].len()},
            data: array,
            _phantom: PhantomData,
        }   
    }

    pub(crate) fn flatten_to_vec(&self) -> Vec<T> 
    where 
        T: Copy
    {
        self.data.iter().flatten().copied().collect()
    }

    pub fn add_checked(&self, other: Self) -> Result<Self, ()> 
    where 
        T: Add<Output = T> + Copy + From<i32>
    {
        if self.shape != other.shape {
            Err(())
        } else {
            Ok(self.clone() + other)
        }
    }

    pub fn multiply_checked(&self, other: Self) -> Result<Self, ()> 
    where
        T: Add<Output = T> + AddAssign + Mul<Output = T> + Copy + From<i32>,
    {
        if self.shape.n != other.shape.m {
            Err(())
        } else {
            Ok(self.clone() * other)
        }
    }
    
    pub fn sub_checked(&self, other: Self) -> Result<Self, ()> 
    where
        T: Sub<Output = T> + Copy + From<i32>,
    {
        if self.shape != other.shape {
            Err(())
        } else {
            Ok(self.clone() - other)
        }
    }
}


impl<'a, T: 'a> Index<(usize, usize)> for Matrix<'a, T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.data[index.0][index.1]
    }
}

impl<'a, T: 'a> IndexMut<(usize, usize)> for Matrix<'a, T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.data[index.0][index.1]
    }
}

impl<'a, T> Add for Matrix<'a, T>
where
    T: Add<Output = T> + Copy + From<i32>,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        if self.shape != rhs.shape {
            panic!("TODO")
        }

        let mut out: Self = Matrix::new((self.shape.m, self.shape.n));
        for i in 0..self.shape.m {
            for j in 0..self.shape.n {
                out[(i, j)] = self[(i, j)] + rhs[(i, j)];
            }
        }
        out
    }
}

impl<'a, T> AddAssign for Matrix<'a, T>
where
    T: Add<Output = T> + Copy + From<i32> + AddAssign,
{
    fn add_assign(&mut self, rhs: Self) {
        if self.shape != rhs.shape {
            panic!("TODO")
        }
        for i in 0..self.shape.m {
            for j in 0..self.shape.n {
                self[(i, j)] += rhs[(i, j)];
            }
        }
    }
}

impl<'a, T> Sub for Matrix<'a, T>
where
    T: Sub<Output = T> + Copy + From<i32>,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        if self.shape != rhs.shape {
            panic!("Shape error");
        }

        let mut out: Self = Matrix::new((self.shape.m, self.shape.n));
        for i in 0..self.shape.m {
            for j in 0..self.shape.n {
                out[(i, j)] = self[(i, j)] - rhs[(i, j)];
            }
        }
        out
    }
}

impl<'a, T> SubAssign for Matrix<'a, T>
where
    T: Sub<Output = T> + Copy + From<i32> + SubAssign,
{
    fn sub_assign(&mut self, rhs: Self) {
        if self.shape != rhs.shape {
            panic!("Shape error");
        }
        for i in 0..self.shape.m {
            for j in 0..self.shape.n {
                self[(i, j)] -= rhs[(i, j)];
            }
        }
    }
}

impl<'a, T> Mul<T> for Matrix<'a, T>
where
    T: Mul<Output = T> + Copy + From<i32>,
{
    type Output = Self;
    /// Implementation of matrix scaling
    fn mul(self, scalar: T) -> Self::Output {
        let mut out: Self = Matrix::new((self.shape.m, self.shape.n));
        for i in 0..self.shape.m {
            for j in 0..self.shape.n {
                out[(i, j)] = self[(i, j)] * scalar;
            }
        }
        out
    }
}


impl<'a, T> MulAssign<T> for Matrix<'a, T>
where
    T: Mul<Output = T> + Copy + From<i32> + MulAssign,
{
    fn mul_assign(&mut self, scalar: T) {
        for i in 0..self.shape.m {
            for j in 0..self.shape.n {
                self[(i, j)] *= scalar;
            }
        }
    }
}

/*

A_mn * B_np = C_mp
*/

impl<'a, T> Mul<Matrix<'a, T>> for Matrix<'a, T>
where
    T: Add<Output = T> + AddAssign + Mul<Output = T> + Copy + From<i32>,
{
    type Output = Matrix<'a, T>;

    fn mul(self, rhs: Matrix<'a, T>) -> Self::Output {
        if self.shape.n != rhs.shape.m {
            panic!("Invalid shape to multiply matrices");
        }

        let m = self.shape.m;
        let n = self.shape.n;
        let p = rhs.shape.n;

        let mut out: Self::Output = Matrix::new((n, p));

        for i in 0..m {
            for j in 0..p {
                let mut sum: T = 0.into();
                for k in 0..n {
                    sum += self[(i, k)] * rhs[(k, j)];
                }
                out[(i, j)] = sum;
            }
        }
        out
    }
}

impl<'a, T> Transpose for Matrix<'a, T> 
where 
    T: Copy + From<i32>
{
    type Output = Matrix<'a, T>;

    fn transpose(&self) -> Self::Output {
        let mut transposed_matrix: Self::Output = Matrix::new((self.shape.n, self.shape.m));

        for i in self.data.iter().enumerate() {
            for j in i.1.iter().enumerate() {
                transposed_matrix[(j.0, i.0)] = self[(i.0, j.0)];
            }
        }

        transposed_matrix
    }
}

// impl<'a, T, const N: usize> Square for Matrix<'a, T, N, N> {}

impl<'a, T> Inverse for Matrix<'a, T> 
where 
    T: Copy + From<i32> + Eq + Div<Output = T>
{
    type Output = Self;
    fn inv(&self) -> <Self as crate::traits::Inverse>::Output {
        if self.shape.m != self.shape.n {
            panic!("Matrix is not invertable");
        }
        if self.det() == 0.into() {
            panic!("Matrix is not invertable");
        } else {
            // (1.into() / self.det()) * self.adj()
            todo!()
        }
    }   
}

impl<'a, T> Adjugate for Matrix<'a, T> 
where 
    T: Copy + From<i32>
{
    type Output = Self;
    fn adj(&self) -> <Self as crate::traits::Adjugate>::Output {
        self.cof().transpose()
    }
}

impl<'a, T> CofactorMatrix for Matrix<'a, T> 
where 
    T: Copy + From<i32>
{
    type Output = Self;
    fn cof(&self) -> <Self as crate::traits::CofactorMatrix>::Output {
        if self.shape.m != self.shape.n {
            panic!("Cofactor matrix cannot be found for rectangular matrices");
        }
        todo!()
    }
}


impl<'a, T> Determinant for Matrix<'a, T> 
where
    T: Eq + From<i32>
{
    type Output = T;
    fn det(&self) -> <Self as crate::traits::Determinant>::Output {
        /* let mut n_1: i32;
        let mut n_2: i32;
        let mut det: T = 1.into();
        let mut index: usize;
        let mut total = 2;

        if self.shape.m != self.shape.n {
            panic!("Cannot find the determinant of a rectangular matrix");
        }

        for i in self.data.iter().enumerate() {
            index = i.0;

            while self[(index as usize, i.0)] == 0.into() && index < self.shape.n {
                index += 1;
            }

            if index == self.shape.n {
                continue;
            }

            if index != i.0 {
                for j in 0 .. self.shape.n {
                    std::mem::swap(&mut self[(index, j)], &mut self[(i.0, j)]);
                }
            }
        }
        det */
        todo!()
    }
}


impl<'a, T> Display for Matrix<'a, T>
where
    T: Display,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut display = String::new();

        for i in self.data.iter() {
            for j in i.iter() {
                display.push_str(&format!("{}\t", j));
            }
            display.push('\n');
        }

        write!(f, "{}", display)
    }
}




#[cfg(test)]
mod tests {
    use crate::*;
    #[test]
    fn test_init() {
        let m: Matrix<i32, 2, 2> = Matrix::new();
        assert_eq!(m, Matrix::from_array([[0, 0], [0, 0]]))
    }

    #[test]
    fn test_identity_init() {
        let m: Matrix<i32, 3, 3> = Matrix::identity();
        assert_eq!(m, Matrix::from_array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    }

    /* #[test]
    fn test_get_col() {
        let m: Matrix<i32, 3, 3> = Matrix::from_array([[1, 2, 3], [5, 6, 3], [5, 1, 5]]);
        let mr: [&'a ]
        assert_eq!(m.get_col(0), MatrixSlice::Column([1, 5, 5]))
    } */

    #[test]
    fn test_add() {
        let a = Matrix::from_array([[1, 2], [3, 4]]);
        let b = Matrix::from_array([[4, 3], [2, 1]]);
        assert_eq!(a + b, Matrix::from_array([[5, 5], [5, 5]]))
    }

    #[test]
    fn test_add_assign() {
        let mut a = Matrix::from_array([[1, 2], [3, 4]]);
        let b = Matrix::from_array([[4, 3], [2, 1]]);
        a += b;
        assert_eq!(a, Matrix::from_array([[5, 5], [5, 5]]))
    }

    #[test]
    fn test_sub() {
        let a = Matrix::from_array([[1, 2], [3, 4]]);
        let b = Matrix::from_array([[4, 3], [2, 1]]);
        assert_eq!(a - b, Matrix::from_array([[-3, -1], [1, 3]]))
    }

    #[test]
    fn test_sub_assign() {
        let mut a = Matrix::from_array([[1, 2], [3, 4]]);
        let b = Matrix::from_array([[4, 3], [2, 1]]);
        a -= b;
        assert_eq!(a, Matrix::from_array([[-3, -1], [1, 3]]))
    }

    #[test]
    fn test_scalar_mul() {
        let a = Matrix::from_array([[3, 1], [2, 6]]);
        let s = 5;
        assert_eq!(a * s, Matrix::from_array([[15, 5], [10, 30]]))
    }

    #[test]
    fn test_scalar_mul_assign() {
        let mut a = Matrix::from_array([[3, 1], [2, 6]]);
        a *= 5;
        assert_eq!(a, Matrix::from_array([[15, 5], [10, 30]]))
    }

    #[test]
    fn test_matrix_multiplication_1() {
        // test multiplication with two NxN matrices
        let a = Matrix::from_array([[3, 1], [2, 6]]);
        let b = Matrix::from_array([[6, 2], [0, 5]]);
        assert_eq!(a * b, Matrix::from_array([[18, 11], [12, 34]]))
    }

    #[test]
    fn test_matrix_multiplication_2() {
        // test multiplication with MxN and NxP matrices
        let a = Matrix::from_array([[3, 6], [3, 1], [6, 3]]);
        let b = Matrix::from_array([[2, 5, 7], [6, 2, 0]]);
        let c = Matrix::from_array([[42, 27, 21], [12, 17, 21], [30, 36, 42]]);
        assert_eq!(a * b, c)
    }

    #[test]
    fn test_2x2_determinant() {
        let a = Matrix::from_array([[3, 1], [2, 6]]);
        assert_eq!(a.det(), 16);
    }
}
