#![allow(incomplete_features, dead_code, unused_variables)]
#![feature(const_generics, const_evaluatable_checked)]

// #![recursion_limit="25"]

extern crate prettytable;


use std::{convert::TryInto, ops::Div};
use std::fmt;
use std::fmt::{Display, Formatter};
// use std::iter::{IntoIterator, Iterator};
use std::marker::PhantomData;
use std::ops::{Add, AddAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};

pub mod iter;
pub mod traits;
pub mod matrix_slice;

pub use traits::*;
pub use matrix_slice::*;

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub struct Matrix<'a, T, const M: usize, const N: usize> {
    data: [[T; N]; M],
    _phantom: PhantomData<&'a T>,
}

impl<'a, T, const N: usize> Matrix<'a, T, N, N> {
    pub fn identity() -> Self 
    where 
        T: From<i32> + Copy 
    {
        let mut matrix_data: [[T; N]; N] = [[0.into(); N]; N];
        for i in 0 .. N {
            matrix_data[i][i] = 1.into();
        }
        Self::from_array(matrix_data)
    }
}

impl<'a, T, const M: usize, const N: usize> Matrix<'a, T, M, N> {
    pub fn new() -> Self
    where
        T: From<i32> + Copy,
    {
        Self {
            data: [[0.into(); N]; M],
            _phantom: PhantomData,
        }
    }

    pub fn from_array(array: [[T; N]; M]) -> Self {
        Self {
            data: array,
            _phantom: PhantomData,
        }   
    }

    /// Returns a zero indexed row of the matrix
    pub fn get_row(&'a self, row: usize) -> MatrixSlice<'a, T, M, N> {
        let mut row_slice_vec: Vec<&'a T> = Vec::with_capacity(M);
        let row_slice: Box<[&'a T; M]>;
        for i in self.data.iter().enumerate() {
            row_slice_vec.push(&i.1[row]);
        }
        row_slice = match row_slice_vec.into_boxed_slice().try_into() {
            Ok(rs) => rs,
            Err(_) => panic!("[ERROR] Length error"),
        };

        MatrixSlice::Row(*row_slice)
    }

    pub fn get_col(&'a self, col: usize) -> MatrixSlice<'a, T, M, N> {
        let mut column_slice_vec: Vec<&'a T> = Vec::new();
        let column_slice: Box<[&'a T; N]>;
        for i in self.data[col].iter() {
            column_slice_vec.push(i);
        }
        column_slice = match column_slice_vec.into_boxed_slice().try_into() {
            Ok(cs) => cs,
            Err(o) => panic!("[Error] Length error"),
        };
        MatrixSlice::Column(*column_slice)
    }

    pub(crate) fn flatten_to_vec(&self) -> Vec<T> 
    where 
        T: Copy
    {
        self.data.iter().flatten().copied().collect()
    }
}


impl<'a, T: 'a, const M: usize, const N: usize> Index<(usize, usize)> for Matrix<'a, T, M, N> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.data[index.0][index.1]
    }
}

impl<'a, T: 'a, const M: usize, const N: usize> IndexMut<(usize, usize)> for Matrix<'a, T, M, N> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.data[index.0][index.1]
    }
}

impl<'a, T, const M: usize, const N: usize> Add for Matrix<'a, T, M, N>
where
    T: Add<Output = T> + Copy + From<i32>,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut out: Self = Matrix::new();
        for i in 0..M {
            for j in 0..N {
                out[(i, j)] = self[(i, j)] + rhs[(i, j)];
            }
        }
        out
    }
}

impl<'a, T, const M: usize, const N: usize> AddAssign for Matrix<'a, T, M, N>
where
    T: Add<Output = T> + Copy + From<i32> + AddAssign,
{
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..M {
            for j in 0..N {
                self[(i, j)] += rhs[(i, j)];
            }
        }
    }
}

impl<'a, T, const M: usize, const N: usize> Sub for Matrix<'a, T, M, N>
where
    T: Sub<Output = T> + Copy + From<i32>,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut out: Self = Matrix::new();
        for i in 0..M {
            for j in 0..N {
                out[(i, j)] = self[(i, j)] - rhs[(i, j)];
            }
        }
        out
    }
}

impl<'a, T, const M: usize, const N: usize> SubAssign for Matrix<'a, T, M, N>
where
    T: Sub<Output = T> + Copy + From<i32> + SubAssign,
{
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..M {
            for j in 0..N {
                self[(i, j)] -= rhs[(i, j)];
            }
        }
    }
}

impl<'a, T, const M: usize, const N: usize> Mul<T> for Matrix<'a, T, M, N>
where
    T: Mul<Output = T> + Copy + From<i32>,
{
    type Output = Self;
    /// Implementation of matrix scaling
    fn mul(self, scalar: T) -> Self::Output {
        let mut out: Self = Matrix::new();
        for i in 0..M {
            for j in 0..N {
                out[(i, j)] = self[(i, j)] * scalar;
            }
        }
        out
    }
}


impl<'a, T, const M: usize, const N: usize> MulAssign<T> for Matrix<'a, T, M, N>
where
    T: Mul<Output = T> + Copy + From<i32> + MulAssign,
{
    fn mul_assign(&mut self, scalar: T) {
        for i in 0..M {
            for j in 0..N {
                self[(i, j)] *= scalar;
            }
        }
    }
}

impl<'a, T, const M: usize, const N: usize, const P: usize> Mul<Matrix<'a, T, N, P>>
    for Matrix<'a, T, M, N>
where
    T: Add<Output = T> + AddAssign + Mul<Output = T> + Copy + From<i32>,
{
    type Output = Matrix<'a, T, M, P>;

    fn mul(self, rhs: Matrix<T, N, P>) -> Self::Output {
        let mut out: Self::Output = Matrix::new();

        for i in 0..M {
            for j in 0..P {
                let mut sum: T = 0.into();
                for k in 0..N {
                    sum += self[(i, k)] * rhs[(k, j)];
                }
                out[(i, j)] = sum;
            }
        }
        out
    }
}

impl<'a, T, const M: usize, const N: usize> Transpose for Matrix<'a, T, M, N> 
where 
    T: Copy + From<i32>
{
    type Output = Matrix<'a, T, N, M>;

    fn transpose(&self) -> Self::Output {
        let mut transposed_matrix: Self::Output = Matrix::new();

        for i in self.data.iter().enumerate() {
            for j in i.1.iter().enumerate() {
                transposed_matrix[(j.0, i.0)] = self[(i.0, j.0)];
            }
        }

        transposed_matrix
    }
}

impl<'a, T, const N: usize> Square for Matrix<'a, T, N, N> {}

impl<'a, T, const N: usize> Inverse for Matrix<'a, T, N, N> 
where 
    T: Copy + From<i32> + Eq + Div
{
    type Output = Self;
    fn inv(&self) -> <Self as crate::traits::Inverse>::Output {
        if self.det() == 0.into() {
            panic!("Matrix is not invertable");
        }

        // 1.into() / self.det() * self.adj()
        todo!()
    }   
}

impl<'a, T, const N: usize> Adjugate for Matrix<'a, T, N, N> 
where 
    T: Copy + From<i32>
{
    type Output = Self;
    fn adj(&self) -> <Self as crate::traits::Adjugate>::Output {
        self.cof().transpose()
    }
}

impl<'a, T, const N: usize> CofactorMatrix for Matrix<'a, T, N, N> 
where 
    T: Copy + From<i32>
{
    type Output = Self;
    fn cof(&self) -> <Self as crate::traits::CofactorMatrix>::Output {
        todo!()
    }
}


impl<'a, T, const N: usize> Determinant for Matrix<'a, T, N, N> 
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

        for i in self.data.iter().enumerate() {
            index = i.0;

            while self[(index as usize, i.0)] == 0.into() && index < N {
                index += 1;
            }

            if index == N {
                continue;
            }

            if index != i.0 {
                for j in 0 .. N {
                    std::mem::swap(&mut self[(index, j)], &mut self[(i.0, j)]);
                }
            }
        }
        det */
        todo!()
    }
}


impl<'a, T, const M: usize, const N: usize> Display for Matrix<'a, T, M, N>
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
