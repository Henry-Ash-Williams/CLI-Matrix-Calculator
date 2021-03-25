extern crate num;

use std::ops::Div;
use std::fmt;
use std::fmt::{Display, Formatter, Debug};
use std::marker::PhantomData;
use std::ops::{Add, AddAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};
use std::str::FromStr;

use num::traits::*;

pub mod traits;
pub mod matrix_macro;

pub use traits::*;

#[derive(Debug, Eq, PartialEq, PartialOrd, Ord, Clone)]
pub struct Matrix<'a, T> {
    data: Vec<Vec<T>>,
    shape: Shape,
    _phantom: PhantomData<&'a T>,
}

#[derive(Debug, Eq, PartialEq, PartialOrd, Ord, Clone, Copy)]
pub struct Shape {
    pub m: usize,
    pub n: usize
}

impl Shape {
    pub fn new(m: usize, n: usize) -> Self {
        Self {
            m, 
            n,
        }
    }

    pub fn as_tuple(&self) -> (usize, usize) {
        (self.m, self.n)
    }
}

impl<'a, T: Float> Matrix<'a, T> {
    /// Creates an identity matrix
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
    
    /// Wrapper function for From trait    
    pub fn from_string<S>(input: S) -> Self 
    where
        S: AsRef<str>,
        T: FromStr + Copy + Debug,
        <T as FromStr>::Err: Debug,

    {
        From::from(input)
    }
    
    /// Gets the minor matrix 
    pub fn get_minor(&self, m: usize, n: usize) -> T
    where
        T: Copy + From<i32> + PartialEq + Float,
        Matrix<'a, T>: Determinant<Output = T> 
    {
        let mut minor_data: Vec<Vec<T>> = Vec::new();

        for i in self.data.iter().enumerate() {
            let mut minor_data_row: Vec<T> = Vec::new();
            for j in i.1.iter().enumerate() {
                if i.0 != m && j.0 != n {
                    minor_data_row.push(self[(i.0, j.0)]);
                }
            }
            minor_data.push(minor_data_row); 
        }

        match self.shape_tuple() {
            (2, 2) => (*minor_data.iter().flatten().nth(0).unwrap()).abs(),
            _ => Matrix::from_array(minor_data).det()
        }
    }
    
    /// Creates a new matrix of mxn, filled with zeros
    pub fn new(m: usize, n: usize) -> Self
    where
        T: From<i32> + Copy,
    {
        let mut temp_m = Vec::new();
        
        for _ in 0 .. m {
            let mut temp_n = Vec::new();
            for _ in 0 .. n {
                temp_n.push(0.into());
            }
            temp_m.push(temp_n);
        }

        Self {
            data: temp_m,
            shape: Shape {m, n},
            _phantom: PhantomData,
        }
    }
    
    /// Creates a matrix from an existing 2D vector
    pub fn from_array(array: Vec<Vec<T>>) -> Self 
    where 
        T: Copy 
    {
        Self {
            shape: Shape {m: array.len(), n: array[0].len()},
            data: array,
            _phantom: PhantomData,
        }   
    }
    
    /// Getter method for the matrix shape
    pub fn shape(&self) -> Shape {
        self.shape 
    }
    
    /// Same as above, but returns it as a tuple of values 
    pub fn shape_tuple(&self) -> (usize, usize) {
        self.shape.as_tuple()
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

    pub fn mul_checked(&self, other: Self) -> Result<Self, ()> 
    where
        T: Add<Output = T> + AddAssign + Mul<Output = T> + Copy + From<i32>,
    {
        if self.shape.n != other.shape.m {
            println!("Cannot multiply matrix of shape {:?} by matrix of shape {:?}", self.shape, other.shape);
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
    T: Add<Output = T> + Copy + From<i32> + Float,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        if self.shape != rhs.shape {
            panic!("TODO")
        }

        let mut out: Self = Matrix::new(self.shape.m, self.shape.n);
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
    T: Sub<Output = T> + Copy + From<i32> + Float,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        if self.shape != rhs.shape {
            panic!("Shape error");
        }

        let mut out: Self = Matrix::new(self.shape.m, self.shape.n);
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
    T: Mul<Output = T> + Copy + From<i32> + Float,
{
    type Output = Self;
    /// Implementation of matrix scaling
    fn mul(self, scalar: T) -> Self::Output {
        let mut out: Self = Matrix::new(self.shape.m, self.shape.n);
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

impl<'a, T> Mul<Matrix<'a, T>> for Matrix<'a, T>
where
    T: Add<Output = T> + AddAssign + Mul<Output = T> + Copy + From<i32> + Float,
{
    type Output = Matrix<'a, T>;

    fn mul(self, rhs: Matrix<'a, T>) -> Self::Output {
        if self.shape.n != rhs.shape.m {
            panic!("Invalid shape to multiply matrices");
        }

        let m = self.shape.m;
        let n = self.shape.n;
        let p = rhs.shape.n;

        let mut out: Self::Output = Matrix::new(m, p);

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
    T: Copy + From<i32> + Float
{
    type Output = Matrix<'a, T>;

    fn transpose(&self) -> Self::Output {
        let mut transposed_matrix: Self::Output = Matrix::new(self.shape.n, self.shape.m);

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
    T: Copy + From<i32> + PartialEq + Div<Output = T> + MulAssign + Mul<Output = T> + Sub<Output = T> + Float + Debug + AddAssign + Display + SubAssign,
    Matrix<'a, T>: Mul<Output = Self>
{
    type Output = Self;
    fn inv(&self) -> <Self as crate::traits::Inverse>::Output {
        if self.shape.m != self.shape.n {
            panic!("Matrix is not invertable");
        }
        if self.det() == 0.into() {
            panic!("Matrix is not invertable");
        } else {
            // (num::NumCast::from(1.0).unwrap() / self.det()) * self.adj()
            todo!()
        }
    }   
}

impl<'a, T> Adjugate for Matrix<'a, T> 
where 
    T: Copy + From<i32> + PartialEq + MulAssign + Sub<Output = T> + Mul<Output = T> + Float + Debug + AddAssign + Display + SubAssign
{
    type Output = Self;
    fn adj(&self) -> <Self as crate::traits::Adjugate>::Output {
        self.cof().transpose()
    }
}

impl<'a, T> CofactorMatrix for Matrix<'a, T> 
where 
    T: Copy + From<i32> + PartialEq + MulAssign + Sub<Output = T> + Mul<Output = T> + Float + Debug + AddAssign + Display + SubAssign, 
{
    type Output = Self;
    fn cof(&self) -> <Self as crate::traits::CofactorMatrix>::Output {
        let mut cofactor_matrix = Matrix::new(self.shape.m, self.shape.n);
        if self.shape.m != self.shape.n {
            panic!("Cofactor matrix cannot be found for rectangular matrices");
        }
        

        if self.shape_tuple() == (2, 2) {
            // Janky solution for 2x2 edge case, i would simplify this to an n * get_minor, but it
            // causes an issue regarding generic types and such 
            cofactor_matrix[(0, 0)] = num::NumCast::from(1.0).unwrap();
            cofactor_matrix[(0, 1)] = num::NumCast::from(-1.0).unwrap();
            cofactor_matrix[(1, 0)] = num::NumCast::from(-1.0).unwrap();
            cofactor_matrix[(1, 1)] = num::NumCast::from(1.0).unwrap(); 
            cofactor_matrix[(0, 0)] *= self.get_minor(0, 0);
            cofactor_matrix[(0, 1)] *= self.get_minor(0, 1);
            cofactor_matrix[(1, 0)] *= self.get_minor(1, 0);
            cofactor_matrix[(1, 1)] *= self.get_minor(1, 1);

        } else {
            for i in self.data.iter().enumerate() {
                for j in i.1.iter().enumerate() {
                    let curr_pos = (i.0 * self.shape.m) + j.0 + 1; 

                    if curr_pos % 2 == 0 {
                        cofactor_matrix[(i.0, j.0)] = num::NumCast::from(-1.0).unwrap();
                    } else {
                        cofactor_matrix[(i.0, j.0)] = num::NumCast::from(1.0).unwrap(); 
                    }

                    cofactor_matrix[(i.0, j.0)] *= self.get_minor(i.0, j.0);
                }
            }
        }
        cofactor_matrix
    }
}

impl<'a, T: Copy + FromStr + Debug, S: AsRef<str>> From<S> for Matrix<'a, T> 
where
    T: Copy + FromStr + Debug + Float,
    <T as FromStr>::Err: Debug
{
    fn from(input: S) -> Matrix<'a, T> {
        let mut input: String = input.as_ref().to_string();
        
        input.remove(0); 
        input.remove(input.len() - 1);

        let rows: Vec<&str> = input.split(';').collect();
        let data_raw: Vec<Vec<&str>> = rows.iter().map(|r| r.split(',').collect()).collect();
        let mut data: Vec<Vec<T>> = Vec::new();

        for r in data_raw.iter() {
            let mut data_row: Vec<T> = Vec::new();
            for e in r.iter() {
                data_row.push(e.parse::<T>().unwrap());
            }
            data.push(data_row);
        }


        Matrix::from_array(data)
    }
}

impl<'a, T: Float> Determinant for Matrix<'a, T> 
where
    T: PartialEq + From<i32> + Mul<Output = T> + Sub<Output = T> + Copy + Debug + AddAssign + MulAssign + Display + SubAssign 
{
    type Output = T;
    fn det(&self) -> <Self as crate::traits::Determinant>::Output {
        if self.shape_tuple() == (1, 1) {
            self[(0, 0)]
        } else if self.shape_tuple() == (2, 2) {
            self[(0,0)] * self[(1,1)] - self[(0,1)] * self[(1,0)] 
        } else {
            let coefficients: &Vec<T> = &self.data[0];
            let mut det: T = num::NumCast::from(0.0).unwrap();

            for c in coefficients.iter().enumerate() { 
                let current_coefficient: T = *c.1;
                let mut minor_data: Vec<Vec<T>> = Vec::with_capacity(self.shape.m);
                for i in self.data.iter().enumerate() {
                    if i.0 == 0 {
                        continue ;
                    }
                    let mut minor_row: Vec<T> = Vec::with_capacity(self.shape.m);
                    for j in i.1.iter().enumerate() {
                        if j.0 == c.0 {
                            continue ;
                        }
                        minor_row.push(*j.1);
                    }
                    minor_data.push(minor_row);
                }
                let minor = Matrix::from_array(minor_data);
                if c.0 % 2 == 0 {
                    det += current_coefficient * minor.det();
                } else {
                    det -= current_coefficient * minor.det();
                }
            }

            det 
        }
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
                display.push_str(&format!("\t{}", j));
            }
            display.push('\n');
        }

        write!(f, "{}", display)
    }
}



// Rewrite tests, they still use const generics 
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
