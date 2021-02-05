/// Matrix determinant trait, requires `Square` trait in order to be implemented
pub trait Determinant: Square {
    type Output;
    fn det(&self) -> Self::Output;
}
/// Matrix transposition trait, can be done for any MxN matrix, hence the `Square` trait is not required
pub trait Transpose {
    type Output;
    fn transpose(&self) -> Self::Output;
}

/// Matrix inverse trait, requires `Square` trait in order to be implemented
pub trait Inverse: Square + Determinant + Adjugate {
    type Output;
    fn inv(&self) -> <Self as crate::traits::Inverse>::Output;
}
pub trait Adjugate: Square + CofactorMatrix {
    type Output;
    fn adj(&self) -> <Self as crate::traits::Adjugate>::Output;
}

pub trait CofactorMatrix: Square {
    type Output;
    fn cof(&self) -> <Self as crate::traits::CofactorMatrix>::Output;
}

/// simple marker trait to indicate the identity matrix,
/// requires the `Square` trait to be implemented
pub trait Identity: Square {}

/// Simple marker trait to indicate a square matrix
/// implemented for all `Matrix<T, N, N>`
pub trait Square {}

