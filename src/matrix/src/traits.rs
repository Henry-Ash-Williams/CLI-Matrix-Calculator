/// Matrix determinant trait, requires `Square` trait in order to be implemented
pub trait Determinant {
    type Output;
    fn det(&self) -> Self::Output;
}
/// Matrix transposition trait, can be done for any MxN matrix, hence the `Square` trait is not required
pub trait Transpose {
    type Output;
    fn transpose(&self) -> Self::Output;
}

/// Matrix inverse trait, requires `Square` trait in order to be implemented
pub trait Inverse: Determinant + Adjugate {
    type Output;
    fn inv(&self) -> <Self as crate::traits::Inverse>::Output;
}
pub trait Adjugate: CofactorMatrix {
    type Output;
    fn adj(&self) -> <Self as crate::traits::Adjugate>::Output;
}

pub trait CofactorMatrix {
    type Output;
    fn cof(&self) -> <Self as crate::traits::CofactorMatrix>::Output;
}

