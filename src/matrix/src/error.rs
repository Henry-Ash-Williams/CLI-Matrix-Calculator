use std::fmt;
use std::fmt::Display;

#[derive(Debug)]
pub enum ShapeErrorKind {
    ShapeNotEqual,
    MatrixMultiplicationShapeError,
}

#[derive(Debug)]
pub enum Error {
    ShapeError(ShapeErrorKind),
    NotInvertable,
    OtherError,
}

impl Display for ShapeErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShapeErrorKind::ShapeNotEqual => write!(f, "Matrix shapes are not equal"),
            ShapeErrorKind::MatrixMultiplicationShapeError => write!(f, "Matrix shapes not valid for multiplication"),
        }
    }
}
impl Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::ShapeError(e) => write!(f, "Shape error: {}", e),
            Error::NotInvertable => write!(f, "Matrix isn't invertable"),
            Error::OtherError => write!(f, "Other error"),
        }
    }
}

impl std::error::Error for Error { }
