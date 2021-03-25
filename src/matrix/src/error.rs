pub enum ShapeErrorKind {
    ShapeNotEqual,
    MatrixMultiplicationShapeError,
}

pub enum Error {
    ShapeError(ShapeErrorKind),
    NotInvertable,
    OtherError,
}

