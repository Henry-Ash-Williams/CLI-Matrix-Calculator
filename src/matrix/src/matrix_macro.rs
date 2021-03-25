// this isn't going to be used in the final project, it's just to assist with testing


/// Create a matrix
#[macro_export]
macro_rules! matrix {
    ($(
        [$($x:expr),*]
    ),*) => {{
        let mut tmp: Vec<Vec<_>> = Vec::new();
        
        $(
            let mut tmp_row: Vec<_> = Vec::new();
            $(
                tmp_row.push($x);
            )*
            tmp.push(tmp_row);
        )*
        crate::Matrix::from_array(tmp)
    }}
}
