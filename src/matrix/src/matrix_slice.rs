use std::fmt;

#[derive(Debug)]
pub enum MatrixSlice<'a, T, const M: usize, const N: usize> {
    Row([&'a T; M]),
    Column([&'a T; N]),
    Cell(&'a T),
}

impl<'a, T, const M: usize, const N: usize> fmt::Display for MatrixSlice<'a, T, M, N> 
where 
    T: fmt::Display 
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use MatrixSlice::*;
        let mut display = String::new();
        match self {
            Row(r) => r.iter().for_each(|e| display.push_str(&format!("{}\n", e))),            
            Column(c) => c.iter().for_each(|e| display.push_str(&format!("{}\t", e))),
            Cell(c) => display.push_str(&format!("{}", c)),            
        };
        write!(f, "{}", display)
    }   
}