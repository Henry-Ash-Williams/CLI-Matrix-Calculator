use std::{iter::FromIterator, marker::PhantomData};

use crate::Matrix;

pub struct Iter<'a, T> {
    iter: std::vec::IntoIter<T>,
    _phantom: PhantomData<&'a T>,
    shape: (usize, usize),
}

impl<'a, T: Copy> IntoIterator for &Matrix<'a, T> {
    type Item = T;
    type IntoIter = Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        Iter {
            iter: self.flatten_to_vec().into_iter(),
            _phantom: PhantomData,
            shape: (self.shape.m, self.shape.n),
        }
    }
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

// impl<'a, T, const M: usize, const N: usize> FromIterator<T> for Matrix<'a, T, M, N> {
//     fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
//         let mut data: Vec<T> = Vec::new();
//         // store the data in a Vec<T>
//         for i in iter {
//             data.push(i);
//         }

//         for i in data.chunks(M).enumerate() {
//         }


//         todo!()
//     }
// }