use std::str::Chars;

#[macro_use]
use matrix::{Matrix, Inverse, Transpose, CofactorMatrix, Adjugate, Determinant};

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Operator {
    Addition,
    Subtraction,
    Division,
    Multiplication,
    Inverse,
    Determinant,
    Adjugate,
    Cofactor,
    Transpose,
    Unknown,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Value<'a> {
    Number(f64),
    Matrix(Matrix<'a, f64>),
}

impl<'a> Value<'a> {
    pub fn is_number(&self) -> bool {
        if let Value::Number(_) = *self { true } else { false } 
    }
    
    pub fn is_matrix(&self) -> bool {
        if let Value::Matrix(_) = *self { true } else { false } 
    }

    pub fn unwrap_number(&self) -> f64 {
        match *self {
            Value::Number(n) => n,
            Value::Matrix(_) => panic!("Cannot unwrap matrix from number variant")
        }
    }
    
    pub fn unwrap_matrixk(&self) -> Matrix<'a, f64> {
        match self {
            Value::Matrix(m) => m.clone(), 
            Value::Number(_) => panic!("Cannot unwrap matrix from number variant"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Token<'a> {
    Operator(Operator),
    OpenScope,
    CloseScope,
    Value(Value<'a>),
    Unknown(char),
    Identifier(String),
}


pub fn get_matrix(iterator: &mut Chars) -> String {
    let mut slice = String::from('[');
    
    
    while let Some(c) = iterator.next() {
        if c != ']' {
            slice.push(c);
        } else if c == ']' {
            slice.push(c);
            break ;
        }
    }

    slice
}

pub fn get_function_call(iterator: &mut Chars, start_char: char) -> String {
    let mut slice = String::from(start_char);
    
    while let Some(c) = iterator.next() {
        if c != '(' {
            slice.push(c);
        } else if c == '(' {
            break ;
        }
    }


    slice
}

pub fn get_number(iterator: &mut Chars, start: char) -> f64 {
    let mut slice: String = String::from(start);

    while let Some(c) = iterator.next() {
        if c.is_ascii_digit() || c == '.' {
            slice.push(c); 
        } else {
            break ;
        }
    }

    slice.parse::<f64>().unwrap()
}

pub fn get_identifier(iterator: &mut Chars) -> String {
    let mut slice: String = String::new();

    while let Some(c) = iterator.next() {
        if !c.is_ascii_whitespace() {
            slice.push(c);
        } else {
            slice.push(c);
            break ;
        }
    }

    slice 
}
pub fn get_matrix_from_token<'a>(token: Token<'a>) -> Matrix<'a, f64> {
    match token {
        Token::Value(v) => match v {
            Value::Matrix(m) => m,
            _ => panic!("Value must be a matrix"),
        },
        _ => panic!("Token must be a value"),
    }
}

/// Tokenize a raw string input, no support for variables yet lol
pub fn tokenize<'a, S: AsRef<str>>(raw: &mut S) -> Vec<Token<'a>> {
    let mut tokens: Vec<Token<'a>> = Vec::new();
    let mut raw = clean_input_string(raw.as_ref());
    // let raw: String = raw.as_ref().chars().filter(|c| !c.is_whitespace()).collect();
    let mut iterator = raw.chars();

    while let Some(c) = iterator.next() {
        tokens.push(match c {
            '(' => Token::OpenScope,
            ')' => Token::CloseScope,
            '+' => Token::Operator(Operator::Addition),
            '-' => Token::Operator(Operator::Subtraction),
            '*' => Token::Operator(Operator::Multiplication),
            '/' => Token::Operator(Operator::Division),
            '[' => Token::Value(Value::Matrix(Matrix::from_string(get_matrix(&mut iterator)))),
            'a'..='z' => Token::Operator(
                    match get_function_call(&mut iterator, c).as_ref() {
                        "det" => Operator::Determinant,
                        "adj" => Operator::Adjugate,
                        "inv" => Operator::Inverse,
                        "cof" => Operator::Cofactor,
                        "transpose" => Operator::Transpose,
                        _ => Operator::Unknown,
                    }
                ),
            '0'..='9' => Token::Value(Value::Number(get_number(&mut iterator, c))),
            _ => Token::Unknown(c),
        });
    }

    tokens
}

pub fn evaluate_unary<'a>(tokens: &'a mut Vec<Token<'a>>) -> Vec<Token<'a>> {
    let mut unary_evaluated = Vec::new();
    let mut iter = tokens.iter_mut();
    while let Some(i) = iter.next() {
        if let Token::Operator(t) = i {
            unary_evaluated.push(match t {
                Operator::Determinant => Token::Value(Value::Number(get_matrix_from_token(iter.next().unwrap().clone()).det())),
                Operator::Adjugate => Token::Value(Value::Matrix(get_matrix_from_token(iter.next().unwrap().clone()).adj())),
                Operator::Cofactor => Token::Value(Value::Matrix(get_matrix_from_token(iter.next().unwrap().clone()).cof())),
                Operator::Inverse => Token::Value(Value::Matrix(get_matrix_from_token(iter.next().unwrap().clone()).inv())),
                Operator::Transpose => Token::Value(Value::Matrix(get_matrix_from_token(iter.next().unwrap().clone()).transpose())),
                _ => Token::Operator(t.clone()), 
            })
        } else if let Token::Unknown(w) = i {
            if !w.is_whitespace() {
                unary_evaluated.push(Token::Unknown(*w)); 
            }
        } else {
            unary_evaluated.push(i.clone()); 
        }
    } 
    unary_evaluated
}

pub fn strip_unmatched_scope_operators<'a>(tokens: &'a mut Vec<Token<'a>>) -> Vec<Token<'a>> {
    let mut count = 0; 
    for i in tokens.into_iter() {
        if let Token::OpenScope = i {
            count += 1; 
        } else if let Token::CloseScope = i {
            count -= 1; 
        }
    }

    if count == 0 {
        tokens.clone()
    } else {
        let mut stripped: Vec<Token<'a>> = Vec::new(); 
        let mut iter = tokens.iter(); 
        let mut scope_open = false; 

        while let Some(t) = iter.next() {
            scope_open = if let Token::OpenScope = t { true } else { false }; 

            if let Token::CloseScope = t {
                if !scope_open {
                    stripped.push(t.clone()); 
                    scope_open = !scope_open; 
                    continue ;
                } else {
                    continue ;
                }
            }

            stripped.push(t.clone()); 
        }
        stripped
    }
}

pub fn clean_input_string<S: AsRef<str>>(input: S) -> String {
    let tmp: &str = input.as_ref(); 
    let mut clean = String::new(); 
    
    for c in tmp.chars() {
        if c == '(' {
            clean.push_str(" ( ");
        } else if c == ')' {
            clean.push_str(" ) "); 
        } else {
            clean.push(c);
        }
    }

    clean 
}

/* pub fn evaluate_addition(operand_1: Value<'a>, operand_2: Value<'a>, result: &mut Value<'a>) -> Value<'a> {
    let res = match (operand_1.is_number(), operand_2.is_number()) {
        (true, true) => Value::Number(operand_1.unwrap_number() + operand_2.unwrap_number()), 
        (false, true) | (true, false) => panic!("cannot add matrix to number"),
        (false, false) => Value::Matrix(operand_1.unwrap_matrix().add_checked(operand_2.unwrap_matrix()).unwrap()),
    }
    


} */ 

/* pub fn evaluate<'a>(tokens: &'a mut Vec<Token<'a>>) -> Token<'a> {
    let mut operator_stack: Vec<Operator> = Vec::new(); 
    let mut operand_stack: Vec<Value<'a>> = Vec::new(); 
    let mut iterator = tokens.into_iter(); 

    let mut result: Value = Value::Number(0.0);

    while let Some(t) = iterator.next() {
        match t {
            Token::Value(v) => operand_stack.push_back(v.clone()),
            Token::Operator(o) => { 
                let operand_1 = operand_stack.pop_back();
                let operand_2 = operand_stack.pop_back(); 

                match o { 
                    Operator::Addition => 
                    _ => () 
                }
            },
            _ => ()

        }
    }

    Token::Value(result)
} */ 

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_tokenizer_1() {
        // Passes 
        let mut input_buffer = clean_input_string(String::from("1 / 10 * [1,2,3;4,5,6;7,8,9]")); 
        let expected = vec![
            Token::Value(Value::Number(1.0)),
            Token::Operator(Operator::Division),
            Token::Value(Value::Number(10.0)),
            Token::Operator(Operator::Multiplication),
            Token::Value(Value::Matrix(matrix::matrix![[1.0,2.0,3.0], [4.0,5.0,6.0], [7.0,8.0,9.0]]))
        ]; 

        let mut tokens = tokenize(&mut input_buffer); 
        let tokens = evaluate_unary(&mut tokens); 

        assert_eq!(tokens, expected);
    }
    
    #[test]
    fn test_tokenizer_2() {
        let mut input_buffer = clean_input_string(String::from("inv([1,2,3;4,5,6;7,8,9]) * [1,2,3;4,5,6;7,8,9]")); 
        let expected = vec![
            Token::Operator(Operator::Inverse),
            Token::Value(Value::Matrix(matrix::matrix![[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]])),
            Token::Operator(Operator::Multiplication),
            Token::Value(Value::Matrix(matrix::matrix![[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]])),
        ]; 

        let mut tokens = tokenize(&mut input_buffer); 

        assert_eq!(tokens, expected);
    }
}
