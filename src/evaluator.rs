use std::str::Chars;

use matrix::*;

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
pub enum Value<'a> {
    Number(f64),
    Matrix(Matrix<'a, f64>),
}

#[derive(Debug, Clone)]
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
    let mut raw = raw.as_ref();
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
            '$' => Token::Identifier(get_identifier(&mut iterator)),
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
    let mut stripped: Vec<Token<'a>> = Vec::new(); 
    let mut iter = tokens.iter(); 
    let mut scope_open = false; 

    while let Some(t) = iter.next() {
        scope_open = if let Token::OpenScope = t { true } else { false }; 

        if let Token::CloseScope = t {
            if scope_open {
                stripped.push(t.clone()); 
                continue ;
            } else {
                continue ;
            }
        }

        stripped.push(t.clone()); 
    }
    stripped
}

// pub fn find_open_scope<'a>(iterator: &mut Iter)

