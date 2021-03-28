use std::str::Chars;

use matrix::*;

#[derive(Debug)]
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

#[derive(Debug)]
pub enum Value<'a> {
    Number(f64),
    Matrix(Matrix<'a, f64>),
}

#[derive(Debug)]
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

/// Tokenize a raw string input, no support for variables yet lol
pub fn tokenize<'a, S: AsRef<str>>(raw: &mut S) -> Vec<Token<'a>> {
    let mut tokens: Vec<Token<'a>> = Vec::new();
    let raw: String = raw.as_ref().chars().filter(|c| !c.is_whitespace()).collect();
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
