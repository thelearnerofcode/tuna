extern crate tuna;

use tuna::{ir::{Scope, Statement}, runtime::{BasicValue, Interpreter, Runtime, RuntimeValue},
           tokenizer::{tokenize, Tree}};

use std::sync::Arc;

fn main() {
    let source = include_str!("factorial.tuna");

    let tree = Tree::from_tokens(&tokenize(source));

    let mut scope = Scope::new();
    for block in tree.get_branches().unwrap() {
        let statement = Statement::from_tree(&scope, block).unwrap();
        scope.load_statement(&statement).unwrap();
    }

    let interpreter = Interpreter::new(scope);
    let factorial = interpreter.run_function(
        "factorial",
        &[Arc::new(RuntimeValue::BasicValue(BasicValue::F32(4.0)))],
    );

    // should be 24
    println!("factorial: {:#?}", factorial);
}
