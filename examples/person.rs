extern crate tuna;

use tuna::{Scope, Statement, tokenizer::{tokenize, Tree}};

fn main() {
    let source = include_str!("person.tuna");

    let tree = Tree::from_tokens(&tokenize(source));
    let mut scope = Scope::new();
    for block in tree.get_branches().unwrap() {
        let statement = Statement::from_tree(&scope, block).unwrap();
        statement.check_type(&scope).unwrap();
        statement.execute(&mut scope);
    }

    let person = scope.run_function("create_person", &[]);
    let old_person = scope.run_function("age_person", &[person]);
}
