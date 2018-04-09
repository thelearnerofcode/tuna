extern crate tuna;

use tuna::{ir::{Scope, Statement}, parser::{tokenize, Tree}, runtime::llvm::LLVMJit};

fn main() {
    let source = include_str!("factorial.tuna");

    let tree = Tree::from_tokens(&tokenize(source));
    let mut scope = Scope::new();
    for branch in tree.get_branches().unwrap() {
        let statement = Statement::from_tree(&scope, branch).unwrap();
        scope.load_statement(&statement).unwrap();
    }

    let jit = LLVMJit::new(scope);
    let factorial: f32 = jit.run_function_1_arg("factorial", 4.0f32);

    // should be 24
    println!("factorial: {}", factorial);
}
