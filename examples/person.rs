extern crate tuna;

use tuna::{ir::{Scope, Statement}, parser::{tokenize, Tree}, runtime::llvm::LLVMJit};

fn main() {
    let source = include_str!("person.tuna");

    let tree = Tree::from_tokens(&tokenize(source));
    println!("{}", tree.to_string_pretty());

    let mut scope = Scope::new();
    for block in tree.get_branches().unwrap() {
        let statement = Statement::from_tree(&scope, block).unwrap();
        scope.load_statement(&statement).unwrap();
    }

    let llvmjit = LLVMJit::new(scope);
    let person: Person = llvmjit.run_function_no_arg("create_person");
    let old_person: Person = llvmjit.run_function_1_arg("age_person", person);

    println!("person: {:#?}", person);
    println!("old_person: {:#?}", old_person);
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct Person {
    id: u64,
    alive: bool,
    age: u16,
}

/* equivalent rust:
struct Person { 
    id: u64,
    name: String,
    age: u16,
}

fn create_person() -> Person {
    Person {
        id: 0,
        name: "Owen".to_owned(),
        age: 15,
    }
}

fn age_person(person: Person) -> Person {
    Person {
        id: person.id,
        name: person.name.clone(),
        age: person.age + 1,
    }
}*/
