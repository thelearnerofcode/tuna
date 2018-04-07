extern crate tuna;

use tuna::{ir::{Scope, Statement}, runtime::Runtime, tokenizer::{tokenize, Tree}};

fn main() {
    let source = include_str!("person.tuna");

    let tree = Tree::from_tokens(&tokenize(source));
    let mut scope = Scope::new();
    for block in tree.get_branches().unwrap() {
        let statement = Statement::from_tree(&scope, block).unwrap();
        scope.load_statement(&statement).unwrap();
    }

    let runtime = Runtime::new(scope);
    let person = runtime.run_function("create_person", &[]);
    let old_person = runtime.run_function("age_person", &[person.clone()]);

    println!("person: {:#?}", person);
    println!("old_person: {:#?}", old_person);
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
