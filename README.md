# A small, statically-typed programming language.
Heavily inspired by LISP, the syntax is based on s-expressions.
SHOULD NOT BE used in a real program (for now).

Focus is on embedability and simplicity.

# Tutorial
1. Number types
    These are the number types in tuna: u16, u32, u64, f32, f64, i16, i32, i64.
2. Structs
    Structs types are created like this: 
    `(def_struct Name (member_name member_type))`
        Ex:
        `(def_struct Person (id u64) (name string) (age u16))`
    Structs are created like this:
    `(new StructName (member_name expression))`
        Ex: 
            `(new Person
                (id (get_member person id))
                (name (get_member person name))
                (age (+ (get_member person age) 1u16)))`
    And members are accessed like this:
    `(get_member struct field)`
        Ex:
            `(get_member person id)`

3. Functions
    Functions are defined like this:
    (def_function name (argument_name argument_type) result_type expression)
        Ex:
            `(def_function age_person
                ((person Person) Person)
                    (new Person
                        (id (get_member person id))
                        (name (get_member person name))
                        (age (+ (get_member person age) 1u16))))`