mod tokenizer;
use tokenizer::Tree;

use std::collections::HashMap;
use std::str::FromStr;
use std::sync::Arc;

// type syntax is:
//     BASIC TYPE:
//          u16, u32, u64, f32, f64, i16, i32, i64, string, void
//     STRUCT TYPE:
//          ((name TYPE)..) or the name of an already defined type
//     CLOSURE TYPE:
//          ((name result) TYPE)

fn main() {
    let source = include_str!("../source.tuna");

    let tree = Tree::from_tokens(&::tokenizer::tokenize(source));
    println!("{:#?}", tree);
    println!("{}", tree.to_string_pretty());

    let mut scope = Scope::new();
    for block in tree.get_branches().unwrap() {
        let conversion_context = ConversionContext {
            variables: HashMap::new(),
            functions: scope
                .functions
                .iter()
                .map(|(name, closure_value)| (name.clone(), closure_value.ty.clone()))
                .collect(),
            structs: scope
                .struct_types
                .iter()
                .map(|(name, struct_type)| (name.clone(), struct_type.clone()))
                .collect(),
        };

        let statement = Statement::from_tree(&conversion_context, block).unwrap();
        statement.get_type(&conversion_context).unwrap();
        statement.execute(&mut scope);
    }

    let function_result = scope.run_function("main", vec![]);
    println!("function result: {:?}", function_result);
}

#[derive(Debug, PartialEq, Clone)]
pub struct ClosureType {
    arguments: Vec<(String, Type)>,
    result: Box<Type>,
}

impl ClosureType {
    pub fn from_tree(
        conversion_context: Option<&ConversionContext>,
        tree: &Tree,
    ) -> Result<ClosureType, TreeConvertError> {
        let mut arguments: Vec<(String, Type)> = Vec::new();
        let nodes = tree.get_branches().unwrap();

        let (return_node, argument_node) = nodes.split_last().unwrap();

        for node in argument_node.iter() {
            let argument_branch = node.get_branches().unwrap();
            let argument_name = argument_branch[0].as_atom().unwrap();
            let argument_type = Type::from_tree(conversion_context, &argument_branch[1])?;

            arguments.push((argument_name, argument_type));
        }

        let result = Box::new(Type::from_tree(conversion_context, return_node)?);
        Ok(ClosureType { arguments, result })
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct StructType {
    fields: HashMap<String, Type>,
}

impl StructType {
    pub fn from_tree(
        convert_context: Option<&ConversionContext>,
        tree: &Tree,
    ) -> Result<StructType, TreeConvertError> {
        match *tree {
            Tree::Branch { ref nodes } => {
                let mut fields = HashMap::new();
                for node in nodes {
                    match *node {
                        Tree::Branch { ref nodes } => {
                            let name = match nodes.get(0).unwrap() {
                                Tree::Atom(ref string) => Ok(string),
                                Tree::Branch { ref nodes } => Err(TreeConvertError::ExpectedAtom),
                            }?;

                            let ty = Type::from_tree(convert_context, nodes.get(1).unwrap())?;

                            fields.insert(name.clone(), ty);
                        }
                        _ => return Err(TreeConvertError::ExpectedBranch),
                    }
                }

                Ok(StructType { fields })
            }
            Tree::Atom(ref string) => Err(TreeConvertError::ExpectedBranch),
        }
    }
}

#[derive(Debug)]
pub enum TreeConvertError {
    NoSuchType(Tree),
    ExpectedBranch,
    ExpectedStructOrFunction,
    ExpectedAtom,
    InvalidDot,
}

#[derive(Debug, PartialEq, Clone)]
pub enum BasicType {
    U16,
    U32,
    U64,

    F32,
    F64,

    I16,
    I32,
    I64,

    Void,
    Bool,
    String,
}

impl BasicType {
    pub fn from_tree(tree: &Tree) -> Result<BasicType, TreeConvertError> {
        match *tree {
            Tree::Branch { ref nodes } => Err(TreeConvertError::ExpectedAtom),
            Tree::Atom(ref string) => match string.as_ref() {
                "u16" => Ok(BasicType::U16),
                "u32" => Ok(BasicType::U32),
                "u64" => Ok(BasicType::U64),

                "f32" => Ok(BasicType::F32),
                "f64" => Ok(BasicType::F64),

                "i16" => Ok(BasicType::I16),
                "i32" => Ok(BasicType::I32),
                "i64" => Ok(BasicType::I64),

                "string" => Ok(BasicType::String),
                "void" => Ok(BasicType::Void),
                "bool" => Ok(BasicType::Bool),

                _ => Err(TreeConvertError::NoSuchType(tree.clone())),
            },
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Type {
    Basic(BasicType),
    Closure(ClosureType),
    Struct(StructType),
}

impl Type {
    pub fn from_tree(
        convert_context: Option<&ConversionContext>,
        tree: &Tree,
    ) -> Result<Type, TreeConvertError> {
        match *tree {
            Tree::Branch { nodes: _ } => Ok(Type::Struct(StructType::from_tree(convert_context, tree)?)),
            Tree::Atom(ref string) => { 
                if let Some(convert_context) = convert_context {
                    if let Some(struct_type) = convert_context.structs.get(string) {
                        return Ok(Type::Struct(struct_type.clone()));
                    }
                }
                Ok(Type::Basic(BasicType::from_tree(tree)?))
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum ConstantValue {
    U16(u16),
    U32(u32),
    U64(u64),

    F32(f32),
    F64(f64),

    I16(i16),
    I32(i32),
    I64(i64),

    Void,
    String(String),
    Bool(bool),
}

#[derive(Debug)]
pub enum TypeCheckError {
    ExpectedBasicType,
    ExpectedClosure,
    ExpectedStruct,

    WrongType { expected: Type, got: Type },
    InvalidArgumentLength { expected: usize, got: usize },
    NoSuchMember(String),
    MissingMember(String),
    NoSuchVariable(String),
    NoSuchFunction(String),
}

#[derive(Debug, Clone)]
pub enum Expression {
    // result type is determined by lhs type, which must be a basic type.
    BinaryExpression(Box<Expression>, BinaryOperator, Box<Expression>),
    // result type is determined by type of closure
    CallClosure(Box<Expression>, Vec<Expression>),
    // result type is determined by type of member
    GetMember(Box<Expression>, String),

    // result type is determined by type of constant value
    CreateConstantValue(ConstantValue),
    // result type is determined by StructType
    CreateStruct(StructType, HashMap<String, Expression>),
    // result type is the closure type of function at the strNoneing
    GetFunction(String),
    // result type is the value of the variable
    GetVariable(String),

    // result type is a bool
    Compare(Box<Expression>, ComparisonOperator, Box<Expression>),

    // main body and else body must return the same type
    If {
        condition: Box<Expression>,
        main_body: Box<Expression>,
        else_body: Box<Expression>,
    },
}

#[derive(Clone, Debug)]
pub struct ConversionContext {
    variables: HashMap<String, Type>,
    functions: HashMap<String, ClosureType>,
    structs: HashMap<String, StructType>,
}

impl Expression {
    pub fn get_type(
        &self,
        conversion_context: &mut ConversionContext,
    ) -> Result<Type, TypeCheckError> {
        match *self {
            Expression::BinaryExpression(ref lhs, ref _op, ref rhs) => {
                let lhs_type = lhs.get_type(conversion_context)?;
                let rhs_type = rhs.get_type(conversion_context)?;

                match lhs_type.clone() {
                    Type::Basic(ref _ty) => {
                        if lhs_type != rhs_type {
                            Err(TypeCheckError::WrongType {
                                expected: lhs_type,
                                got: rhs_type,
                            })
                        } else {
                            Ok(lhs_type)
                        }
                    }
                    _ => Err(TypeCheckError::ExpectedBasicType),
                }
            }
            Expression::CallClosure(ref closure_expr, ref arguments) => {
                // first lets make sure closure_expr evaluates to a closure
                let closure_expr_type = closure_expr.get_type(conversion_context)?;
                match closure_expr_type.clone() {
                    Type::Closure(ref closure_type) => {
                        // now we check to make sure that the argument count lines up
                        if arguments.len() != closure_type.arguments.len() {
                            return Err(TypeCheckError::InvalidArgumentLength {
                                expected: closure_type.arguments.len(),
                                got: arguments.len(),
                            });
                        }

                        // now we make sure that the argument types line up
                        for ((_field_name, expected_type), got_expr) in
                            closure_type.arguments.iter().zip(arguments.iter())
                        {
                            let got_type = got_expr.get_type(conversion_context)?;
                            if *expected_type != got_type {
                                return Err(TypeCheckError::WrongType {
                                    expected: expected_type.clone(),
                                    got: got_type,
                                });
                            }
                        }

                        Ok(Type::clone(&closure_type.result))
                    }
                    _ => Err(TypeCheckError::ExpectedClosure),
                }
            }
            Expression::GetMember(ref struct_expr, ref member_name) => {
                let struct_expr_type = struct_expr.get_type(conversion_context)?;
                match struct_expr_type {
                    Type::Struct(ref struct_type) => {
                        let member_type = struct_type
                            .fields
                            .get(member_name)
                            .ok_or(TypeCheckError::NoSuchMember(member_name.clone()))?;
                        return Ok(member_type.clone());
                    }
                    _ => Err(TypeCheckError::ExpectedStruct),
                }
            }
            Expression::CreateConstantValue(ref constant_value) => {
                Ok(Type::Basic(match *constant_value {
                    ConstantValue::U16(_) => BasicType::U16,
                    ConstantValue::U32(_) => BasicType::U32,
                    ConstantValue::U64(_) => BasicType::U64,

                    ConstantValue::F32(_) => BasicType::F32,
                    ConstantValue::F64(_) => BasicType::F64,

                    ConstantValue::I16(_) => BasicType::I16,
                    ConstantValue::I32(_) => BasicType::I32,
                    ConstantValue::I64(_) => BasicType::I64,

                    ConstantValue::Void => BasicType::Void,
                    ConstantValue::String(_) => BasicType::String,
                    ConstantValue::Bool(_) => BasicType::Bool,
                }))
            }
            Expression::CreateStruct(ref struct_type, ref member_expressions) => {
                for (member_name, member_type) in &struct_type.fields {
                    let member_expression = member_expressions
                        .get(member_name)
                        .ok_or(TypeCheckError::MissingMember(member_name.clone()))?;

                    let member_expr_type = member_expression.get_type(conversion_context)?;
                    if *member_type != member_expr_type {
                        return Err(TypeCheckError::WrongType {
                            expected: member_type.clone(),
                            got: member_expr_type,
                        });
                    }
                }

                for (member_name, _member_type) in member_expressions {
                    struct_type
                        .fields
                        .get(member_name)
                        .ok_or(TypeCheckError::NoSuchMember(member_name.clone()))?;
                }

                Ok(Type::Struct(struct_type.clone()))
            }
            Expression::GetVariable(ref name) => match conversion_context.variables.get(name) {
                Some(ref ty) => Ok(Type::clone(ty)),
                None => Err(TypeCheckError::NoSuchVariable(name.clone())),
            },
            Expression::GetFunction(ref name) => match conversion_context.functions.get(name) {
                Some(ref ty) => Ok(Type::Closure(ClosureType::clone(ty))),
                None => Err(TypeCheckError::NoSuchFunction(name.clone())),
            },
            Expression::Compare(ref lhs, ref op, ref rhs) => {
                let lhs_type = lhs.get_type(conversion_context)?;
                let rhs_type = rhs.get_type(conversion_context)?;

                match lhs_type.clone() {
                    Type::Basic(ref _ty) => {
                        if lhs_type != rhs_type {
                            Err(TypeCheckError::WrongType {
                                expected: lhs_type,
                                got: rhs_type,
                            })
                        } else {
                            Ok(Type::Basic(BasicType::Bool))
                        }
                    }
                    _ => Err(TypeCheckError::ExpectedBasicType),
                }
            }
            Expression::If {
                ref condition,
                ref main_body,
                ref else_body,
            } => {
                let main_type = main_body.get_type(conversion_context)?;
                let else_type = else_body.get_type(conversion_context)?;

                if main_type != else_type {
                    Err(TypeCheckError::WrongType {
                        expected: main_type,
                        got: else_type,
                    })
                } else {
                    Ok(main_type)
                }
            }
        }
    }

    pub fn from_tree(
        conversion_context: &ConversionContext,
        tree: &Tree,
    ) -> Result<Expression, TreeConvertError> {
        match *tree {
            Tree::Atom(ref string) => {
                // check if it is a constant value
                if string.ends_with("u16") {
                    let number = u16::from_str(&string.replace("u16", "")).unwrap();
                    return Ok(Expression::CreateConstantValue(ConstantValue::U16(number)));
                }

                if string.ends_with("u32") {
                    let number = u32::from_str(&string.replace("u32", "")).unwrap();
                    return Ok(Expression::CreateConstantValue(ConstantValue::U32(number)));
                }

                if string.ends_with("u64") {
                    let number = u64::from_str(&string.replace("u64", "")).unwrap();
                    return Ok(Expression::CreateConstantValue(ConstantValue::U64(number)));
                }

                if string.ends_with("f32") {
                    let number = f32::from_str(&string.replace("f32", "")).unwrap();
                    return Ok(Expression::CreateConstantValue(ConstantValue::F32(number)));
                }

                if string.ends_with("f64") {
                    let number = f64::from_str(&string.replace("f64", "")).unwrap();
                    return Ok(Expression::CreateConstantValue(ConstantValue::F64(number)));
                }

                if string.ends_with("i16") {
                    let number = i16::from_str(&string.replace("i16", "")).unwrap();
                    return Ok(Expression::CreateConstantValue(ConstantValue::I16(number)));
                }

                if string.ends_with("i32") {
                    let number = i32::from_str(&string.replace("i32", "")).unwrap();
                    return Ok(Expression::CreateConstantValue(ConstantValue::I32(number)));
                }

                if string.ends_with("i64") {
                    let number = i64::from_str(&string.replace("i64", "")).unwrap();
                    return Ok(Expression::CreateConstantValue(ConstantValue::I64(number)));
                }

                if string == "true" {
                    return Ok(Expression::CreateConstantValue(ConstantValue::Bool(true)));
                }

                if string == "false" {
                    return Ok(Expression::CreateConstantValue(ConstantValue::Bool(false)));
                }

                if conversion_context.variables.contains_key(string) {
                    return Ok(Expression::GetVariable(string.clone()));
                }

                if conversion_context.functions.contains_key(string) {
                    return Ok(Expression::GetFunction(string.clone()));
                }

                println!("{}", string);
                panic!()
            }
            Tree::Branch { ref nodes } => match nodes[0].as_atom().unwrap().as_ref() {
                "+" => {
                    let op = BinaryOperator::Add;
                    let lhs = Expression::from_tree(conversion_context, &nodes[1])?;
                    let rhs = Expression::from_tree(conversion_context, &nodes[2])?;

                    return Ok(Expression::BinaryExpression(
                        Box::new(lhs),
                        op,
                        Box::new(rhs),
                    ));
                }
                "-" => {
                    let op = BinaryOperator::Subtract;
                    let lhs = Expression::from_tree(conversion_context, &nodes[1])?;
                    let rhs = Expression::from_tree(conversion_context, &nodes[2])?;

                    return Ok(Expression::BinaryExpression(
                        Box::new(lhs),
                        op,
                        Box::new(rhs),
                    ));
                }
                "/" => {
                    let op = BinaryOperator::Divide;
                    let lhs = Expression::from_tree(conversion_context, &nodes[1])?;
                    let rhs = Expression::from_tree(conversion_context, &nodes[2])?;

                    return Ok(Expression::BinaryExpression(
                        Box::new(lhs),
                        op,
                        Box::new(rhs),
                    ));
                }
                "*" => {
                    let op = BinaryOperator::Multiply;
                    let lhs = Expression::from_tree(conversion_context, &nodes[1])?;
                    let rhs = Expression::from_tree(conversion_context, &nodes[2])?;

                    return Ok(Expression::BinaryExpression(
                        Box::new(lhs),
                        op,
                        Box::new(rhs),
                    ));
                }
                "call" => {
                    let closure = Expression::from_tree(conversion_context, &nodes[1])?;
                    let mut arguments = vec![];

                    for branch in nodes[2].get_branches().unwrap() {
                        let expr = Expression::from_tree(conversion_context, branch)?;
                        arguments.push(expr);
                    }
                    return Ok(Expression::CallClosure(Box::new(closure), arguments));
                }
                "if" => {
                    let condition = Box::new(Expression::from_tree(conversion_context, &nodes[1])?);
                    let main_body = Box::new(Expression::from_tree(conversion_context, &nodes[2])?);
                    let else_body = Box::new(Expression::from_tree(conversion_context, &nodes[3])?);

                    return Ok(Expression::If {
                        condition,
                        main_body,
                        else_body,
                    });
                }
                "==" => {
                    let op = ComparisonOperator::EqualTo;
                    let lhs = Expression::from_tree(conversion_context, &nodes[1])?;
                    let rhs = Expression::from_tree(conversion_context, &nodes[2])?;

                    return Ok(Expression::Compare(Box::new(lhs), op, Box::new(rhs)));
                }
                "<" => {
                    let op = ComparisonOperator::LessThan;
                    let lhs = Expression::from_tree(conversion_context, &nodes[1])?;
                    let rhs = Expression::from_tree(conversion_context, &nodes[2])?;

                    return Ok(Expression::Compare(Box::new(lhs), op, Box::new(rhs)));
                }
                "<=" => {
                    let op = ComparisonOperator::LessThanEqualTo;
                    let lhs = Expression::from_tree(conversion_context, &nodes[1])?;
                    let rhs = Expression::from_tree(conversion_context, &nodes[2])?;

                    return Ok(Expression::Compare(Box::new(lhs), op, Box::new(rhs)));
                }
                ">" => {
                    let op = ComparisonOperator::GreaterThan;
                    let lhs = Expression::from_tree(conversion_context, &nodes[1])?;
                    let rhs = Expression::from_tree(conversion_context, &nodes[2])?;

                    return Ok(Expression::Compare(Box::new(lhs), op, Box::new(rhs)));
                }
                ">=" => {
                    let op = ComparisonOperator::GreaterThanEqualTo;
                    let lhs = Expression::from_tree(conversion_context, &nodes[1])?;
                    let rhs = Expression::from_tree(conversion_context, &nodes[2])?;

                    return Ok(Expression::Compare(Box::new(lhs), op, Box::new(rhs)));
                }
                "!=" => {
                    let op = ComparisonOperator::NotEqualTo;
                    let lhs = Expression::from_tree(conversion_context, &nodes[1])?;
                    let rhs = Expression::from_tree(conversion_context, &nodes[2])?;

                    return Ok(Expression::Compare(Box::new(lhs), op, Box::new(rhs)));
                }
                "new" => {
                    let struct_type_name = nodes[1].as_atom().unwrap();
                    let struct_type = conversion_context
                        .structs
                        .get(&struct_type_name)
                        .unwrap()
                        .clone();

                    let mut member_expressions = HashMap::new();
                    for branch in nodes[2].get_branches().unwrap() {
                        let branch = branch.get_branches().unwrap();
                        let name = &branch[0].as_atom().unwrap();
                        let expr = Expression::from_tree(conversion_context, &branch[1])?;

                        member_expressions.insert(name.clone(), expr);
                    }
                    return Ok(Expression::CreateStruct(struct_type, member_expressions));
                }
                _ => panic!(),
            },
        }
    }

    pub fn execute(&self, state: &mut State) -> Arc<RuntimeValue> {
        match *self {
            Expression::BinaryExpression(ref lhs, ref op, ref rhs) => {
                let lhs_value = lhs.execute(state);
                let lhs_value = lhs_value.as_basic_value().unwrap();
                let rhs_value = rhs.execute(state);
                let rhs_value = rhs_value.as_basic_value().unwrap();

                match *lhs_value {
                    BasicValue::U16(ref lhs) => {
                        let rhs = rhs_value.as_u16().unwrap();

                        Arc::new(RuntimeValue::BasicValue(BasicValue::U16(match *op {
                            BinaryOperator::Add => lhs + rhs,
                            BinaryOperator::Subtract => lhs - rhs,
                            BinaryOperator::Multiply => lhs * rhs,
                            BinaryOperator::Divide => lhs / rhs,
                        })))
                    }
                    BasicValue::U32(ref lhs) => {
                        let rhs = rhs_value.as_u32().unwrap();

                        Arc::new(RuntimeValue::BasicValue(BasicValue::U32(match *op {
                            BinaryOperator::Add => lhs + rhs,
                            BinaryOperator::Subtract => lhs - rhs,
                            BinaryOperator::Multiply => lhs * rhs,
                            BinaryOperator::Divide => lhs / rhs,
                        })))
                    }
                    BasicValue::U64(ref lhs) => {
                        let rhs = rhs_value.as_u64().unwrap();

                        Arc::new(RuntimeValue::BasicValue(BasicValue::U64(match *op {
                            BinaryOperator::Add => lhs + rhs,
                            BinaryOperator::Subtract => lhs - rhs,
                            BinaryOperator::Multiply => lhs * rhs,
                            BinaryOperator::Divide => lhs / rhs,
                        })))
                    }

                    BasicValue::F32(ref lhs) => {
                        let rhs = rhs_value.as_f32().unwrap();

                        Arc::new(RuntimeValue::BasicValue(BasicValue::F32(match *op {
                            BinaryOperator::Add => lhs + rhs,
                            BinaryOperator::Subtract => lhs - rhs,
                            BinaryOperator::Multiply => lhs * rhs,
                            BinaryOperator::Divide => lhs / rhs,
                        })))
                    }
                    BasicValue::F64(ref lhs) => {
                        let rhs = rhs_value.as_f64().unwrap();

                        Arc::new(RuntimeValue::BasicValue(BasicValue::F64(match *op {
                            BinaryOperator::Add => lhs + rhs,
                            BinaryOperator::Subtract => lhs - rhs,
                            BinaryOperator::Multiply => lhs * rhs,
                            BinaryOperator::Divide => lhs / rhs,
                        })))
                    }

                    BasicValue::I16(ref lhs) => {
                        let rhs = rhs_value.as_i16().unwrap();

                        Arc::new(RuntimeValue::BasicValue(BasicValue::I16(match *op {
                            BinaryOperator::Add => lhs + rhs,
                            BinaryOperator::Subtract => lhs - rhs,
                            BinaryOperator::Multiply => lhs * rhs,
                            BinaryOperator::Divide => lhs / rhs,
                        })))
                    }
                    BasicValue::I32(ref lhs) => {
                        let rhs = rhs_value.as_i32().unwrap();

                        Arc::new(RuntimeValue::BasicValue(BasicValue::I32(match *op {
                            BinaryOperator::Add => lhs + rhs,
                            BinaryOperator::Subtract => lhs - rhs,
                            BinaryOperator::Multiply => lhs * rhs,
                            BinaryOperator::Divide => lhs / rhs,
                        })))
                    }
                    BasicValue::I64(ref lhs) => {
                        let rhs = rhs_value.as_i64().unwrap();

                        Arc::new(RuntimeValue::BasicValue(BasicValue::I64(match *op {
                            BinaryOperator::Add => lhs + rhs,
                            BinaryOperator::Subtract => lhs - rhs,
                            BinaryOperator::Multiply => lhs * rhs,
                            BinaryOperator::Divide => lhs / rhs,
                        })))
                    }
                    BasicValue::String(ref lhs) => {
                        let rhs = rhs_value.as_string().unwrap();

                        Arc::new(RuntimeValue::BasicValue(BasicValue::String(match *op {
                            BinaryOperator::Add => {
                                let mut new_string = lhs.clone();
                                new_string.push_str(rhs);
                                new_string
                            }
                            _ => panic!(),
                        })))
                    }
                    _ => panic!(),
                }
            }
            Expression::CreateConstantValue(ref constant) => {
                Arc::new(RuntimeValue::BasicValue(match *constant {
                    ConstantValue::U16(ref c) => BasicValue::U16(*c),
                    ConstantValue::U32(ref c) => BasicValue::U32(*c),
                    ConstantValue::U64(ref c) => BasicValue::U64(*c),

                    ConstantValue::F32(ref c) => BasicValue::F32(*c),
                    ConstantValue::F64(ref c) => BasicValue::F64(*c),

                    ConstantValue::I16(ref c) => BasicValue::I16(*c),
                    ConstantValue::I32(ref c) => BasicValue::I32(*c),
                    ConstantValue::I64(ref c) => BasicValue::I64(*c),

                    ConstantValue::Void => BasicValue::Void,
                    ConstantValue::String(ref string) => BasicValue::String(string.clone()),
                    ConstantValue::Bool(ref bool) => BasicValue::Bool(*bool),
                }))
            }
            Expression::CreateStruct(ref _ty, ref member_expressions) => {
                let mut members = HashMap::new();
                for (name, expr) in member_expressions {
                    members.insert(name.clone(), expr.execute(state));
                }

                Arc::new(RuntimeValue::StructValue(StructValue { members }))
            }
            Expression::GetVariable(ref name) => state.variables.get(name).unwrap().clone(),
            Expression::GetFunction(ref name) => Arc::new(RuntimeValue::ClosureValue(
                state.scope.functions.get(name).unwrap().clone(),
            )),
            Expression::CallClosure(ref closure_expr, ref argument_expressions) => {
                let closure_value = closure_expr.execute(state);
                let closure_value = closure_value.as_closure_value().unwrap();

                let mut arguments: Vec<Arc<RuntimeValue>> = vec![];
                for argument_expr in argument_expressions {
                    arguments.push(argument_expr.execute(state));
                }

                closure_value.execute(state.scope, arguments)
            }
            Expression::GetMember(ref struct_expr, ref member_name) => unimplemented!(),
            Expression::Compare(ref lhs, ref op, ref rhs) => {
                let lhs_value = lhs.execute(state);
                let lhs_value = lhs_value.as_basic_value().unwrap();
                let rhs_value = rhs.execute(state);
                let rhs_value = rhs_value.as_basic_value().unwrap();

                Arc::new(RuntimeValue::BasicValue(BasicValue::Bool(match *op {
                    ComparisonOperator::EqualTo => lhs_value == rhs_value,
                    ComparisonOperator::LessThan => lhs_value < rhs_value,
                    ComparisonOperator::LessThanEqualTo => lhs_value <= rhs_value,
                    ComparisonOperator::GreaterThan => lhs_value > rhs_value,
                    ComparisonOperator::GreaterThanEqualTo => lhs_value >= rhs_value,
                    ComparisonOperator::NotEqualTo => lhs_value != rhs_value,
                })))
            }
            Expression::If {
                ref condition,
                ref main_body,
                ref else_body,
            } => {
                let condition_value = condition.execute(state);
                let condition_value = condition_value.as_basic_value().unwrap();
                let condition_value = match *condition_value {
                    BasicValue::Bool(ref b) => *b,
                    _ => panic!(),
                };

                if condition_value {
                    main_body.execute(state)
                } else {
                    else_body.execute(state)
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum Statement {
    DefineFunction {
        name: String,
        ty: ClosureType,
        body: Expression,
    },
    DefineStruct {
        name: String,
        ty: StructType,
    },
}

impl Statement {
    pub fn execute(&self, scope: &mut Scope) {
        match *self {
            Statement::DefineFunction {
                ref name,
                ref ty,
                ref body,
            } => {
                scope.functions.insert(
                    name.clone(),
                    ClosureValue {
                        body: body.clone(),
                        ty: ty.clone(),
                    },
                );
            }
            Statement::DefineStruct { ref name, ref ty } => {
                scope.struct_types.insert(name.clone(), ty.clone());
            }
        }
    }

    pub fn get_type(&self, conversion_context: &ConversionContext) -> Result<(), TypeCheckError> {
        match *self {
            Statement::DefineStruct { name: _, ty: _ } => Ok(()),
            Statement::DefineFunction {
                ref name,
                ref ty,
                ref body,
            } => {
                let result_ty = Type::clone(&ty.result);

                let mut real_conversion_context = conversion_context.clone();
                real_conversion_context.variables = ty.arguments.iter().cloned().collect();
                real_conversion_context
                    .functions
                    .insert(name.clone(), ty.clone());

                let body_ty = body.get_type(&mut real_conversion_context)?;

                if result_ty != body_ty {
                    return Err(TypeCheckError::WrongType {
                        expected: result_ty,
                        got: body_ty,
                    });
                }

                Ok(())
            }
        }
    }

    pub fn from_tree(
        conversion_context: &ConversionContext,
        tree: &Tree,
    ) -> Result<Statement, TreeConvertError> {
        match *tree {
            Tree::Branch { ref nodes } => {
                let function_node = nodes.get(0).unwrap();
                match function_node {
                    Tree::Atom(ref atom_string) => match atom_string.as_ref() {
                        "def_struct" => {
                            let name = nodes
                                .get(1)
                                .unwrap()
                                .as_atom()
                                .ok_or(TreeConvertError::ExpectedAtom)?;
                            let ty = StructType::from_tree(
                                Some(conversion_context),
                                nodes.get(2).unwrap(),
                            )?;
                            Ok(Statement::DefineStruct {
                                name: name.clone(),
                                ty,
                            })
                        }
                        "def_function" => {
                            let name = nodes
                                .get(1)
                                .unwrap()
                                .as_atom()
                                .ok_or(TreeConvertError::ExpectedAtom)?;

                            let mut conversion_context = conversion_context.clone();
                            let ty = ClosureType::from_tree(
                                Some(&conversion_context),
                                nodes.get(2).unwrap(),
                            )?;
                            conversion_context
                                .functions
                                .insert(name.clone(), ty.clone());

                            for (argument_name, argument_type) in &ty.arguments {
                                conversion_context
                                    .variables
                                    .insert(argument_name.clone(), argument_type.clone());
                            }

                            let body =
                                Expression::from_tree(&conversion_context, nodes.get(3).unwrap())?;

                            Ok(Statement::DefineFunction { name, ty, body })
                        }
                        _ => Err(TreeConvertError::ExpectedStructOrFunction),
                    },
                    _ => Err(TreeConvertError::ExpectedAtom),
                }
            }
            Tree::Atom(ref _node) => Err(TreeConvertError::ExpectedBranch),
        }
    }
}

#[derive(Debug, Clone)]
pub enum ComparisonOperator {
    EqualTo,
    LessThan,
    LessThanEqualTo,
    GreaterThan,
    GreaterThanEqualTo,
    NotEqualTo,
}

#[derive(Debug, Clone)]
pub enum BinaryOperator {
    Add,
    Subtract,
    Multiply,
    Divide,
}

#[derive(Debug, PartialEq, PartialOrd, Clone)]
pub enum BasicValue {
    U16(u16),
    U32(u32),
    U64(u64),

    F32(f32),
    F64(f64),

    I16(i16),
    I32(i32),
    I64(i64),

    String(String),
    Bool(bool),
    Void,
}

impl BasicValue {
    pub fn as_u16(&self) -> Option<u16> {
        match *self {
            BasicValue::U16(ref v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_u32(&self) -> Option<u32> {
        match *self {
            BasicValue::U32(ref v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            BasicValue::U64(ref v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match *self {
            BasicValue::F32(ref v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            BasicValue::F64(ref v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_i16(&self) -> Option<i16> {
        match *self {
            BasicValue::I16(ref v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_i32(&self) -> Option<i32> {
        match *self {
            BasicValue::I32(ref v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match *self {
            BasicValue::I64(ref v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_string(&self) -> Option<&String> {
        match *self {
            BasicValue::String(ref string) => Some(string),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ClosureValue {
    ty: ClosureType,
    body: Expression,
}

impl ClosureValue {
    pub fn execute(
        &self,
        parent_scope: &Scope,
        arguments: Vec<Arc<RuntimeValue>>,
    ) -> Arc<RuntimeValue> {
        assert!(arguments.len() == self.ty.arguments.len());

        let mut state = State::new(parent_scope);
        // load arguments into state
        for ((argument_name, _), provided_argument) in
            self.ty.arguments.iter().zip(arguments.iter())
        {
            state
                .variables
                .insert(argument_name.clone(), provided_argument.clone());
        }

        self.body.execute(&mut state)
    }
}

#[derive(Debug, Clone)]
pub struct StructValue {
    members: HashMap<String, Arc<RuntimeValue>>,
}

#[derive(Debug, Clone)]
pub enum RuntimeValue {
    BasicValue(BasicValue),
    ClosureValue(ClosureValue),
    StructValue(StructValue),
}

impl RuntimeValue {
    pub fn as_basic_value(&self) -> Option<&BasicValue> {
        match *self {
            RuntimeValue::BasicValue(ref basic) => Some(basic),
            _ => None,
        }
    }

    pub fn as_closure_value(&self) -> Option<&ClosureValue> {
        match *self {
            RuntimeValue::ClosureValue(ref closure) => Some(closure),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct State<'a> {
    scope: &'a Scope<'a>,
    parent: Option<&'a State<'a>>,

    variables: HashMap<String, Arc<RuntimeValue>>,
}

impl<'a> State<'a> {
    pub fn new(scope: &'a Scope<'a>) -> State<'a> {
        State::new_impl(scope, None)
    }

    pub fn with_parent(scope: &'a Scope<'a>, parent: &'a State<'a>) -> State<'a> {
        State::new_impl(scope, Some(parent))
    }

    pub fn new_impl(scope: &'a Scope<'a>, parent: Option<&'a State<'a>>) -> State<'a> {
        State {
            scope,
            parent,
            variables: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Scope<'a> {
    parent: Option<&'a Scope<'a>>,

    functions: HashMap<String, ClosureValue>,
    struct_types: HashMap<String, StructType>,
}

impl<'a> Scope<'a> {
    pub fn new() -> Scope<'a> {
        Scope::new_impl(None)
    }

    pub fn with_parent(parent: &'a Scope<'a>) -> Scope<'a> {
        Scope::new_impl(Some(parent))
    }

    pub fn new_impl(parent: Option<&'a Scope<'a>>) -> Scope<'a> {
        Scope {
            parent,
            functions: HashMap::new(),
            struct_types: HashMap::new(),
        }
    }

    pub fn run_function(&self, name: &str, arguments: Vec<Arc<RuntimeValue>>) -> Arc<RuntimeValue> {
        self.functions.get(name).unwrap().execute(self, arguments)
    }
}
