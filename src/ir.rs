use std::collections::HashMap;
use std::str::FromStr;

use runtime::ClosureValue;
use tokenizer::Tree;

#[derive(Debug, PartialEq, Clone)]
pub struct ClosureType {
    arguments: Vec<(String, Type)>,
    result: Box<Type>,
}

impl ClosureType {
    pub fn required_arguments(&self) -> &[(String, Type)] {
        &self.arguments
    }

    pub fn result(&self) -> &Type {
        &self.result
    }

    fn from_tree(
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

#[derive(Debug)]
pub enum TreeConvertError {
    NoSuchType(Tree),
    NotExpected(String),
    ExpectedBranch,
    ExpectedStructOrFunction,
    ExpectedAtom,
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
            Tree::Branch { .. } => Err(TreeConvertError::ExpectedAtom),
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
            Tree::String(ref string) => Err(TreeConvertError::NotExpected(string.clone())),
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
    fn from_tree(
        convert_context: Option<&ConversionContext>,
        tree: &Tree,
    ) -> Result<Type, TreeConvertError> {
        match *tree {
            Tree::Branch { .. } => Err(TreeConvertError::ExpectedAtom),
            Tree::Atom(ref string) => {
                if let Some(convert_context) = convert_context {
                    if let Some(struct_type) = convert_context.structs.get(string) {
                        return Ok(Type::Struct(struct_type.clone()));
                    }
                }
                Ok(Type::Basic(BasicType::from_tree(tree)?))
            }
            Tree::String(ref string) => Err(TreeConvertError::NotExpected(string.clone())),
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
struct ConversionContext {
    variables: HashMap<String, Type>,
    functions: HashMap<String, ClosureType>,
    structs: HashMap<String, StructType>,
}

impl Expression {
    fn get_type(&self, conversion_context: &mut ConversionContext) -> Result<Type, TypeCheckError> {
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
                            .ok_or_else(|| TypeCheckError::NoSuchMember(member_name.clone()))?;
                        Ok(member_type.clone())
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
                        .ok_or_else(|| TypeCheckError::MissingMember(member_name.clone()))?;

                    let member_expr_type = member_expression.get_type(conversion_context)?;
                    if *member_type != member_expr_type {
                        return Err(TypeCheckError::WrongType {
                            expected: member_type.clone(),
                            got: member_expr_type,
                        });
                    }
                }

                for member_name in member_expressions.keys() {
                    struct_type
                        .fields
                        .get(member_name)
                        .ok_or_else(|| TypeCheckError::NoSuchMember(member_name.clone()))?;
                }

                Ok(Type::Struct(struct_type.clone()))
            }
            Expression::GetVariable(ref name) => match conversion_context.variables.get(name) {
                Some(ty) => Ok(Type::clone(ty)),
                None => Err(TypeCheckError::NoSuchVariable(name.clone())),
            },
            Expression::GetFunction(ref name) => match conversion_context.functions.get(name) {
                Some(ty) => Ok(Type::Closure(ClosureType::clone(ty))),
                None => Err(TypeCheckError::NoSuchFunction(name.clone())),
            },
            Expression::Compare(ref lhs, ref _op, ref rhs) => {
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
                let condition_type = condition.get_type(conversion_context)?;
                match condition_type {
                    Type::Basic(BasicType::Bool) => {}
                    ty => {
                        return Err(TypeCheckError::WrongType {
                            expected: Type::Basic(BasicType::Bool),
                            got: ty,
                        });
                    }
                }

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

    fn from_tree(
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

                // check if it is a variable
                if conversion_context.variables.contains_key(string) {
                    return Ok(Expression::GetVariable(string.clone()));
                }

                // check if it is a function
                if conversion_context.functions.contains_key(string) {
                    return Ok(Expression::GetFunction(string.clone()));
                }

                Err(TreeConvertError::NotExpected(string.clone()))
            }
            Tree::Branch { ref nodes } => match nodes[0].as_atom().unwrap().as_ref() {
                "+" => {
                    let op = BinaryOperator::Add;
                    let lhs = Expression::from_tree(conversion_context, &nodes[1])?;
                    let rhs = Expression::from_tree(conversion_context, &nodes[2])?;

                    Ok(Expression::BinaryExpression(
                        Box::new(lhs),
                        op,
                        Box::new(rhs),
                    ))
                }
                "-" => {
                    let op = BinaryOperator::Subtract;
                    let lhs = Expression::from_tree(conversion_context, &nodes[1])?;
                    let rhs = Expression::from_tree(conversion_context, &nodes[2])?;

                    Ok(Expression::BinaryExpression(
                        Box::new(lhs),
                        op,
                        Box::new(rhs),
                    ))
                }
                "/" => {
                    let op = BinaryOperator::Divide;
                    let lhs = Expression::from_tree(conversion_context, &nodes[1])?;
                    let rhs = Expression::from_tree(conversion_context, &nodes[2])?;

                    Ok(Expression::BinaryExpression(
                        Box::new(lhs),
                        op,
                        Box::new(rhs),
                    ))
                }
                "*" => {
                    let op = BinaryOperator::Multiply;
                    let lhs = Expression::from_tree(conversion_context, &nodes[1])?;
                    let rhs = Expression::from_tree(conversion_context, &nodes[2])?;

                    Ok(Expression::BinaryExpression(
                        Box::new(lhs),
                        op,
                        Box::new(rhs),
                    ))
                }
                "call" => {
                    let closure = Expression::from_tree(conversion_context, &nodes[1])?;
                    let mut arguments = vec![];

                    for branch in nodes[2].get_branches().unwrap() {
                        let expr = Expression::from_tree(conversion_context, branch)?;
                        arguments.push(expr);
                    }
                    Ok(Expression::CallClosure(Box::new(closure), arguments))
                }
                "if" => {
                    let condition = Box::new(Expression::from_tree(conversion_context, &nodes[1])?);
                    let main_body = Box::new(Expression::from_tree(conversion_context, &nodes[2])?);
                    let else_body = Box::new(Expression::from_tree(conversion_context, &nodes[3])?);

                    Ok(Expression::If {
                        condition,
                        main_body,
                        else_body,
                    })
                }
                "==" => {
                    let op = ComparisonOperator::EqualTo;
                    let lhs = Expression::from_tree(conversion_context, &nodes[1])?;
                    let rhs = Expression::from_tree(conversion_context, &nodes[2])?;

                    Ok(Expression::Compare(Box::new(lhs), op, Box::new(rhs)))
                }
                "<" => {
                    let op = ComparisonOperator::LessThan;
                    let lhs = Expression::from_tree(conversion_context, &nodes[1])?;
                    let rhs = Expression::from_tree(conversion_context, &nodes[2])?;

                    Ok(Expression::Compare(Box::new(lhs), op, Box::new(rhs)))
                }
                "<=" => {
                    let op = ComparisonOperator::LessThanEqualTo;
                    let lhs = Expression::from_tree(conversion_context, &nodes[1])?;
                    let rhs = Expression::from_tree(conversion_context, &nodes[2])?;

                    Ok(Expression::Compare(Box::new(lhs), op, Box::new(rhs)))
                }
                ">" => {
                    let op = ComparisonOperator::GreaterThan;
                    let lhs = Expression::from_tree(conversion_context, &nodes[1])?;
                    let rhs = Expression::from_tree(conversion_context, &nodes[2])?;

                    Ok(Expression::Compare(Box::new(lhs), op, Box::new(rhs)))
                }
                ">=" => {
                    let op = ComparisonOperator::GreaterThanEqualTo;
                    let lhs = Expression::from_tree(conversion_context, &nodes[1])?;
                    let rhs = Expression::from_tree(conversion_context, &nodes[2])?;

                    Ok(Expression::Compare(Box::new(lhs), op, Box::new(rhs)))
                }
                "!=" => {
                    let op = ComparisonOperator::NotEqualTo;
                    let lhs = Expression::from_tree(conversion_context, &nodes[1])?;
                    let rhs = Expression::from_tree(conversion_context, &nodes[2])?;

                    Ok(Expression::Compare(Box::new(lhs), op, Box::new(rhs)))
                }
                "new" => {
                    let (first_node, rest) = nodes[1..].split_first().unwrap();
                    let struct_type_name = first_node.as_atom().unwrap();
                    let struct_type = conversion_context.structs[&struct_type_name].clone();

                    let mut member_expressions = HashMap::new();
                    for node in rest {
                        let branch = node.get_branches().unwrap();
                        let name = &branch[0].as_atom().unwrap();
                        let expr = Expression::from_tree(conversion_context, &branch[1])?;

                        member_expressions.insert(name.clone(), expr);
                    }

                    Ok(Expression::CreateStruct(struct_type, member_expressions))
                }
                "get_member" => {
                    let struct_expression = Expression::from_tree(conversion_context, &nodes[1])?;
                    let member_name = nodes[2].as_atom().unwrap();

                    Ok(Expression::GetMember(
                        Box::new(struct_expression),
                        member_name,
                    ))
                }
                string => Err(TreeConvertError::NotExpected(string.to_owned())),
            },
            Tree::String(ref string) => Ok(Expression::CreateConstantValue(ConstantValue::String(
                string.clone(),
            ))),
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
    pub fn check_type(&self, scope: &Scope) -> Result<(), TypeCheckError> {
        let conversion_context = ConversionContext {
            variables: HashMap::new(),
            functions: scope
                .functions
                .iter()
                .map(|(name, closure_value)| (name.clone(), closure_value.ty().clone()))
                .collect(),
            structs: scope
                .struct_types
                .iter()
                .map(|(name, struct_type)| (name.clone(), struct_type.clone()))
                .collect(),
        };

        match *self {
            Statement::DefineStruct { .. } => Ok(()),
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

    pub fn from_tree(scope: &Scope, tree: &Tree) -> Result<Statement, TreeConvertError> {
        let conversion_context = ConversionContext {
            variables: HashMap::new(),
            functions: scope
                .functions
                .iter()
                .map(|(name, closure_value)| (name.clone(), closure_value.ty().clone()))
                .collect(),
            structs: scope
                .struct_types
                .iter()
                .map(|(name, struct_type)| (name.clone(), struct_type.clone()))
                .collect(),
        };

        match *tree {
            Tree::Branch { ref nodes } => {
                let function_node = &nodes[0];
                match function_node {
                    Tree::Atom(ref atom_string) => match atom_string.as_ref() {
                        "def_struct" => {
                            let (first_node, rest) = nodes[1..].split_first().unwrap();
                            let name = first_node.as_atom().ok_or(TreeConvertError::ExpectedAtom)?;

                            let mut fields = HashMap::new();
                            for node in rest {
                                match *node {
                                    Tree::Branch { ref nodes } => {
                                        let name = match nodes[0] {
                                            Tree::Atom(ref string) => Ok(string),
                                            Tree::Branch { .. } => {
                                                Err(TreeConvertError::ExpectedAtom)
                                            }
                                            Tree::String(ref _string) => {
                                                Err(TreeConvertError::ExpectedAtom)
                                            }
                                        }?;

                                        let ty =
                                            Type::from_tree(Some(&conversion_context), &nodes[1])?;

                                        fields.insert(name.clone(), ty);
                                    }
                                    _ => return Err(TreeConvertError::ExpectedBranch),
                                }
                            }

                            let ty = StructType { fields };

                            Ok(Statement::DefineStruct {
                                name: name.clone(),
                                ty,
                            })
                        }
                        "def_function" => {
                            let name = nodes[1].as_atom().ok_or(TreeConvertError::ExpectedAtom)?;

                            let mut conversion_context = conversion_context.clone();
                            let ty = ClosureType::from_tree(Some(&conversion_context), &nodes[2])?;
                            conversion_context
                                .functions
                                .insert(name.clone(), ty.clone());

                            for (argument_name, argument_type) in &ty.arguments {
                                conversion_context
                                    .variables
                                    .insert(argument_name.clone(), argument_type.clone());
                            }

                            let body = Expression::from_tree(&conversion_context, &nodes[3])?;

                            Ok(Statement::DefineFunction { name, ty, body })
                        }
                        _ => Err(TreeConvertError::ExpectedStructOrFunction),
                    },
                    _ => Err(TreeConvertError::ExpectedAtom),
                }
            }
            Tree::String(ref _node) => Err(TreeConvertError::ExpectedBranch),
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

#[derive(Debug, Default, Clone)]
pub struct Scope {
    functions: HashMap<String, ClosureValue>,
    struct_types: HashMap<String, StructType>,
}

impl Scope {
    pub fn new() -> Scope {
        Scope {
            functions: HashMap::new(),
            struct_types: HashMap::new(),
        }
    }

    pub fn functions(&self) -> &HashMap<String, ClosureValue> {
        &self.functions
    }

    pub fn struct_types(&self) -> &HashMap<String, StructType> {
        &self.struct_types
    }

    pub fn get_struct_type(&self, name: &str) -> Option<&StructType> {
        self.struct_types.get(name)
    }

    pub fn load_statement(&mut self, statement: &Statement) -> Result<(), TypeCheckError> {
        statement.check_type(self)?;

        match *statement {
            Statement::DefineFunction {
                ref name,
                ref ty,
                ref body,
            } => {
                self.functions.insert(
                    name.clone(),
                    ClosureValue {
                        body: body.clone(),
                        ty: ty.clone(),
                    },
                );
            }
            Statement::DefineStruct { ref name, ref ty } => {
                self.struct_types.insert(name.clone(), ty.clone());
            }
        }
        Ok(())
    }
}
