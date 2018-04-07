use ir::{BinaryOperator, ClosureType, ComparisonOperator, ConstantValue, Expression, Scope};

use std::collections::HashMap;
use std::sync::Arc;

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
    pub(crate) ty: ClosureType,
    pub(crate) body: Expression,
}

#[derive(Debug, Clone)]
pub struct StructValue {
    pub members: HashMap<String, Arc<RuntimeValue>>,
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

    pub fn as_struct_value(&self) -> Option<&StructValue> {
        match *self {
            RuntimeValue::StructValue(ref struc) => Some(struc),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
struct State<'a> {
    parent: Option<&'a State<'a>>,
    variables: HashMap<String, Arc<RuntimeValue>>,
}

impl<'a> State<'a> {
    pub fn new() -> State<'a> {
        State::new_impl(None)
    }

    pub fn with_parent(parent: &'a State<'a>) -> State<'a> {
        State::new_impl(Some(parent))
    }

    pub fn new_impl(parent: Option<&'a State<'a>>) -> State<'a> {
        State {
            parent,
            variables: HashMap::new(),
        }
    }

    pub fn get_variable(&self, name: &str) -> Option<&Arc<RuntimeValue>> {
        match self.variables.get(name).as_ref() {
            Some(var) => Some(var),
            None => match self.parent {
                Some(parent) => parent.get_variable(name),
                None => None,
            },
        }
    }
}

pub trait Runtime {
    fn run_function(&self, name: &str, arguments: &[Arc<RuntimeValue>]) -> Arc<RuntimeValue>;
}

pub struct Interpreter {
    scope: Scope,
}

impl Interpreter {
    pub fn new(scope: Scope) -> Interpreter {
        Interpreter { scope }
    }

    fn execute(&self, expression: &Expression, state: &mut State) -> Arc<RuntimeValue> {
        match *expression {
            Expression::BinaryExpression(ref lhs, ref op, ref rhs) => {
                let lhs_value = self.execute(lhs, state);
                let lhs_value = lhs_value.as_basic_value().unwrap();
                let rhs_value = self.execute(rhs, state);
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
                    members.insert(name.clone(), self.execute(expr, state));
                }

                Arc::new(RuntimeValue::StructValue(StructValue { members }))
            }
            Expression::GetVariable(ref name) => state.get_variable(name).unwrap().clone(),
            Expression::GetFunction(ref name) => Arc::new(RuntimeValue::ClosureValue(
                self.scope.functions[name].clone(),
            )),
            Expression::CallClosure(ref closure_expr, ref argument_expressions) => {
                let closure_value = self.execute(closure_expr, state);
                let closure_value = closure_value.as_closure_value().unwrap();

                let mut arguments: Vec<Arc<RuntimeValue>> = vec![];
                for argument_expr in argument_expressions {
                    arguments.push(self.execute(argument_expr, state));
                }

                run_closure(self, closure_value, &arguments)
            }
            Expression::GetMember(ref struct_expr, ref member_name) => {
                let struc = self.execute(struct_expr, state);
                let struct_value = struc.as_struct_value().unwrap();

                struct_value.members[member_name].clone()
            }
            Expression::Compare(ref lhs, ref op, ref rhs) => {
                let lhs_value = self.execute(lhs, state);
                let lhs_value = lhs_value.as_basic_value().unwrap();
                let rhs_value = self.execute(rhs, state);
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
                let condition_value = {
                    let mut child_state = State::with_parent(state);
                    self.execute(condition, &mut child_state)
                };

                let condition_value = condition_value.as_basic_value().unwrap();
                let condition_value = match *condition_value {
                    BasicValue::Bool(ref b) => *b,
                    _ => panic!(),
                };

                if condition_value {
                    let mut child_state = State::with_parent(state);
                    self.execute(main_body, &mut child_state)
                } else {
                    let mut child_state = State::with_parent(state);
                    self.execute(else_body, &mut child_state)
                }
            }
        }
    }
}

impl Runtime for Interpreter {
    fn run_function(&self, name: &str, arguments: &[Arc<RuntimeValue>]) -> Arc<RuntimeValue> {
        let closure_value = &self.scope.functions[name];
        run_closure(self, closure_value, arguments)
    }
}

fn run_closure(
    interpreter: &Interpreter,
    closure: &ClosureValue,
    arguments: &[Arc<RuntimeValue>],
) -> Arc<RuntimeValue> {
    // todo: check argument types.
    assert!(arguments.len() == closure.ty.arguments.len());
    let mut state = State::new();

    // load arguments into state
    for ((argument_name, _), provided_argument) in closure.ty.arguments.iter().zip(arguments.iter())
    {
        state
            .variables
            .insert(argument_name.clone(), provided_argument.clone());
    }

    interpreter.execute(&closure.body, &mut state)
}
