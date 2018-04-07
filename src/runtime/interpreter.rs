use std::collections::HashMap;
use std::sync::Arc;

use ir::{BinaryOperator, Closure, ComparisonOperator, ConstantValue, Expression, Scope};
use runtime::{BasicValue, Runtime, RuntimeValue, StructValue};

#[derive(Debug, Clone)]
struct LocalInterpreterState<'a> {
    parent: Option<&'a LocalInterpreterState<'a>>,
    variables: HashMap<String, Arc<RuntimeValue>>,
}

impl<'a> LocalInterpreterState<'a> {
    pub fn new() -> LocalInterpreterState<'a> {
        LocalInterpreterState::new_impl(None)
    }

    pub fn with_parent(parent: &'a LocalInterpreterState<'a>) -> LocalInterpreterState<'a> {
        LocalInterpreterState::new_impl(Some(parent))
    }

    pub fn new_impl(parent: Option<&'a LocalInterpreterState<'a>>) -> LocalInterpreterState<'a> {
        LocalInterpreterState {
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

pub struct Interpreter {
    scope: Scope,
}

impl Interpreter {
    pub fn new(scope: Scope) -> Interpreter {
        Interpreter { scope }
    }

    fn execute(
        &self,
        expression: &Expression,
        local_interpreter_state: &mut LocalInterpreterState,
    ) -> Arc<RuntimeValue> {
        match *expression {
            Expression::BinaryExpression(ref lhs, ref op, ref rhs) => {
                let lhs_value = self.execute(lhs, local_interpreter_state);
                let lhs_value = lhs_value.as_basic_value().unwrap();
                let rhs_value = self.execute(rhs, local_interpreter_state);
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
                    members.insert(name.clone(), self.execute(expr, local_interpreter_state));
                }

                Arc::new(RuntimeValue::StructValue(StructValue { members }))
            }
            Expression::GetVariable(ref name) => {
                local_interpreter_state.get_variable(name).unwrap().clone()
            }
            Expression::GetFunction(ref name) => Arc::new(RuntimeValue::Closure(
                self.scope.functions().get(name).unwrap().clone(),
            )),
            Expression::CallClosure(ref closure_expr, ref argument_expressions) => {
                let closure_value = self.execute(closure_expr, local_interpreter_state);
                let closure_value = closure_value.as_closure_value().unwrap();

                let mut arguments: Vec<Arc<RuntimeValue>> = vec![];
                for argument_expr in argument_expressions {
                    arguments.push(self.execute(argument_expr, local_interpreter_state));
                }

                run_closure(self, closure_value, &arguments)
            }
            Expression::GetMember(ref struct_expr, ref member_name) => {
                let struc = self.execute(struct_expr, local_interpreter_state);
                let struct_value = struc.as_struct_value().unwrap();

                struct_value.members[member_name].clone()
            }
            Expression::Compare(ref lhs, ref op, ref rhs) => {
                let lhs_value = self.execute(lhs, local_interpreter_state);
                let lhs_value = lhs_value.as_basic_value().unwrap();
                let rhs_value = self.execute(rhs, local_interpreter_state);
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
                    let mut child_state =
                        LocalInterpreterState::with_parent(local_interpreter_state);
                    self.execute(condition, &mut child_state)
                };

                let condition_value = condition_value.as_basic_value().unwrap();
                let condition_value = match *condition_value {
                    BasicValue::Bool(ref b) => *b,
                    _ => panic!(),
                };

                if condition_value {
                    let mut child_state =
                        LocalInterpreterState::with_parent(local_interpreter_state);
                    self.execute(main_body, &mut child_state)
                } else {
                    let mut child_state =
                        LocalInterpreterState::with_parent(local_interpreter_state);
                    self.execute(else_body, &mut child_state)
                }
            }
        }
    }
}

impl Runtime for Interpreter {
    fn run_function(&self, name: &str, arguments: &[Arc<RuntimeValue>]) -> Arc<RuntimeValue> {
        let closure_value = &self.scope.functions().get(name).unwrap();
        run_closure(self, closure_value, arguments)
    }
}

fn run_closure(
    interpreter: &Interpreter,
    closure: &Closure,
    arguments: &[Arc<RuntimeValue>],
) -> Arc<RuntimeValue> {
    // todo: check argument types.
    assert!(arguments.len() == closure.ty.required_arguments().len());
    let mut local_interpreter_state = LocalInterpreterState::new();

    // load arguments into local_interpreter_state
    for ((argument_name, _), provided_argument) in
        closure.ty.required_arguments().iter().zip(arguments.iter())
    {
        local_interpreter_state
            .variables
            .insert(argument_name.clone(), provided_argument.clone());
    }

    interpreter.execute(&closure.body, &mut local_interpreter_state)
}
