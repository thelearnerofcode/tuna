use llvm_sys as llvm;

use indexmap::IndexMap;
use ir::{BasicType, BinaryOperator, ClosureType, ComparisonOperator, ConstantValue,
         ConversionContext, Expression, Scope, StructType, Type};
use runtime::RuntimeValue;
use std::ffi::CString;
use std::sync::Arc;

struct CompileContext {
    variables: IndexMap<String, llvm::prelude::LLVMValueRef>,
    conversion_context: ConversionContext,
}

pub struct LLVMJit {
    ee: llvm::execution_engine::LLVMExecutionEngineRef,
    context: *mut llvm::LLVMContext,
}

impl LLVMJit {
    pub fn new(scope: Scope) -> LLVMJit {
        #[not(cfg(debug_assertions))]
        panic!("Don't use the LLVM Backend! It is incomplete!");
        unsafe {
            // setup llvm
            let context = llvm::core::LLVMContextCreate();
            let module = llvm::core::LLVMModuleCreateWithNameInContext(
                b"main_module\0".as_ptr() as *const _,
                context,
            );
            llvm::core::LLVMSetDataLayout(
                module,
                b"e-m:e-i64:64-f80:128-n8:16:32:64-S128\0".as_ptr() as *const _,
            );

            let builder = llvm::core::LLVMCreateBuilderInContext(context);

            let mut compile_context = CompileContext {
                variables: IndexMap::new(),
                conversion_context: ConversionContext {
                    variables: IndexMap::new(),
                    functions: IndexMap::new(),
                    structs: IndexMap::new(),
                },
            };

            let mut functions = vec![];

            // setup functions
            for (function_name, closure) in scope.functions() {
                // compile the closure type
                let function_type = Self::compile_closure_type(closure.ty(), context);

                // now we add the function to the module
                let function_name_cstring = CString::new(function_name.clone()).unwrap();
                functions.push((
                    llvm::core::LLVMAddFunction(
                        module,
                        function_name_cstring.as_ptr() as *const _,
                        function_type,
                    ),
                    function_name,
                    closure,
                ));
            }

            // give function bodies
            for (function, function_name, closure) in functions {
                compile_context.conversion_context = ConversionContext {
                    structs: scope.struct_types().clone(),
                    variables: closure.ty().required_arguments().iter().cloned().collect(),
                    functions: {
                        let mut functions: IndexMap<String, ClosureType> = scope
                            .functions()
                            .iter()
                            .map(|(name, closure_value)| (name.clone(), closure_value.ty().clone()))
                            .collect();
                        functions.insert(function_name.clone(), closure.ty().clone());
                        functions
                    },
                };

                // load the arguments into the compile context
                compile_context.variables = IndexMap::new();
                for (index, (argument_name, _)) in
                    closure.ty().required_arguments().iter().enumerate()
                {
                    let value_ref = llvm::core::LLVMGetParam(function, index as u32);
                    let name = CString::new(argument_name.clone()).unwrap();
                    llvm::core::LLVMSetValueName(value_ref, name.as_ptr());

                    compile_context
                        .variables
                        .insert(argument_name.clone(), value_ref);
                }

                // give it a body
                let bb = llvm::core::LLVMAppendBasicBlockInContext(
                    context,
                    function,
                    b"entry\0".as_ptr() as *const _,
                );
                llvm::core::LLVMPositionBuilderAtEnd(builder, bb);

                let result = Self::compile_expression(
                    module,
                    context,
                    builder,
                    closure.body(),
                    &mut compile_context,
                );
                llvm::core::LLVMBuildRet(builder, result);

                llvm::analysis::LLVMVerifyFunction(
                    function,
                    llvm::analysis::LLVMVerifierFailureAction::LLVMPrintMessageAction,
                );
            }

            // dump the module
            #[cfg(debug_assertions)]
            llvm::core::LLVMDumpModule(module);

            // clean up llvm
            llvm::core::LLVMDisposeBuilder(builder);

            // build an execution engine
            let mut ee = ::std::mem::uninitialized();
            let mut out = ::std::mem::zeroed();

            // robust code should check that these calls complete successfully
            // each of these calls is necessary to setup an execution engine which compiles to native
            // code
            llvm::execution_engine::LLVMLinkInMCJIT();
            llvm::target::LLVM_InitializeNativeTarget();
            llvm::target::LLVM_InitializeNativeAsmPrinter();

            llvm::execution_engine::LLVMCreateExecutionEngineForModule(&mut ee, module, &mut out);

            LLVMJit { ee, context }
        }
    }

    unsafe fn compile_expression(
        module: llvm::prelude::LLVMModuleRef,
        context: llvm::prelude::LLVMContextRef,
        builder: llvm::prelude::LLVMBuilderRef,
        expression: &Expression,
        compile_context: &mut CompileContext,
    ) -> llvm::prelude::LLVMValueRef {
        match *expression {
            Expression::CreateConstantValue(ref constant_value) => match *constant_value {
                ConstantValue::U16(ref v) => {
                    let ty = llvm::core::LLVMInt16TypeInContext(context);
                    llvm::core::LLVMConstInt(ty, u64::from(*v), 0)
                }
                ConstantValue::U32(ref v) => {
                    let ty = llvm::core::LLVMInt32TypeInContext(context);
                    llvm::core::LLVMConstInt(ty, u64::from(*v), 0)
                }
                ConstantValue::U64(ref v) => {
                    let ty = llvm::core::LLVMInt64TypeInContext(context);
                    llvm::core::LLVMConstInt(ty, *v, 0)
                }
                ConstantValue::F32(ref v) => {
                    let ty = llvm::core::LLVMFloatTypeInContext(context);
                    llvm::core::LLVMConstReal(ty, f64::from(*v))
                }
                ConstantValue::F64(ref v) => {
                    let ty = llvm::core::LLVMDoubleTypeInContext(context);
                    llvm::core::LLVMConstReal(ty, *v)
                }
                ConstantValue::I16(ref v) => {
                    let ty = llvm::core::LLVMInt16TypeInContext(context);
                    llvm::core::LLVMConstInt(ty, *v as u64, 0)
                }
                ConstantValue::I32(ref v) => {
                    let ty = llvm::core::LLVMInt32TypeInContext(context);
                    llvm::core::LLVMConstInt(ty, *v as u64, 0)
                }
                ConstantValue::I64(ref v) => {
                    let ty = llvm::core::LLVMInt64TypeInContext(context);
                    llvm::core::LLVMConstInt(ty, *v as u64, 0)
                }
                ConstantValue::String(ref str) => {
                    let string = CString::new(str.clone()).unwrap();
                    llvm::core::LLVMBuildGlobalString(
                        builder,
                        string.as_ptr(),
                        b"\0".as_ptr() as *const _,
                    )
                }
                ConstantValue::Bool(ref b) => {
                    let ty = llvm::core::LLVMInt8TypeInContext(context);
                    match b {
                        true => llvm::core::LLVMConstInt(ty, 1, 0),
                        false => llvm::core::LLVMConstInt(ty, 0, 0),
                    }
                }
            },
            Expression::BinaryExpression(ref lhs_expr, ref op, ref rhs_expr) => {
                let expr_type = expression
                    .get_type(&mut compile_context.conversion_context)
                    .unwrap();

                let lhs =
                    Self::compile_expression(module, context, builder, lhs_expr, compile_context);
                let rhs =
                    Self::compile_expression(module, context, builder, rhs_expr, compile_context);

                match *op {
                    BinaryOperator::Add => match expr_type {
                        Type::Basic(BasicType::U16)
                        | Type::Basic(BasicType::U32)
                        | Type::Basic(BasicType::U64)
                        | Type::Basic(BasicType::I16)
                        | Type::Basic(BasicType::I32)
                        | Type::Basic(BasicType::I64) => {
                            llvm::core::LLVMBuildAdd(builder, lhs, rhs, b"\0".as_ptr() as *const _)
                        }
                        Type::Basic(BasicType::F32) | Type::Basic(BasicType::F64) => {
                            llvm::core::LLVMBuildFAdd(builder, lhs, rhs, b"\0".as_ptr() as *const _)
                        }
                        _ => panic!(),
                    },
                    BinaryOperator::Subtract => match expr_type {
                        Type::Basic(BasicType::U16)
                        | Type::Basic(BasicType::U32)
                        | Type::Basic(BasicType::U64)
                        | Type::Basic(BasicType::I16)
                        | Type::Basic(BasicType::I32)
                        | Type::Basic(BasicType::I64) => {
                            llvm::core::LLVMBuildSub(builder, lhs, rhs, b"\0".as_ptr() as *const _)
                        }
                        Type::Basic(BasicType::F32) | Type::Basic(BasicType::F64) => {
                            llvm::core::LLVMBuildFSub(builder, lhs, rhs, b"\0".as_ptr() as *const _)
                        }
                        _ => panic!(),
                    },
                    BinaryOperator::Multiply => match expr_type {
                        Type::Basic(BasicType::U16)
                        | Type::Basic(BasicType::U32)
                        | Type::Basic(BasicType::U64)
                        | Type::Basic(BasicType::I16)
                        | Type::Basic(BasicType::I32)
                        | Type::Basic(BasicType::I64) => {
                            llvm::core::LLVMBuildMul(builder, lhs, rhs, b"\0".as_ptr() as *const _)
                        }
                        Type::Basic(BasicType::F32) | Type::Basic(BasicType::F64) => {
                            llvm::core::LLVMBuildFMul(builder, lhs, rhs, b"\0".as_ptr() as *const _)
                        }
                        _ => panic!(),
                    },
                    BinaryOperator::Divide => match expr_type {
                        Type::Basic(BasicType::U16)
                        | Type::Basic(BasicType::U32)
                        | Type::Basic(BasicType::U64) => {
                            llvm::core::LLVMBuildUDiv(builder, lhs, rhs, b"\0".as_ptr() as *const _)
                        }
                        Type::Basic(BasicType::I16)
                        | Type::Basic(BasicType::I32)
                        | Type::Basic(BasicType::I64) => {
                            llvm::core::LLVMBuildSDiv(builder, lhs, rhs, b"\0".as_ptr() as *const _)
                        }
                        Type::Basic(BasicType::F32) | Type::Basic(BasicType::F64) => {
                            llvm::core::LLVMBuildFDiv(builder, lhs, rhs, b"\0".as_ptr() as *const _)
                        }
                        _ => panic!(),
                    },
                }
            }
            Expression::CallClosure(ref closure_expr, ref argument_expresions) => {
                let closure = Self::compile_expression(
                    module,
                    context,
                    builder,
                    closure_expr,
                    compile_context,
                );
                let mut compiled_arguments = vec![];
                for argument_expr in argument_expresions {
                    compiled_arguments.push(Self::compile_expression(
                        module,
                        context,
                        builder,
                        argument_expr,
                        compile_context,
                    ))
                }
                llvm::core::LLVMBuildCall(
                    builder,
                    closure,
                    compiled_arguments.as_mut_ptr(),
                    compiled_arguments.len() as u32,
                    b"\0".as_ptr() as *const _,
                )
            }
            Expression::GetFunction(ref name) => {
                let name = CString::new(name.clone()).unwrap();
                llvm::core::LLVMGetNamedFunction(module, name.as_ptr())
            }
            Expression::GetMember(ref struct_expr, ref member_name) => {
                let struct_value = Self::compile_expression(
                    module,
                    context,
                    builder,
                    struct_expr,
                    compile_context,
                );

                let struct_type = struct_expr
                    .get_type(&mut compile_context.conversion_context)
                    .unwrap();
                let struct_type = struct_type.as_struct().unwrap();

                // copy to pointer on stack
                let result_value = llvm::core::LLVMBuildAlloca(
                    builder,
                    Self::compile_struct_type(&struct_type, context),
                    b"\0".as_ptr() as *const _,
                );
                llvm::core::LLVMSetAlignment(result_value, 8);
                llvm::core::LLVMBuildStore(builder, struct_value, result_value);

                let field_index = struct_type.fields().get_full(member_name).unwrap().0;

                let ty = llvm::core::LLVMInt32TypeInContext(context);
                let mut indices = [
                    llvm::core::LLVMConstInt(ty, 0, 0),
                    llvm::core::LLVMConstInt(ty, field_index as u64, 0),
                ];

                let gep = llvm::core::LLVMBuildGEP(
                    builder,
                    result_value,
                    indices.as_mut_ptr(),
                    indices.len() as u32,
                    b"\0".as_ptr() as *const _,
                );

                llvm::core::LLVMBuildLoad(builder, gep, b"\0".as_ptr() as *const _)
            }
            Expression::GetVariable(ref name) => *compile_context.variables.get(name).unwrap(),
            Expression::Compare(ref lhs_expr, ref op, ref rhs_expr) => {
                let lhs =
                    Self::compile_expression(module, context, builder, lhs_expr, compile_context);

                let rhs =
                    Self::compile_expression(module, context, builder, rhs_expr, compile_context);

                let ty = lhs_expr
                    .get_type(&mut compile_context.conversion_context)
                    .unwrap();
                let ty = ty.as_basic().unwrap();

                match *op {
                    ComparisonOperator::EqualTo => match ty {
                        BasicType::F32 | BasicType::F64 => llvm::core::LLVMBuildFCmp(
                            builder,
                            llvm::LLVMRealPredicate::LLVMRealOEQ,
                            lhs,
                            rhs,
                            b"\0".as_ptr() as *const _,
                        ),
                        _ => llvm::core::LLVMBuildICmp(
                            builder,
                            llvm::LLVMIntPredicate::LLVMIntEQ,
                            lhs,
                            rhs,
                            b"\0".as_ptr() as *const _,
                        ),
                    },
                    ComparisonOperator::LessThan => match ty {
                        BasicType::U16
                        | BasicType::U32
                        | BasicType::U64
                        | BasicType::Bool
                        | BasicType::String => llvm::core::LLVMBuildICmp(
                            builder,
                            llvm::LLVMIntPredicate::LLVMIntULT,
                            lhs,
                            rhs,
                            b"\0".as_ptr() as *const _,
                        ),
                        BasicType::F32 | BasicType::F64 => llvm::core::LLVMBuildFCmp(
                            builder,
                            llvm::LLVMRealPredicate::LLVMRealOLT,
                            lhs,
                            rhs,
                            b"\0".as_ptr() as *const _,
                        ),
                        BasicType::I16 | BasicType::I32 | BasicType::I64 => {
                            llvm::core::LLVMBuildICmp(
                                builder,
                                llvm::LLVMIntPredicate::LLVMIntSLT,
                                lhs,
                                rhs,
                                b"\0".as_ptr() as *const _,
                            )
                        }
                    },
                    ComparisonOperator::LessThanEqualTo => match ty {
                        BasicType::U16
                        | BasicType::U32
                        | BasicType::U64
                        | BasicType::Bool
                        | BasicType::String => llvm::core::LLVMBuildICmp(
                            builder,
                            llvm::LLVMIntPredicate::LLVMIntULE,
                            lhs,
                            rhs,
                            b"\0".as_ptr() as *const _,
                        ),
                        BasicType::F32 | BasicType::F64 => llvm::core::LLVMBuildFCmp(
                            builder,
                            llvm::LLVMRealPredicate::LLVMRealOLE,
                            lhs,
                            rhs,
                            b"\0".as_ptr() as *const _,
                        ),
                        BasicType::I16 | BasicType::I32 | BasicType::I64 => {
                            llvm::core::LLVMBuildICmp(
                                builder,
                                llvm::LLVMIntPredicate::LLVMIntSLE,
                                lhs,
                                rhs,
                                b"\0".as_ptr() as *const _,
                            )
                        }
                    },
                    ComparisonOperator::GreaterThan => match ty {
                        BasicType::U16
                        | BasicType::U32
                        | BasicType::U64
                        | BasicType::Bool
                        | BasicType::String => llvm::core::LLVMBuildICmp(
                            builder,
                            llvm::LLVMIntPredicate::LLVMIntUGT,
                            lhs,
                            rhs,
                            b"\0".as_ptr() as *const _,
                        ),
                        BasicType::F32 | BasicType::F64 => llvm::core::LLVMBuildFCmp(
                            builder,
                            llvm::LLVMRealPredicate::LLVMRealOGT,
                            lhs,
                            rhs,
                            b"\0".as_ptr() as *const _,
                        ),
                        BasicType::I16 | BasicType::I32 | BasicType::I64 => {
                            llvm::core::LLVMBuildICmp(
                                builder,
                                llvm::LLVMIntPredicate::LLVMIntSGT,
                                lhs,
                                rhs,
                                b"\0".as_ptr() as *const _,
                            )
                        }
                    },
                    ComparisonOperator::GreaterThanEqualTo => match ty {
                        BasicType::U16
                        | BasicType::U32
                        | BasicType::U64
                        | BasicType::Bool
                        | BasicType::String => llvm::core::LLVMBuildICmp(
                            builder,
                            llvm::LLVMIntPredicate::LLVMIntUGE,
                            lhs,
                            rhs,
                            b"\0".as_ptr() as *const _,
                        ),
                        BasicType::F32 | BasicType::F64 => llvm::core::LLVMBuildFCmp(
                            builder,
                            llvm::LLVMRealPredicate::LLVMRealOGE,
                            lhs,
                            rhs,
                            b"\0".as_ptr() as *const _,
                        ),
                        BasicType::I16 | BasicType::I32 | BasicType::I64 => {
                            llvm::core::LLVMBuildICmp(
                                builder,
                                llvm::LLVMIntPredicate::LLVMIntSGE,
                                lhs,
                                rhs,
                                b"\0".as_ptr() as *const _,
                            )
                        }
                    },
                    ComparisonOperator::NotEqualTo => match ty {
                        BasicType::F32 | BasicType::F64 => llvm::core::LLVMBuildFCmp(
                            builder,
                            llvm::LLVMRealPredicate::LLVMRealONE,
                            lhs,
                            rhs,
                            b"\0".as_ptr() as *const _,
                        ),
                        _ => llvm::core::LLVMBuildICmp(
                            builder,
                            llvm::LLVMIntPredicate::LLVMIntEQ,
                            lhs,
                            rhs,
                            b"\0".as_ptr() as *const _,
                        ),
                    },
                }
            }
            Expression::If {
                ref condition,
                ref main_body,
                ref else_body,
            } => {
                // compile the condition expression
                let condition_value =
                    Self::compile_expression(module, context, builder, condition, compile_context);

                // get the result of the if block
                let result_type = Self::compile_type(
                    &expression
                        .get_type(&mut compile_context.conversion_context)
                        .unwrap(),
                    context,
                );

                // this is where we will store our result
                let result_value =
                    llvm::core::LLVMBuildAlloca(builder, result_type, b"\0".as_ptr() as *const _);
                llvm::core::LLVMSetAlignment(result_value, 8);

                // get a reference to the parent function so we can insert basic blocks
                let function =
                    llvm::core::LLVMGetBasicBlockParent(llvm::core::LLVMGetInsertBlock(builder));

                // create the block that is jumped to when condition is true
                let then_block = llvm::core::LLVMAppendBasicBlockInContext(
                    context,
                    function,
                    b"\0".as_ptr() as *const _,
                );

                // create the block that is jumped to when condition is false
                let else_block = llvm::core::LLVMAppendBasicBlockInContext(
                    context,
                    function,
                    b"\0".as_ptr() as *const _,
                );

                // create the block that is jumped to at the end of executing the two earlier blocks
                let merge_block = llvm::core::LLVMAppendBasicBlockInContext(
                    context,
                    function,
                    b"\0".as_ptr() as *const _,
                );

                // build a condition branch that jumps to the corresponding block
                llvm::core::LLVMBuildCondBr(builder, condition_value, then_block, else_block);

                // move our builder to the end of the then block
                llvm::core::LLVMPositionBuilderAtEnd(builder, then_block);
                // compile the body
                let mut else_value =
                    Self::compile_expression(module, context, builder, main_body, compile_context);
                // store the result in th epointer
                llvm::core::LLVMBuildStore(builder, else_value, result_value);
                // branch to the merge block
                llvm::core::LLVMBuildBr(builder, merge_block);

                // do the same thing for the then_block
                let mut then_block = llvm::core::LLVMGetInsertBlock(builder);
                llvm::core::LLVMPositionBuilderAtEnd(builder, else_block);
                let mut then_value =
                    Self::compile_expression(module, context, builder, else_body, compile_context);
                llvm::core::LLVMBuildStore(builder, then_value, result_value);
                llvm::core::LLVMBuildBr(builder, merge_block);

                // position buidler at the end of the merge block
                llvm::core::LLVMPositionBuilderAtEnd(builder, merge_block);
                // load result value
                llvm::core::LLVMBuildLoad(builder, result_value, b"\0".as_ptr() as *const _)
            }
            Expression::CreateStruct(ref struct_ty, ref member_expressions) => {
                let mut member_expressions = member_expressions.clone();
                member_expressions.sort_by(|ref fkey, ref fval, ref skey, ref sval| {
                    let struct_fields = struct_ty.fields();
                    (struct_fields.get_full(*fkey).unwrap().0)
                        .cmp(&struct_fields.get_full(*skey).unwrap().0)
                });

                // get the llvm type of the struct
                let llvm_struct_type = Self::compile_struct_type(struct_ty, context);

                // this is where we will store our result
                let result_value = llvm::core::LLVMBuildAlloca(
                    builder,
                    llvm_struct_type,
                    b"\0".as_ptr() as *const _,
                );
                llvm::core::LLVMSetAlignment(result_value, 8);

                for (ref member_name, ref member_expr) in &member_expressions {
                    let (index, _, _) = struct_ty.fields().get_full(*member_name).unwrap();
                    if index != 0 {
                        let member_type = member_expr
                            .get_type(&mut compile_context.conversion_context)
                            .unwrap();

                        let member_value = Self::compile_expression(
                            module,
                            context,
                            builder,
                            member_expr,
                            compile_context,
                        );

                        let ty = llvm::core::LLVMInt32TypeInContext(context);
                        let mut indices = [
                            llvm::core::LLVMConstInt(ty, 0, 0),
                            llvm::core::LLVMConstInt(ty, index as u64, 0),
                        ];
                        let member_pointer = llvm::core::LLVMBuildGEP(
                            builder,
                            result_value,
                            indices.as_mut_ptr(),
                            indices.len() as u32,
                            b"\0".as_ptr() as *const _,
                        );
                        llvm::core::LLVMSetIsInBounds(member_pointer, true as i32);
                        let store_val =
                            llvm::core::LLVMBuildStore(builder, member_value, member_pointer);
                        llvm::core::LLVMSetAlignment(
                            store_val,
                            get_alignment_of_basic_type(member_type.as_basic().unwrap()),
                        );
                    } else {
                        // bitcast the struct
                        let bitcasted_struct = llvm::core::LLVMBuildBitCast(
                            builder,
                            result_value,
                            build_u8_pointer(context),
                            b"\0".as_ptr() as *const _,
                        );
                        let first_type = member_expr
                            .get_type(&mut compile_context.conversion_context)
                            .unwrap();

                        let first_value = Self::compile_expression(
                            module,
                            context,
                            builder,
                            member_expr,
                            compile_context,
                        );

                        let store_val =
                            llvm::core::LLVMBuildStore(builder, first_value, bitcasted_struct);
                        llvm::core::LLVMSetAlignment(
                            store_val,
                            get_alignment_of_basic_type(first_type.as_basic().unwrap()),
                        );
                    }
                }

                llvm::core::LLVMBuildLoad(builder, result_value, b"\0".as_ptr() as *const _)
            }
        }
    }

    unsafe fn compile_closure_type(
        ty: &ClosureType,
        context: *mut llvm::LLVMContext,
    ) -> llvm::prelude::LLVMTypeRef {
        let result_type = Self::compile_type(ty.result(), context);
        let mut argument_types = vec![];
        for (_, argument_type) in ty.required_arguments() {
            argument_types.push(LLVMJit::compile_type(argument_type, context))
        }

        llvm::core::LLVMFunctionType(
            result_type,
            argument_types.as_mut_ptr(),
            ty.required_arguments().len() as u32,
            0,
        )
    }

    unsafe fn compile_basic_type(
        ty: &BasicType,
        context: *mut llvm::LLVMContext,
    ) -> llvm::prelude::LLVMTypeRef {
        match *ty {
            BasicType::U16 => llvm::core::LLVMInt16TypeInContext(context),
            BasicType::U32 => llvm::core::LLVMInt32TypeInContext(context),
            BasicType::U64 => llvm::core::LLVMInt64TypeInContext(context),

            BasicType::F32 => llvm::core::LLVMFloatTypeInContext(context),
            BasicType::F64 => llvm::core::LLVMDoubleTypeInContext(context),

            BasicType::I16 => llvm::core::LLVMInt16TypeInContext(context),
            BasicType::I32 => llvm::core::LLVMInt32TypeInContext(context),
            BasicType::I64 => llvm::core::LLVMInt64TypeInContext(context),

            BasicType::Bool => llvm::core::LLVMInt8TypeInContext(context),
            BasicType::String => {
                llvm::core::LLVMPointerType(llvm::core::LLVMInt8TypeInContext(context), 0)
            }
        }
    }
    unsafe fn compile_struct_type(
        ty: &StructType,
        context: *mut llvm::LLVMContext,
    ) -> llvm::prelude::LLVMTypeRef {
        let mut llvm_fields = Vec::new();
        for field_type in ty.fields().values() {
            llvm_fields.push(Self::compile_type(field_type, context))
        }

        llvm::core::LLVMStructTypeInContext(
            context,
            llvm_fields.as_mut_ptr(),
            llvm_fields.len() as u32,
            0,
        )
    }

    unsafe fn compile_type(
        ty: &Type,
        context: *mut llvm::LLVMContext,
    ) -> llvm::prelude::LLVMTypeRef {
        match *ty {
            Type::Basic(ref basic_ty) => Self::compile_basic_type(basic_ty, context),
            Type::Closure(ref closure_ty) => Self::compile_closure_type(closure_ty, context),
            Type::Struct(ref struct_ty) => Self::compile_struct_type(struct_ty, context),
        }
    }

    pub fn run_function_no_arg<A>(&self, name: &str) -> A {
        unsafe {
            let name = CString::new(name).unwrap();
            let addr =
                llvm::execution_engine::LLVMGetFunctionAddress(self.ee, name.as_ptr() as *const _);
            let f: extern "C" fn() -> A = ::std::mem::transmute(addr);
            f()
        }
    }

    pub fn run_function_1_arg<T, A>(&self, name: &str, args: T) -> A {
        unsafe {
            let name = CString::new(name).unwrap();
            let addr =
                llvm::execution_engine::LLVMGetFunctionAddress(self.ee, name.as_ptr() as *const _);
            let f: extern "C" fn(T) -> A = ::std::mem::transmute(addr);
            f(args)
        }
    }
}

impl Drop for LLVMJit {
    fn drop(&mut self) {
        unsafe {
            llvm::execution_engine::LLVMDisposeExecutionEngine(self.ee);
            llvm::core::LLVMContextDispose(self.context);
        }
    }
}

fn get_alignment_of_basic_type(ty: &BasicType) -> u32 {
    match *ty {
        BasicType::U16 | BasicType::I16 => 2,
        BasicType::U32 | BasicType::I32 | BasicType::F32 => 4,
        BasicType::U64 | BasicType::I64 | BasicType::F64 => 8,
        BasicType::Bool => 1,
        BasicType::String => unimplemented!(),
    }
}

unsafe fn build_u8_pointer(context: llvm::prelude::LLVMContextRef) -> llvm::prelude::LLVMTypeRef {
    llvm::core::LLVMPointerType(llvm::core::LLVMInt8TypeInContext(context), 0)
}
