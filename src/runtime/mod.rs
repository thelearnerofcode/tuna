pub mod interpreter;
pub mod llvm;

use ir::Closure;

use indexmap::IndexMap;
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

    pub fn as_bool(&self) -> Option<bool> {
        match *self {
            BasicValue::Bool(ref bool) => Some(*bool),
            _ => None,
        }
    }

    pub fn as_void(&self) -> Option<()> {
        match *self {
            BasicValue::Void => Some(()),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StructValue {
    pub members: IndexMap<String, Arc<RuntimeValue>>,
}

#[derive(Debug, Clone)]
pub enum RuntimeValue {
    BasicValue(BasicValue),
    Closure(Closure),
    StructValue(StructValue),
}

impl RuntimeValue {
    pub fn as_basic_value(&self) -> Option<&BasicValue> {
        match *self {
            RuntimeValue::BasicValue(ref basic) => Some(basic),
            _ => None,
        }
    }

    pub fn as_closure_value(&self) -> Option<&Closure> {
        match *self {
            RuntimeValue::Closure(ref closure) => Some(closure),
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
