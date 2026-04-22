use std::{
    convert::Infallible,
    fmt::{self, Debug, Display, Formatter, Write, format},
    str::FromStr,
};

use babble::{learn::{LibId, ParseLibIdError}, teachable::DeBruijnIndex};
use egg::*;
use serde::Serialize;

// Old in Lemonade
// pub enum MLIROps {
//     // Primatives in MLIR (Variables, Constant - Look Like %0, %1, %c_32 ...)
//     Constant(i32),
//     Variable(Symbol),

//     // Minimal Set Of Predefine Arithmetic Functions
//     Add,
//     Sub,
//     Mult,
//     Div,  // arith.divsi (signed integer division)
//     And,
//     Or,
//     LeftShift,

//     Neg, // -a
//     Not, // ~a

//     // Integer Comparison (arith.cmpi predicates)
//     CmpEq,  // eq  — equal
//     CmpNe,  // ne  — not equal
//     CmpSlt, // slt — signed less than
//     CmpSle, // sle — signed less than or equal
//     CmpSgt, // sgt — signed greater than
//     CmpSge, // sge — signed greater than or equal
//     CmpUlt, // ult — unsigned less than
//     CmpUle, // ule — unsigned less than or equal
//     CmpUgt, // ugt — unsigned greater than
//     CmpUge, // uge — unsigned greater than or equal

//     // Selection
//     Select, // arith.select condition, true_val, false_val

//     // Load Store operations 1D/2D/3D/4D
//     Load, // Array, Addr
//     Load2D, // Array, Addr1, Addr2
//     Load3D, // Array, Addr1, Addr2, Addr3
//     Load4D, // Array, Addr1, Addr2, Addr3, Addr4

//     Store, // variable, Array, Addr
//     Store2D, // variable, Array, Addr1, Addr2
//     Store3D, // variable, Array, Addr1, Addr2, Addr3
//     Store4D, // variable, Array, Addr1, Addr2, Addr3, Addr4

//     Transpose, // Array // To learn more general load patterns

//     ForLoop, // For Loop construct -> this has an iterator, start, end, step, and body
//                                 // String for Loop Name/ID,
//                                 // Vec of Loop Name/IDs that this loop can be reordered with
//                                 // The tile size for this loop

//     YieldForLoop, // For Loop with a Yield and Iterator Arguments
//     // Not 100% sure I need this yet maybe i can just have it implicity as child of the YieldForLoop
//     // Yield, // A Yield Node
//     // IteratingVar, // Iterator Arg in a YieldingForLoop

//     IfThen,
//     IfThenElse,

//     YieldingIfThen,
//     YieldingIfThenElse,

//     Block, // Group of instructions

//     ProgList, // Group of programs

//     Call, // Function Call
//     FDef(LibId), // Function Defn

//     FReturn, // Return from a function

//     /// A library function binding
//     Lib(LibId),
//     /// A reference to a lib var
//     LibVar(LibId),
//     // DeBruijnIndex - no clue what this is but needed?
//     DBVar(DeBruijnIndex),
// }

#[derive(Debug, Serialize, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum MLIRType {
    Integer32,
    Index32,
    Void,
}

impl Display for MLIRType {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let type_str = match self {
            MLIRType::Index32 => "idx32",
            MLIRType::Integer32 => "int32",
            MLIRType::Void => "void",
        };

        write!(f, "{type_str}")
    }
}

impl FromStr for MLIRType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "int32" => Ok(MLIRType::Integer32),
            "idx32" => Ok(MLIRType::Index32),
            "void" => Ok(MLIRType::Void),
            _ => Err("Failed to parse unsupported type".to_string()),
        }
    }
}

// I think all other ops can be derived from children?
#[derive(Debug, Serialize, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum MLIROP {
    // Type of the variabel itself
    Constant(i32),
    Variable(Symbol),
    // DBVar(DeBruijnIndex),

    // Load and Stores in MLIR have weird typing
    Load,
    Store,

    // This will have its own type
    FunctionCall,

    // If we are following babble
    // I think the LibVar Node has no children and therefore would actually type the Function Call
    FunctionRef(LibId), // Babble Library Name part of Function Call Chain

    // Minimal Set Of Predefine Arithmetic Functions
    Add,
    Sub,
    Mult,

    And,
    Or,
    LeftShift,

    Neg, // -a
    Not, // ~a

    Select, // arith.select condition, true_val, false_val

    ForLoop, // For Loop construct
    IfThen,
    IfThenElse,

    Block,
    LibraryTop(LibId), // Babble Library Top
    FunctionDef(LibId),
    FunctionReturn, // Return from a function
}

impl Display for MLIROP {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let type_str = match self {
            MLIROP::Constant(val) => format!("{}", val),
            MLIROP::Variable(sym) => format!("{}", sym),
            // MLIROP::DBVar(idx) => format!("dbvar({})", idx),
            MLIROP::Load => "load".to_string(),
            MLIROP::Store => "store".to_string(),
            MLIROP::FunctionCall => "call".to_string(),
            MLIROP::FunctionRef(lib_id) => format!("{}", lib_id),
            MLIROP::Add => "+".to_string(),
            MLIROP::Sub => "--".to_string(),
            MLIROP::Mult => "*".to_string(),
            MLIROP::And => "&".to_string(),
            MLIROP::Or => "|".to_string(),
            MLIROP::LeftShift => "<<".to_string(),
            MLIROP::Neg => "-".to_string(),
            MLIROP::Not => "~".to_string(),
            MLIROP::Select => "select".to_string(),
            MLIROP::ForLoop => "for_loop".to_string(),
            MLIROP::IfThen => "if_then".to_string(),
            MLIROP::IfThenElse => "if_then_else".to_string(),
            MLIROP::Block => "block".to_string(),
            MLIROP::LibraryTop(lib_id) => format!("Lib-{}", lib_id),
            MLIROP::FunctionDef(lib_id) => format!("Fn-{}", lib_id),
            MLIROP::FunctionReturn => "return".to_string(),
        };
        write!(f, "{type_str}")
    }
}


impl FromStr for MLIROP {
    type Err = Infallible;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        let op = match input {
        "+" => MLIROP::Add,
        "--" => MLIROP::Sub,
        "*" => MLIROP::Mult,
        "&" => MLIROP::And,
        "|" => MLIROP::Or,
        "<<" => MLIROP::LeftShift,
        "-" => MLIROP::Neg,
        "~" => MLIROP::Not,
        "select" => MLIROP::Select,
        "for_loop" => MLIROP::ForLoop,
        "if_then" => MLIROP::IfThen,
        "if_then_else" => MLIROP::IfThenElse,
        "block" => MLIROP::Block,
        "call" => MLIROP::FunctionCall,
        "return" => MLIROP::FunctionReturn,

        input => {
            input
            .parse()
            .map(MLIROP::Constant)
            // .or_else(|_| input.parse().map(MLIROps::DBVar))
            .or_else(|_| input.parse().map(MLIROP::FunctionRef))
            .or_else(|_| {
                input
                .strip_prefix("lib-")
                .ok_or(ParseLibIdError::NoLeadingL)
                .and_then(|x| x.parse().map(MLIROP::LibraryTop))
            })
            .or_else(|_| {
                // println!("input {} -> {:?}", input, input.strip_prefix("lib-"));
                input.strip_prefix("Fn-").ok_or(ParseLibIdError::NoLeadingL).and_then(|x| x.parse().map(MLIROP::FunctionDef))
            })
            .unwrap_or_else(|_| MLIROP::Variable(input.into()))
            // .unwrap(),
        }
    };

    Ok(op)
}
}

#[derive(Clone, Serialize, Hash)]
pub struct MLIRNode {
    pub op: MLIROP,
    pub name: Symbol,
    pub mlir_type: Option<MLIRType>, // idk if it should have a type or not...
                                     // other option is to propogate type from variables/constants/casts/functioncalls?
    pub children: Vec<Id>,
}

impl MLIRNode {
    pub fn get_fields(&self) -> (MLIROP, Symbol, Option<MLIRType>) {
        (self.op, self.name, self.mlir_type)
    }
}

impl Debug for MLIRNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Decide on a format string, e.g., "name:width"
        match self.mlir_type {
            Some(t) => write!(f, "{}:{}:{}", self.op, self.name, t),
            None => write!(f, "{}:{}:None", self.op, self.name)
        }
    }
}

impl Display for MLIRNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Decide on a format string, e.g., "name:width"
        match self.mlir_type {
            Some(t) => write!(f, "{}:{}:{}", self.op, self.name, t),
            None => write!(f, "{}:{}", self.op, self.name)
        }
    }
}

// impl FromStr for MLIRNode {
//     type Err = String;

//     fn from_str(s: &str) -> Result<Self, Self::Err> {
//         // Parse the format string back into the struct
//         let parts: Vec<&str> = s.split(':').collect();


//         match parts.len() {
//             2 => {
//                 let op = MLIROP::from_str(parts[0]).map_err(|e| e.to_string())?;
//                 let name = Symbol::from(parts[1]);
//                 Ok(MLIRNode {
//                     op,
//                     name,
//                     mlir_type : None,
//                 })
//             }
//             3 => {
//                 let op = MLIROP::from_str(parts[0]).map_err(|e| e.to_string())?;
//                 let name = Symbol::from(parts[1]);
//                 let mlir_type = MLIRType::from_str(parts[2]).map_err(|e| e.to_string())?;
//                 Ok(MLIRNode {
//                     op,
//                     name,
//                     mlir_type : Some(mlir_type),
//                 })
//             }
//             _ => Err("Expected format opt:name or op:name:type".to_string())
//         }

//     }
// }

define_language! {
    enum MlirLang {
        Op(MLIROP, Vec<Id>),
    }
}

impl MlirLang {
    fn new_const() {}
    fn new_variable() {}
    fn new_bin_op() {} // accepts the bin ops
    fn new_tern_op() {}
    fn new_generic() {}
    fn new_block() {}
    
    // other common ones should go here?
    // new_loop
    // new_if

    // typed variants
}