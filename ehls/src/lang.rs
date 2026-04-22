use std::{
    convert::Infallible, fmt::{self, Debug, Display, Formatter, Write, format}, str::FromStr
};

use egg::*;
use serde::Serialize;

#[derive(Debug, Serialize, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum MLIRType {
    Integer32,
    Index32,
    Void
}

pub enum MLIRTyped {
    
    // Type of the variabel itself
    Constant(i32),
    Variable(Symbol),
    
    // Load and Stores in MLIR have weird typing
    Load,
    Store,

    // This will have its own type
    FunctionCall,

    // If we are following babble
    // I think the LibVar Node has no children and therefore would actually type the Function Call
    LibVar(LibID)
}

// I think all other ops can be derived from children?
pub enum MLIRUntyped {

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

#[derive(Clone, Serialize, Hash)]
pub struct TypedNode {
    pub op: MLIRTyped,
    pub name: Symbol, 
    pub mlir_type: MLIRType, // idk if it should have a type or not...
    // other option is to propogate type from variables/constants/casts/functioncalls?
}

#[derive(Clone, Serialize, Hash)]
pub struct UnTypedNode {
    pub op: MLIRUntyped,
    pub name: Symbol
}


impl TypedNode {
    pub fn get_fields(&self) -> (Symbol, MLIRType) {
        (self.name, self.mlir_type)
    }
}

impl Debug for TypedNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Decide on a format string, e.g., "name:width"
        write!(f, "{}:{}", self.name, self.mlir_type)
    }
}

impl Display for TypedNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Decide on a format string, e.g., "name:width"
        write!(f, "{}:{}", self.name, self.mlir_type)
    }
}

impl FromStr for TypedNode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // Parse the format string back into the struct
        let parts: Vec<&str> = s.split(':').collect();
        if parts.len() != 2 {
            return Err("Expected format name:type".to_string());
        }
        let name = Symbol::from(parts[0]);
        let mlir_type = parts[1].parse::<u8>().map_err(|e| e.to_string())?;
        Ok(IntegerMeta { name, bitwidth })
    }
}

impl UnTypedNode {
    pub fn get_fields(&self) -> (Symbol, MLIRType) {
        (self.name, self.mlir_type)
    }
}

impl Debug for UnTypedNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Decide on a format string, e.g., "name:width"
        write!(f, "{}", self.name)
    }
}

impl Display for UnTypedNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Decide on a format string, e.g., "name:width"
        write!(f, "{}", self.name, self.mlir_type)
    }
}

impl FromStr for UnTypedNode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // Parse the format string back into the struct
        let parts: Vec<&str> = s.split(':').collect();
        if parts.len() != 2 {
            return Err("Expected format name:type".to_string());
        }
        let name = Symbol::from(parts[0]);
        let mlir_type = parts[1].parse::<u8>().map_err(|e| e.to_string())?;
        Ok(IntegerMeta { name, bitwidth })
    }
}
