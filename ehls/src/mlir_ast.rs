use nom::{
    branch::alt,
    bytes::complete::{tag, take_while, take_while1},
    character::complete::{char, digit1, multispace0},
    combinator::{map, opt, recognize},
    multi::{many0, separated_list0},
    sequence::{delimited, tuple},
    IResult,
};

// ============================================================================
// AST Data Structures
// ============================================================================

#[derive(Debug, Clone, PartialEq)]
pub struct MLIRModule {
    pub functions: Vec<MLIRFunction>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MLIRFunction {
    pub name: String,
    pub arguments: Vec<MLIRArgument>,
    pub return_ty: Option<String>,
    pub body: Vec<MLIRStatement>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MLIRArgument {
    pub name: String,
    pub ty: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MLIRStatement {
    /// memref.load or memref.store (only memref dialect now, no affine)
    Load {
        result: String,
        memref: String,
        indices: Vec<String>,
        ty: String,
    },
    Store {
        value: String,
        memref: String,
        indices: Vec<String>,
        ty: String,
    },
    /// Binary arith ops: arith.muli, arith.addi, arith.subi, etc.
    ArithOp {
        result: String,
        op: ArithOpKind,
        lhs: String,
        rhs: String,
        ty: String,
    },
    /// func.call / call @name(...)
    FunctionCall {
        result: Option<String>,
        function_name: String,
        arguments: Vec<String>,
        ty: String,
    },
    /// arith.constant
    Constant {
        result: String,
        value: String,
        ty: String,
    },
    /// `arith.index_cast`
    IndexCast {
        result: String,
        operand: String,
        from_ty: String,
        to_ty: String,
    },
    /// arith.cmpi
    CmpIOp {
        result: String,
        predicate: String,
        lhs: String,
        rhs: String,
        ty: String,
    },
    /// arith.select
    SelectOp {
        result: String,
        condition: String,
        true_val: String,
        false_val: String,
        ty: String,
    },
    /// `memref.alloca()` : memref<...>
    MemrefAlloca {
        result: String,
        ty: String,
    },
    /// `memref.get_global` @name : memref<...>
    MemrefGetGlobal {
        result: String,
        sym_name: String,
        ty: String,
    },
    /// scf.for %i = %lb to %ub step %s { ... }  (no `iter_args`, no yield)
    ScfFor {
        iterator: String,
        lower: String,
        upper: String,
        step: String,
        body: Vec<MLIRStatement>,
    },
    /// scf.for with `iter_args` and yield (loop-carried values, exactly one `iter_arg` assumed)
    ScfForYield {
        result: String,
        iterator: String,
        lower: String,
        upper: String,
        step: String,
        iter_args: (String, String),
        result_tys: Vec<String>,
        yield_val: String,
        body: Vec<MLIRStatement>,
    },
    /// scf.if %cond { ... } else { ... }  (no yield, side-effect only)
    ScfIf {
        condition: String,
        then_body: Vec<MLIRStatement>,
        else_body: Vec<MLIRStatement>,
    },
    /// scf.if with yield (produces values)
    ScfIfYield {
        result: String,
        result_tys: Vec<String>,
        condition: String,
        then_body: Vec<MLIRStatement>,
        else_body: Vec<MLIRStatement>,
        then_yield: String,
        else_yield: String,
    },
    /// scf.yield (internal — extracted into parent ScfForYield/ScfIfYield during parsing)
    ScfYield {
        values: Vec<String>,
    },
    /// return / return %v : ty
    Return {
        values: Vec<String>,
    },
}

impl MLIRStatement {
    #[must_use]
    pub fn get_var_assign(&self) -> Option<String> {
        match self {
            Self::ArithOp { result, .. }
            | Self::Load { result, .. }
            | Self::Constant { result, .. }
            | Self::IndexCast { result, .. }
            | Self::CmpIOp { result, .. }
            | Self::SelectOp { result, .. }
            | Self::MemrefAlloca { result, .. }
            | Self::MemrefGetGlobal { result, .. } => Some(result.clone()),
            Self::FunctionCall { result, .. } => result.clone(),
            Self::ScfForYield { result, .. }
            | Self::ScfIfYield { result, .. } => Some(result.clone()),
            Self::ScfFor { .. }
            | Self::ScfIf { .. } => None,
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Copy)]
pub enum ArithOpKind {
    Muli,
    Addi,
    Subi,
    Mulf,
    Addf,
    Andi,
    Ori,
    Divsi,
    Shli,
}

// ============================================================================
// Lexer / Token Parsers
// ============================================================================

/// Consume surrounding whitespace (including newlines).
fn ws<'a, F, O>(inner: F) -> impl FnMut(&'a str) -> IResult<&'a str, O>
where
    F: FnMut(&'a str) -> IResult<&'a str, O>,
{
    delimited(multispace0, inner, multispace0)
}

/// %arg0, %0, %c0_i32, %c-1_i32 (Polygeist names for constants like -1), etc.
fn variable(input: &str) -> IResult<&str, String> {
    map(
        recognize(tuple((
            char('%'),
            alt((
                take_while1(|c: char| c.is_alphanumeric() || c == '_' || c == '-'),
                digit1,
            )),
        ))),
        |s: &str| s.to_string(),
    )(input)
}

/// Parse a variable that may carry an SSA result number: %5#1
fn variable_with_hash(input: &str) -> IResult<&str, String> {
    map(
        recognize(tuple((
            char('%'),
            alt((
                take_while1(|c: char| c.is_alphanumeric() || c == '_' || c == '-'),
                digit1,
            )),
            opt(tuple((char('#'), digit1))),
        ))),
        |s: &str| s.to_string(),
    )(input)
}

/// Parse an integer literal (possibly negative).
fn integer(input: &str) -> IResult<&str, i32> {
    map(
        recognize(tuple((opt(char('-')), digit1))),
        |s: &str| s.parse().unwrap(),
    )(input)
}

/// @`symbol_name` (function / global names).
fn symbol(input: &str) -> IResult<&str, String> {
    map(
        recognize(tuple((
            char('@'),
            take_while1(|c: char| c.is_alphanumeric() || c == '_'),
        ))),
        |s: &str| s.to_string(),
    )(input)
}

/// Simplified MLIR type parser: memref<?x128xi32>, i32, index, f32, etc.
fn mlir_type(input: &str) -> IResult<&str, String> {
    map(
        take_while1(|c: char| {
            c.is_alphanumeric()
                || c == '<'
                || c == '>'
                || c == '?'
                || c == 'x'
                || c == '_'
        }),
        |s: &str| s.to_string(),
    )(input)
}

/// Parse a parenthesised, comma-separated list of types: (i32, i32, i32)
fn type_tuple(input: &str) -> IResult<&str, Vec<String>> {
    delimited(
        ws(char('(')),
        separated_list0(ws(char(',')), ws(mlir_type)),
        ws(char(')')),
    )(input)
}

// ============================================================================
// Statement Parsers
// ============================================================================

// --- arith.constant ---------------------------------------------------------
// %c32 = arith.constant 32 : i32
// %cst = arith.constant 0.000000e+00 : f32
// %true = arith.constant true
// %false = arith.constant false
fn parse_arith_constant(input: &str) -> IResult<&str, MLIRStatement> {
    let (input, result) = ws(variable)(input)?;
    let (input, _) = ws(char('='))(input)?;
    let (input, _) = ws(tag("arith.constant"))(input)?;
    let (input, val) = ws(take_while1(|c: char| {
        c.is_alphanumeric() || c == '.' || c == '+' || c == '-' || c == 'e' || c == 'E'
    }))(input)?;
    // type annotation is optional for bool constants (true / false)
    let (input, ty) = opt(tuple((ws(char(':')), ws(mlir_type))))(input)?;
    let ty = ty.map(|(_, t)| t).unwrap_or_else(|| {
        if val == "true" || val == "false" {
            "i1".to_string()
        } else {
            "unknown".to_string()
        }
    });

    Ok((
        input,
        MLIRStatement::Constant {
            result,
            value: val.to_string(),
            ty,
        },
    ))
}

// --- arith.index_cast -------------------------------------------------------
// %0 = arith.index_cast %arg11 : i32 to index
fn parse_arith_index_cast(input: &str) -> IResult<&str, MLIRStatement> {
    let (input, result) = ws(variable)(input)?;
    let (input, _) = ws(char('='))(input)?;
    let (input, _) = ws(tag("arith.index_cast"))(input)?;
    let (input, operand) = ws(variable)(input)?;
    let (input, _) = ws(char(':'))(input)?;
    let (input, from_ty) = ws(mlir_type)(input)?;
    let (input, _) = ws(tag("to"))(input)?;
    let (input, to_ty) = ws(mlir_type)(input)?;

    Ok((
        input,
        MLIRStatement::IndexCast {
            result,
            operand,
            from_ty,
            to_ty,
        },
    ))
}

// --- arith binary ops -------------------------------------------------------
// %2 = arith.muli %0, %1 : i32
fn parse_arith_op(input: &str) -> IResult<&str, MLIRStatement> {
    let (input, result) = ws(variable)(input)?;
    let (input, _) = ws(char('='))(input)?;
    let (input, op_str) = ws(alt((
        tag("arith.muli"),
        tag("arith.addi"),
        tag("arith.subi"),
        tag("arith.mulf"),
        tag("arith.addf"),
        tag("arith.andi"),
        tag("arith.ori"),
        tag("arith.divsi"),
        tag("arith.shli"),
    )))(input)?;

    let op = match op_str {
        "arith.muli" => ArithOpKind::Muli,
        "arith.addi" => ArithOpKind::Addi,
        "arith.subi" => ArithOpKind::Subi,
        "arith.mulf" => ArithOpKind::Mulf,
        "arith.addf" => ArithOpKind::Addf,
        "arith.andi" => ArithOpKind::Andi,
        "arith.ori" => ArithOpKind::Ori,
        "arith.divsi" => ArithOpKind::Divsi,
        "arith.shli" => ArithOpKind::Shli,
        _ => unreachable!(),
    };

    let (input, lhs) = ws(variable)(input)?;
    let (input, _) = ws(char(','))(input)?;
    let (input, rhs) = ws(variable)(input)?;
    let (input, _) = ws(char(':'))(input)?;
    let (input, ty) = ws(mlir_type)(input)?;

    Ok((
        input,
        MLIRStatement::ArithOp {
            result,
            op,
            lhs,
            rhs,
            ty,
        },
    ))
}

// --- arith.cmpi -------------------------------------------------------------
// %0 = arith.cmpi ne, %arg4, %c0_i32 : i32
fn parse_arith_cmpi(input: &str) -> IResult<&str, MLIRStatement> {
    let (input, result) = ws(variable)(input)?;
    let (input, _) = ws(char('='))(input)?;
    let (input, _) = ws(tag("arith.cmpi"))(input)?;
    let (input, predicate) = ws(take_while1(|c: char| c.is_alphanumeric() || c == '_'))(input)?;
    let (input, _) = ws(char(','))(input)?;
    let (input, lhs) = ws(variable)(input)?;
    let (input, _) = ws(char(','))(input)?;
    let (input, rhs) = ws(variable)(input)?;
    let (input, _) = ws(char(':'))(input)?;
    let (input, ty) = ws(mlir_type)(input)?;

    Ok((
        input,
        MLIRStatement::CmpIOp {
            result,
            predicate: predicate.to_string(),
            lhs,
            rhs,
            ty,
        },
    ))
}

// --- arith.select -----------------------------------------------------------
// %10 = arith.select %9, %c0_i32, %8 : i32
fn parse_arith_select(input: &str) -> IResult<&str, MLIRStatement> {
    let (input, result) = ws(variable)(input)?;
    let (input, _) = ws(char('='))(input)?;
    let (input, _) = ws(tag("arith.select"))(input)?;
    let (input, condition) = ws(variable)(input)?;
    let (input, _) = ws(char(','))(input)?;
    let (input, true_val) = ws(variable)(input)?;
    let (input, _) = ws(char(','))(input)?;
    let (input, false_val) = ws(variable)(input)?;
    let (input, _) = ws(char(':'))(input)?;
    let (input, ty) = ws(mlir_type)(input)?;

    Ok((
        input,
        MLIRStatement::SelectOp {
            result,
            condition,
            true_val,
            false_val,
            ty,
        },
    ))
}

// --- memref.alloca ----------------------------------------------------------
// %alloca = memref.alloca() : memref<64x64xi32>
fn parse_memref_alloca(input: &str) -> IResult<&str, MLIRStatement> {
    let (input, result) = ws(variable)(input)?;
    let (input, _) = ws(char('='))(input)?;
    let (input, _) = ws(tag("memref.alloca"))(input)?;
    let (input, _) = ws(char('('))(input)?;
    let (input, _) = ws(char(')'))(input)?;
    let (input, _) = ws(char(':'))(input)?;
    let (input, ty) = ws(mlir_type)(input)?;

    Ok((input, MLIRStatement::MemrefAlloca { result, ty }))
}

// --- hls.dataflow.buffer ----------------------------------------------------
// %2 = hls.dataflow.buffer {depth = 1 : i32} : memref<1xi32>
fn parse_hls_dataflow_buffer(input: &str) -> IResult<&str, MLIRStatement> {
    let (input, result) = ws(variable)(input)?;
    let (input, _) = ws(char('='))(input)?;
    let (input, _) = ws(tag("hls.dataflow.buffer"))(input)?;
    // parse attributes (starts with '{')
    let (input, _) = ws(char('{'))(input)?;
    let (input, _) = skip_balanced_braces(input)?;
    let (input, _) = ws(char(':'))(input)?;
    let (input, ty) = ws(mlir_type)(input)?;

    Ok((input, MLIRStatement::MemrefAlloca { result, ty }))
}

// --- memref.get_global ------------------------------------------------------
// %2 = memref.get_global @name : memref<3xi32>
fn parse_memref_get_global(input: &str) -> IResult<&str, MLIRStatement> {
    let (input, result) = ws(variable)(input)?;
    let (input, _) = ws(char('='))(input)?;
    let (input, _) = ws(tag("memref.get_global"))(input)?;
    let (input, sym_name) = ws(symbol)(input)?;
    let (input, _) = ws(char(':'))(input)?;
    let (input, ty) = ws(mlir_type)(input)?;

    Ok((
        input,
        MLIRStatement::MemrefGetGlobal {
            result,
            sym_name,
            ty,
        },
    ))
}

// --- memref.load ------------------------------------------------------------
// %10 = memref.load %arg10[%8] : memref<?xi32>
fn parse_memref_load(input: &str) -> IResult<&str, MLIRStatement> {
    let (input, result) = ws(variable)(input)?;
    let (input, _) = ws(char('='))(input)?;
    let (input, _) = ws(tag("memref.load"))(input)?;
    let (input, memref) = ws(variable)(input)?;
    let (input, _) = ws(char('['))(input)?;
    let (input, indices) = separated_list0(ws(char(',')), ws(variable))(input)?;
    let (input, _) = ws(char(']'))(input)?;
    let (input, _) = ws(char(':'))(input)?;
    let (input, ty) = ws(mlir_type)(input)?;

    Ok((
        input,
        MLIRStatement::Load {
            result,
            memref,
            indices,
            ty,
        },
    ))
}

// --- memref.store -----------------------------------------------------------
// memref.store %14, %arg2[%11, %c0] : memref<?x2x100xi32>
fn parse_memref_store(input: &str) -> IResult<&str, MLIRStatement> {
    let (input, _) = ws(tag("memref.store"))(input)?;
    let (input, val) = ws(variable)(input)?;
    let (input, _) = ws(char(','))(input)?;
    let (input, memref) = ws(variable)(input)?;
    let (input, _) = ws(char('['))(input)?;
    let (input, indices) = separated_list0(ws(char(',')), ws(variable))(input)?;
    let (input, _) = ws(char(']'))(input)?;
    let (input, _) = ws(char(':'))(input)?;
    let (input, ty) = ws(mlir_type)(input)?;

    Ok((
        input,
        MLIRStatement::Store {
            value: val,
            memref,
            indices,
            ty,
        },
    ))
}

// --- scf.for ----------------------------------------------------------------
// scf.for %arg15 = %c0 to %4 step %c1 { ... }
// %9 = scf.for %arg17 = %c0 to %3 step %c1 iter_args(%arg18 = %c0_i32) -> (i32) {
// %12:4 = scf.for %arg15 = %c0 to %8 step %c1 iter_args(...) -> (i32, i32, i32, i32) {
fn parse_scf_for(input: &str) -> IResult<&str, MLIRStatement> {
    // Optional result: %9 = or %12:4 =
    let (input, result_assign) = opt(tuple((
        ws(variable_with_hash),
        opt(tuple((char(':'), digit1))),
        ws(char('=')),
    )))(input)?;
    let result = result_assign.map(|(v, _, _)| {
        // strip ":N" suffix if present (e.g. %12:4 -> %12)
        if let Some(pos) = v.find(':') {
            v[..pos].to_string()
        } else {
            v
        }
    });

    let (input, _) = ws(tag("scf.for"))(input)?;
    let (input, iterator) = ws(variable)(input)?;
    let (input, _) = ws(char('='))(input)?;
    let (input, lower) = ws(variable)(input)?;
    let (input, _) = ws(tag("to"))(input)?;
    let (input, upper) = ws(variable)(input)?;
    let (input, _) = ws(tag("step"))(input)?;
    let (input, step) = ws(variable)(input)?;

    // Optional iter_args(%arg18 = %c0_i32)  — exactly one iter_arg assumed; multiple will panic.
    // Multi-arg variant: iter_args(%a = %x, %b = %y, ...) is NOT supported.
    let (input, iter_args_parsed) = opt(tuple((
        ws(tag("iter_args")),
        ws(char('(')),
        // Single iter_arg only; multi-arg parsing intentionally disabled:
        // separated_list0(ws(char(',')), map(tuple((ws(variable), ws(char('=')), ws(variable))), |(n,_,i)| (n,i))),
        map(
            tuple((ws(variable), ws(char('=')), ws(variable))),
            |(name, _, init): (String, _, String)| (name, init),
        ),
        ws(char(')')),
    )))(input)?;
    let iter_args: Option<(String, String)> = iter_args_parsed.map(|(_, _, arg, _)| arg);

    // Optional return types -> (i32, i32)
    let (input, result_tys) = opt(tuple((ws(tag("->")), type_tuple)))(input)?;
    let result_tys = result_tys.map(|(_, tys)| tys).unwrap_or_default();

    let (input, _) = ws(char('{'))(input)?;
    let (input, body) = many0(ws(parse_statement))(input)?;
    let (input, _) = ws(char('}'))(input)?;

    // If iter_args present, extract trailing scf.yield and build ScfForYield
    if let Some(iter_args) = iter_args {
        let (body, yield_val) = extract_trailing_yield(body);
        Ok((
            input,
            MLIRStatement::ScfForYield {
                result: result.expect("scf.for with iter_args must have a result"),
                iterator,
                lower,
                upper,
                step,
                iter_args,
                result_tys,
                yield_val,
                body,
            },
        ))
    } else {
        Ok((
            input,
            MLIRStatement::ScfFor {
                iterator,
                lower,
                upper,
                step,
                body,
            },
        ))
    }
}

// --- scf.if -----------------------------------------------------------------
// scf.if %7 { ... } else { ... }
// %17 = scf.if %16 -> (i32) { ... } else { ... }
// %11:3 = scf.if %10 -> (i32, i32, i1) { ... } else { ... }
fn parse_scf_if(input: &str) -> IResult<&str, MLIRStatement> {
    // Optional result
    let (input, result_assign) = opt(tuple((
        ws(variable_with_hash),
        opt(tuple((char(':'), digit1))),
        ws(char('=')),
    )))(input)?;
    let result = result_assign.map(|(v, _, _)| {
        if let Some(pos) = v.find(':') {
            v[..pos].to_string()
        } else {
            v
        }
    });

    let (input, _) = ws(tag("scf.if"))(input)?;
    let (input, condition) = ws(variable)(input)?;

    // Optional return types -> (i32, i32)
    let (input, result_tys) = opt(tuple((ws(tag("->")), type_tuple)))(input)?;
    let result_tys = result_tys.map(|(_, tys)| tys).unwrap_or_default();

    let (input, _) = ws(char('{'))(input)?;
    let (input, then_body) = many0(ws(parse_statement))(input)?;
    let (input, _) = ws(char('}'))(input)?;

    // Optional else block
    let (input, else_body) = opt(tuple((
        ws(tag("else")),
        ws(char('{')),
        many0(ws(parse_statement)),
        ws(char('}')),
    )))(input)?;
    let else_body = else_body
        .map(|(_, _, stmts, _)| stmts)
        .unwrap_or_default();

    // If result present, extract yields from both branches -> ScfIfYield
    if let Some(result) = result {
        let (then_body, then_yield) = extract_trailing_yield(then_body);
        let (else_body, else_yield) = extract_trailing_yield(else_body);
        // Single yield value per branch is enforced; multiple values will panic.
        Ok((
            input,
            MLIRStatement::ScfIfYield {
                result,
                result_tys,
                condition,
                then_body,
                else_body,
                then_yield,
                else_yield,
            },
        ))
    } else {
        Ok((
            input,
            MLIRStatement::ScfIf {
                condition,
                then_body,
                else_body,
            },
        ))
    }
}

/// Extract the trailing `ScfYield` from a body, returning (`remaining_body`, `yield_value`).
/// Panics if no `ScfYield` is found or if the yield carries more than one value.
fn extract_trailing_yield(mut body: Vec<MLIRStatement>) -> (Vec<MLIRStatement>, String) {
    if let Some(MLIRStatement::ScfYield { .. }) = body.last() {
        if let Some(MLIRStatement::ScfYield { values }) = body.pop() {
            assert_eq!(
                values.len(),
                1,
                "expected exactly one yield value, got {}",
                values.len()
            );
            return (body, values.into_iter().next().unwrap());
        }
    }
    panic!("expected a trailing scf.yield with exactly one value");
}

// --- scf.yield --------------------------------------------------------------
// scf.yield
// scf.yield %25 : i32
// Multi-value yield (e.g. scf.yield %arg10, %arg11 : i32, i32) is NOT supported;
// exactly one yield value is assumed; multiple values will panic at extraction.
fn parse_scf_yield(input: &str) -> IResult<&str, MLIRStatement> {
    let (input, _) = ws(tag("scf.yield"))(input)?;
    // Single-value yield only; multi-value parsing is intentionally disabled.
    // let (input, values) = separated_list0(ws(char(',')), ws(variable_with_hash))(input)?;
    let (input, value) = opt(ws(variable_with_hash))(input)?;
    let values = value.into_iter().collect::<Vec<_>>();
    // Consume optional type annotation (single type only)
    let (input, _) = opt(tuple((
        ws(char(':')),
        // separated_list0(ws(char(',')), ws(mlir_type)),
        ws(mlir_type),
    )))(input)?;

    Ok((input, MLIRStatement::ScfYield { values }))
}

// --- return -----------------------------------------------------------------
// return
// return %2 : i32
fn parse_return(input: &str) -> IResult<&str, MLIRStatement> {
    let (input, _) = ws(tag("return"))(input)?;
    let (input, values) = separated_list0(ws(char(',')), ws(variable))(input)?;
    // Consume optional type annotations
    let (input, _) = opt(tuple((
        ws(char(':')),
        separated_list0(ws(char(',')), ws(mlir_type)),
    )))(input)?;

    Ok((input, MLIRStatement::Return { values }))
}

// --- function call ----------------------------------------------------------
// %result = call @function_name(%arg0, %arg1) : (i32, i32) -> i32
// %result = func.call @function_name(%arg0) : (i32) -> i32   (Polygeist / func dialect)
// call @function_name(%arg0) : (i32) -> ()
fn parse_function_call(input: &str) -> IResult<&str, MLIRStatement> {
    let (input, result) = opt(tuple((ws(variable), ws(char('=')))))(input)?;
    let result = result.map(|(var, _)| var);

    let (input, _) = ws(alt((tag("func.call"), tag("call"))))(input)?;
    let (input, function_name) = ws(symbol)(input)?;
    let (input, _) = ws(char('('))(input)?;
    let (input, arguments) = separated_list0(ws(char(',')), variable)(input)?;
    let (input, _) = ws(char(')'))(input)?;
    let (input, _) = ws(char(':'))(input)?;
    let (input, ty) = ws(take_while1(|c: char| c != '\n' && c != '}'))(input)?;

    Ok((
        input,
        MLIRStatement::FunctionCall {
            result,
            function_name,
            arguments,
            ty: ty.trim().to_string(),
        },
    ))
}

// ============================================================================
// Top-level statement dispatcher
// ============================================================================

fn parse_statement(input: &str) -> IResult<&str, MLIRStatement> {
    alt((
        parse_arith_constant,
        parse_arith_index_cast,
        parse_arith_cmpi,
        parse_arith_select,
        parse_arith_op,
        parse_memref_alloca,
        parse_hls_dataflow_buffer,
        parse_memref_get_global,
        parse_memref_load,
        parse_memref_store,
        parse_scf_for,
        parse_scf_if,
        parse_scf_yield,
        parse_function_call,
        parse_return,
    ))(input)
}

// ============================================================================
// Function & Module Parsers
// ============================================================================

/// %arg0: memref<?x192xi32>
fn parse_argument(input: &str) -> IResult<&str, MLIRArgument> {
    let (input, name) = ws(variable)(input)?;
    let (input, _) = ws(char(':'))(input)?;
    let (input, ty) = ws(mlir_type)(input)?;
    Ok((input, MLIRArgument { name, ty }))
}

/// Skip a balanced { ... } block (including nested braces).
fn skip_balanced_braces(input: &str) -> IResult<&str, ()> {
    let mut depth = 1usize;
    let mut consumed = 0usize;
    for ch in input.chars() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    consumed += 1;
                    return Ok((&input[consumed..], ()));
                }
            }
            _ => {}
        }
        consumed += ch.len_utf8();
    }
    Err(nom::Err::Error(nom::error::Error::new(
        input,
        nom::error::ErrorKind::Eof,
    )))
}

/// Skip `attributes { ... }`.
fn skip_attributes(input: &str) -> IResult<&str, ()> {
    let (input, _) = ws(tag("attributes"))(input)?;
    let (input, _) = ws(char('{'))(input)?;
    skip_balanced_braces(input)
}

/// Parse optional `-> type` return annotation on func signature.
/// Returns the return type string (e.g. "i32" or "(i32, i32)").
fn parse_func_return_type(input: &str) -> IResult<&str, String> {
    let (input, _) = ws(tag("->"))(input)?;
    let (input, ty) = ws(alt((
        map(type_tuple, |tys| {
            if tys.len() == 1 { tys.into_iter().next().unwrap() }
            else { format!("({})", tys.join(", ")) }
        }),
        mlir_type,
    )))(input)?;
    Ok((input, ty))
}

/// func.func [@visibility] @name(%arg0: type, ...) -> rettype attributes {...} { body }
/// Polygeist uses `private` for internal helpers (e.g. `polybench_isqrt`).
fn parse_function(input: &str) -> IResult<&str, MLIRFunction> {
    let (input, _) = ws(tag("func.func"))(input)?;
    let (input, _) = opt(ws(alt((tag("private"), tag("public")))))(input)?;
    let (input, name) = ws(symbol)(input)?;
    let (input, _) = ws(char('('))(input)?;
    let (input, arguments) = separated_list0(ws(char(',')), parse_argument)(input)?;
    let (input, _) = ws(char(')'))(input)?;

    // Optional return type
    let (input, return_ty) = opt(parse_func_return_type)(input)?;
    // Optional attributes
    let (input, _) = opt(skip_attributes)(input)?;

    let (input, _) = ws(char('{'))(input)?;
    let (input, body) = many0(ws(parse_statement))(input)?;
    let (input, _) = ws(char('}'))(input)?;

    Ok((
        input,
        MLIRFunction {
            name,
            arguments,
            return_ty,
            body,
        },
    ))
}

/// Skip `module attributes { ... } {` header.
fn skip_module_header(input: &str) -> IResult<&str, ()> {
    let (input, _) = ws(tag("module"))(input)?;
    let (input, _) = opt(skip_attributes)(input)?;
    let (input, _) = ws(char('{'))(input)?;
    Ok((input, ()))
}

/// Skip `#set = affine_set<...>` or `#map = affine_map<...>` lines.
fn skip_set_or_map_def(input: &str) -> IResult<&str, ()> {
    let (input, _) = ws(char('#'))(input)?;
    let (input, _) = take_while1(|c: char| c.is_alphanumeric() || c == '_')(input)?;
    let (input, _) = ws(char('='))(input)?;
    let (input, _) = take_while(|c: char| c != '\n')(input)?;
    Ok((input, ()))
}

/// Skip `memref.global "private" @name : memref<3xi32> = dense<...>`.
fn skip_memref_global(input: &str) -> IResult<&str, ()> {
    let (input, _) = ws(tag("memref.global"))(input)?;
    let (input, _) = take_while(|c: char| c != '\n')(input)?;
    Ok((input, ()))
}

fn skip_module_level_decl(input: &str) -> IResult<&str, ()> {
    alt((skip_set_or_map_def, skip_memref_global))(input)
}

pub fn parse_mlir_module(input: &str) -> IResult<&str, MLIRModule> {
    let (input, ()) = skip_module_header(input)?;
    let (input, _) = many0(ws(skip_module_level_decl))(input)?;
    let (input, functions) = many0(ws(parse_function))(input)?;
    let (input, _) = ws(char('}'))(input)?;
    Ok((input, MLIRModule { functions }))
}

// ============================================================================
// Public API
// ============================================================================

pub fn parse_mlir(input: &str) -> Result<MLIRModule, String> {
    match parse_mlir_module(input) {
        Ok((_, module)) => Ok(module),
        Err(e) => Err(format!("Parse error: {e:?}")),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variable_parser() {
        assert_eq!(variable("%arg0"), Ok(("", "%arg0".to_string())));
        assert_eq!(variable("%0"), Ok(("", "%0".to_string())));
        assert_eq!(variable("%arg_123"), Ok(("", "%arg_123".to_string())));
        // Polygeist: constants like -1 get names such as %c-1_i32
        assert_eq!(
            variable("%c-1_i32"),
            Ok(("", "%c-1_i32".to_string()))
        );
    }

    #[test]
    fn test_variable_with_hash() {
        assert_eq!(
            variable_with_hash("%5#1"),
            Ok(("", "%5#1".to_string()))
        );
        assert_eq!(
            variable_with_hash("%arg10"),
            Ok(("", "%arg10".to_string()))
        );
    }

    #[test]
    fn test_integer_parser() {
        assert_eq!(integer("42"), Ok(("", 42)));
        assert_eq!(integer("-1"), Ok(("", -1)));
    }

    #[test]
    fn test_parse_arith_constant() {
        let input = "%c32 = arith.constant 32 : i32";
        let (_, stmt) = parse_arith_constant(input).unwrap();
        match stmt {
            MLIRStatement::Constant { result, value, ty } => {
                assert_eq!(result, "%c32");
                assert_eq!(value, "32");
                assert_eq!(ty, "i32");
            }
            _ => panic!("Expected Constant"),
        }
    }

    #[test]
    fn test_parse_arith_constant_negative_polygeist_name() {
        let input = "%c-1_i32 = arith.constant -1 : i32";
        let (_, stmt) = parse_arith_constant(input).unwrap();
        match stmt {
            MLIRStatement::Constant { result, value, ty } => {
                assert_eq!(result, "%c-1_i32");
                assert_eq!(value, "-1");
                assert_eq!(ty, "i32");
            }
            _ => panic!("Expected Constant"),
        }
    }

    #[test]
    fn test_parse_arith_op() {
        let input = "%2 = arith.muli %0, %1 : i32";
        let (_, stmt) = parse_arith_op(input).unwrap();
        match stmt {
            MLIRStatement::ArithOp {
                result,
                op,
                lhs,
                rhs,
                ty,
            } => {
                assert_eq!(result, "%2");
                assert_eq!(op, ArithOpKind::Muli);
                assert_eq!(lhs, "%0");
                assert_eq!(rhs, "%1");
                assert_eq!(ty, "i32");
            }
            _ => panic!("Expected ArithOp"),
        }
    }

    #[test]
    fn test_parse_memref_load() {
        let input = "%0 = memref.load %arg1[%arg3, %arg5] : memref<?x128xi32>";
        let (_, stmt) = parse_memref_load(input).unwrap();
        match stmt {
            MLIRStatement::Load {
                result,
                memref,
                indices,
                ty,
            } => {
                assert_eq!(result, "%0");
                assert_eq!(memref, "%arg1");
                assert_eq!(indices, vec!["%arg3", "%arg5"]);
                assert_eq!(ty, "memref<?x128xi32>");
            }
            _ => panic!("Expected Load"),
        }
    }

    #[test]
    fn test_parse_memref_store() {
        let input = "memref.store %4, %arg0[%arg3, %arg4] : memref<?x192xi32>";
        let (_, stmt) = parse_memref_store(input).unwrap();
        match stmt {
            MLIRStatement::Store {
                value,
                memref,
                indices,
                ty,
            } => {
                assert_eq!(value, "%4");
                assert_eq!(memref, "%arg0");
                assert_eq!(indices, vec!["%arg3", "%arg4"]);
                assert_eq!(ty, "memref<?x192xi32>");
            }
            _ => panic!("Expected Store"),
        }
    }

    #[test]
    fn test_parse_scf_for_simple() {
        let input = "scf.for %arg5 = %c0 to %c64 step %c1 { }";
        let (_, stmt) = parse_scf_for(input).unwrap();
        match stmt {
            MLIRStatement::ScfFor {
                iterator,
                lower,
                upper,
                step,
                body,
            } => {
                assert_eq!(iterator, "%arg5");
                assert_eq!(lower, "%c0");
                assert_eq!(upper, "%c64");
                assert_eq!(step, "%c1");
                assert!(body.is_empty());
            }
            _ => panic!("Expected ScfFor"),
        }
    }

    #[test]
    fn test_parse_scf_for_with_yield() {
        let input = r#"
            %9 = scf.for %arg17 = %c0 to %c10 step %c1 iter_args(%arg18 = %c0_i32) -> (i32) {
                %v = arith.addi %arg18, %c1_i32 : i32
                scf.yield %v : i32
            }
        "#;
        let (_, stmt) = parse_scf_for(input.trim()).unwrap();
        match stmt {
            MLIRStatement::ScfForYield {
                result,
                iterator,
                iter_args,
                result_tys,
                yield_val,
                body,
                ..
            } => {
                assert_eq!(result, "%9");
                assert_eq!(iterator, "%arg17");
                assert_eq!(iter_args, ("%arg18".to_string(), "%c0_i32".to_string()));
                assert_eq!(result_tys, vec!["i32"]);
                assert_eq!(yield_val, "%v");
                assert_eq!(body.len(), 1); // just the arith.addi, yield extracted
            }
            _ => panic!("Expected ScfForYield"),
        }
    }

    #[test]
    fn test_parse_scf_yield_internal() {
        // scf.yield still parseable (used internally before extraction)
        // Multi-value yield is not supported; only single-value is tested.
        // let input = "scf.yield %arg10, %arg11 : i32, i32";
        let input = "scf.yield %arg10 : i32";
        let (_, stmt) = parse_scf_yield(input).unwrap();
        match stmt {
            MLIRStatement::ScfYield { values } => {
                assert_eq!(values, vec!["%arg10"]);
            }
            _ => panic!("Expected ScfYield"),
        }
    }

    #[test]
    fn test_parse_return_with_value() {
        let input = "return %2 : i32";
        let (_, stmt) = parse_return(input).unwrap();
        match stmt {
            MLIRStatement::Return { values } => {
                assert_eq!(values, vec!["%2"]);
            }
            _ => panic!("Expected Return"),
        }
    }

    #[test]
    fn test_parse_return_void() {
        let input = "return";
        let (_, stmt) = parse_return(input).unwrap();
        match stmt {
            MLIRStatement::Return { values } => {
                assert!(values.is_empty());
            }
            _ => panic!("Expected Return"),
        }
    }

    #[test]
    fn test_parse_function_call_with_result() {
        let input = "%0 = call @my_func(%arg0, %arg1) : (i32, i32) -> i32";
        let (_, stmt) = parse_function_call(input).unwrap();
        match stmt {
            MLIRStatement::FunctionCall {
                result,
                function_name,
                arguments,
                ty,
            } => {
                assert_eq!(result, Some("%0".to_string()));
                assert_eq!(function_name, "@my_func");
                assert_eq!(arguments, vec!["%arg0", "%arg1"]);
                assert_eq!(ty, "(i32, i32) -> i32");
            }
            _ => panic!("Expected FunctionCall"),
        }
    }

    #[test]
    fn test_parse_func_call_polygeist() {
        let input = "%3 = func.call @polybench_isqrt(%2) : (i32) -> i32";
        let (_, stmt) = parse_function_call(input).unwrap();
        match stmt {
            MLIRStatement::FunctionCall {
                result,
                function_name,
                arguments,
                ty,
            } => {
                assert_eq!(result, Some("%3".to_string()));
                assert_eq!(function_name, "@polybench_isqrt");
                assert_eq!(arguments, vec!["%2"]);
                assert_eq!(ty, "(i32) -> i32");
            }
            _ => panic!("Expected FunctionCall"),
        }
    }

    #[test]
    fn test_parse_module_with_private_func_and_llvm_linkage() {
        let module = r#"module {
  func.func private @polybench_isqrt(%arg0: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c0_i32 = arith.constant 0 : i32
    return %c0_i32 : i32
  }
}"#;
        let m = parse_mlir(module).expect("parse module with private func");
        assert_eq!(m.functions.len(), 1);
        assert_eq!(m.functions[0].name, "@polybench_isqrt");
    }

    #[test]
    fn test_parse_function_call_void() {
        let input = "call @void_func(%arg0) : (i32) -> ()";
        let (_, stmt) = parse_function_call(input).unwrap();
        match stmt {
            MLIRStatement::FunctionCall {
                result,
                function_name,
                arguments,
                ..
            } => {
                assert_eq!(result, None);
                assert_eq!(function_name, "@void_func");
                assert_eq!(arguments, vec!["%arg0"]);
            }
            _ => panic!("Expected FunctionCall"),
        }
    }

    #[test]
    fn test_parse_scf_if_with_yield() {
        let input = r#"
            %3 = scf.if %0 -> (i32) {
                scf.yield %5 : i32
            } else {
                scf.yield %2 : i32
            }
        "#;
        let (_, stmt) = parse_scf_if(input.trim()).unwrap();
        match stmt {
            MLIRStatement::ScfIfYield {
                result,
                result_tys,
                condition,
                then_body,
                else_body,
                then_yield,
                else_yield,
            } => {
                assert_eq!(result, "%3");
                assert_eq!(result_tys, vec!["i32"]);
                assert_eq!(condition, "%0");
                assert!(then_body.is_empty()); // yield extracted
                assert!(else_body.is_empty()); // yield extracted
                assert_eq!(then_yield, "%5");
                assert_eq!(else_yield, "%2");
            }
            _ => panic!("Expected ScfIfYield"),
        }
    }

    #[test]
    fn test_parse_scf_if_no_yield() {
        let input = r#"
            scf.if %cond {
                memref.store %x, %buf[%c0] : memref<1xi32>
            }
        "#;
        let (_, stmt) = parse_scf_if(input.trim()).unwrap();
        match stmt {
            MLIRStatement::ScfIf {
                condition,
                then_body,
                else_body,
            } => {
                assert_eq!(condition, "%cond");
                assert_eq!(then_body.len(), 1);
                assert!(else_body.is_empty());
            }
            _ => panic!("Expected ScfIf"),
        }
    }
}
