/*

In here I will write a way to parse MLIR into my very basic IR
The plan
first every variable should be padded with p_i (to make sure unique across programs I think)

MLIR will look like:
f(%x, %y)
%c = %x op %y
%d = %c op %c
...

We will parse this into a RecExpr (which is basically our ast)
For each assign we will get the ir add to the recexpr and then save a mapping to the its recexpr id
when callin to_ir on a binop(x, y) we will effceitvely look up the the id for x and y and create a recexpr with binop(id_x1, id_x2)
when assigning this will be added the recexpr, get an id, and c will map to it

Figure out:

Function Defs/Calls
For Loops
Returns

We technically have IR support for all except returns,
In IR land function application is like a lambda so I think the returns should be ok

if I remember MLIR for loops use some sort of yield to say a value is prodcued or something
so in my gemm cases we need to think abt that


*/

use std::{
    collections::{HashMap, HashSet},
    fmt::format,
    path::PathBuf,
};

// use babble::{ast_node::AstNode, learn::LibId, teachable::DeBruijnIndex};
use egg::{Id, RecExpr};
use indexmap::IndexSet;
use itertools::{Either, Itertools};

use crate::{
    lang::{MLIRLang, MLIROps},
    mlir_ast::{parse_mlir, ArithOpKind, MLIRFunction, MLIRModule, MLIRStatement},
};

type HLSIR = AstNode<MLIRLang>;
type HLSAST = RecExpr<HLSIR>;

/// Single line in output: either an SSA def (deduped) or structural (not deduped).
#[derive(Clone)]
enum MlirLine {
    Def(String),
    Structural(String),
}

/// Collects MLIR lines in insertion order. Defs are deduplicated (first occurrence
/// wins); structural lines are not, so "return" and "}" can repeat per function.
#[derive(Default)]
struct OrderedMlirLines {
    /// Lines in output order; defs are deduped via `defs_seen`.
    lines: Vec<MlirLine>,
    /// Tracks which def strings we've already emitted so we don't repeat.
    defs_seen: IndexSet<String>,
}

impl OrderedMlirLines {
    fn new() -> Self {
        Self::default()
    }

    fn extend(&mut self, other: &Self) {
        for line in &other.lines {
            match line {
                MlirLine::Def(s) => self.push_def(s.clone()),
                MlirLine::Structural(s) => self.push_structural(s.clone()),
            }
        }
    }

    fn push_def(&mut self, s: String) {
        if self.defs_seen.insert(s.clone()) {
            self.lines.push(MlirLine::Def(s));
        }
    }

    fn push_structural(&mut self, s: String) {
        self.lines.push(MlirLine::Structural(s));
    }

    /// Remove and return the last line in output order.
    fn pop_last(&mut self) -> Option<String> {
        let last = self.lines.pop()?;
        let s = match &last {
            MlirLine::Def(x) => {
                self.defs_seen.remove(x);
                x.clone()
            }
            MlirLine::Structural(x) => x.clone(),
        };
        Some(s)
    }

    fn to_vec(&self) -> Vec<String> {
        self.lines
            .iter()
            .map(|l| match l {
                MlirLine::Def(s) | MlirLine::Structural(s) => s.clone(),
            })
            .collect()
    }
}

#[derive(Debug, Default)]
pub struct MLIRRecExprParser {
    mlir_recexpr: HLSAST,
    mlir_to_expr_id_map: HashMap<String, Id>,
    fn_to_lib_map: HashMap<String, LibId>,
    loop_count: usize,
}

impl MLIRRecExprParser {
    // I think this is more of a stitching of all the functions
    // Could probably do the scoping here --
    // oh slotted egraphs...
    fn mlir_module_to_expr(&mut self, mlir_module: &MLIRModule) -> Id {
        // first pass to just get all libs
        for (i, func) in mlir_module.functions.iter().enumerate() {
            // When we encounter a function defintion we must also create its lib id so it can be applied later
            let func_name = func.name.clone();
            let fn_lib_id = LibId(i);
            self.fn_to_lib_map.insert(func_name, fn_lib_id); // This function now has a valid lib-id so it can be referenced with an application
        }

        // fairly certain the nect step here is to make this the child of a Lib (node)
        // Which should kind of happen recurssively until the top function

        // something like

        /*  lib-f1 --> f1-def()
              :
            lib-f2 --> f2-def()
              :
            lib-f3 --> f3-def()
              :
              :
              :
              :
            top_block {
                ...
                ...
            }
        */

        // This will require finding 'main'
        // let mut main_block_def = AstNode::new(MLIRLang::Block, vec![]); // just temporary for the top block def
        // let mut main_id = self.mlir_main_function_to_id(mlir_module);

        // // second pass creates the expr for each function pass to just get all libs
        // for (i, func) in mlir_module.functions.iter().enumerate() {
        //     let function_name = func.name.as_str(); // This will add the function defintion to the recurssive expression
        //     if Self::is_main(function_name) {
        //         continue;
        //     } // skip main as alr handled above with main id
        //     let tag = format!("f_{i}");
        //     let fn_def_id = self.mlir_function_to_id(func, tag.as_str());
        //     let fn_lib_id = self
        //         .fn_to_lib_map
        //         .get(function_name)
        //         .unwrap_or_else(|| panic!("Function {} not in Fn-LibID Map!", function_name));
        //     let main_block_def = AstNode::new(
        //         MLIRLang::new(MLIROps::Lib(*fn_lib_id)),
        //         vec![fn_def_id, main_id],
        //     );
        //     main_id = self.mlir_recexpr.add(main_block_def);
        // }

        // main_id

        // mlir_module
        //     .functions
        //     .iter()
        //     .find(|func| Self::is_main(&func.name)).map(|f| self.mlir_function_to_id(f, "top"))

        // let (base_id, fns) = match mlir_module.functions.iter().position(|f| Self::is_main(&f.name)) {
        //     Some(i) => {
        //         let main = &mlir_module.functions[i];
        //         let main_id = self.mlir_function_to_id(main, "top");
        //         let fns = mlir_module.functions.into_iter().filter(|f| !Self::is_main(&f.name)).into_iter();
        //         (main_id, fns)
        //     }
        //     None => {
        //         let mut module_iters =  mlir_module.functions.iter();
        //         let first = module_iters.next().expect("Expect atleast 1 function");
        //         let first_id = self.mlir_function_to_id(first, "f_0");
        //         (first_id, module_iters.into_iter())
        //     }
        // };

        let (pos, base_id) = mlir_module
            .functions
            .iter()
            .enumerate()
            .find_or_last(|(_pos, func)| Self::is_main(&func.name))
            .map(|(pos, f)| (pos, self.mlir_function_to_id(f, "top")))
            .expect("Expect Atleast One function");

        let mut fns = mlir_module.functions.clone();
        fns.remove(pos);

        let fn_id = fns.iter().enumerate().fold(base_id, |fn_ids, (i, func)| {
            let function_name = func.name.as_str();
            let tag = format!("f_{i}");
            let fn_def_id = self.mlir_function_to_id(func, tag.as_str());
            let fn_lib_id = self
                .fn_to_lib_map
                .get(function_name)
                .unwrap_or_else(|| panic!("Function {} not in Fn-LibID Map!", function_name));
            let main_block_def = AstNode::new(
                MLIRLang::new(MLIROps::Lib(*fn_lib_id)),
                vec![fn_def_id, fn_ids],
            );
            self.mlir_recexpr.add(main_block_def)
        });

        fn_id
    }

    // Adds the main function
    // if it does not exist just use an empty block
    // There is a seperate procedure because we dont want a lib for main
    fn mlir_main_function_to_id(&mut self, mlir_module: &MLIRModule) -> Option<Id> {
        // if let Some(main_func) = mlir_module
        //     .functions
        //     .iter()
        //     .find(|func| Self::is_main(&func.name))
        // {
        //     Some(self.mlir_function_to_id(main_func, "top"))
        // } else {
        //     // let main_expr = AstNode::new(MLIRLang::new(MLIROps::Block), vec![]); // main is empty by default
        //     // self.mlir_recexpr.add(main_expr)
        //     None
        // }

        mlir_module
            .functions
            .iter()
            .find(|func| Self::is_main(&func.name))
            .map(|f| self.mlir_function_to_id(f, "top"))
    }

    // Wrap the statements into a lambda function
    fn mlir_function_to_id(&mut self, mlir_function: &MLIRFunction, name_tag: &str) -> Id {
        // Function header - args - etc here ...
        // almost def needed bc of debruijn indices in Fdef
        // build recexpr mapping var strings to db indices
        // pass this rec_expr down the line...

        // Really what we are creating is the RecExpr for the function Definition
        // Which will be used/named during a function call?

        // let function_name = mlir_function.name.as_str();

        // Create the DbVars for each MLIR argument, preserving their types
        for (i, arg) in mlir_function.arguments.iter().enumerate() {
            let db_var = DeBruijnIndex::new(i, name_tag);
            let db_var_expr = AstNode::new(
                MLIRLang::new_with_name_and_type(MLIROps::DBVar(db_var), &arg.name, &arg.ty),
                vec![],
            );
            let db_var_id = self.mlir_recexpr.add(db_var_expr);
            let var_name = arg.name.clone();
            self.mlir_to_expr_id_map.insert(var_name, db_var_id);
        }

        // This is now each statement in the function body as list of ids
        let body_exprs = self.mlir_block_to_ids(mlir_function.body.as_slice(), name_tag);

        // The block of statements is added to the RecExpr
        let body_block = AstNode::new(MLIRLang::new(MLIROps::Block), body_exprs);
        let body_id = self.mlir_recexpr.add(body_block);

        // We now need to create Fn -> Fn -> ... wrapper around the body
        // There is one Fn for every argument
        let mut fdef_id = body_id;
        let lib_id = self.fn_to_lib_map.get(mlir_function.name.as_str()).expect("MLIR Function Not in Lib Map");
        // println!("Fdef for LibId: {lib_id}");
        for _arg in &mlir_function.arguments {
            let fn_node = AstNode::new(MLIRLang::new(MLIROps::FDef(*lib_id)), vec![fdef_id]);
            fdef_id = self.mlir_recexpr.add(fn_node);
        }

        // fdef_id is the start of the Function Definition
        // This one of the children to the Lib, the other is the argument application list during a function call

        let libdef = format!("{}_def", mlir_function.name);
        self.mlir_to_expr_id_map.insert(libdef, fdef_id);

        fdef_id
        // self.mlir_recexpr.clone()
    }

    fn is_main(function_name: &str) -> bool {
        function_name.contains("top")
    }

    fn mlir_block_to_ids(&mut self, mlir_statements: &[MLIRStatement], name_tag: &str) -> Vec<Id> {
        // let mut expr_ids = vec![];
        let mut referenced_ids = HashSet::new();
        let expr_ids = mlir_statements
            .iter()
            .map(|s| {
                let (x, local_ref_ids) = self.mlir_statement_to_expr(s, name_tag);
                let x_id = self.mlir_recexpr.add(x);

                referenced_ids.extend(local_ref_ids);
                if let Some(var_assign) = s.get_var_assign() {
                    // map the expr id to the variable its assigned to
                    self.mlir_to_expr_id_map.insert(var_assign, x_id);
                }
                x_id
            })
            .collect_vec(); // this is right now all the ids in a body

        // we now need to trim the body_ids to be only those which are not referrenced earlier - I think the best way to do this with a HashSet of referenced IDs
        let body_ids = expr_ids
            .into_iter()
            .filter(
                |&def_id| !(referenced_ids.contains(&def_id)), // if expr_id is referenced in the body we do not want to include it in the final body as it is implicitly defined in a later ID AST
            )
            .collect_vec();

        body_ids
    }

    fn str_to_id(&mut self, s: &String) -> Id {
        // strings may be vars in which case have IDs
        // but if theyre constants this may be the first time they are encountered

        match self.mlir_to_expr_id_map.get(s) {
            Some(expr_id) => expr_id.to_owned(),
            None => {
                // check if the string is an integer that could be a constant
                if let Ok(const_val) = s.parse::<i32>() {
                    let const_node = AstNode::new(MLIRLang::new_const(const_val), vec![]);
                    let const_id = self.mlir_recexpr.add(const_node);
                    self.mlir_to_expr_id_map.insert(s.to_owned(), const_id);
                    const_id
                } else {
                    panic!("Variable {} not found in mapping and is not a constant!", s)
                }
            }
        }
    }

    // fn indices_to_id(&self, s: &String) -> Id {
    //     // an index may be a constant
    // }

    #[allow(clippy::too_many_lines)]
    // Return the AST and a vector of the refernced IDs
    fn mlir_statement_to_expr(
        &mut self,
        mlir_statement: &MLIRStatement,
        name_tag: &str,
    ) -> (HLSIR, Vec<Id>) {
        // Where the real mapping will happen! - This only needs to be recursive with For loops I think
        match mlir_statement {
            MLIRStatement::ArithOp {
                result,
                op,
                lhs,
                rhs,
                ty,
            } => {
                let left_id = self.str_to_id(lhs);
                let right_id = self.str_to_id(rhs);
                let mk = |mlir_op| {
                    (
                        AstNode::new(
                            MLIRLang::new_with_name_and_type(mlir_op, result, ty),
                            vec![left_id, right_id],
                        ),
                        vec![left_id, right_id],
                    )
                };
                match op {
                    ArithOpKind::Addi => mk(MLIROps::Add),
                    ArithOpKind::Muli => mk(MLIROps::Mult),
                    ArithOpKind::Subi => mk(MLIROps::Sub),
                    ArithOpKind::Andi => mk(MLIROps::And),
                    ArithOpKind::Ori => mk(MLIROps::Or),
                    ArithOpKind::Divsi => mk(MLIROps::Div),
                    ArithOpKind::Shli => mk(MLIROps::LeftShift),
                    ArithOpKind::Mulf | ArithOpKind::Addf => {
                        panic!("Float arithmetic operations are not supported")
                    }
                }
            }
            MLIRStatement::Load {
                result,
                memref,
                indices,
                ty,
            } => {
                let array_id = self.str_to_id(memref);
                let indices_ids = indices.iter().map(|s| self.str_to_id(s)).collect_vec();
                let load_args = vec![array_id].into_iter().chain(indices_ids).collect_vec();
                let load_args_clone = load_args.clone();
                let load_ast = match indices.len() {
                    0 => panic!("Load Word with 0 Index Vars!"),
                    1 => AstNode::new(
                        MLIRLang::new_with_name_and_type(MLIROps::Load, result, ty),
                        load_args_clone,
                    ),
                    2 => AstNode::new(
                        MLIRLang::new_with_name_and_type(MLIROps::Load2D, result, ty),
                        load_args_clone,
                    ),
                    3 => AstNode::new(
                        MLIRLang::new_with_name_and_type(MLIROps::Load3D, result, ty),
                        load_args_clone,
                    ),
                    4 => AstNode::new(
                        MLIRLang::new_with_name_and_type(MLIROps::Load4D, result, ty),
                        load_args_clone,
                    ),
                    _ => panic!("Does not support Loads of over 4 dims!"),
                };

                (load_ast, load_args)
            }
            MLIRStatement::Store {
                value,
                memref,
                indices,
                ty,
            } => {
                let value_id = self.str_to_id(value);
                let array_id = self.str_to_id(memref);
                let indices_ids = indices.iter().map(|s| self.str_to_id(s)).collect_vec();
                let store_args = vec![array_id]
                    .into_iter()
                    .chain(indices_ids)
                    .chain(vec![value_id])
                    .collect_vec();
                let store_args_clone = store_args.clone();

                let store_ast = match indices.len() {
                    0 => panic!("Store Word with 0 Index Vars!"),
                    1 => AstNode::new(
                        MLIRLang::new_with_type(MLIROps::Store, ty),
                        store_args_clone,
                    ),
                    2 => AstNode::new(
                        MLIRLang::new_with_type(MLIROps::Store2D, ty),
                        store_args_clone,
                    ),
                    3 => AstNode::new(
                        MLIRLang::new_with_type(MLIROps::Store3D, ty),
                        store_args_clone,
                    ),
                    4 => AstNode::new(
                        MLIRLang::new_with_type(MLIROps::Store4D, ty),
                        store_args_clone,
                    ),
                    _ => panic!("Does not support Stores of over 4 dims!"),
                };

                (store_ast, store_args)
            }
            MLIRStatement::ScfFor {
                iterator,
                lower,
                upper,
                step,
                body,
            } => {
                // let to_mlirlang = |var: &String| self.str_to_id(s)

                let lb_id = self.str_to_id(lower);
                let ub_id = self.str_to_id(upper);
                let stp_id = self.str_to_id(step);

                let iter_name = format!("{iterator}_{name_tag}"); // tag the loop variable with the function name to keep unique across functions
                let arg_expr = AstNode::new(MLIRLang::new_var(iter_name.as_str()), vec![]);

                // Give each loop a new name
                // Could maybe do a thing where loop names only count per function
                // but im lazy and think its eq anw :)

                let loop_name: String = format!("loop-{}", self.loop_count);
                self.loop_count += 1;

                let loop_name_expr = AstNode::new(MLIRLang::new_var(loop_name.as_str()), vec![]);

                // The constants may exist also...
                // but assume they dont I guess..
                let loop_name_id = self.mlir_recexpr.add(loop_name_expr);
                let arg_id = self.mlir_recexpr.add(arg_expr);
                // let lb_id = self.mlir_recexpr.add(lb_const);
                // let ub_id = self.mlir_recexpr.add(ub_const);
                // let stp_id = self.mlir_recexpr.add(stp_const);

                // for scoping:
                // This would allow multiple for loops to use the same var name
                // Is that even smthn that may happen in mlir...
                let saved_mapping = self.mlir_to_expr_id_map.clone();
                self.mlir_to_expr_id_map.insert(iterator.to_owned(), arg_id);

                let body_exprs = self.mlir_block_to_ids(body, name_tag);
                let body_block = AstNode::new(MLIRLang::new(MLIROps::Block), body_exprs);
                let body_id = self.mlir_recexpr.add(body_block);

                // Restore the mapping after processing the loop body
                self.mlir_to_expr_id_map = saved_mapping;

                let for_node = AstNode::new(
                    MLIRLang::new(MLIROps::ForLoop),
                    vec![loop_name_id, arg_id, lb_id, ub_id, stp_id, body_id],
                );

                (for_node, vec![lb_id, ub_id, stp_id, arg_id])
            }
            MLIRStatement::ScfForYield {
                result,
                iterator,
                lower,
                upper,
                step,
                iter_args: (iter_arg, init_val),
                result_tys: _,
                yield_val,
                body,
            } => {
                // let to_mlirlang = |var: &String| self.str_to_id(s)

                let lb_id = self.str_to_id(lower);
                let ub_id = self.str_to_id(upper);
                let stp_id = self.str_to_id(step);
                let iter_arg_init_id = self.str_to_id(init_val);

                let iter_name = format!("{iterator}_{name_tag}"); // tag the loop variable with the function name to keep unique across functions
                let iter_arg_name = format!("{iter_arg}_{name_tag}"); // tag the loop variable with the function name to keep unique across functions

                let iter_expr = AstNode::new(MLIRLang::new_var(iter_name.as_str()), vec![]);
                let iter_arg_expr = AstNode::new(MLIRLang::new_var(iter_arg_name.as_str()), vec![]);

                // Give each loop a new name
                // Could maybe do a thing where loop names only count per function
                // but im lazy and think its eq anw :)

                let loop_name: String = format!("loop-{}", self.loop_count);
                self.loop_count += 1;

                let loop_name_expr = AstNode::new(MLIRLang::new_var(loop_name.as_str()), vec![]);

                // The constants may exist also...
                // but assume they dont I guess..
                let loop_name_id = self.mlir_recexpr.add(loop_name_expr);
                let iter_id = self.mlir_recexpr.add(iter_expr);
                let arg_id = self.mlir_recexpr.add(iter_arg_expr);
                // let lb_id = self.mlir_recexpr.add(lb_const);
                // let ub_id = self.mlir_recexpr.add(ub_const);
                // let stp_id = self.mlir_recexpr.add(stp_const);

                // for scoping:
                // This would allow multiple for loops to use the same var name
                // Is that even smthn that may happen in mlir...
                let saved_mapping = self.mlir_to_expr_id_map.clone();
                self.mlir_to_expr_id_map.insert(iterator.to_owned(), arg_id);

                let body_exprs = self.mlir_block_to_ids(body, name_tag);
                let body_block = AstNode::new(MLIRLang::new(MLIROps::Block), body_exprs);
                let body_id = self.mlir_recexpr.add(body_block);

                // after the body has been processed we can now point to the yield ID

                let yield_id = self.str_to_id(yield_val);

                // Restore the mapping after processing the loop body
                self.mlir_to_expr_id_map = saved_mapping;

                let for_node = AstNode::new(
                    MLIRLang::new_with_name(MLIROps::YieldForLoop, result),
                    vec![
                        loop_name_id,
                        arg_id,
                        lb_id,
                        ub_id,
                        stp_id,
                        arg_id,
                        iter_arg_init_id,
                        body_id,
                        yield_id,
                    ],
                );

                (for_node, vec![lb_id, ub_id, stp_id, arg_id])
            }
            MLIRStatement::ScfIf {
                condition,
                then_body,
                else_body,
            } => {
                let condition_id = self.str_to_id(condition);
                let then_body_exprs = self.mlir_block_to_ids(then_body, name_tag);
                let then_body_block = AstNode::new(MLIRLang::new(MLIROps::Block), then_body_exprs);
                let then_body_id = self.mlir_recexpr.add(then_body_block);

                let body_expr = if else_body.is_empty() {
                    AstNode::new(
                        MLIRLang::new(MLIROps::IfThen),
                        vec![condition_id, then_body_id],
                    )
                } else {
                    let else_body_exprs = self.mlir_block_to_ids(else_body, name_tag);
                    let else_body_block =
                        AstNode::new(MLIRLang::new(MLIROps::Block), else_body_exprs);
                    let else_body_id = self.mlir_recexpr.add(else_body_block);
                    AstNode::new(
                        MLIRLang::new(MLIROps::IfThenElse),
                        vec![condition_id, then_body_id, else_body_id],
                    )
                };

                (body_expr, vec![condition_id])
            }
            MLIRStatement::ScfIfYield {
                result,
                result_tys,
                condition,
                then_body,
                else_body,
                then_yield,
                else_yield,
            } => {
                let condition_id = self.str_to_id(condition);
                let then_body_exprs = self.mlir_block_to_ids(then_body, name_tag);
                let then_body_block = AstNode::new(MLIRLang::new(MLIROps::Block), then_body_exprs);
                let then_body_id = self.mlir_recexpr.add(then_body_block);
                let then_yield_id = self.str_to_id(then_yield);

                let body_expr = if else_body.is_empty() {
                    AstNode::new(
                        MLIRLang::new(MLIROps::YieldingIfThen),
                        vec![condition_id, then_body_id, then_yield_id],
                    )
                } else {
                    let else_body_exprs = self.mlir_block_to_ids(else_body, name_tag);
                    let else_body_block =
                        AstNode::new(MLIRLang::new(MLIROps::Block), else_body_exprs);
                    let else_body_id = self.mlir_recexpr.add(else_body_block);
                    let else_yield_id = self.str_to_id(else_yield);
                    AstNode::new(
                        MLIRLang::new(MLIROps::YieldingIfThenElse),
                        vec![
                            condition_id,
                            then_body_id,
                            then_yield_id,
                            else_body_id,
                            else_yield_id,
                        ],
                    )
                };

                (body_expr, vec![condition_id])
            }
            MLIRStatement::FunctionCall {
                result: _,
                function_name,
                arguments,
                ty,
            } => {
                // A function call is an application chain
                // with the last application calling the lib-var

                let fn_libid = self
                    .fn_to_lib_map
                    .get(function_name)
                    .unwrap_or_else(|| panic!("Function {} not in Fn-LibID Map!", function_name));

                let lib_node = AstNode::new(MLIRLang::new(MLIROps::LibVar(*fn_libid)), vec![]);
                let mut application_expr = lib_node;
                // here we now need a for loop to iterate over the args and make the function expr
                for arg in arguments {
                    let application_id = self.mlir_recexpr.add(application_expr);
                    let arg_id = self.str_to_id(arg);
                    application_expr = AstNode::new(
                        MLIRLang::new_with_type(MLIROps::Call, ty),
                        vec![application_id, arg_id],
                    );
                }

                (application_expr, vec![])
            }

            MLIRStatement::Return { .. } => (
                AstNode::new(MLIRLang::new(MLIROps::FReturn), vec![]),
                vec![],
            ),

            // --- Constant: register the variable name → id mapping, produce a Constant node ---
            MLIRStatement::Constant { result, value, ty } => {
                // Try to parse as i32; if it fails, store as a named variable
                let node = if let Ok(const_val) = value.parse::<i32>() {
                    AstNode::new(
                        MLIRLang::new_with_name_and_type(MLIROps::Constant(const_val), result, ty),
                        vec![],
                    )
                } else {
                    // bool constants (true/false) or float literals — treat as typed variable
                    let vname = format!("{result}_{name_tag}");
                    let varname = vname.as_str(); // tage with function name to keep unique across functions
                    AstNode::new(
                        MLIRLang::new_with_name_and_type(
                            MLIROps::Variable(varname.into()),
                            varname,
                            ty,
                        ),
                        vec![],
                    )
                };

                (node, vec![])
            }

            // --- IndexCast: transparent pass-through (just alias the result to the operand) ---
            MLIRStatement::IndexCast {
                result,
                operand,
                from_ty: _,
                to_ty,
            } => {
                // Index casts don't change the value semantically for our purposes;
                // just alias result to the same id as the operand
                let op_id = self.str_to_id(operand);
                self.mlir_to_expr_id_map.insert(result.clone(), op_id);
                // Return a dummy variable node with target type; the id mapping is what matters
                let vname = format!("{result}_{name_tag}");
                let varname = vname.as_str(); // tage with function name to keep unique across functions
                (
                    AstNode::new(
                        MLIRLang::new_with_name_and_type(
                            MLIROps::Variable(varname.into()),
                            varname,
                            to_ty,
                        ),
                        vec![],
                    ),
                    vec![],
                )
            }

            // --- CmpI: integer comparison producing an i1 result ---
            MLIRStatement::CmpIOp {
                result,
                predicate,
                lhs,
                rhs,
                ty,
            } => {
                let left_id = self.str_to_id(lhs);
                let right_id = self.str_to_id(rhs);
                let cmp_op = match predicate.as_str() {
                    "eq" => MLIROps::CmpEq,
                    "ne" => MLIROps::CmpNe,
                    "slt" => MLIROps::CmpSlt,
                    "sle" => MLIROps::CmpSle,
                    "sgt" => MLIROps::CmpSgt,
                    "sge" => MLIROps::CmpSge,
                    "ult" => MLIROps::CmpUlt,
                    "ule" => MLIROps::CmpUle,
                    "ugt" => MLIROps::CmpUgt,
                    "uge" => MLIROps::CmpUge,
                    other => panic!("{}", "Unsupported cmpi predicate: {other}"),
                };
                (
                    AstNode::new(
                        MLIRLang::new_with_name_and_type(cmp_op, result, ty),
                        vec![left_id, right_id],
                    ),
                    vec![left_id, right_id],
                )
            }

            // --- Select: conditional value selection ---
            MLIRStatement::SelectOp {
                result,
                condition,
                true_val,
                false_val,
                ty,
            } => {
                let cond_id = self.str_to_id(condition);
                let true_id = self.str_to_id(true_val);
                let false_id = self.str_to_id(false_val);
                (
                    AstNode::new(
                        MLIRLang::new_with_name_and_type(MLIROps::Select, result, ty),
                        vec![cond_id, true_id, false_id],
                    ),
                    vec![cond_id, true_id, false_id],
                )
            }

            // --- MemrefAlloca: variable with its memref type ---
            MLIRStatement::MemrefAlloca { result, ty }
            | MLIRStatement::MemrefGetGlobal {
                result,
                sym_name: _,
                ty,
            } => {
                let vname = format!("{result}_{name_tag}");
                let varname = vname.as_str(); // tage with function name to keep unique across functions
                (
                    AstNode::new(
                        MLIRLang::new_with_name_and_type(
                            MLIROps::Variable(varname.into()),
                            varname,
                            ty,
                        ),
                        vec![],
                    ),
                    vec![],
                )
            }

            // --- MemrefGetGlobal: variable with its memref type ---
            // ScfYield should have been consumed by for/if block parsing;
            // if we see it here, something is wrong.
            MLIRStatement::ScfYield { .. } => {
                panic!("Unexpected top-level scf.yield — should have been consumed during block parsing")
            }
        }
    }

    /// Returns the root expr and its ID
    pub fn mlir_to_expr(&mut self, mlir_text: &str) -> (HLSAST, Id, usize) {
        let module = parse_mlir(mlir_text).expect("Error Parsing MLIR!");
        let root_id = self.mlir_module_to_expr(&module);
        let lib_offset = count_lib_id_extent(&self.mlir_recexpr);
        (self.mlir_recexpr.clone(), root_id, lib_offset)
    }
}

/// Converts a `RecExpr` AST back to MLIR text format
#[derive(Debug, Default)]
pub struct RecExprToMLIR {
    /// Maps `RecExpr` Id to the MLIR variable name assigned to it.
    /// Also doubles as the "visited" set — if a node index is present,
    /// it has already been processed and its expressions emitted.
    id_to_var_name: HashMap<usize, String>,
    /// Counter for generating fresh variable names when none exists
    var_counter: usize,
    /// Maps `LibId` → function name, populated when processing Lib nodes
    lib_to_fn_name: HashMap<LibId, String>,
    /// Tracks whether each function (by `LibId`) is void (returns nothing)
    lib_is_void: HashMap<LibId, bool>,
}

impl RecExprToMLIR {
    #[must_use]
    pub fn new() -> Self {
        Self {
            id_to_var_name: HashMap::new(),
            var_counter: 0,
            lib_to_fn_name: HashMap::new(),
            lib_is_void: HashMap::new(),
        }
    }

    /// Convert a `RecExpr` to MLIR text. Returns the generated MLIR code as a string.
    pub fn convert(&mut self, rec_expr: &HLSAST) -> String {
        let root_idx = rec_expr.as_ref().len() - 1;
        let (ordered_exprs, _) = self.convert_node(rec_expr, root_idx);
        ordered_exprs.to_vec().join("\n")
    }

    /// Recursively convert a node and all its dependencies into an ordered set
    /// of MLIR expressions.
    ///
    /// Returns `(ordered_mlir_lines, var_name)` where:
    /// - `ordered_mlir_lines` uses `IndexSet` for definitions (deduped) and a Vec
    ///   for structural lines (return, "}") so those can repeat per function.
    /// - `var_name` is the MLIR variable name assigned to this node.
    ///
    /// Nodes that have already been visited return an empty set and their
    /// previously-assigned variable name (deduplication).
    #[allow(clippy::too_many_lines)]
    fn convert_node(&mut self, rec_expr: &HLSAST, node_idx: usize) -> (OrderedMlirLines, String) {
        // Already visited — no new expressions, just return the var name
        if let Some(var_name) = self.id_to_var_name.get(&node_idx) {
            return (OrderedMlirLines::new(), var_name.clone());
        }

        let node = &rec_expr.as_ref()[node_idx];
        let op = node.operation().get_op();

        // Assign a variable name for this node
        let var_name = self.get_or_create_var_name(node, node_idx);
        self.id_to_var_name.insert(node_idx, var_name.clone());

        match op {
            // Leaf nodes — no MLIR line, no children to recurse into
            MLIROps::Constant(val) => {
                let mut lines = OrderedMlirLines::new();
                let ty = Self::get_type_str(node);
                let const_assign_expr = format!("{var_name} = arith.constant {val} : {ty}");
                lines.push_def(const_assign_expr);
                (lines, var_name)
            }

            // These should return an empty set; they are defined elsewhere (loop scope, function args, etc.)
            MLIROps::Variable(_) | MLIROps::DBVar(_) | MLIROps::LibVar(_) => {
                (OrderedMlirLines::new(), var_name)
            }

            // Ops that produce an MLIR definition (or FReturn — structural)
            MLIROps::Add
            | MLIROps::Sub
            | MLIROps::Mult
            | MLIROps::Div
            | MLIROps::And
            | MLIROps::Or
            | MLIROps::LeftShift
            | MLIROps::Neg
            | MLIROps::Not
            | MLIROps::CmpEq
            | MLIROps::CmpNe
            | MLIROps::CmpSlt
            | MLIROps::CmpSle
            | MLIROps::CmpSgt
            | MLIROps::CmpSge
            | MLIROps::CmpUlt
            | MLIROps::CmpUle
            | MLIROps::CmpUgt
            | MLIROps::CmpUge
            | MLIROps::Select
            | MLIROps::Load
            | MLIROps::Load2D
            | MLIROps::Load3D
            | MLIROps::Load4D
            | MLIROps::Store
            | MLIROps::Store2D
            | MLIROps::Store3D
            | MLIROps::Store4D
            | MLIROps::Transpose
            | MLIROps::FReturn => {
                let mut lines = OrderedMlirLines::new();
                for &child_id in node.args() {
                    let (child_exprs, _) = self.convert_node(rec_expr, usize::from(child_id));
                    lines.extend(&child_exprs);
                }
                if let Some(mlir_line) = self.node_to_mlir(node, node_idx) {
                    if op == MLIROps::FReturn {
                        lines.push_structural(mlir_line);
                    } else {
                        lines.push_def(mlir_line);
                    }
                }
                (lines, var_name)
            }

            MLIROps::Block => {
                let mut lines = OrderedMlirLines::new();
                for &child_id in node.args() {
                    let (child_exprs, _) = self.convert_node(rec_expr, usize::from(child_id));
                    lines.extend(&child_exprs);
                }
                (lines, var_name)
            }

            MLIROps::ForLoop => {
                let mut lines = OrderedMlirLines::new();
                let mut loop_names = vec![];

                let (body, args_no_body) = node.args().split_last().unwrap();

                for &child_id in args_no_body {
                    let (child_exprs, cvar_name) =
                        self.convert_node(rec_expr, usize::from(child_id));
                    lines.extend(&child_exprs);
                    loop_names.push(cvar_name);
                }

                let iter_var = &loop_names[1];
                let lower_bound = &loop_names[2];
                let upper_bound = &loop_names[3];
                let step = &loop_names[4];

                let for_expr =
                    format!("scf.for {iter_var} = {lower_bound} to {upper_bound} step {step} {{");
                lines.push_structural(for_expr);

                let (mut body_exprs, _) = self.convert_node(rec_expr, usize::from(*body));
                if let Some(last) = body_exprs.pop_last() {
                    body_exprs.push_def(format!("{last} }}"));
                }
                lines.extend(&body_exprs);

                (lines, var_name)
            }

            MLIROps::YieldForLoop => {
                let mut lines = OrderedMlirLines::new();
                let mut loop_names = vec![];

                let (yield_id, args_no_yield) = node.args().split_last().unwrap();
                let (body, args_no_body) = args_no_yield.split_last().unwrap();

                for &child_id in args_no_body {
                    let (child_exprs, cvar_name) =
                        self.convert_node(rec_expr, usize::from(child_id));
                    lines.extend(&child_exprs);
                    loop_names.push(cvar_name);
                }

                let iter_var = &loop_names[1];
                let lower_bound = &loop_names[2];
                let upper_bound = &loop_names[3];
                let step = &loop_names[4];
                let iter_arg = &loop_names[5];
                let init_val = &loop_names[6];

                let for_expr = format!("{var_name} = scf.for {iter_var} = {lower_bound} to {upper_bound} step {step} iter_args({iter_arg} = {init_val}) -> (i32) {{");
                lines.push_structural(for_expr);

                let (mut body_exprs, _) = self.convert_node(rec_expr, usize::from(*body));
                let (_, yield_var) = self.convert_node(rec_expr, usize::from(*yield_id));
                let yield_expr = format!("scf.yield {yield_var} : i32 }}");
                body_exprs.push_structural(yield_expr);

                lines.extend(&body_exprs);

                (lines, var_name)
            }

            // IfThen: args = [condition, then_body]
            MLIROps::IfThen => {
                let mut lines = OrderedMlirLines::new();

                let condition_id = node.args()[0];
                let then_body_id = node.args()[1];

                let (cond_exprs, cond_var) = self.convert_node(rec_expr, usize::from(condition_id));
                lines.extend(&cond_exprs);

                let if_expr = format!("scf.if {cond_var} {{");
                lines.push_structural(if_expr);

                let (mut then_exprs, _) = self.convert_node(rec_expr, usize::from(then_body_id));
                if let Some(last) = then_exprs.pop_last() {
                    then_exprs.push_def(format!("{last} }}"));
                }
                lines.extend(&then_exprs);

                (lines, var_name)
            }

            // IfThenElse: args = [condition, then_body, else_body]
            MLIROps::IfThenElse => {
                let mut lines = OrderedMlirLines::new();

                let condition_id = node.args()[0];
                let then_body_id = node.args()[1];
                let else_body_id = node.args()[2];

                let (cond_exprs, cond_var) = self.convert_node(rec_expr, usize::from(condition_id));
                lines.extend(&cond_exprs);

                let if_expr = format!("scf.if {cond_var} {{");
                lines.push_structural(if_expr);

                let (mut then_exprs, _) = self.convert_node(rec_expr, usize::from(then_body_id));
                if let Some(last) = then_exprs.pop_last() {
                    then_exprs.push_def(format!("{last} }} else {{"));
                }
                lines.extend(&then_exprs);

                let (mut else_exprs, _) = self.convert_node(rec_expr, usize::from(else_body_id));
                if let Some(last) = else_exprs.pop_last() {
                    else_exprs.push_def(format!("{last} }}"));
                }
                lines.extend(&else_exprs);

                (lines, var_name)
            }

            // YieldingIfThen: args = [condition, then_body, then_yield]
            MLIROps::YieldingIfThen => {
                let mut lines = OrderedMlirLines::new();

                let condition_id = node.args()[0];
                let then_body_id = node.args()[1];
                let then_yield_id = node.args()[2];

                let (cond_exprs, cond_var) = self.convert_node(rec_expr, usize::from(condition_id));
                lines.extend(&cond_exprs);

                let if_expr = format!("{var_name} = scf.if {cond_var} -> (i32) {{");
                lines.push_structural(if_expr);

                let (mut then_exprs, _) = self.convert_node(rec_expr, usize::from(then_body_id));
                let (_, then_yield_var) = self.convert_node(rec_expr, usize::from(then_yield_id));
                let yield_expr = format!("scf.yield {then_yield_var} : i32 }}");
                then_exprs.push_structural(yield_expr);

                lines.extend(&then_exprs);

                (lines, var_name)
            }

            // YieldingIfThenElse: args = [condition, then_body, then_yield, else_body, else_yield]
            MLIROps::YieldingIfThenElse => {
                let mut lines = OrderedMlirLines::new();

                let condition_id = node.args()[0];
                let then_body_id = node.args()[1];
                let then_yield_id = node.args()[2];
                let else_body_id = node.args()[3];
                let else_yield_id = node.args()[4];

                let (cond_exprs, cond_var) = self.convert_node(rec_expr, usize::from(condition_id));
                lines.extend(&cond_exprs);

                let if_expr = format!("{var_name} = scf.if {cond_var} -> (i32) {{");
                lines.push_structural(if_expr);

                let (mut then_exprs, _) = self.convert_node(rec_expr, usize::from(then_body_id));
                let (_, then_yield_var) = self.convert_node(rec_expr, usize::from(then_yield_id));
                let then_yield_expr = format!("scf.yield {then_yield_var} : i32 }} else {{");
                then_exprs.push_structural(then_yield_expr);

                lines.extend(&then_exprs);

                let (mut else_exprs, _) = self.convert_node(rec_expr, usize::from(else_body_id));
                let (_, else_yield_var) = self.convert_node(rec_expr, usize::from(else_yield_id));
                let else_yield_expr = format!("scf.yield {else_yield_var} : i32 }}");
                else_exprs.push_structural(else_yield_expr);

                lines.extend(&else_exprs);

                (lines, var_name)
            }

            // ── Function infrastructure ──────────────────────────────────

            // Lib(lib_id): args = [fn_def_id, rest_id]
            // Represents a named function binding.  Peel off the FDef chain
            // to discover arity & body, emit `func.func @name(...) { body }`,
            // then continue converting the rest of the program.
            MLIROps::Lib(lib_id) => {
                let mut lines = OrderedMlirLines::new();
                let fn_def_id = node.args()[0];
                let rest_id = node.args()[1];

                let (num_args, body_idx) = self.unwrap_fdef_chain(rec_expr, usize::from(fn_def_id));

                let fn_name = format!("fn_{}", lib_id.0);
                self.lib_to_fn_name.insert(lib_id, fn_name.clone());

                let is_void = self.is_void_function_body(rec_expr, body_idx);
                self.lib_is_void.insert(lib_id, is_void);

                {
                    let mut idx = usize::from(fn_def_id);
                    loop {
                        let n = &rec_expr.as_ref()[idx];
                        match n.operation().get_op() {
                            MLIROps::FDef(_) => {
                                self.id_to_var_name.insert(idx, var_name.clone());
                                idx = usize::from(n.args()[0]);
                            }
                            _ => break,
                        }
                    }
                }

                let arg_types = self.collect_arg_types(rec_expr, usize::from(fn_def_id), num_args);
                let args_str = arg_types
                    .iter()
                    .enumerate()
                    .map(|(i, ty)| format!("%arg{i}: {ty}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                let func_header = if is_void {
                    format!("func.func @{fn_name}({args_str}) {{")
                } else {
                    format!("func.func @{fn_name}({args_str}) -> i32 {{")
                };
                lines.push_def(func_header);

                let (body_exprs, _) = self.convert_node(rec_expr, body_idx);
                lines.extend(&body_exprs);
                lines.push_structural("}".to_string());

                let (rest_exprs, rest_var) = self.convert_node(rec_expr, usize::from(rest_id));
                lines.extend(&rest_exprs);

                (lines, rest_var)
            }

            // FDef: args = [child_id]
            // A standalone FDef at the root of the rest-chain is the
            // main / top-level function.  Unwrap the nesting to discover
            // arity, then emit `func.func @top(...) { body }`.
            MLIROps::FDef(_) => {
                let mut lines = OrderedMlirLines::new();
                let (num_args, body_idx) = self.unwrap_fdef_chain(rec_expr, node_idx);

                let is_void = self.is_void_function_body(rec_expr, body_idx);

                {
                    let mut idx = usize::from(node.args()[0]);
                    loop {
                        let n = &rec_expr.as_ref()[idx];
                        match n.operation().get_op() {
                            MLIROps::FDef(_) => {
                                self.id_to_var_name.insert(idx, var_name.clone());
                                idx = usize::from(n.args()[0]);
                            }
                            _ => break,
                        }
                    }
                }

                let arg_types = self.collect_arg_types(rec_expr, node_idx, num_args);
                let args_str = arg_types
                    .iter()
                    .enumerate()
                    .map(|(i, ty)| format!("%arg{i}: {ty}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                let func_header = if is_void {
                    format!("func.func @top({args_str}) {{")
                } else {
                    format!("func.func @top({args_str}) -> i32 {{")
                };
                lines.push_def(func_header);

                let (body_exprs, _) = self.convert_node(rec_expr, body_idx);
                lines.extend(&body_exprs);
                lines.push_structural("}".to_string());

                (lines, var_name)
            }

            // Call: args = [fn_expr_id, arg_id]
            MLIROps::Call => {
                let mut lines = OrderedMlirLines::new();

                let (base_idx, arg_indices) = self.unwrap_call_chain(rec_expr, node_idx);

                {
                    let mut idx = usize::from(node.args()[0]);
                    loop {
                        let n = &rec_expr.as_ref()[idx];
                        match n.operation().get_op() {
                            MLIROps::Call => {
                                self.id_to_var_name.insert(idx, var_name.clone());
                                idx = usize::from(n.args()[0]);
                            }
                            _ => break,
                        }
                    }
                }

                let (base_exprs, _) = self.convert_node(rec_expr, base_idx);
                lines.extend(&base_exprs);

                let mut arg_vars = vec![];
                for &arg_idx in &arg_indices {
                    let (arg_exprs, arg_var) = self.convert_node(rec_expr, arg_idx);
                    lines.extend(&arg_exprs);
                    arg_vars.push(arg_var);
                }

                // Resolve function name and void-ness from the base node
                let base_node = &rec_expr.as_ref()[base_idx];
                let (fn_name, is_void) = match base_node.operation().get_op() {
                    MLIROps::LibVar(lib_id) => {
                        let name = self
                            .lib_to_fn_name
                            .get(&lib_id)
                            .cloned()
                            .unwrap_or_else(|| format!("fn_{}", lib_id.0));
                        let void = self.lib_is_void.get(&lib_id).copied().unwrap_or(false);
                        (name, void)
                    }
                    _ => ("unknown_fn".to_string(), false),
                };

                // Build per-argument type list from the RecExpr nodes
                let arg_type_strs: Vec<&str> = arg_indices
                    .iter()
                    .map(|&ai| {
                        let n = &rec_expr.as_ref()[ai];
                        n.operation().get_ty().unwrap_or("i32")
                    })
                    .collect();
                let args_str = arg_vars.join(", ");
                let arg_types = arg_type_strs.join(", ");

                // Use the full call type signature stored on the Call node if available
                let call_expr = if is_void {
                    format!("call @{fn_name}({args_str}) : ({arg_types}) -> ()")
                } else {
                    format!("{var_name} = call @{fn_name}({args_str}) : ({arg_types}) -> i32")
                };
                lines.push_def(call_expr);

                (lines, var_name)
            }

            // ProgList: variable-arity list of sub-programs, similar to Block
            MLIROps::ProgList => {
                let mut lines = OrderedMlirLines::new();
                for &child_id in node.args() {
                    let (child_exprs, _) = self.convert_node(rec_expr, usize::from(child_id));
                    lines.extend(&child_exprs);
                }
                (lines, var_name)
            }
        }
    }

    /// Get or create a variable name for a node
    fn get_or_create_var_name(&mut self, node: &HLSIR, _idx: usize) -> String {
        let mlir_lang = node.operation();

        // Use existing var_name if available
        if let Some(name) = mlir_lang.get_name() {
            // Ensure it has the % suffix for MLIR
            if name.starts_with('%') {
                return name;
            }
            return format!("%{name}");
        }

        // Need to add handling here so that if array is used as the var name to index at 0

        // For constants and variables, use their inherent representation
        match mlir_lang.get_op() {
            MLIROps::Constant(val) => format!("%c{val}"),
            MLIROps::Variable(sym) => {
                let s = sym.to_string();
                if s.starts_with('%') {
                    s
                } else {
                    format!("%{s}")
                }
            }
            MLIROps::DBVar(arg) => format!("%arg{}_{}", arg.arg_pos, arg.lib_name),
            MLIROps::LibVar(lib_id) => format!("%fn_{lib_id}"),
            _ => {
                // Generate a fresh variable name
                let name = format!("%{}", self.var_counter);
                self.var_counter += 1;
                name
            }
        }
    }

    /// Collect the MLIR types of function arguments by inspecting `DBVar` nodes
    /// reachable from the body Block.  Falls back to "i32" when no type is stored.
    fn collect_arg_types(
        &self,
        rec_expr: &HLSAST,
        start_idx: usize,
        num_args: usize,
    ) -> Vec<String> {
        let mut types = vec!["i32".to_string(); num_args];
        // Walk the FDef chain to find DBVar nodes referenced in the body
        self.scan_for_dbvar_types(
            rec_expr,
            start_idx,
            &mut types,
            &mut std::collections::HashSet::new(),
        );
        types
    }

    fn scan_for_dbvar_types(
        &self,
        rec_expr: &HLSAST,
        idx: usize,
        types: &mut [String],
        visited: &mut std::collections::HashSet<usize>,
    ) {
        if !visited.insert(idx) {
            return;
        }
        let node = &rec_expr.as_ref()[idx];
        if let MLIROps::DBVar(db_idx) = node.operation().get_op() {
            let i = db_idx.arg_pos;
            if let Some(ty) = node.operation().get_ty() {
                if i < types.len() {
                    types[i] = ty.to_string();
                }
            }
        }
        for &child_id in node.args() {
            self.scan_for_dbvar_types(rec_expr, usize::from(child_id), types, visited);
        }
    }

    /// Check whether the function body at `body_idx` represents a void function.
    /// A function is void if its body (a Block node) ends with `FReturn` that has no arguments.
    fn is_void_function_body(&self, rec_expr: &HLSAST, body_idx: usize) -> bool {
        let body_node = &rec_expr.as_ref()[body_idx];
        if !matches!(body_node.operation().get_op(), MLIROps::Block) {
            return true;
        }
        if let Some(&last_child_id) = body_node.args().last() {
            let last_child = &rec_expr.as_ref()[usize::from(last_child_id)];
            matches!(last_child.operation().get_op(), MLIROps::FReturn)
                && last_child.args().is_empty()
        } else {
            true
        }
    }

    /// Unwrap a chain of `FDef` nodes starting at `start_idx`.
    /// Returns `(num_args, body_idx)` where `num_args` is the nesting depth
    /// (one per function argument) and `body_idx` is the innermost child
    /// (the actual function body, typically a Block node).
    fn unwrap_fdef_chain(&self, rec_expr: &HLSAST, start_idx: usize) -> (usize, usize) {
        let mut num_args = 0;
        let mut current_idx = start_idx;
        loop {
            let node = &rec_expr.as_ref()[current_idx];
            match node.operation().get_op() {
                MLIROps::FDef(_) => {
                    num_args += 1;
                    current_idx = usize::from(node.args()[0]);
                }
                _ => break,
            }
        }
        (num_args, current_idx)
    }

    /// Unwrap a chain of curried Call nodes starting at `start_idx`.
    /// Returns `(base_idx, arg_indices)` where `base_idx` is the index of the
    /// base function node (typically a `LibVar`) and `arg_indices` are the
    /// argument node indices in application order (left to right).
    fn unwrap_call_chain(&self, rec_expr: &HLSAST, start_idx: usize) -> (usize, Vec<usize>) {
        let mut arg_indices = vec![];
        let mut current_idx = start_idx;
        loop {
            let node = &rec_expr.as_ref()[current_idx];
            match node.operation().get_op() {
                MLIROps::Call => {
                    let fn_id = node.args()[0];
                    let arg_id = node.args()[1];
                    arg_indices.push(usize::from(arg_id));
                    current_idx = usize::from(fn_id);
                }
                _ => break,
            }
        }
        arg_indices.reverse(); // collected outermost-first, need innermost-first
        (current_idx, arg_indices)
    }

    /// Get the variable name for a child Id
    fn get_var_for_id(&self, id: Id) -> String {
        let idx = usize::from(id);
        self.id_to_var_name
            .get(&idx)
            .cloned()
            .unwrap_or_else(|| format!("%unknown_{idx}"))
    }

    /// Get the MLIR type string for a node, falling back to `"i32"` when unknown.
    fn get_type_str(node: &HLSIR) -> &str {
        node.operation().get_ty().unwrap_or("i32")
    }

    /// Generate MLIR for a single node. Returns None for leaf nodes that don't need assignments.
    ///
    #[allow(clippy::too_many_lines)]
    fn node_to_mlir(&self, node: &HLSIR, idx: usize) -> Option<String> {
        let mlir_lang = node.operation();
        let op = mlir_lang.get_op();
        let args = node.args();
        let result_var = self.id_to_var_name.get(&idx)?;

        match op {
            // Leaf nodes - no assignment needed, they are just referenced
            MLIROps::Constant(_)
            | MLIROps::Variable(_)
            | MLIROps::DBVar(_)
            | MLIROps::LibVar(_) => None,

            // Arithmetic operations
            MLIROps::Add => {
                let (lhs, rhs) = (self.get_var_for_id(args[0]), self.get_var_for_id(args[1]));
                let ty = Self::get_type_str(node);
                Some(format!("{result_var} = arith.addi {lhs}, {rhs} : {ty}"))
            }
            MLIROps::Sub => {
                let (lhs, rhs) = (self.get_var_for_id(args[0]), self.get_var_for_id(args[1]));
                let ty = Self::get_type_str(node);
                Some(format!("{result_var} = arith.subi {lhs}, {rhs} : {ty}"))
            }
            MLIROps::Mult => {
                let (lhs, rhs) = (self.get_var_for_id(args[0]), self.get_var_for_id(args[1]));
                let ty = Self::get_type_str(node);
                Some(format!("{result_var} = arith.muli {lhs}, {rhs} : {ty}"))
            }
            MLIROps::And => {
                let (lhs, rhs) = (self.get_var_for_id(args[0]), self.get_var_for_id(args[1]));
                let ty = Self::get_type_str(node);
                Some(format!("{result_var} = arith.andi {lhs}, {rhs} : {ty}"))
            }
            MLIROps::Or => {
                let (lhs, rhs) = (self.get_var_for_id(args[0]), self.get_var_for_id(args[1]));
                let ty = Self::get_type_str(node);
                Some(format!("{result_var} = arith.ori {lhs}, {rhs} : {ty}"))
            }
            MLIROps::LeftShift => {
                let (lhs, rhs) = (self.get_var_for_id(args[0]), self.get_var_for_id(args[1]));
                let ty = Self::get_type_str(node);
                Some(format!("{result_var} = arith.shli {lhs}, {rhs} : {ty}"))
            }
            MLIROps::Neg => {
                let operand = self.get_var_for_id(args[0]);
                let ty = Self::get_type_str(node);
                Some(format!("{result_var} = arith.subi %c0, {operand} : {ty}"))
            }
            MLIROps::Not => {
                let operand = self.get_var_for_id(args[0]);
                let ty = Self::get_type_str(node);
                Some(format!(
                    "{result_var} = arith.xori {operand}, %c_neg1 : {ty}"
                ))
            }

            // Load operations — type is the memref type
            MLIROps::Load => {
                let memref = self.get_var_for_id(args[0]);
                let idx0 = self.get_var_for_id(args[1]);
                let ty = Self::get_type_str(node);
                Some(format!(
                    "{result_var} = memref.load {memref}[{idx0}] : {ty}"
                ))
            }
            MLIROps::Load2D => {
                let memref = self.get_var_for_id(args[0]);
                let (idx0, idx1) = (self.get_var_for_id(args[1]), self.get_var_for_id(args[2]));
                let ty = Self::get_type_str(node);
                Some(format!(
                    "{result_var} = memref.load {memref}[{idx0}, {idx1}] : {ty}"
                ))
            }
            MLIROps::Load3D => {
                let memref = self.get_var_for_id(args[0]);
                let (idx0, idx1, idx2) = (
                    self.get_var_for_id(args[1]),
                    self.get_var_for_id(args[2]),
                    self.get_var_for_id(args[3]),
                );
                let ty = Self::get_type_str(node);
                Some(format!(
                    "{result_var} = memref.load {memref}[{idx0}, {idx1}, {idx2}] : {ty}"
                ))
            }
            MLIROps::Load4D => {
                let memref = self.get_var_for_id(args[0]);
                let (idx0, idx1, idx2, idx3) = (
                    self.get_var_for_id(args[1]),
                    self.get_var_for_id(args[2]),
                    self.get_var_for_id(args[3]),
                    self.get_var_for_id(args[4]),
                );
                let ty = Self::get_type_str(node);
                Some(format!(
                    "{result_var} = memref.load {memref}[{idx0}, {idx1}, {idx2}, {idx3}] : {ty}"
                ))
            }

            // Store operations — type is the memref type
            MLIROps::Store => {
                let memref = self.get_var_for_id(args[0]);
                let idx0 = self.get_var_for_id(args[1]);
                let value = self.get_var_for_id(args[2]);
                let ty = Self::get_type_str(node);
                Some(format!("memref.store {value}, {memref}[{idx0}] : {ty}"))
            }
            MLIROps::Store2D => {
                let memref = self.get_var_for_id(args[0]);
                let (idx0, idx1) = (self.get_var_for_id(args[1]), self.get_var_for_id(args[2]));
                let value = self.get_var_for_id(args[3]);
                let ty = Self::get_type_str(node);
                Some(format!(
                    "memref.store {value}, {memref}[{idx0}, {idx1}] : {ty}"
                ))
            }
            MLIROps::Store3D => {
                let memref = self.get_var_for_id(args[0]);
                let (idx0, idx1, idx2) = (
                    self.get_var_for_id(args[1]),
                    self.get_var_for_id(args[2]),
                    self.get_var_for_id(args[3]),
                );
                let value = self.get_var_for_id(args[4]);
                let ty = Self::get_type_str(node);
                Some(format!(
                    "memref.store {value}, {memref}[{idx0}, {idx1}, {idx2}] : {ty}"
                ))
            }
            MLIROps::Store4D => {
                let memref = self.get_var_for_id(args[0]);
                let (idx0, idx1, idx2, idx3) = (
                    self.get_var_for_id(args[1]),
                    self.get_var_for_id(args[2]),
                    self.get_var_for_id(args[3]),
                    self.get_var_for_id(args[4]),
                );
                let value = self.get_var_for_id(args[5]);
                let ty = Self::get_type_str(node);
                Some(format!(
                    "memref.store {value}, {memref}[{idx0}, {idx1}, {idx2}, {idx3}] : {ty}"
                ))
            }

            MLIROps::Transpose => {
                let array = self.get_var_for_id(args[0]);
                let ty = Self::get_type_str(node);
                Some(format!("{result_var} = linalg.transpose {array} : {ty}"))
            }

            // Return statement
            MLIROps::FReturn => {
                if args.is_empty() {
                    Some("return".to_string())
                } else {
                    let return_vals: Vec<String> =
                        args.iter().map(|a| self.get_var_for_id(*a)).collect();
                    Some(format!("return {}", return_vals.join(", ")))
                }
            }

            MLIROps::Block => None,

            MLIROps::ForLoop => {
                let _loop_name = self.get_var_for_id(args[0]);
                let iter_var = self.get_var_for_id(args[1]);
                let lower_bound = self.get_var_for_id(args[2]);
                let upper_bound = self.get_var_for_id(args[3]);
                let _step = self.get_var_for_id(args[4]);
                Some(format!(
                    "affine.for {iter_var} = {lower_bound} to {upper_bound})"
                ))
            }

            MLIROps::FDef(_)
            | MLIROps::Call
            | MLIROps::Lib(_)
            | MLIROps::ProgList
            | MLIROps::YieldForLoop
            | MLIROps::IfThen
            | MLIROps::IfThenElse
            | MLIROps::YieldingIfThen
            | MLIROps::YieldingIfThenElse => None,

            MLIROps::Div => {
                let (lhs, rhs) = (self.get_var_for_id(args[0]), self.get_var_for_id(args[1]));
                let ty = Self::get_type_str(node);
                Some(format!("{result_var} = arith.divsi {lhs}, {rhs} : {ty}"))
            }

            // Integer comparisons — type annotation is the operand type
            MLIROps::CmpEq => {
                let (lhs, rhs) = (self.get_var_for_id(args[0]), self.get_var_for_id(args[1]));
                let ty = Self::get_type_str(node);
                Some(format!("{result_var} = arith.cmpi eq, {lhs}, {rhs} : {ty}"))
            }
            MLIROps::CmpNe => {
                let (lhs, rhs) = (self.get_var_for_id(args[0]), self.get_var_for_id(args[1]));
                let ty = Self::get_type_str(node);
                Some(format!("{result_var} = arith.cmpi ne, {lhs}, {rhs} : {ty}"))
            }
            MLIROps::CmpSlt => {
                let (lhs, rhs) = (self.get_var_for_id(args[0]), self.get_var_for_id(args[1]));
                let ty = Self::get_type_str(node);
                Some(format!(
                    "{result_var} = arith.cmpi slt, {lhs}, {rhs} : {ty}"
                ))
            }
            MLIROps::CmpSle => {
                let (lhs, rhs) = (self.get_var_for_id(args[0]), self.get_var_for_id(args[1]));
                let ty = Self::get_type_str(node);
                Some(format!(
                    "{result_var} = arith.cmpi sle, {lhs}, {rhs} : {ty}"
                ))
            }
            MLIROps::CmpSgt => {
                let (lhs, rhs) = (self.get_var_for_id(args[0]), self.get_var_for_id(args[1]));
                let ty = Self::get_type_str(node);
                Some(format!(
                    "{result_var} = arith.cmpi sgt, {lhs}, {rhs} : {ty}"
                ))
            }
            MLIROps::CmpSge => {
                let (lhs, rhs) = (self.get_var_for_id(args[0]), self.get_var_for_id(args[1]));
                let ty = Self::get_type_str(node);
                Some(format!(
                    "{result_var} = arith.cmpi sge, {lhs}, {rhs} : {ty}"
                ))
            }
            MLIROps::CmpUlt => {
                let (lhs, rhs) = (self.get_var_for_id(args[0]), self.get_var_for_id(args[1]));
                let ty = Self::get_type_str(node);
                Some(format!(
                    "{result_var} = arith.cmpi ult, {lhs}, {rhs} : {ty}"
                ))
            }
            MLIROps::CmpUle => {
                let (lhs, rhs) = (self.get_var_for_id(args[0]), self.get_var_for_id(args[1]));
                let ty = Self::get_type_str(node);
                Some(format!(
                    "{result_var} = arith.cmpi ule, {lhs}, {rhs} : {ty}"
                ))
            }
            MLIROps::CmpUgt => {
                let (lhs, rhs) = (self.get_var_for_id(args[0]), self.get_var_for_id(args[1]));
                let ty = Self::get_type_str(node);
                Some(format!(
                    "{result_var} = arith.cmpi ugt, {lhs}, {rhs} : {ty}"
                ))
            }
            MLIROps::CmpUge => {
                let (lhs, rhs) = (self.get_var_for_id(args[0]), self.get_var_for_id(args[1]));
                let ty = Self::get_type_str(node);
                Some(format!(
                    "{result_var} = arith.cmpi uge, {lhs}, {rhs} : {ty}"
                ))
            }

            MLIROps::Select => {
                let cond = self.get_var_for_id(args[0]);
                let true_val = self.get_var_for_id(args[1]);
                let false_val = self.get_var_for_id(args[2]);
                let ty = Self::get_type_str(node);
                Some(format!(
                    "{result_var} = arith.select {cond}, {true_val}, {false_val} : {ty}"
                ))
            }
        }
    }
}

// ─── Multi-program merging ──────────────────────────────────────────────

/// Strip leading `%` from an MLIR variable name so tagging doesn't produce
/// `p0_%0` (which would be printed as `%p0_%0`).  After tagging we store
/// `p0_0` so the printer's `%` suffix yields `%p0_0`.
// fn strip_mlir_percent(s: &str) -> &str {
//     s.strip_suffix('%').unwrap_or(s)
// }

/// Tags all named entities in an `MLIRLang` node with a program-specific
/// suffix to prevent cross-program name collisions.  Constants are left
/// untouched so that identical literal values remain comparable.
/// MLIR names that already have a `%` suffix are stripped before tagging
/// so the result is `p0_0` not `p0_%0` (the `%` is re-added when printing).
fn tag_mlirlang(lang: &MLIRLang, suffix: &str, lib_offset: usize) -> MLIRLang {
    let mk_name = |s: &str| format!("{s}_{suffix}");
    let default_rename = lang.get_name().map(|s| mk_name(s.as_str()));

    let (tagged_op, tagged_name) = match lang.get_op() {
        MLIROps::Constant(v) => (MLIROps::Constant(v), lang.get_name()),
        MLIROps::Variable(sym) => {
            let tagged_name = mk_name(sym.as_str());
            // let s = strip_mlir_percent(&var_name);
            (MLIROps::Variable(tagged_name.into()), default_rename)
        }
        MLIROps::DBVar(idx) => {
            let db_lib_name = idx.lib_name;
            let tagged_lib_name = format!("{db_lib_name}_{suffix}");
            let new_dbvar = DeBruijnIndex::new(idx.arg_pos, tagged_lib_name.as_str());
            (MLIROps::DBVar(new_dbvar), Some(new_dbvar.to_string()))
        }
        MLIROps::LibVar(lib_id) => (
            MLIROps::LibVar(LibId(lib_id.0 + lib_offset)),
            default_rename,
        ),
        MLIROps::Lib(lib_id) => {
            // let inp = lang;
            // let new_lib = MLIROps::Lib(LibId(lib_id.0 + lib_offset));
            // println!("from: {inp} to: {new_lib}");
            (MLIROps::Lib(LibId(lib_id.0 + lib_offset)), default_rename)
        },
        MLIROps::FDef(lib_id) => {
            // let inp = lang;
            // let new_fdef = MLIROps::FDef(LibId(lib_id.0 + lib_offset));
            // println!("from: {inp} to: {new_fdef}");
            (MLIROps::FDef(LibId(lib_id.0 + lib_offset)), default_rename)
        },
        other => (other, default_rename),
    };

    let ty = lang.get_ty().map(String::from);

    MLIRLang::new_full(tagged_op, tagged_name, ty)
}

/// Returns 1 + the maximum `LibId` index found in the `RecExpr`, or 0 if
/// no Lib/LibVar nodes exist.
fn count_lib_id_extent(expr: &HLSAST) -> usize {
    let mut max_id: Option<usize> = None;
    for node in expr.as_ref() {
        match node.operation().get_op() {
            MLIROps::Lib(lib_id) | MLIROps::LibVar(lib_id) | MLIROps::FDef(lib_id) => {
                max_id = Some(max_id.map_or(lib_id.0, |cur| cur.max(lib_id.0)));
            }
            _ => {}
        }
    }
    max_id.map_or(0, |m| m + 1)
}

// /// Merges multiple independently-parsed `RecExpr`s into a single `RecExpr`
// /// whose root is a `ProgList` node.
// ///
// /// Each sub-expression's named entities (variables, functions, loop names)
// /// are suffixed with `p{i}_` to ensure no cross-program collisions.
// /// Constants (which carry no meaningful name) are **not** suffixed.
// /// `LibId`s are offset so that function bindings from different programs
// /// never share the same id.
// #[must_use]
// pub fn merge_into_proglist(recexprs: Vec<HLSAST>) -> HLSAST {
//     let mut combined = HLSAST::default();
//     let mut root_ids: Vec<Id> = Vec::new();
//     let mut lib_id_offset: usize = 0;

//     for (prog_idx, sub_expr) in recexprs.iter().enumerate() {
//         let suffix = format!("p{prog_idx}_");
//         let id_offset = combined.as_ref().len();

//         for node in sub_expr.as_ref() {
//             let tagged_lang = tag_mlirlang(node.operation(), &suffix, lib_id_offset);
//             let shifted_children: Vec<Id> = node
//                 .args()
//                 .iter()
//                 .map(|&id| Id::from(usize::from(id) + id_offset))
//                 .collect();
//             let new_node = AstNode::new(tagged_lang, shifted_children);
//             combined.add(new_node);
//         }

//         let root_id = Id::from(id_offset + sub_expr.as_ref().len() - 1);
//         root_ids.push(root_id);
//         lib_id_offset += count_lib_id_extent(sub_expr);
//     }

//     let proglist_node = AstNode::new(MLIRLang::new(MLIROps::ProgList), root_ids);
//     combined.add(proglist_node);
//     combined
// }

/// Tags all names in a single `RecExpr` with the given suffix and lib
/// offset.  Returns a new `RecExpr` with the same structure but tagged
/// names.  Child `Id`s are unchanged since the structure is identical.
#[must_use]
pub fn tag_recexpr(expr: &HLSAST, suffix: &str, lib_offset: usize) -> HLSAST {
    let tagged_nodes: Vec<HLSIR> = expr
        .as_ref()
        .iter()
        .map(|node| {
            let tagged_lang = tag_mlirlang(node.operation(), suffix, lib_offset);
            AstNode::new(tagged_lang, node.args().to_vec())
        })
        .collect();
    tagged_nodes.into()
}

/// Parses multiple MLIR files into a `Vec` of independently tagged
/// `RecExpr`s — one per file.  Each program's names are suffixed with
/// `p{i}_` and `LibId`s are offset so the expressions can later be
/// safely combined (e.g. via `combine_exprs`).
#[must_use]
pub fn mlir_files_to_recexprs(mlir_paths: &[PathBuf]) -> (Vec<HLSAST>, usize) {
    let mut lib_id_offset: usize = 0;

    (
        mlir_paths
            .iter()
            .enumerate()
            .map(|(i, path)| {
                let mlir_text = std::fs::read_to_string(path)
                    .unwrap_or_else(|e| panic!("Failed to read MLIR file {}: {e}", path.display()));
                let mut parser = MLIRRecExprParser::default();
                let (expr, _root, lib_count) = parser.mlir_to_expr(&mlir_text);
                println!(
                    "[mlir] Parsed program {} ({} nodes) from {}",
                    i,
                    expr.as_ref().len(),
                    path.display()
                );

                let suffix = format!("p{i}");
                let tagged = tag_recexpr(&expr, &suffix, lib_id_offset);
                lib_id_offset += lib_count;
                tagged
            })
            .collect(),
        lib_id_offset,
    )
}
