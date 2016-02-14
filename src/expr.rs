use op::{OpArith, OpLogic, OpUnary, OpBool, OpCast};
use num::bigint::{BigUint, ToBigUint};

pub type Address = u64;
pub type Width = u32;
pub type Literal = BigUint;
pub type ILiteral = i16;

pub type Name = String;

#[derive(Copy,Clone,Debug,PartialEq,Eq,Hash,PartialOrd,Ord)]
pub enum FloatType {
    Single,
    Double,
    Fp80
}

#[derive(Copy,Clone,Debug,PartialEq,Eq,Hash,PartialOrd,Ord)]
pub enum ExprType {
    Float(FloatType),
    Int(u32)
}

impl ExprType {
    /// Get ExprType::Int(val) or fail trying
    pub fn get_int_width(&self) -> u32 {
        match *self {
            ExprType::Int(w) => w,
            _ => panic!("not supported")
        }
    }
}

/// Expressions allowed in our IR
#[derive(Clone,Debug,PartialEq,Eq,Hash,PartialOrd,Ord)]
pub enum Expr {
    // Basics
    Reg(Name, u32),
    // These both are bitvectors transformations
    Float(Literal, FloatType),
    Int(Literal),

    // A bitvector constant of specified width
    Const(Literal, u32),

    // The idea is that this value is expnded then to the width needed
    IInt(ILiteral),

    // Deref a piece of memory of width
    Deref(Box<Expr>, u32),

    // Ops
    //ArithOp(OpArith, Box<Expr>, Box<Expr>, ExprType),
    ArithOp(OpArith, Box<Expr>, Box<Expr>, ExprType),
    LogicOp(OpLogic, Box<Expr>, Box<Expr>),
    UnOp(OpUnary, Box<Expr>),
    BoolOp(OpBool, Box<Expr>, Box<Expr>),

    // XXX_ not sure what NoOp has to do here, what does it express?
    NoOp,
    // Get if the operation overflows/underflows (only the arithmetics ones)
    Overflow(OpArith, Box<Expr>, Box<Expr>, ExprType),
    // Control
    ITE(Box<Expr>, Box<Expr>, Box<Expr>),
    // Casting
    Cast(OpCast, ExprType, Box<Expr>),
    Bits(u32, u32, Box<Expr>),
    // Syntax sugar
    Bit(u32, Box<Expr>) // := Bits(i, i, Box<Expr>)
}

#[allow(dead_code)]
fn test() -> Expr {
    Expr::ArithOp(OpArith::Add,
          Box::new(Expr::Reg("ESP".to_owned(), 32)),
          Box::new(Expr::Int(0x10000.to_biguint().unwrap())),
          ExprType::Int(32))
}


use self::Expr::*;
impl Expr {
    pub fn is_reg(&self) -> bool {
        match *self {
            Reg(_,_) => true,
            _ => false
        }
    }
    pub fn is_deref(&self) -> bool {
        match *self {
            Deref(_,_) => true,
            _ => false
        }
    }
    pub fn is_const(&self) -> bool {
        match *self {
            Float(_, _) | Int(_) =>
                true,
            _ => false
        }
    }
    pub fn is_noop(&self) -> bool {
        match *self {
            NoOp => true,
            _ => false
        }
    }
    pub fn is_arithop(&self) -> bool {
        match *self {
            ArithOp(_, _, _, _) => true,
            _ => false,
        }
    }
    pub fn is_boolop(&self) -> bool {
        match *self {
            BoolOp(_, _, _) => true,
            _ => false,
        }
    }
    pub fn is_cast(&self) -> bool {
        match *self {
            Cast(_, _, _) => true,
            _ => false,
        }
    }
    pub fn is_ite(&self) -> bool {
        match *self {
            ITE(_, _, _) => true,
            _ => false,
        }
    }

    /// Get register name (will panic if it's not a reg)
    pub fn get_reg_name(&self) -> &str {
        match *self {
            Reg(ref name, _) => name,
            _ => panic!("Only supports Expr::Reg")
        }
    }

    /// Return if it's a terminal expression (we can't recurse it any
    /// more)
    pub fn is_last(&self) -> bool {
        match *self {
            Reg(_,_) => true,
            Float(_,_) => true,
            Int(_) => true,
            IInt(_) => true,
            Deref(_,_) => true,
            _ => false
        }
    }

    /// Get the bit width from the Expr (restricted)
    pub fn get_width(&self) -> Option<u32> {
        match *self {
            Reg(_, w) | Deref(_, w) |
            ArithOp(_, _, _, ExprType::Int(w)) => Some(w),
            LogicOp(_, ref e1, _) | UnOp(_, ref e1) |
            ITE(_, ref e1, _)
                => e1.get_width(),
            BoolOp(_, _, _) | Overflow(_, _, _, _) |
            Bit(_, _) => Some(1),
            Bits(b1, b2, _) => Some(b2 - b1 + 1),
            NoOp => None,
            Int(_) | IInt(_) => None,
            // XXX_ cast is width-defined actually
            Cast(_, _, _) => None,
            _ => unreachable!()
            //_ => panic!(format!("not supported: {:?}", self))
        }
    }

    /// Get the largest path in the tree
    pub fn get_size(&self) -> u32 {
        let mut esize = ESize::new();
        traverse_get_size(&mut esize, self);
        esize.get_largest()
    }

    /// Get all the regs in the expr
    pub fn get_regs(&self) -> Vec<&Expr> {
        let mut res = Vec::new();
        let filter_regs = |e: &Expr| e.is_reg();
        self.get_something(&mut res, &filter_regs);
        res
    }

    /// Get all the derefs in the expr
    pub fn get_derefs(&self) -> Vec<&Expr> {
        let mut res = Vec::new();
        let filter_regs = |e: &Expr| e.is_deref();
        self.get_something(&mut res, &filter_regs);
        res
    }

    /// Get the leafs of the current Expr if there's any
    pub fn get_leafs<'a>(&'a self) -> Vec<&'a Expr> {
        let mut res = Vec::new();
        match *self {
            ArithOp(_, ref e1, ref e2, _) |
            LogicOp(_, ref e1, ref e2) |
            BoolOp(_, ref e1, ref e2) => {
                res.push(&**e1);
                res.push(&**e2);
            }
            Cast(_, _, ref e) |
            Bit(_, ref e) | Bits(_, _, ref e) => {
                res.push(&**e);
            }
            _ => ()
        }
        res
    }

    /// Get the leafs (copy) of the current Expr if there's any
    pub fn get_leafs_copy(&self) -> Vec<Expr> {
        self.get_leafs()
            .into_iter()
            .cloned()
            .collect()
    }

    /// Walk the tree filtering which expression to save with a
    /// closure
    pub fn get_something<'a>(
        &'a self,
        res: &mut Vec<&'a Expr>,
        f: &Fn(&Expr) -> bool)
    {
        if f(self) {
            res.push(self);
        }
        match *self {
            ArithOp(_, ref e1, ref e2, _) |
            LogicOp(_, ref e1, ref e2) |
            BoolOp(_, ref e1, ref e2) => {
                e1.get_something(res, f);
                e2.get_something(res, f);
            }
            Cast(_, _, ref e) |
            Bit(_, ref e) | Bits(_, _, ref e) => {
                e.get_something(res, f);
            }
            _ => ()
        }
    }
}
use std::string::ToString;
impl ToString for Expr {
    fn to_string(&self) -> String {
        format!("{:?}", self)
    }
}
// +++++++ Experiment starting +++++
use op::AnyOp;
enum EBType {
    Op(AnyOp),
    Reg
}
enum EBArg {
    E(EBuild),
    ETy(ExprType),
}
struct EBuild {
    ty: EBType,
    args: Vec<EBArg>
}
impl EBArg {
    fn to_expr(&self) -> Expr {
        match *self {
            EBArg::E(ref e) => e.to_expr(),
            _ => panic!("only E can be converted to Expr")
        }
    }
    fn to_type(&self) -> ExprType {
        match *self {
            EBArg::ETy(ety) => ety,
            _ => panic!("only ETy can be converted to ExprType")
        }
    }
}

impl EBuild {
    fn to_expr(&self) -> Expr {
        use op::AnyOp::*;
        use self::EBType::*;

        match self.ty {
            Op(Arith(o)) =>
                ArithOp(o,
                        Box::new(self.args[0].to_expr()),
                        Box::new(self.args[1].to_expr()),
                        self.args[2].to_type()),
            _ => panic!("not supported yet")
        }
    }
}
// #################################

/// Points in the Expr tree
#[derive(Debug)]
pub struct EPoints {
    pos: Vec<u8>,
    op: Vec<Vec<u8>>,
    expr: Vec<Vec<u8>>,
    ty: Vec<Vec<u8>>,
    width: Vec<Vec<u8>>,
    bit: Vec<Vec<u8>>,
    bin: Vec<Vec<u8>>,
    dep: Vec<Vec<u8>>,
    cnst: Vec<Vec<u8>>,
}

impl EPoints {
    pub fn new() -> EPoints {
        EPoints {
            pos: Vec::with_capacity(50),
            op: Vec::with_capacity(50),
            expr: Vec::with_capacity(50),
            ty: Vec::with_capacity(50),
            width: Vec::with_capacity(50),
            bit: Vec::with_capacity(50),
            bin: Vec::with_capacity(50),
            dep: Vec::with_capacity(50),
            cnst: Vec::with_capacity(50)
        }
    }

    pub fn get_expr(&self) -> &[Vec<u8>] {
        &self.expr
    }
    pub fn get_op(&self) -> &[Vec<u8>] {
        &self.op
    }
    pub fn get_bin(&self) -> &[Vec<u8>] {
        &self.bin
    }
    pub fn insert_expr(&mut self) {
        self.expr.push(self.pos.to_vec())
    }
    pub fn insert_op(&mut self) {
        self.op.push(self.pos.to_vec())
    }
    pub fn insert_dep(&mut self) {
        self.dep.push(self.pos.to_vec())
    }
    pub fn insert_ty(&mut self) {
        self.ty.push(self.pos.to_vec())
    }
    pub fn insert_const(&mut self) {
        self.cnst.push(self.pos.to_vec())
    }
    pub fn insert_bit(&mut self) {
        self.bit.push(self.pos.to_vec())
    }
    pub fn insert_bin(&mut self) {
        self.bin.push(self.pos.to_vec())
    }
    pub fn push(&mut self, u: u8) { self.pos.push(u) }
    pub fn pop(&mut self) { self.pos.pop(); }
}

pub fn traverse_get_points(
    state: &mut EPoints,
    e: &Expr)
{
    state.push(0);
    state.insert_expr();
    match *e {
        Reg(_, _) | Deref(_, _) =>
            state.insert_dep(),
        Int(_) | IInt(_) =>
            state.insert_const(),
        ArithOp(_, ref e1, ref e2, _) |
        LogicOp(_, ref e1, ref e2) |
        BoolOp(_, ref e1, ref e2) => {
            if e.is_arithop() {
                state.insert_ty();
            }
            state.insert_op();
            state.insert_bin();
            state.push(1); traverse_get_points(state, e1); state.pop();
            state.push(2); traverse_get_points(state, e2); state.pop();
        },
        UnOp(_, ref e1) | Cast(_, _, ref e1) => {
            if e.is_cast() {
                state.insert_ty();
            }
            state.insert_op();
            state.push(1); traverse_get_points(state, e1); state.pop();
        },
        Bit(_, ref e1) | Bits(_, _, ref e1) => {
            state.insert_bit();
            state.push(1); traverse_get_points(state, e1); state.pop();
        },
        ITE(ref e1, ref e2, ref e3) => {
            state.push(1); traverse_get_points(state, e1); state.pop();
            state.push(2); traverse_get_points(state, e2); state.pop();
            state.push(3); traverse_get_points(state, e3); state.pop();
        },
        // XXX_ this should mean an error, no generator should leave a
        // NoOp as it's just undefined result
        NoOp => (),
        _ => panic!(format!("not supported/implement me: {:?}", e))
    }
    state.pop();
}

pub struct ESize {
    tree: Vec<Vec<u8>>,
    pos: Vec<u8>
}
impl ESize {
    pub fn new() -> ESize { ESize { tree: Vec::new(), pos: Vec::new() } }
    pub fn insert_expr(&mut self) { self.tree.push(self.pos.clone()) }
    pub fn push(&mut self, u: u8) { self.pos.push(u) }
    pub fn pop(&mut self) { self.pos.pop(); }
    pub fn get_largest(&self) -> u32 {
        self.tree.iter()
            .map(|v| v.len())
            .max().unwrap() as u32
    }
}
pub fn traverse_get_size (
    state: &mut ESize,
    e: &Expr)
{
    state.insert_expr();
    match *e {
        ArithOp(_, ref e1, ref e2, _) |
        LogicOp(_, ref e1, ref e2) |
        BoolOp(_, ref e1, ref e2) => {
            state.push(1); traverse_get_size(state, e1); state.pop();
            state.push(2); traverse_get_size(state, e2); state.pop();
        },
        UnOp(_, ref e1) | Cast(_, _, ref e1) |
        Bit(_, ref e1) | Bits(_, _, ref e1) => {
            state.push(1); traverse_get_size(state, e1); state.pop();
        },
        ITE(ref e1, ref e2, ref e3) => {
            state.push(1); traverse_get_size(state, e1); state.pop();
            state.push(2); traverse_get_size(state, e2); state.pop();
            state.push(3); traverse_get_size(state, e3); state.pop();
        },
        _ => ()
    }
}

/// Macro to help visit the operations
macro_rules! visit_ops {
    ($matched:expr, $v_left:expr, $v_right:expr) => {
        match *$matched {
            ArithOp(o, ref e1, ref e2, et) =>
                ArithOp(o, $v_left(e1), $v_right(e2), et),
            LogicOp(o, ref e1, ref e2) =>
                LogicOp(o, $v_left(e1), $v_right(e2)),
            BoolOp(o, ref e1, ref e2) =>
                BoolOp(o, $v_left(e1), $v_right(e2)),
            UnOp(o, ref e) =>
                UnOp(o, $v_right(e)),
            ITE(ref eb, ref e1, ref e2) =>
                ITE(eb.clone(), $v_left(e1), $v_right(e2)),
            _ => $matched.clone()
        }
    }
}

/// A helper for building large expressions
pub struct ExprBuilder {
    expr: Expr
}

impl ExprBuilder {
    pub fn new() -> ExprBuilder {
        ExprBuilder {
            expr: Expr::NoOp
        }
    }

    pub fn from_expr(e: &Expr) -> ExprBuilder {
        ExprBuilder {
            expr: e.clone()
        }
    }

    /// Destroy the ExprBuilder and return the Expr
    pub fn finalize(self) -> Expr {
        self.expr
    }

    /// Delete all the other expressions and insert this one
    pub fn insert_first(&mut self, e: &Expr) {
        self.expr = e.clone()
    }

    /// Insert all the tree in the first argument ot the new operation
    pub fn preinsert_with_second(&mut self, e: &Expr) {
        let old_e = Box::new(self.expr.clone());
        self.expr = visit_ops!(e,
                               |e1: &Box<Expr>| old_e,
                               |e2: &Box<Expr>| e2.clone())
    }

    /// Private function for walking the tree and inserting `ins` in
    /// the leftmost node of the tree
    fn insert_last_right_e(&self, e: &Expr, ins: &Expr) -> Expr {
        if e.is_last() {
            ins.clone()
        } else {
            visit_ops!(e,
                       |e1: &Box<Expr>| e1.clone(),
                       |e2: &Expr| Box::new(self.insert_last_right_e(e2, ins)))
        }
    }

    /// Insert in the last right node of the tree
    /// (insert_last_right_e)
    pub fn insert_last_right(&mut self, e: &Expr) {
        self.expr = self.insert_last_right_e(&self.expr, e);
    }

    /// Insert in the last right node or replace all the tree if it's
    /// NoOp (insert_last_right_e)
    pub fn insert_last_right_or_replace(&mut self, e: &Expr) {
        if self.expr.is_noop() {
            self.expr = e.clone();
        } else {
            self.expr = self.insert_last_right_e(&self.expr, e);
        }
    }

}
/*
// Typed expression
pub struct TExpr {
    ty: ExprType,
    width: u32,
    expr: Expr
}

//use rand::distributions::Sample;
use rand::distributions::RandSample;
use rand::Rng;

pub trait Sample<T> {
    fn sample<R: Rng>(rng: &mut R) -> T;
}

impl Sample<TExpr>{
    fn sample<R: Rng>(rng: &mut R) -> TExpr {
        let caca = rng.gen::<u64>();
        TExpr {
            ty: ExprType::Int(10),
            width: 32,
            expr: Deref(Box::new(Int(0x10.to_biguint().unwrap())))
        }
    }
}
*/
