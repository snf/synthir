use num::bigint::BigUint;
use num::traits::{ToPrimitive};
use std::cell::RefCell;
use std::cmp;
use std::collections::HashMap;
use std::string::ToString;

use expr::{Expr, ExprType};
use op::{OpArith, OpLogic, OpUnary, OpBool, OpCast};
use z3::{Z3, Z3Ast};

/// Owns a Z3 context and has a cache for storing Registers
pub struct Z3Store<'a> {
    z3: Z3,
    store: RefCell<HashMap<Expr, Z3Ast<'a>>>,
}
impl<'a> Z3Store<'a> {

    /// Create new Z3 Context plus empty cache
    pub fn new() -> Z3Store<'a> {
        Z3Store {
            z3: Z3::new(),
            store: RefCell::new(HashMap::new())
        }
    }

    /// Borrow the Z3 context
    pub fn z3(&'a self) -> &'a Z3 {
        &self.z3
    }

    /// Get an expression from the cache or create a new one and save
    /// it before returning it
    pub fn get_expr(&'a self, e: &Expr) -> Z3Ast<'a> {
        let mut store = self.store.borrow_mut();
        if store.contains_key(e) {
            store.get(e).unwrap().clone()
        } else {
            let reg = self.z3.mk_bv_str(&e.to_string(),
                                        e.get_width().unwrap());
            debugln!("reg: {:?}, width: {}",
                     e, reg.get_bv_width());
            store.insert(e.clone(), reg.clone());
            reg
        }
    }

    /// Get an Iterator on the stored expressions
    pub fn get_exprs(&'a self) -> HashMap<Expr, Z3Ast<'a>> {
        self.store.borrow().clone()
    }
}

/// Translate ITE to SMT logic
pub fn translate_ite<'a>(z3: &'a Z3Store<'a>, b: &Z3Ast, a1: &Z3Ast, a2: &Z3Ast) -> Z3Ast<'a> {
    let ctx = z3.z3();

    let (a1, a2) = if a1.get_bv_width() != a2.get_bv_width() {
        let max = cmp::max(a1.get_bv_width(), a2.get_bv_width());
        (adjust_width(z3, a1, max, false),
         adjust_width(z3, a2, max, false))
    } else {
        (a1.clone(), a2.clone())
    };

    return ctx.ite(b, &a1, &a2);
    // Convert first comparision to bool
    // XXX_ clean
    // let one  = ctx.mk_bv_const_i(1, 32);
    // let and1 = ctx.bvand(b, &one);
    // let eq1  = ctx.eq(&and1, &one);
    // ctx.ite(&eq1, a1, a2)
}

/// Translate BoolOp to SMT logic
pub fn translate_boolop<'a>(z3: &'a Z3Store<'a>, op: OpBool, a1: &Z3Ast, a2: &Z3Ast, w: u32) -> Z3Ast<'a> {
    let ctx = z3.z3();
    let (a1, a2) = match op {
        OpBool::LT | OpBool::LE | OpBool::EQ | OpBool::NEQ
            => (adjust_width(z3, a1, w, false),
                adjust_width(z3, a2, w, false)),
        OpBool::SLT | OpBool::SLE
            => (adjust_width(z3, a1, w, true),
                adjust_width(z3, a2, w, true)),
    };
    match op {
        OpBool::LT  => ctx.bvult(&a1, &a2),
        OpBool::LE  => ctx.bvule(&a1, &a2),
        OpBool::SLT => ctx.bvslt(&a1, &a2),
        OpBool::SLE => ctx.bvsle(&a1, &a2),
        OpBool::EQ  => ctx.eq(&a1, &a2),
        OpBool::NEQ => {
            let eq = ctx.eq(&a1, &a2);
            ctx.not(&eq)
        }
    }
}

/// Translate UnOp to SMT logic
pub fn translate_unop<'a>(z3: &'a Z3Store<'a>, op: OpUnary, a: &Z3Ast, width: u32) -> Z3Ast<'a> {
    let ctx = z3.z3();
    let a = adjust_width(&z3, &a, width, false);
    match op {
        OpUnary::Neg => ctx.bvneg(&a),
        OpUnary::Not => ctx.bvnot(&a)
    }
}

/// Translate LogicOp to SMT logic
pub fn translate_logicop<'a>(z3: &'a Z3Store<'a>, op: OpLogic, a1: &Z3Ast, a2: &Z3Ast, width: u32) -> Z3Ast<'a> {
    let ctx = z3.z3();
    let a1 = adjust_width(&z3, &a1, width, false);
    let a2 = adjust_width(&z3, &a2, width, false);
    match op {
        OpLogic::And => ctx.bvand(&a1, &a2),
        OpLogic::Xor => ctx.bvxor(&a1, &a2),
        OpLogic::Or  => ctx.bvor(&a1, &a2),
        OpLogic::LLShift => ctx.bvshl(&a1, &a2),
        OpLogic::LRShift => ctx.bvlshr(&a1, &a2),
    }
}

/// Adjust width
pub fn adjust_width<'a>(z3: &'a Z3Store<'a>, a: &Z3Ast,
                        width: u32, sign_ext: bool) -> Z3Ast<'a>
{
    if width > a.get_bv_width() {
        if sign_ext {
            z3.z3().sign_ext(width - a.get_bv_width(), a)
        } else {
            z3.z3().zero_ext(width - a.get_bv_width(), a)
        }
    } else if width < a.get_bv_width() {
        z3.z3().extract(width - 1, 0, a)
    } else {
        z3.z3().clone_ast(a)
    }
}

/// Translate ArithOp to SMT logic
pub fn translate_arithop<'a>(z3: &'a Z3Store<'a>, op: OpArith, a1: &Z3Ast, a2: &Z3Ast, ty: ExprType) -> Z3Ast<'a> {
    let ctx = z3.z3();
    // First extend BitVectors to match result
    // debugln!("adjusting width");
    let (a1p, a2p) = match ty {
        ExprType::Int(i) => {
            match op {
                // If using signed operations sign extend
                OpArith::SDiv | OpArith::SMod | OpArith::ARShift =>
                    (adjust_width(z3, a1, i, true),
                     adjust_width(z3, a2, i, true))
                    ,
                _ =>
                    (adjust_width(z3, a1, i, false),
                     adjust_width(z3, a2, i, false))
            }
        },
        _ => panic!("not supported")
    };
    match op {
        OpArith::Add => ctx.bvadd(&a1p, &a2p),
        OpArith::Sub => ctx.bvsub(&a1p, &a2p),
        OpArith::Mul => ctx.bvmul(&a1p, &a2p),
        OpArith::Div => ctx.bvudiv(&a1p, &a2p),
        OpArith::SDiv => ctx.bvsdiv(&a1p, &a2p),
        // URem = UMod
        OpArith::Mod => ctx.bvurem(&a1p, &a2p),
        OpArith::SMod => ctx.bvsmod(&a1p, &a2p),
        OpArith::ARShift => ctx.bvashr(&a1p, &a2p),
        OpArith::ALShift => ctx.bvshl(&a1p, &a2p),
    }
}

/// Translate BigUint to SMT logic
pub fn translate_biguint<'a>(z3: &'a Z3Store<'a>, a: &BigUint, w: u32) -> Z3Ast<'a> {
    let ctx = z3.z3();
    let numstr = a.to_string();
    ctx.mk_bv_const_str(&numstr, w)
}
/// Translate Number to SMT logic
pub fn translate_int<'a>(z3: &'a Z3Store<'a>, a: i32, w: u32) -> Z3Ast<'a> {
    let ctx = z3.z3();
    ctx.mk_bv_const_i(a.to_i32().unwrap(), w)
}

/// Translate bit extraction to SMT logic
pub fn translate_bits<'a>(z3: &'a Z3Store<'a>, high: u32, low: u32, e: &Z3Ast)
    -> Z3Ast<'a>
{
    let ctx = z3.z3();
    let e = if high > e.get_bv_width() {
        adjust_width(z3, e, high + 1, false)
    } else {
        e.clone()
    };
    ctx.extract(high, low, &e)
}

/// Translate Cast to SMT logic
fn translate_cast<'a>(z3: &'a Z3Store<'a>, op: OpCast, e: &Z3Ast, et: ExprType)
    -> Z3Ast<'a>
{
    // XXX_ implement float
    let ctx = z3.z3();
    let i_w = et.get_width();
    match op {
        OpCast::CastLow => {
            ctx.extract(i_w - 1, 0, e)
        }
        OpCast::CastHigh => {
            let width = e.get_bv_width();
            let low = width - i_w;
            ctx.extract(width - 1, low, e)
        }
        OpCast::CastSigned => {
            if i_w < e.get_bv_width() {
                panic!("CastSigned is only for increasing the width");
            } else {
                adjust_width(z3, e, i_w, true)
            }
        }
    }
}

/// Translate to SMT logic
pub fn translate<'a>(z3: &'a Z3Store<'a>, e: &Expr, w: u32) -> Z3Ast<'a> {
    use expr::Expr::*;
    //debugln!("processing: {:?}", e);
    let res = match *e {
        Reg(_, _) => z3.get_expr(e),
        Int(ref i) => translate_biguint(z3, i, w),
        IInt(i) => translate_int(z3, i as i32, w),
        ArithOp(o, ref e1, ref e2, et) => {
            let w = et.get_width();
            translate_arithop(z3,
                              o,
                              &translate(z3, &*e1, w),
                              &translate(z3, &*e2, w),
                              et)
        },
        LogicOp(o, ref e1, ref e2, w) =>
            translate_logicop(z3,
                              o,
                              &translate(z3, &*e1, w),
                              &translate(z3, &*e2, w),
                              w),
        BoolOp(o, ref e1, ref e2, w) =>
            translate_boolop(z3,
                             o,
                             &translate(z3, &*e1, w),
                             &translate(z3, &*e2, w),
                             w),
        UnOp(o, ref e, w) =>
            translate_unop(z3,
                           o,
                           &translate(z3, &*e, w),
                           w),
        ITE(ref eb, ref e1, ref e2) =>
            translate_ite(z3,
                          &translate(z3, &*eb, 1),
                          &translate(z3, &*e1, w),
                          &translate(z3, &*e2, w)),
        Bit(b, ref e) => translate_bits(z3,
                                        b, b,
                                        &translate(z3, &*e, b + 1)),
        Bits(high, low, ref e) => translate_bits(z3,
                                              high, low,
                                              &translate(z3, &*e, high + 1)),
        Cast(o, ref e, et) => translate_cast(z3,
                                             o,
                                             // XXX_ width
                                             &translate(z3, &*e, 1024),
                                             et),
        _ => panic!(format!("not supported: {:?}", e))
    };
    //debugln!("res: {:?}", res);
    //debugln!("res_width: {:?}", res.get_bv_width());
    res
}

/// Check if e1 and e1 are equal
pub fn are_equal(e1: &Expr, e2: &Expr, width: u32) -> bool {
    let z3 = Z3Store::new();
    let ast1 = translate(&z3, e1, width);
    let ast2 = translate(&z3, e2, width);
    let ctx = z3.z3();
    let eq = ctx.eq(&ast1, &ast2);
    let model = ctx.check_and_get_model(&eq);
    model.is_valid()
}

/// Check if e1 and e1 are equal and return a counterexample
pub fn equal_or_counter(e1: &Expr, e2: &Expr, width: u32)
                        -> Option<HashMap<Expr,BigUint>>
{
    let z3 = Z3Store::new();
    let ast1 = translate(&z3, e1, width);
    let ast2 = translate(&z3, e2, width);
    let ctx = z3.z3();

    // Adjust both to the same bit width
    let ast1 = adjust_width(&z3, &ast1, width, false);
    let ast2 = adjust_width(&z3, &ast2, width, false);

    debugln!("ast1: {:?}\nast2: {:?}", ast1, ast2);

    let eq = ctx.eq(&ast1, &ast2);
    let model = ctx.check_and_get_model(&eq);
    if model.is_valid() {
        None
    } else {
        let mut map = HashMap::new();
        // Get the AST values
        debugln!("ast1_res: {}",
                 model.eval(&ast1).unwrap().get_numstring().unwrap());
        debugln!("ast2_res: {}",
                 model.eval(&ast2).unwrap().get_numstring().unwrap());
        // Get the expressions values
        let exprs = z3.get_exprs();
        for (e, e_ast) in exprs {
            let e_numstr = model.eval(&e_ast).unwrap().get_numstring().unwrap();
            let e_bigu = BigUint::parse_bytes(e_numstr.as_bytes(), 10).unwrap();
            map.insert(e, e_bigu);
        }
        Some(map)
    }
}

#[cfg(test)]
mod test {
    use num::bigint::ToBigUint;
    use expr::{Expr, ExprType};
    use expr::Expr::*;
    use op::{OpArith};
    use verify::{are_equal, equal_or_counter};
    #[test]
    fn test_expr_equal() {
        let e1 = Reg("EAX".to_owned(), 32);
        let e2 = Reg("EBX".to_owned(), 32);
        let b1 = Box::new(e1);
        let b2 = Box::new(e2);
        let a1 = Expr::ArithOp(OpArith::Add, b1.clone(), b2.clone(), ExprType::Int(32));
        let a2 = Expr::ArithOp(OpArith::Add, b2.clone(), b1.clone(), ExprType::Int(32));
        assert_eq!(are_equal(&a1, &a2, 32), true);
    }

    #[test]
    fn test_expr_sub_not_equal1() {
        let e1 = Reg("EAX".to_owned(), 32);
        let e2 = Reg("EBX".to_owned(), 32);
        let b1 = Box::new(e1);
        let b2 = Box::new(e2);
        let a1 = Expr::ArithOp(OpArith::Add, b1.clone(), b2.clone(), ExprType::Int(32));
        let a2 = Expr::ArithOp(OpArith::Sub, b1.clone(), b2.clone(), ExprType::Int(32));
        assert_eq!(are_equal(&a1, &a2, 32), false);
    }

    #[test]
    fn test_expr_sub_not_equal_2() {
        let e1 = Reg("EAX".to_owned(), 32);
        let e2 = Reg("EBX".to_owned(), 32);
        let b1 = Box::new(e1);
        let b2 = Box::new(e2);
        let a1 = Expr::ArithOp(OpArith::Sub, b1.clone(), b2.clone(), ExprType::Int(32));
        let a2 = Expr::ArithOp(OpArith::Sub, b2.clone(), b1.clone(), ExprType::Int(32));
        assert_eq!(are_equal(&a1, &a2, 32), false);
    }

    #[test]
    fn test_expr_add_const_not_equal() {
        let e1 = Reg("EAX".to_owned(), 32);
        let e2 = Reg("EBX".to_owned(), 32);
        let one = Box::new(Int(1.to_biguint().unwrap()));
        let two = Box::new(Int(2.to_biguint().unwrap()));
        let b1 = Box::new(e1);
        let b2 = Box::new(e2);
        let a1 = ArithOp(
            OpArith::Sub, b1.clone(), one.clone(), ExprType::Int(32));
        let a2 = ArithOp(
            OpArith::Sub, b2.clone(), two.clone(), ExprType::Int(32));
        assert_eq!(are_equal(&a1, &a2, 32), false);
        let diffs = equal_or_counter(&a1, &a2, 32);
        debugln!("diffs: {:?}", diffs);
    }
    #[test]
    fn test_expr_mul_div_const_equal() {
        let e1 = Reg("EAX".to_owned(), 32);
        let e2 = Reg("EBX".to_owned(), 32);
        let one = Box::new(Int(1.to_biguint().unwrap()));
        let b1 = Box::new(e1);
        let b2 = Box::new(e2);
        let a0 = Box::new(
            ArithOp(
                OpArith::Sub,
                b1.clone(),
                one.clone(),
                ExprType::Int(32)));
        // *1
        let a1 = ArithOp(OpArith::Mul,
                         a0.clone(),
                         one.clone(),
                         ExprType::Int(32));
        // /1
        let a2 = ArithOp(OpArith::Div,
                         a0.clone(),
                         one.clone(),
                         ExprType::Int(32));
        assert_eq!(are_equal(&a1, &a2, 32), true);
        assert_eq!(equal_or_counter(&a1, &a2, 32), None);
        //debugln!("diffs: {:?}", diffs);
    }

}
