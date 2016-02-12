use num::bigint::BigUint;
use num::traits::{ToPrimitive};
use std::cell::RefCell;
use std::collections::HashMap;
use std::string::ToString;

use expr::{Expr, ExprType};
use op::{OpArith, OpLogic, OpUnary, OpBool};//, OpCast};
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
    // Convert first comparision to bool
    let one  = ctx.mk_bv_const_i(1, 32);
    let and1 = ctx.bvand(b, &one);
    let eq1  = ctx.eq(&and1, &one);
    ctx.ite(&eq1, a1, a2)
}

/// Translate BoolOp to SMT logic
pub fn translate_boolop<'a>(z3: &'a Z3Store<'a>, op: OpBool, a1: &Z3Ast, a2: &Z3Ast) -> Z3Ast<'a> {
    let ctx = z3.z3();
    // XXX_ convert to same width
    //println!("sort1: {}, sort2: {}", a1.get_bv_width(), a2.get_bv_width());
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
pub fn translate_unop<'a>(z3: &'a Z3Store<'a>, op: OpUnary, a: &Z3Ast) -> Z3Ast<'a> {
    let ctx = z3.z3();
    match op {
        OpUnary::Neg => ctx.bvneg(&a),
        OpUnary::Not => ctx.bvnot(&a)
    }
}

/// Translate LogicOp to SMT logic
pub fn translate_logicop<'a>(z3: &'a Z3Store<'a>, op: OpLogic, a1: &Z3Ast, a2: &Z3Ast) -> Z3Ast<'a> {
    let ctx = z3.z3();
    match op {
        OpLogic::And => ctx.bvand(&a1, &a2),
        OpLogic::Xor => ctx.bvxor(&a1, &a2),
        OpLogic::Or  => ctx.bvor(&a1, &a2),
        OpLogic::LLShift => ctx.bvshl(&a1, &a2),
        OpLogic::LRShift => ctx.bvlshr(&a1, &a2),
    }
}

/// Translate ArithOp to SMT logic
pub fn translate_arithop<'a>(z3: &'a Z3Store<'a>, op: OpArith, a1: &Z3Ast, a2: &Z3Ast, ty: ExprType) -> Z3Ast<'a> {
    let ctx = z3.z3();
    // First extend BitVectors to match result
    let (a1p, a2p) = match ty {
        ExprType::Int(i) => {
            match op {
                // If using signed operations sign extend
                OpArith::SDiv | OpArith::SMod | OpArith::ARShift =>
                    (a1.sign_ext(i - a1.get_bv_width()),
                     a2.sign_ext(i - a2.get_bv_width()))
                    ,
                _ =>
                    (a1.zero_ext(i - a1.get_bv_width()),
                     a2.zero_ext(i - a2.get_bv_width()))
            }
        },
        _ => panic!("not supported")
    };
    let res = match op {
        OpArith::Add => ctx.bvadd(&a1p, &a2p),
        OpArith::Sub => ctx.bvsub(&a1p, &a2p),
        OpArith::Mul => ctx.bvmul(&a1p, &a2p),
        OpArith::Div => ctx.bvudiv(&a1p, &a2p),
        OpArith::SDiv => ctx.bvsdiv(&a1, &a2p),
        // URem = UMod
        OpArith::Mod => ctx.bvurem(&a1p, &a2p),
        OpArith::SMod => ctx.bvsmod(&a1p, &a2p),
        OpArith::ARShift => ctx.bvashr(&a1p, &a2p),
        OpArith::ALShift => ctx.bvshl(&a1p, &a2p),
        // XXX_ MUL with res > a1 or a2, how to do it?,
        // signed?
    };
    res
}

/// Translate BigUint to SMT logic
pub fn translate_biguint<'a>(z3: &'a Z3Store<'a>, a: &BigUint) -> Z3Ast<'a> {
    let ctx = z3.z3();
    let numstr = a.to_string();
    // XXX_ width
    ctx.mk_bv_const_str(&numstr, 32)
}
/// Translate Number to SMT logic
pub fn translate_int<'a>(z3: &'a Z3Store<'a>, a: i32) -> Z3Ast<'a> {
    let ctx = z3.z3();
    // XXX_ width
    ctx.mk_bv_const_i(a.to_i32().unwrap(), 32)
}

/// Translate bit extraction to SMT logic
pub fn translate_bits<'a>(z3: &'a Z3Store<'a>, b1: u32, b2: u32, e: &Z3Ast)
    -> Z3Ast<'a>
{
    let ctx = z3.z3();
    ctx.extract(b2, b1, e)
}

/// Translate to SMT logic
pub fn translate<'a>(z3: &'a Z3Store<'a>, e: &Expr) -> Z3Ast<'a> {
    use expr::Expr::*;
    match *e {
        Reg(_, _) => z3.get_expr(e),
        // XXX_ check
        Int(ref i) => translate_biguint(z3, i),
        // XXX_ check
        IInt(i) => translate_int(z3, i as i32),
        ArithOp(o, ref e1, ref e2, et) =>
            translate_arithop(z3,
                              o,
                              &translate(z3, &*e1),
                              &translate(z3, &*e2),
                              et),
        LogicOp(o, ref e1, ref e2) =>
            translate_logicop(z3,
                              o,
                              &translate(z3, &*e1),
                              &translate(z3, &*e2)),
        BoolOp(o, ref e1, ref e2) =>
            translate_boolop(z3,
                             o,
                              &translate(z3, &*e1),
                              &translate(z3, &*e2)),
        UnOp(o, ref e) =>
            translate_unop(z3,
                           o,
                           &translate(z3, &*e)),
        ITE(ref eb, ref e1, ref e2) =>
            translate_ite(z3,
                          &translate(z3, &*eb),
                          &translate(z3, &*e1),
                          &translate(z3, &*e2)),
        Bit(b, ref e) => translate_bits(z3,
                                        b, b+1,
                                        &translate(z3, &*e)),
        Bits(b1, b2, ref e) => translate_bits(z3,
                                              b1, b2,
                                              &translate(z3, &*e)),
        _ => panic!(format!("not supported: {:?}", e))
    }
}

/// Check if e1 and e1 are equal
pub fn are_equal(e1: &Expr, e2: &Expr) -> bool {
    let z3 = Z3Store::new();
    let ast1 = translate(&z3, e1);
    let ast2 = translate(&z3, e2);
    let ctx = z3.z3();
    let eq = ctx.eq(&ast1, &ast2);
    let model = ctx.check_and_get_model(&eq);
    //ctx.prove(&eq, true);
    // println!("model: {}", model.get_str());
    // let regs = z3.get_regs();
    // for (r, r_ast) in regs {
    //     println!("{}: {}", r, model.eval(&r_ast).unwrap().get_u64().unwrap());
    // }
    model.is_valid()
}

/// Check if e1 and e1 are equal and return a counterexample
pub fn equal_or_counter(e1: &Expr, e2: &Expr)
                        -> Option<HashMap<Expr,BigUint>>
{
    let z3 = Z3Store::new();
    let ast1 = translate(&z3, e1);
    // XXX_ remove all these prints
    println!("translation 1 done");
    let ast2 = translate(&z3, e2);
    println!("translation 2 done");
    let ctx = z3.z3();
    let eq = ctx.eq(&ast1, &ast2);
    println!("eq done");
    let model = ctx.check_and_get_model(&eq);
    println!("model done");
    if model.is_valid() {
        None
    } else {
        let mut map = HashMap::new();
        let exprs = z3.get_exprs();
        for (e, e_ast) in exprs {
            let e_numstr = model.eval(&e_ast).unwrap().get_numstring().unwrap();
            let e_bigu = BigUint::parse_bytes(e_numstr.as_bytes(), 16).unwrap();
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
        assert_eq!(are_equal(&a1, &a2), true);
    }

    #[test]
    fn test_expr_sub_not_equal1() {
        let e1 = Reg("EAX".to_owned(), 32);
        let e2 = Reg("EBX".to_owned(), 32);
        let b1 = Box::new(e1);
        let b2 = Box::new(e2);
        let a1 = Expr::ArithOp(OpArith::Add, b1.clone(), b2.clone(), ExprType::Int(32));
        let a2 = Expr::ArithOp(OpArith::Sub, b1.clone(), b2.clone(), ExprType::Int(32));
        assert_eq!(are_equal(&a1, &a2), false);
    }

    #[test]
    fn test_expr_sub_not_equal_2() {
        let e1 = Reg("EAX".to_owned(), 32);
        let e2 = Reg("EBX".to_owned(), 32);
        let b1 = Box::new(e1);
        let b2 = Box::new(e2);
        let a1 = Expr::ArithOp(OpArith::Sub, b1.clone(), b2.clone(), ExprType::Int(32));
        let a2 = Expr::ArithOp(OpArith::Sub, b2.clone(), b1.clone(), ExprType::Int(32));
        assert_eq!(are_equal(&a1, &a2), false);
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
        assert_eq!(are_equal(&a1, &a2), false);
        let diffs = equal_or_counter(&a1, &a2);
        println!("diffs: {:?}", diffs);
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
        assert_eq!(are_equal(&a1, &a2), true);
        assert_eq!(equal_or_counter(&a1, &a2), None);
        //println!("diffs: {:?}", diffs);
    }

}
