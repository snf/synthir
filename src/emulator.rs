use std::collections::{HashMap};
use std::mem;

use op::{OpArith, OpLogic, OpUnary, OpBool, OpCast};
use expr::{Expr, ExprType};
use stmt::Stmt;

use num::bigint::{ToBigUint, BigUint};
use num::traits::{ToPrimitive, Zero, One};
use num::Num;

pub struct State<'a> {
    values: &'a HashMap<Expr, BigUint>,
}


impl<'a> State<'a> {
    pub fn borrow(map: &'a HashMap<Expr, BigUint>) -> State<'a> {
        State {
            values: map
        }
    }
    pub fn get_expr_value(&self, e: &Expr) -> Value {
        let width = e.get_width().unwrap();
        Value::new(self.values.get(e).unwrap().clone(), width)
    }
}

trait BinOper<S, D> {
    fn execute(S, S) -> D;
}

enum BinOps {}

trait Execute<T> {
    fn execute() -> T;
}

/*
impl<S, D> BinOper<S, D> for BinOps
    where S: ToBigUint + Num, D: ToPrimitive {
    fn execute(e1: S, e2: S) -> D {
        //e1.to_biguint().unwrap()
        (e1.to_biguint().unwrap() + e2.to_biguint().unwrap()).
            to_u64().unwrap()
            //(e1 + e2).to_biguint().unwrap()
    }
}
*/
/*
impl<S, D> Execute<BigUint> for BinOper<S, D>
    where S: ToBigUint, D: ToBigUint
{
    fn execute() -> BigUint {
        0x10.to_biguint().unwrap()
    }
}
*/

/*
fn get_type_for_width<T>(w: i32) -> T {
    match w {
        1 => bool,
        8 => u8,
        16 => u16,
        32 => u32
    }
}
*/
use std::ops::{BitXor, BitOr, BitAnd, Shl, Shr, Mul};

pub trait PlusOne<D> {
    fn plus_one(s: Self) -> D;
}

impl PlusOne<u64> for u32 {
    fn plus_one(s: u32) -> u64 {
        s.to_u64().unwrap()
    }
}

pub trait SrcOp: Num + BitXor<Output=Self> + BitAnd<Output=Self> +
    BitOr<Output=Self> + Shl<usize, Output=Self> + Shr<usize, Output=Self> +
    ToBigUint + ToPrimitive
{
}

/*
pub trait Ops<T> where T: SrcOp {
    fn execute(op: Op, e1: T, e2: T) -> T {
        let e1 = e1.to_biguint().unwrap();
        let e2 = e2.to_biguint().unwrap();
        match op {
            Op::Add => e1 + e2,
            _ => panic!("EEERRRROR!")
        }
    }
}
*/
#[derive(PartialEq, Eq)]
pub enum Sign { Positive, Negative }

impl Mul for Sign {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        if self == rhs {
            Sign::Positive
        } else {
            Sign::Negative
        }
    }
}
// All the emulation works returning values
#[derive(Debug,Clone)]
pub struct Value {
    width: u32,
    value: BigUint
}

impl Value {
    pub fn new(value: BigUint, width: u32) -> Value {
        let size_mask = (BigUint::one() << (width as usize)) - BigUint::one();
        Value { value: value & size_mask, width: width }
    }
    pub fn from_int(value: i32, width: u32) -> Value {
        let abs_value = value.abs();
        let val = Value::new(abs_value.to_biguint().unwrap(), width);
        if value >= 0 {
            val
        } else {
            val.set_sign(Sign::Negative)
        }
    }
    pub fn value(&self) -> &BigUint {
        &self.value
    }
    fn sign(&self) -> Sign {
        if self.width.is_zero() {
            return Sign::Positive
        }
        let sign_bit_b = BigUint::one() << ((self.width as usize) - 1);
        if (&self.value & sign_bit_b).is_zero() {
            Sign::Positive
        } else {
            Sign::Negative
        }
    }
    fn unsign(&self) -> Value {
        if self.sign() == Sign::Positive {
            self.clone()
        } else {
            let sign_mask = BigUint::one() << (self.width as usize);
            Value {
                width: self.width,
                value: sign_mask - &self.value
            }
        }
    }
    fn set_sign(self, sign: Sign) -> Value {
        if sign == Sign::Negative {
            let sign_n_b = BigUint::one() << (self.width as usize);
            Value {
                width: self.width,
                value: sign_n_b - self.value
            }
        } else {
            self
        }
    }
    pub fn width(&self) -> u32 { self.width }
}

fn execute_sub(v1: &Value, v2: &Value, width: u32) -> BigUint {
    let ref b1 = v1.value;
    let ref b2 = v2.value;
    if b1 >= b2 {
        b1 - b2
    } else {
        let ub1 = v1.unsign().value;
        let ub2 = v2.unsign().value;
        if ub1 >= ub2 {
            Value::new((ub1 - ub2), width).set_sign(Sign::Negative).value
        } else {
            Value::new((ub2 - ub1), width).set_sign(Sign::Negative).value
        }
    }

}

fn execute_logicop(op: OpLogic, v1: &Value, v2: &Value) -> Value {
    let ref b1 = v1.value;
    let ref b2 = v2.value;
    let v_res = match op {
        OpLogic::LLShift => {
            let b2_usize =
                if b2 > &v1.width.to_biguint().unwrap() {
                    v1.width as usize
                } else {
                    b2.to_usize().unwrap()
                };
            b1 << b2_usize
        },
        OpLogic::LRShift => {
            let b2_usize =
                if b2 > &v1.width.to_biguint().unwrap() {
                    v1.width as usize
                } else {
                    b2.to_usize().unwrap()
                };
            b1 >> b2_usize
        },
        OpLogic::And => b1 & b2,
        OpLogic::Xor => b1 ^ b2,
        OpLogic::Or  => b1 | b2
    };
    Value::new(v_res, v1.width)
}
pub fn execute_unsigned_arithop(op: OpArith, v1: &Value, v2: &Value, width: u32)
                                -> Result<Value,()>
{
    let ref b1 = v1.value;
    let ref b2 = v2.value;
    let v_res = match op {
        OpArith::ALShift => {
            let b2_usize =
                if b2 > &v1.width.to_biguint().unwrap() {
                    v1.width as usize
                } else {
                    b2.to_usize().unwrap()
                };
            b1 << b2_usize
        },
        OpArith::Add => b1 + b2,
        OpArith::Sub => execute_sub(v1, v2, width),
        OpArith::Mul => b1 * b2,
        OpArith::Div => {
            if b2.is_zero() {
                return Err(());
            }
            b1 / b2
        },
        OpArith::Mod => {
            if b2.is_zero() {
                return Err(());
            }
            b1 % b2
        },
        _ => panic!("not supported")
    };
    Ok(Value::new(v_res, width))
}

pub fn execute_signed_arithop(op: OpArith, v1: &Value, v2: &Value, width: u32)
                              -> Result<Value,()>
{
    let sign_1 = v1.sign();
    match op {
        OpArith::SDiv => {
            let sign = v1.sign() * v2.sign();
            let v2_unsigned = v2.unsign().value;
            if v2_unsigned.is_zero() {
                return Err(());
            }
            let u_res = v1.unsign().value / v2_unsigned;
            let s_res = Value::new(u_res, width).set_sign(sign);
            Ok(s_res)
        },
        OpArith::ARShift => {
            // assert!(v2.sign() == Sign::Positive);
            if v2.sign() == Sign::Negative {
                return Err(());
            }
            let in_bit = match v1.sign() {
                Sign::Positive => 0,
                Sign::Negative => 1,
            };
            let msb = in_bit.to_biguint().unwrap() << (width as usize - 1);
            //let times = v2.value.to_usize().unwrap();
            let times =
                if v2.value > v1.width.to_biguint().unwrap() {
                    v1.width as usize
                } else {
                    v2.value.to_usize().unwrap()
                };
            let mut val1 = v1.value.clone();
            if in_bit == 0 {
                Ok(Value::new(val1 >> times, width))
            } else {
                for i in 0 .. times {
                    val1 = (val1 >> 1) | &msb;
                }
                Ok(Value::new(val1, width))
            }
        },
        OpArith::SMod => {
            let v2_unsigned = v2.unsign().value;
            if v2_unsigned.is_zero() {
                return Err(());
            }
            let res = v1.unsign().value % v2_unsigned;
            Ok(Value::new(res, width).set_sign(v1.sign()))
        },
        _ => panic!("not supported")
    }
}

fn execute_arithop(op: OpArith, v1: &Value, v2: &Value, ty: ExprType)
                   -> Result<Value,()>
{
    let width = match ty {
        ExprType::Int(w) => w,
        _ => panic!("not supported")
    };
    match op {
        // These ones need special treatment due to sign
        OpArith::SMod | OpArith::SDiv | OpArith::ARShift =>
            execute_signed_arithop(op, v1, v2, width),
        // The ones not depending in the sign can be processed
        // generically and then casted
        OpArith::Add | OpArith::Sub | OpArith::Mul |
        OpArith::Div | OpArith::Mod | OpArith::ALShift =>
            execute_unsigned_arithop(op, v1, v2, width)
    }
}

pub fn execute_unop(op: OpUnary, v: &Value) -> Value {
    let ref val = v.value;
    match op {
        OpUnary::Neg => Value::new(
            (BigUint::one() << v.width as usize) - val, v.width),
        OpUnary::Not => Value::new(
            ((BigUint::one() << v.width as usize) - BigUint::one())
                ^ val, v.width)
    }
}

pub fn execute_unsigned_boolop(op: OpBool, v1: &Value, v2: &Value) -> bool {
    let ref v1 = v1.value;
    let ref v2 = v2.value;
    match op {
        OpBool::LT => v1 > v2,
        OpBool::LE => v1 >= v2,
        OpBool::EQ => v1 == v2,
        OpBool::NEQ => v1 != v2,
        _ => panic!("not supported")
    }
}

fn execute_signed_boolop(op: OpBool, v1: &Value, v2: &Value) -> bool {
    let is_larger = match (v1.sign(), v2.sign()) {
        (Sign::Positive, Sign::Negative) => true,
        (Sign::Negative, Sign::Positive) => false,
        (_ , _) => v1.value > v2.value
    };
    match op {
        OpBool::SLT => is_larger,
        OpBool::SLE => is_larger || (v1.value == v2.value),
        _ => panic!("not supported")
    }
}

fn execute_boolop(op: OpBool, v1: &Value, v2: &Value) -> Value {
    let res = match op {
        OpBool::LT | OpBool::LE |
        OpBool::EQ | OpBool::NEQ => execute_unsigned_boolop(op, v1, v2),
        OpBool::SLT | OpBool::SLE => execute_signed_boolop(op, v1, v2)
    };
    match res {
        true => Value::new(BigUint::one(), 1),
        false => Value::new(BigUint::zero(), 1)
    }
}

// XXX_ We should only force Value on the branch taken to save some
// cpu.
fn execute_ite(b: &Value, e1: &Value, e2: &Value) -> Value {
    let ref v = b.value;
    if v == &BigUint::one() {
        e1.clone()
    } else if v == &BigUint::zero() {
        e2.clone()
    } else {
        // XXX_ I know...
        //panic!("ITE first argument should be 1 bit")
        e1.clone()
    }
}
// XXX_ implement
fn execute_cast(op: OpCast, et: ExprType, v: &Value) -> Result<Value,()> {
    let ty_w = et.get_int_width();
    match op {
        OpCast::CastLow => {
            if ty_w > v.width() {
                Err(())
            } else {
                let mask = (BigUint::one() << (ty_w as usize)) - BigUint::one();
                Ok(Value::new(v.value() & mask, ty_w))
            }
        }
        OpCast::CastHigh => {
            if ty_w > v.width() {
                Err(())
            } else {
                let res = v.value() >> ((v.width() - ty_w) as usize);
                Ok(Value::new(res, ty_w))
            }
        }
        // XXX_ implement me
        OpCast::CastSigned => Ok(v.clone()),
    }
}

pub fn execute_bits(l1: u32, l2: u32, v: &Value) -> Result<Value,()> {
    if l1 > v.width() || l2 > v.width() || l1 > l2 {
        return Err(());
    }
    let new_width = l2 - l1;
    let mask_bits = (BigUint::one() << (new_width as usize)) - BigUint::one();
    let mask = mask_bits << (l1 as usize);
    let val = &v.value & mask;
    Ok(Value::new(val >> (l1 as usize), new_width))
}

pub fn execute_expr(state: &State, e: &Expr) -> Result<Value,()> {
    let res = match *e {
        Reg(ref n, _) => state.get_expr_value(e),
        // XXX_ implement me, but I have the feeling that it should be
        // avoided at all cost and only found during dependency.
        Deref(ref e, _) => // try!(execute_expr(state, &*e)),
            state.get_expr_value(e),
        // XXX_ this one should be good to depend on the width of the
        // last expression, so for example if it's -1 for an Int(8)
        // type, then it will go to 0xff, for Int(16): 0xFFff, etc.
        Int(ref i) => Value::new(i.clone(), 32),
        // XXX_ this too
        IInt(i) => Value::new(i.abs().to_biguint().unwrap(), 32),
        ArithOp(o, ref e1, ref e2, et) =>
            try!(execute_arithop(o,
                                 &try!(execute_expr(state, &*e1)),
                                 &try!(execute_expr(state, &*e2)),
                                 et)),
        LogicOp(o, ref e1, ref e2) =>
            execute_logicop(o,
                            &try!(execute_expr(state, &*e1)),
                            &try!(execute_expr(state, &*e2))),
        BoolOp(o, ref e1, ref e2) =>
            execute_boolop(o,
                           &try!(execute_expr(state, &*e1)),
                           &try!(execute_expr(state, &*e2))),
        UnOp(o, ref e) => execute_unop(o,
                                        &try!(execute_expr(state, &*e))),
        ITE(ref eb, ref e1, ref e2) =>
            execute_ite(&try!(execute_expr(state, &*eb)),
                        &try!(execute_expr(state, &*e1)),
                        &try!(execute_expr(state, &*e2))),
        Cast(o, ref et, ref e) => try!(
            execute_cast(o, *et,
                         &try!(execute_expr(state, &*e)))),
        Bits(l1, l2, ref e) => try!(
            execute_bits(l1, l2, &try!(execute_expr(state, &*e)))),
        Bit(b, ref e) => try!(
            execute_bits(b, b+1, &try!(execute_expr(state, &*e)))),
        _ => panic!(format!("not supported: {:?}", e))
    };
    Ok(res)
}

// Fillers
fn execute_to_bool(c: Expr) -> bool {
    true
}

/*
fn execute_ite(c: Expr, e1: Expr, e2: Expr) -> BigUint {
    if execute_to_bool(c) {
        execute_expr(e1)
    } else {
        execute_expr(e2)
    }
}
*/
use expr::Expr::*;

pub fn emulate_stmt(state: State, stmt: Stmt) -> State {
    match stmt {
        Stmt::StoreReg(e1, e2) =>
            if e2.is_reg() {
                state
            } else {
                state
            },
        _ => state
    }
}

// Emulator assumes all the operations are width coeherent so it
// doesn't check types.
pub type Program = Vec<Stmt>;

pub fn emulate_progrma(state: State, program: Program) -> State {
    let mut n_state = state;
    for stmt in program {
        n_state = emulate_stmt(n_state, stmt);
    }
    n_state
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;
    use expr::Expr::*;
    use op::{OpArith, OpLogic, OpUnary, OpBool, OpCast};
    use num::bigint::{ToBigUint, BigUint};
    use num::Num;
    use std::collections::HashMap;

    // Value
    #[test]
    fn test_Value_unsign() {
        let a_n = Value { width: 32, value: 0xffFF_ffFFu32.to_biguint().unwrap() };
        assert_eq!(a_n.unsign().value, 1u32.to_biguint().unwrap());

        let a_n = Value { width: 32, value: 0x7fFF_ffFFu32.to_biguint().unwrap() };
        assert_eq!(a_n.unsign().value, 0x7fFF_ffFFu32.to_biguint().unwrap());

        let a_n = Value { width: 32, value: 0x7fFF_ffFFu32.to_biguint().unwrap() };
        assert_eq!(a_n.unsign().value, 0x7fFF_ffFFu32.to_biguint().unwrap());
    }
    #[test]
    fn test_Value_set_sign_negative1(){
        let a_n = Value { width: 32, value: 1.to_biguint().unwrap() };
        let a_n = a_n.set_sign(Sign::Negative);
        assert_eq!(a_n.value, 0xffFF_ffFFu32.to_biguint().unwrap());
    }
    #[test]
    fn test_Value_set_sign_negative2(){
        // XXX_
        let a_n = Value { width: 32, value: 0x7fFF_ffFF.to_biguint().unwrap() };
        let a_n = a_n.set_sign(Sign::Negative);
        assert_eq!(a_n.value, 0x8000_0001u32.to_biguint().unwrap());
    }
    #[test]
    fn test_Value_from_neg_int() {
        let a_n = Value::from_int(-1, 32);
        assert_eq!(a_n.value, 0xFFff_FFffu32.to_biguint().unwrap());
    }
    #[test]
    fn test_Value_from_pos_int() {
        let a_n = Value::from_int(0x80, 32);
        assert_eq!(a_n.value, 0x80.to_biguint().unwrap());
    }
    // Emulator
    #[test]
    fn test_ARShift_positive() {
        let v1 = Value { width: 32, value: 0x7fFF_ffFF.to_biguint().unwrap() };
        let v2 = Value { width:  8, value: 3.to_biguint().unwrap() };
        let vres = execute_signed_arithop(OpArith::ARShift, &v1, &v2, 32);
        assert_eq!(vres.unwrap().value, 0xfffffff.to_biguint().unwrap());
    }
    #[test]
    fn test_ARShift_negative() {
        let v1 = Value { width: 32, value: 0x8000_1234u32.to_biguint().unwrap() };
        let v2 = Value { width:  8, value: 8.to_biguint().unwrap() };
        let vres = execute_signed_arithop(OpArith::ARShift, &v1, &v2, 32);
        assert_eq!(vres.unwrap().value, 0xff80_0012u32.to_biguint().unwrap());
    }
    #[test]
    fn test_SDiv_positive() {
        let v1 = Value { width: 32, value: 1000.to_biguint().unwrap() };
        let v2 = Value { width: 32, value: 10.to_biguint().unwrap() };
        let vres = execute_signed_arithop(OpArith::SDiv, &v1, &v2, 32);
        assert_eq!(vres.unwrap().value, 100.to_biguint().unwrap());
    }
    #[test]
    fn test_SDiv_negative_positive() {
        let v1 = Value { width: 32, value: 0xffFF_ff00u32.to_biguint().unwrap() };
        let v2 = Value { width: 32, value: 0x10.to_biguint().unwrap() };
        let vres = execute_signed_arithop(OpArith::SDiv, &v1, &v2, 32);
        assert_eq!(vres.unwrap().value, 0xffFF_ffF0u32.to_biguint().unwrap());
    }
    #[test]
    fn test_SDiv_negative_negative() {
        let v1 = Value { width: 32, value: 0xffFF_ff00u32.to_biguint().unwrap() };
        let v2 = Value { width: 32, value: 0xffFF_ffF0u32.to_biguint().unwrap() };
        let vres = execute_signed_arithop(OpArith::SDiv, &v1, &v2, 32);
        assert_eq!(vres.unwrap().value, 0x10.to_biguint().unwrap());
    }
    #[test]
    fn test_SMod_negative_negative() {
        let v1 = Value { width: 32, value: 0xffFF_ff01u32.to_biguint().unwrap() };
        let v2 = Value { width: 32, value: 0xffFF_ffF0u32.to_biguint().unwrap() };
        let vres = execute_signed_arithop(OpArith::SMod, &v1, &v2, 32);
        assert_eq!(vres.unwrap().value, 0xffFF_ffF1u32.to_biguint().unwrap());
    }
    #[test]
    fn test_SMod_positive_negative() {
        let v1 = Value { width: 32, value: 0x101u32.to_biguint().unwrap() };
        let v2 = Value { width: 32, value: 0xffFF_ffF0u32.to_biguint().unwrap() };
        let vres = execute_signed_arithop(OpArith::SMod, &v1, &v2, 32);
        assert_eq!(vres.unwrap().value, 1u32.to_biguint().unwrap());
    }
    #[test]
    fn test_Add() {
        let v1 = Value { width: 32, value: 0x10.to_biguint().unwrap() };
        let v2 = Value { width: 32, value: 0x10.to_biguint().unwrap() };
        let vres = execute_unsigned_arithop(OpArith::Add, &v1, &v2, 32);
        assert_eq!(vres.unwrap().value, 0x20.to_biguint().unwrap());
    }
    #[test]
    fn test_Add_ovf_trim() {
        let v1 = Value { width: 32, value: 0x8000_0000u32.to_biguint().unwrap() };
        let v2 = Value { width: 32, value: 0x8000_0000u32.to_biguint().unwrap() };
        let vres = execute_unsigned_arithop(OpArith::Add, &v1, &v2, 32);
        assert_eq!(vres.unwrap().value, 0.to_biguint().unwrap());
    }
    #[test]
    fn test_Add_ovf_notrim() {
        let v1 = Value { width: 32, value: 0x8000_0000u32.to_biguint().unwrap() };
        let v2 = Value { width: 32, value: 0x8000_0000u32.to_biguint().unwrap() };
        let vres = execute_unsigned_arithop(OpArith::Add, &v1, &v2, 33);
        assert_eq!(vres.unwrap().value, 0x1_0000_0000u64.to_biguint().unwrap());
    }
    #[test]
    fn test_Mul_ovf_notrim() {
        let v1 = Value { width: 32, value: 0x8000_0000u32.to_biguint().unwrap() };
        let v2 = Value { width: 32, value: 0x10.to_biguint().unwrap() };
        let vres = execute_unsigned_arithop(OpArith::Mul, &v1, &v2, 64);
        assert_eq!(vres.unwrap().value, 0x8_0000_0000u64.to_biguint().unwrap());
    }
    #[test]
    fn test_Sub_1() {
        let v1 = Value { width: 32, value: 5.to_biguint().unwrap() };
        let v2 = Value { width: 32, value: 1.to_biguint().unwrap() };
        let vres = execute_unsigned_arithop(OpArith::Sub, &v1, &v2, 32);
        assert_eq!(vres.unwrap().value, 4.to_biguint().unwrap());
    }
    #[test]
    fn test_Sub_2() {
        let v1 = Value { width: 32, value: 0xF000_0005u32.to_biguint().unwrap() };
        let v2 = Value { width: 32, value: 1.to_biguint().unwrap() };
        let vres = execute_unsigned_arithop(OpArith::Sub, &v1, &v2, 32);
        assert_eq!(vres.unwrap().value, 0xF000_0004u32.to_biguint().unwrap());
    }
    #[test]
    fn test_Sub_neg1() {
        let v1 = Value { width: 32, value: 0xFFff_FFf0u32.to_biguint().unwrap() };
        let v2 = Value { width: 32, value: 0xFFff_FFffu32.to_biguint().unwrap() };
        let vres = execute_unsigned_arithop(OpArith::Sub, &v1, &v2, 32);
        assert_eq!(vres.unwrap().value, 0xFFff_FFf1u32.to_biguint().unwrap());
    }
    #[test]
    fn test_Sub_ovf1() {
        let v1 = Value { width: 32, value: 1.to_biguint().unwrap() };
        let v2 = Value { width: 32, value: 2.to_biguint().unwrap() };
        let vres = execute_unsigned_arithop(OpArith::Sub, &v1, &v2, 32);
        assert_eq!(vres.unwrap().value, 0xffFF_ffFFu32.to_biguint().unwrap());
    }
    #[test]
    fn test_Sub_ovf2() {
        let v1 = Value { width: 32, value: 1.to_biguint().unwrap() };
        let v2 = Value { width: 32, value: 0xFFff_FFffu32.to_biguint().unwrap() };
        let vres = execute_unsigned_arithop(OpArith::Sub, &v1, &v2, 32);
        assert_eq!(vres.unwrap().value, 0.to_biguint().unwrap());
    }
    #[test]
    fn test_Sub_ovf3() {
        let v1 = Value { width: 32, value: 1.to_biguint().unwrap() };
        let v2 = Value { width: 32, value: 2.to_biguint().unwrap() };
        let vres = execute_unsigned_arithop(OpArith::Sub, &v1, &v2, 64);
        assert_eq!(vres.unwrap().value, 0xFFff_FFff_FFff_FFffu64.to_biguint().unwrap());
    }
    // XXX_ fill with the other unsigned operations
    #[test]
    fn test_Neg(){
        let v = Value { width: 32, value: 0x1000_0000u32.to_biguint().unwrap() };
        let vres = execute_unop(OpUnary::Neg, &v);
        assert_eq!(vres.value, 0xf000_0000u32.to_biguint().unwrap());
    }
    #[test]
    fn test_Not(){
        let v = Value { width: 32, value: 0x1000_0000u32.to_biguint().unwrap() };
        let vres = execute_unop(OpUnary::Not, &v);
        assert_eq!(vres.value, 0xefFF_ffFFu32.to_biguint().unwrap());
    }
    #[test]
    fn test_Bits_ok1(){
        let v = Value {width: 32, value: 0xff0.to_biguint().unwrap() };
        let vres = execute_bits(8, 16, &v);
        assert_eq!(vres.unwrap().value, 0xf.to_biguint().unwrap());
    }
    #[test]
    fn test_Bits_ok2(){
        let v = Value {width: 32, value: 0x70000.to_biguint().unwrap() };
        let vres = execute_bits(16, 24, &v);
        assert_eq!(vres.unwrap().value, 0x7.to_biguint().unwrap());
    }
    #[test]
    fn test_Bits_ok3(){
        let v = Value {width: 32, value: 0x70000.to_biguint().unwrap() };
        let vres = execute_bits(24, 32, &v);
        assert_eq!(vres.unwrap().value, 0.to_biguint().unwrap());
    }
    #[test]
    fn test_Bits_ok4(){
        let v = Value {width: 32, value: 0xffFF_ffFFu32.to_biguint().unwrap() };
        let vres = execute_bits(0, 32, &v);
        assert_eq!(vres.unwrap().value, 0xffFF_ffFFu32.to_biguint().unwrap());
    }
    #[test]
    fn test_Bits_err1(){
        let v = Value {width: 32, value: 0x70000.to_biguint().unwrap() };
        let vres = execute_bits(32, 36, &v);
        assert_eq!(vres.is_err(), true);
    }
    #[test]
    fn test_Bits_err2(){
        let v = Value {width: 32, value: 0x70000.to_biguint().unwrap() };
        let vres = execute_bits(8, 4, &v);
        assert_eq!(vres.is_err(), true);
    }
    #[test]
    fn test_Bit_ok1(){
        let e = Reg("EAX".to_owned(), 32);
        let v = 0xff00ff00u32.to_biguint().unwrap();
        let mut h = HashMap::new();
        h.insert(e.clone(), v.clone());
        let state = State::borrow(&h);

        let op = Bit(24, Box::new(e.clone()));
        let vres = execute_expr(&state, &op);

        assert_eq!(vres.unwrap().value, 0x1.to_biguint().unwrap());
    }
    #[test]
    fn test_Bit_ok2(){
        let e = Reg("EAX".to_owned(), 32);
        let v = 0xff00ff00u32.to_biguint().unwrap();
        let mut h = HashMap::new();
        h.insert(e.clone(), v.clone());
        let state = State::borrow(&h);

        let op = Bit(7, Box::new(e.clone()));
        let vres = execute_expr(&state, &op);

        assert_eq!(vres.unwrap().value, 0.to_biguint().unwrap());
    }
    // XXX_ tests for boolops
    // XXX_ tests for checking abortions/Err

    // Benchmarks
    use test::{Bencher, black_box};
    #[bench]
    fn bench_Mul(b: &mut Bencher) {
        let v1 = Value { width: 32, value: 0x8000_0000u32.to_biguint().unwrap() };
        let v2 = Value { width: 32, value: 0x10.to_biguint().unwrap() };
        b.iter(|| {
            black_box(execute_unsigned_arithop(OpArith::Mul, &v1, &v2, 64).ok());
        });
    }
    #[bench]
    fn bench_Sub_ovf(b: &mut Bencher) {
        let v1 = Value { width: 32, value: 1.to_biguint().unwrap() };
        let v2 = Value { width: 32, value: 2.to_biguint().unwrap() };
        b.iter(|| {
            black_box(execute_unsigned_arithop(OpArith::Sub, &v1, &v2, 32).ok());
        });
    }
    #[bench]
    fn bench_Add_ovf_trim(b: &mut Bencher) {
        let v1 = Value { width: 32, value: 0x8000_0000u32.to_biguint().unwrap() };
        let v2 = Value { width: 32, value: 0x8000_0000u32.to_biguint().unwrap() };
        b.iter(|| {
            black_box(execute_unsigned_arithop(OpArith::Add, &v1, &v2, 32).ok());
        });
    }
}
