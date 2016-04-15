use std::collections::{HashMap};

use op::{OpArith, OpLogic, OpUnary, OpBool, OpCast};
use expr::{Expr, ExprType};
use stmt::Stmt;

use num::bigint::{ToBigUint, BigUint};
use num::traits::{ToPrimitive, Zero, One};
use num::Num;

static mut MASKS_ALL: *mut Vec<BigUint> = 0 as *mut Vec<BigUint>;
static mut MASKS_BIT: *mut Vec<BigUint> = 0 as *mut Vec<BigUint>;

pub struct State<'a> {
    values: &'a HashMap<Expr, BigUint>,
}

/// Initialize the global vector of BigUint masks
unsafe fn init_masks() {
    let mut v_a = Box::new(Vec::new());
    let mut v_b = Box::new(Vec::new());

    for i in 0..2048 {
        let bit_mask = BigUint::one() << i;
        let all_mask = &bit_mask - BigUint::one();
        v_a.push(all_mask);
        v_b.push(bit_mask);
    }
    MASKS_ALL = Box::into_raw(v_a);
    MASKS_BIT = Box::into_raw(v_b);
}

#[inline(always)]
fn mask_n_bits<'a>(n: u32) -> &'a BigUint {
    unsafe { (*MASKS_ALL).get_unchecked(n as usize) }
}

#[inline(always)]
fn mask_bit_n<'a>(n: u32) -> &'a BigUint {
    unsafe { (*MASKS_BIT).get_unchecked(n as usize) }
}

impl<'a> State<'a> {
    pub fn borrow(map: &'a HashMap<Expr, BigUint>) -> State<'a> {
        State {
            values: map
        }
    }
    pub fn get_expr_value(&self, e: &Expr) -> Result<Value, ()> {
        let width = e.get_width().unwrap();
        match self.values.get(e) {
            Some(e) => Ok(Value::new(e.clone(), width)),
            None => Err(())
        }
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
    #[cfg(any(test, bench))]
    pub fn new(value: BigUint, width: u32) -> Value {
        setup_emulator();
        let size_mask = mask_n_bits(width);
        Value { value: value & size_mask, width: width }
    }

    #[cfg(not(test))]
    pub fn new(value: BigUint, width: u32) -> Value {
        let size_mask = mask_n_bits(width);
        Value { value: value & size_mask, width: width }
    }

    pub fn new_unchecked(value: BigUint, width: u32) -> Value {
        Value { value: value, width: width }
    }
    pub fn from_int(value: i32, width: u32) -> Value {
        let abs_value = value.abs();
        let val = Value::new_unchecked(abs_value.to_biguint().unwrap(), width);
        if value >= 0 {
            val
        } else {
            val.set_sign(Sign::Negative)
        }
    }
    pub fn value(&self) -> &BigUint {
        &self.value
    }
    pub fn get_sign(&self) -> Sign {
        if self.width.is_zero() {
            return Sign::Positive
        }
        let sign_bit_b = mask_bit_n(self.width - 1);
        if (&self.value & sign_bit_b).is_zero() {
            Sign::Positive
        } else {
            Sign::Negative
        }
    }
    pub fn unsign(&self) -> Value {
        if self.get_sign() == Sign::Positive {
            self.clone()
        } else {
            let sign_mask = mask_bit_n(self.width);
            Value {
                width: self.width,
                value: sign_mask - &self.value
            }
        }
    }
    pub fn set_sign(self, sign: Sign) -> Value {
        if sign == Sign::Negative {
            let sign_n_b = mask_bit_n(self.width);
            Value {
                width: self.width,
                value: sign_n_b - self.value
            }
        } else {
            self
        }
    }
    pub fn get_width(&self) -> u32 { self.width }
    pub fn set_width(&mut self, w: u32) { self.width = w }
}

/// Sign extend (should only be used by adjust_width)
fn sign_ext(v: &Value, width: u32) -> Value {
    let sign = v.get_sign();
    if sign == Sign::Positive {
        let mut val = v.clone();
        val.set_width(width);
        val
    } else {
        let mut val = v.unsign();
        val.set_width(width);
        val.set_sign(Sign::Negative)
    }
}

/// Extract bits (should only be used by adjust_width)
fn extract(v: &Value, width: u32) -> Value {
    let inside = v.value().clone();
    Value::new(inside, width)
}

/// Adjust width
fn adjust_width(v: &Value,
                width: u32, s_ext: bool) -> Value
{
    if width > v.get_width() {
        if s_ext {
            sign_ext(v, width)
        } else {
            let mut rv = v.clone();
            rv.set_width(width);
            rv
        }
    } else if width < v.get_width() {
        extract(v, width)
    } else {
        v.clone()
    }
}

fn execute_sub(v1: &Value, v2: &Value, w: u32) -> BigUint {
    let ref b1 = v1.value;
    let ref b2 = v2.value;
    if b1 >= b2 {
        b1 - b2
    } else {
        let ub1 = v1.unsign().value;
        let ub2 = v2.unsign().value;
        if ub1 >= ub2 {
            Value::new((ub1 - ub2), w).set_sign(Sign::Negative).value
        } else {
            Value::new((ub2 - ub1), w).set_sign(Sign::Negative).value
        }
    }
}

fn execute_logicop(op: OpLogic, v1: &Value, v2: &Value, w: u32) -> Value {
    let v1 = adjust_width(v1, w, false);
    let v2 = adjust_width(v2, w, false);
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
    let v1 = adjust_width(v1, width, false);
    let v2 = adjust_width(v2, width, false);
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
        OpArith::Sub => execute_sub(&v1, &v2, width),
        OpArith::Mul => b1 * b2,
        OpArith::Div => {
            if b2.is_zero() {
                return Err(());
            }
            b1 / b2
        },
        OpArith::URem => {
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
    let v1 = adjust_width(v1, width, true);
    let v2 = adjust_width(v2, width, true);
    let sign_1 = v1.get_sign();
    match op {
        OpArith::SDiv => {
            let sign = v1.get_sign() * v2.get_sign();
            let v2_unsigned = v2.unsign().value;
            if v2_unsigned.is_zero() {
                return Err(());
            }
            let u_res = v1.unsign().value / v2_unsigned;
            let s_res = Value::new(u_res, width).set_sign(sign);
            Ok(s_res)
        },
        OpArith::ARShift => {
            // assert!(v2.get_sign() == Sign::Positive);
            if v2.get_sign() == Sign::Negative {
                return Err(());
            }
            let in_bit = match v1.get_sign() {
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
        OpArith::SRem => {
            let v2_unsigned = v2.unsign().value;
            if v2_unsigned.is_zero() {
                return Err(());
            }
            let res = v1.unsign().value % v2_unsigned;
            Ok(Value::new(res, width).set_sign(v1.get_sign()))
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
        OpArith::SRem | OpArith::SDiv | OpArith::ARShift =>
            execute_signed_arithop(op, v1, v2, width),
        // The ones not depending in the sign can be processed
        // generically and then casted
        OpArith::Add | OpArith::Sub | OpArith::Mul |
        OpArith::Div | OpArith::URem | OpArith::ALShift =>
            execute_unsigned_arithop(op, v1, v2, width)
    }
}

pub fn execute_unop(op: OpUnary, v: &Value, w: u32) -> Value {
    let v = adjust_width(v, w, false);
    let ref val = v.value;
    match op {
        OpUnary::Neg => Value::new(
            mask_bit_n(v.width) - val, v.width),
        OpUnary::Not => Value::new(
            mask_n_bits(v.width) ^ val, v.width)
    }
}

fn execute_unsigned_boolop(op: OpBool, v1: &Value, v2: &Value, w: u32) -> bool {
    let v1 = adjust_width(v1, w, false);
    let v2 = adjust_width(v2, w, false);
    let ref v1 = v1.value;
    let ref v2 = v2.value;
    match op {
        OpBool::LT => v1 < v2,
        OpBool::LE => v1 <= v2,
        OpBool::EQ => v1 == v2,
        OpBool::NEQ => v1 != v2,
        _ => panic!("not supported")
    }
}

fn execute_signed_boolop(op: OpBool, v1: &Value, v2: &Value, w: u32) -> bool {
    let v1 = adjust_width(v1, w, true);
    let v2 = adjust_width(v2, w, true);
    let is_less = match (v1.get_sign(), v2.get_sign()) {
        (Sign::Positive, Sign::Negative) => false,
        (Sign::Negative, Sign::Positive) => true,
        (_ , _) => v1.value < v2.value
    };
    match op {
        OpBool::SLT => is_less,
        OpBool::SLE => is_less || (v1.value == v2.value),
        _ => panic!("not supported")
    }
}

fn execute_boolop(op: OpBool, v1: &Value, v2: &Value, w: u32) -> Value {
    let res = match op {
        OpBool::LT | OpBool::LE |
        OpBool::EQ | OpBool::NEQ => execute_unsigned_boolop(op, v1, v2, w),
        OpBool::SLT | OpBool::SLE => execute_signed_boolop(op, v1, v2, w)
    };
    match res {
        true => Value::new_unchecked(BigUint::one(), 1),
        false => Value::new_unchecked(BigUint::zero(), 1)
    }
}

/// Only execute the Expr that is needed to evaluate according to the
/// condition, it's a special case
fn execute_ite(state: &State, b: &Value, e1: &Expr, e2: &Expr, w: u32)
               -> Result<Value,()>
{
    let ref v = b.value;
    if v == &BigUint::one() {
        execute_expr(state, e1, w)
    } else if v == &BigUint::zero() {
        execute_expr(state, e2, w)
    } else {
        panic!("ITE first argument should be 1 bit")
    }
}

/// Cast value to other type/width
fn execute_cast(op: OpCast, et: ExprType, v: &Value) -> Result<Value,()> {
    // XXX_ implement floating point
    let ty_w = et.get_int_width();
    match op {
        OpCast::CastLow => {
            if ty_w > v.get_width() {
                Err(())
            } else {
                Ok(adjust_width(v, ty_w, false))
            }
        }
        OpCast::CastHigh => {
            if ty_w > v.get_width() {
                Err(())
            } else {
                let res = v.value() >> ((v.get_width() - ty_w) as usize);
                Ok(Value::new(res, ty_w))
            }
        }
        OpCast::CastSigned => {
            if ty_w < v.get_width() {
                Err(())
            } else {
                Ok(adjust_width(v, ty_w, true))
            }
        }
    }
}

fn create_iint(i: i16, w: u32) -> Result<Value, ()> {
    if i < 0 {
        let val = mask_bit_n(w);
        let abs = i.abs().to_biguint().unwrap();
        if &abs > val {
            Err(())
        } else {
            Ok(Value::new_unchecked(val - abs, w))
        }
    } else {
        let val = i.to_biguint().unwrap();
        Ok(Value::new(val, w))
    }
}

pub fn execute_bits(high: u32, low: u32, v: &Value) -> Result<Value,()> {
    if high > v.get_width() || low > v.get_width() || low > high {
        return Err(());
    }
    let new_width = high - low + 1;
    let mask_bits = mask_n_bits(new_width);
    let mask = mask_bits << (low as usize);
    let val = &v.value & mask;
    Ok(Value::new(val >> (low as usize), new_width))
}

// static mut E_I: u32 = 0;

fn execute_expr_2(state: &State, e: &Expr, w: u32) -> Result<Value,()> {
    // unsafe { debugln!("[{}] Executing: {:?}", E_I, e) };
    // unsafe { E_I += 1 };
    let res = match *e {
        Reg(ref n, _) => try!(state.get_expr_value(e)),
        // XXX_ implement me, but I have the feeling that it should be
        // avoided at all cost and only found during dependency.
        Deref(ref e, _) => // try!(execute_expr(state, &*e)),
            try!(state.get_expr_value(e)),
        Int(ref i) => Value::new(i.clone(), w),
        IInt(i) => try!(create_iint(i, w)),
        ArithOp(o, ref e1, ref e2, et) => {
            let w = et.get_width();
            try!(execute_arithop(o,
                                 &try!(execute_expr(state, &*e1, w)),
                                 &try!(execute_expr(state, &*e2, w)),
                                 et))
        },
        LogicOp(o, ref e1, ref e2, w) =>
            execute_logicop(o,
                            &try!(execute_expr(state, &*e1, w)),
                            &try!(execute_expr(state, &*e2, w)),
                            w),
        BoolOp(o, ref e1, ref e2, w) =>
            execute_boolop(o,
                           &try!(execute_expr(state, &*e1, w)),
                           &try!(execute_expr(state, &*e2, w)),
                           w),
        UnOp(o, ref e, w) => execute_unop(o,
                                          &try!(execute_expr(state, &*e, w)),
                                          w),
        ITE(ref eb, ref e1, ref e2) =>
            try!(execute_ite(state,
                             &try!(execute_expr(state, &*eb, 1)),
                             e1,
                             e2,
                             w)),
        Cast(o, ref e, ref et) => try!(
            execute_cast(o, *et,
                         // XXX_ width
                         &try!(execute_expr(state, &*e, 1024)))),
        Bits(high, low, ref e) => try!(
            execute_bits(high, low, &try!(execute_expr(state, &*e, high + 1)))),
        Bit(b, ref e) => try!(
            execute_bits(b, b, &try!(execute_expr(state, &*e, b + 1)))),
        _ => panic!(format!("not supported: {:?}", e))
    };
    // unsafe { E_I -= 1 };
    // unsafe { debugln!("[{}] res: {:?}", E_I, res) };
    Ok(res)
}

pub fn setup_emulator() {
    unsafe {
        if MASKS_ALL.is_null() || MASKS_BIT.is_null() {
            init_masks();
        }
    }
}

pub fn execute_expr(state: &State, e: &Expr, w: u32) -> Result<Value,()> {
    setup_emulator();
    execute_expr_2(state, e, w)
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
        let a_n = Value::new(0xffFF_ffFFu32.to_biguint().unwrap(), 32);
        assert_eq!(a_n.unsign().value, 1u32.to_biguint().unwrap());

        let a_n = Value::new(0x7fFF_ffFFu32.to_biguint().unwrap(), 32);
        assert_eq!(a_n.unsign().value, 0x7fFF_ffFFu32.to_biguint().unwrap());

        let a_n = Value::new(0x7fFF_ffFFu32.to_biguint().unwrap(), 32);
        assert_eq!(a_n.unsign().value, 0x7fFF_ffFFu32.to_biguint().unwrap());
    }
    #[test]
    fn test_Value_set_sign_negative1(){
        let a_n = Value::new(1.to_biguint().unwrap(), 32);
        let a_n = a_n.set_sign(Sign::Negative);
        assert_eq!(a_n.value, 0xffFF_ffFFu32.to_biguint().unwrap());
    }
    #[test]
    fn test_Value_set_sign_negative2(){
        // XXX_
        let a_n = Value::new(0x7fFF_ffFF.to_biguint().unwrap(), 32);
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
        let v1 = Value::new(0x7fFF_ffFF.to_biguint().unwrap(), 32);
        let v2 = Value::new(3.to_biguint().unwrap(), 8);
        let vres = execute_signed_arithop(OpArith::ARShift, &v1, &v2, 32);
        assert_eq!(vres.unwrap().value, 0xfffffff.to_biguint().unwrap());
    }
    #[test]
    fn test_ARShift_negative() {
        let v1 = Value::new(0x8000_1234u32.to_biguint().unwrap(), 32);
        let v2 = Value::new(8.to_biguint().unwrap(), 8);
        let vres = execute_signed_arithop(OpArith::ARShift, &v1, &v2, 32);
        assert_eq!(vres.unwrap().value, 0xff80_0012u32.to_biguint().unwrap());
    }
    #[test]
    fn test_SDiv_positive() {
        let v1 = Value::new(1000.to_biguint().unwrap(), 32);
        let v2 = Value::new(10.to_biguint().unwrap(), 32);
        let vres = execute_signed_arithop(OpArith::SDiv, &v1, &v2, 32);
        assert_eq!(vres.unwrap().value, 100.to_biguint().unwrap());
    }
    #[test]
    fn test_SDiv_negative_positive() {
        let v1 = Value::new(0xffFF_ff00u32.to_biguint().unwrap(), 32);
        let v2 = Value::new(0x10.to_biguint().unwrap(), 32);
        let vres = execute_signed_arithop(OpArith::SDiv, &v1, &v2, 32);
        assert_eq!(vres.unwrap().value, 0xffFF_ffF0u32.to_biguint().unwrap());
    }
    #[test]
    fn test_SDiv_negative_negative() {
        let v1 = Value::new(0xffFF_ff00u32.to_biguint().unwrap(), 32);
        let v2 = Value::new(0xffFF_ffF0u32.to_biguint().unwrap(), 32);
        let vres = execute_signed_arithop(OpArith::SDiv, &v1, &v2, 32);
        assert_eq!(vres.unwrap().value, 0x10.to_biguint().unwrap());
    }
    #[test]
    fn test_SRem_negative_negative() {
        let v1 = Value::new(0xffFF_ff01u32.to_biguint().unwrap(), 32);
        let v2 = Value::new(0xffFF_ffF0u32.to_biguint().unwrap(), 32);
        let vres = execute_signed_arithop(OpArith::SRem, &v1, &v2, 32);
        assert_eq!(vres.unwrap().value, 0xffFF_ffF1u32.to_biguint().unwrap());
    }
    #[test]
    fn test_SRem_positive_negative() {
        let v1 = Value::new(0x101u32.to_biguint().unwrap(), 32);
        let v2 = Value::new(0xffFF_ffF0u32.to_biguint().unwrap(), 32);
        let vres = execute_signed_arithop(OpArith::SRem, &v1, &v2, 32);
        assert_eq!(vres.unwrap().value, 1u32.to_biguint().unwrap());
    }
    #[test]
    fn test_Add() {
        let v1 = Value::new(0x10.to_biguint().unwrap(), 32);
        let v2 = Value::new(0x10.to_biguint().unwrap(), 32);
        let vres = execute_unsigned_arithop(OpArith::Add, &v1, &v2, 32);
        assert_eq!(vres.unwrap().value, 0x20.to_biguint().unwrap());
    }
    #[test]
    fn test_Add_ovf_trim() {
        let v1 = Value::new(0x8000_0000u32.to_biguint().unwrap(), 32);
        let v2 = Value::new(0x8000_0000u32.to_biguint().unwrap(), 32);
        let vres = execute_unsigned_arithop(OpArith::Add, &v1, &v2, 32);
        assert_eq!(vres.unwrap().value, 0.to_biguint().unwrap());
    }
    #[test]
    fn test_Add_ovf_notrim() {
        let v1 = Value::new(0x8000_0000u32.to_biguint().unwrap(), 32);
        let v2 = Value::new(0x8000_0000u32.to_biguint().unwrap(), 32);
        let vres = execute_unsigned_arithop(OpArith::Add, &v1, &v2, 33);
        assert_eq!(vres.unwrap().value, 0x1_0000_0000u64.to_biguint().unwrap());
    }
    #[test]
    fn test_Mul_ovf_notrim() {
        let v1 = Value::new(0x8000_0000u32.to_biguint().unwrap(), 32);
        let v2 = Value::new(0x10.to_biguint().unwrap(), 32);
        let vres = execute_unsigned_arithop(OpArith::Mul, &v1, &v2, 64);
        assert_eq!(vres.unwrap().value, 0x8_0000_0000u64.to_biguint().unwrap());
    }
    #[test]
    fn test_Sub_1() {
        let v1 = Value::new(5.to_biguint().unwrap(), 32);
        let v2 = Value::new(1.to_biguint().unwrap(), 32);
        let vres = execute_unsigned_arithop(OpArith::Sub, &v1, &v2, 32);
        assert_eq!(vres.unwrap().value, 4.to_biguint().unwrap());
    }
    #[test]
    fn test_Sub_2() {
        let v1 = Value::new(0xF000_0005u32.to_biguint().unwrap(), 32);
        let v2 = Value::new(1.to_biguint().unwrap(), 32);
        let vres = execute_unsigned_arithop(OpArith::Sub, &v1, &v2, 32);
        assert_eq!(vres.unwrap().value, 0xF000_0004u32.to_biguint().unwrap());
    }
    #[test]
    fn test_Sub_neg1() {
        let v1 = Value::new(0xFFff_FFf0u32.to_biguint().unwrap(), 32);
        let v2 = Value::new(0xFFff_FFffu32.to_biguint().unwrap(), 32);
        let vres = execute_unsigned_arithop(OpArith::Sub, &v1, &v2, 32);
        assert_eq!(vres.unwrap().value, 0xFFff_FFf1u32.to_biguint().unwrap());
    }
    #[test]
    fn test_Sub_ovf1() {
        let v1 = Value::new(1.to_biguint().unwrap(), 32);
        let v2 = Value::new(2.to_biguint().unwrap(), 32);
        let vres = execute_unsigned_arithop(OpArith::Sub, &v1, &v2, 32);
        assert_eq!(vres.unwrap().value, 0xffFF_ffFFu32.to_biguint().unwrap());
    }
    #[test]
    fn test_Sub_ovf2() {
        let v1 = Value::new(1.to_biguint().unwrap(), 32);
        let v2 = Value::new(0xFFff_FFffu32.to_biguint().unwrap(), 32);
        let vres = execute_unsigned_arithop(OpArith::Sub, &v1, &v2, 32);
        assert_eq!(vres.unwrap().value, 0.to_biguint().unwrap());
    }
    #[test]
    fn test_Sub_ovf3() {
        let v1 = Value::new(1.to_biguint().unwrap(), 32);
        let v2 = Value::new(2.to_biguint().unwrap(), 32);
        let vres = execute_unsigned_arithop(OpArith::Sub, &v1, &v2, 64);
        assert_eq!(vres.unwrap().value, 0xFFff_FFff_FFff_FFffu64.to_biguint().unwrap());
    }
    // XXX_ fill with the other unsigned operations
    #[test]
    fn test_Neg(){
        let v = Value::new(0x1000_0000u32.to_biguint().unwrap(), 32);
        let vres = execute_unop(OpUnary::Neg, &v, 32);
        assert_eq!(vres.value, 0xf000_0000u32.to_biguint().unwrap());
    }
    #[test]
    fn test_Not(){
        let v = Value::new(0x1000_0000u32.to_biguint().unwrap(), 32);
        let vres = execute_unop(OpUnary::Not, &v, 32);
        assert_eq!(vres.value, 0xefFF_ffFFu32.to_biguint().unwrap());
    }
    #[test]
    fn test_Bits_ok1(){
        let v = Value::new(0xff0.to_biguint().unwrap(), 32);
        let vres = execute_bits(15, 8, &v);
        assert_eq!(vres.unwrap().value, 0xf.to_biguint().unwrap());
    }
    #[test]
    fn test_Bits_ok2(){
        let v = Value::new(0x70000.to_biguint().unwrap(), 32);
        let vres = execute_bits(23, 16, &v);
        assert_eq!(vres.unwrap().value, 0x7.to_biguint().unwrap());
    }
    #[test]
    fn test_Bits_ok3(){
        let v = Value::new(0x70000.to_biguint().unwrap(), 32);
        let vres = execute_bits(31, 24, &v);
        assert_eq!(vres.unwrap().value, 0.to_biguint().unwrap());
    }
    #[test]
    fn test_Bits_ok4(){
        let v = Value::new(0xffFF_ffFFu32.to_biguint().unwrap(), 32);
        let vres = execute_bits(31, 0, &v);
        assert_eq!(vres.unwrap().value, 0xffFF_ffFFu32.to_biguint().unwrap());
    }
    #[test]
    fn test_Bits_err1(){
        let v = Value::new(0x70000.to_biguint().unwrap(), 32);
        let vres = execute_bits(32, 35, &v);
        assert_eq!(vres.is_err(), true);
    }
    #[test]
    fn test_Bits_err2(){
        let v = Value::new(0x70000.to_biguint().unwrap(), 32);
        let vres = execute_bits(4, 7, &v);
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
        let vres = execute_expr(&state, &op, 32);

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
        let vres = execute_expr(&state, &op, 32);

        assert_eq!(vres.unwrap().value, 0.to_biguint().unwrap());
    }
    // XXX_ tests for boolops
    // XXX_ tests for checking abortions/Err
}
