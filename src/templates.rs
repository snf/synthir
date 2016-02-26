use emulator::{State, execute_expr};
use expr::{Expr, ExprType, ExprBuilder};
use op::{OpArith, OpLogic, OpUnary, OpBool, OpCast};

use num::bigint::{BigUint, ToBigUint};
use num::traits::{One, Zero};//, ToPrimitive};
use std::collections::HashMap;
use std::iter::Iterator;

use permutohedron::LexicalPermutation;
use itertools::Itertools;

/// Trait required for becoming a Template
trait Template2  {
    fn exec(&[&Expr], u32) -> Vec<Expr>;
    fn args() -> u32;
}

trait Template: Template2 {
    fn n_args(&self) -> u32 { Self::args() }
}

impl<T> Template for T where T: Template2 {}
//impl<T> Debug for T where T: Template2 {}

struct ParityTemplate;

impl ParityTemplate {
    /// Mod(Add(Bit(0), Add(Bit(1), Add(Bit....))), 2)
    fn parity_flag(e: &Expr, width: u32) -> Expr {
        let mut expr_builder = ExprBuilder::new();
        for i in 0..width {
            let bit = Expr::Bit(i, Box::new(e.clone()));
            let add = Expr::ArithOp(OpArith::Add,
                                    Box::new(bit),
                                    Box::new(Expr::Int(BigUint::zero())),
                                    ExprType::Int(width));
            expr_builder.insert_last_right_or_replace(&add);
        }
        expr_builder.preinsert_with_second(
            &Expr::ArithOp(
                OpArith::Mod,
                Box::new(Expr::NoOp),
                Box::new(Expr::Int(2.to_biguint().unwrap())),
                ExprType::Int(width))
                );
        expr_builder.finalize()
    }
    fn reverse_parity_flag(e: &Expr, width: u32) -> Expr {
        let e = Self::parity_flag(e, width);
        Expr::ArithOp(OpArith::Sub,
                      Box::new(Expr::Int(BigUint::one())),
                      Box::new(e),
                      ExprType::Int(1))
    }
}

impl Template2 for ParityTemplate {
    fn exec(e: &[&Expr], width: u32) -> Vec<Expr> {
        [1, 4, 8, 16, 32, 64, 128, 256]
            .iter()
            .flat_map(|w| vec![Self::parity_flag(e[0], *w),
                               Self::reverse_parity_flag(e[0], *w)])
            .collect()
    }

    fn args() -> u32 { 1 }
}

struct CarryTemplate;

impl CarryTemplate {
    fn carry_flag(e1: &Expr, e2: &Expr) -> Vec<Expr> {
        Vec::new()
    }
}

impl Template2 for CarryTemplate {
    fn exec(e: &[&Expr], width: u32) -> Vec<Expr> {
        Self::carry_flag(e[0], e[1])
    }

    fn args() -> u32 { 2 }
}

struct SignTemplate;

impl SignTemplate {
    fn sign_flag(e: &Expr, width: u32) -> Expr {
        let width = {
            match e.get_width() {
                Some(w) => w,
                None => width
            }
        };
        let mask = BigUint::one() << ((width as usize) - 1);
        Expr::Bit(
            width - 1,
            Box::new(e.clone()))
    }
}

impl Template2 for SignTemplate {
    fn exec(e: &[&Expr], width: u32) -> Vec<Expr> {
        vec![Self::sign_flag(e[0], width)]
    }

    fn args() -> u32 { 1 }
}

struct ZeroTemplate;

impl ZeroTemplate {
    fn zero_flag(e: &Expr, width: u32) -> Expr {
        Expr::ITE(
            Box::new(Expr::BoolOp(
                OpBool::EQ,
                Box::new(e.clone()),
                Box::new(Expr::Int(BigUint::zero())),
                width)),
            Box::new(Expr::Int(BigUint::one())),
            Box::new(Expr::Int(BigUint::zero())))
    }
}

impl Template2 for ZeroTemplate {
    fn exec(e: &[&Expr], width: u32) -> Vec<Expr> {
        vec![Self::zero_flag(e[0], width)]
    }

    fn args() -> u32 { 1 }
}

    /* Carry flag
    ITE(EQ(MSB(p1), MSB(r1)), 0, 1)
    ITE(EQ(MSB(p1),
    if res < p1 || res < p2
     */
    fn exprs_to_replace(width: u32) -> Vec<Expr> {
        // Carry
        // Overflow
        // Parity
        vec![]
    }

pub struct TemplateSearch<'a> {
    args: &'a[&'a Expr],
    io_sets: &'a [(BigUint,HashMap<Expr, BigUint>)],
    expr_width: u32
}

impl<'a> TemplateSearch<'a> {
    pub fn new(args: &'a [&'a Expr],
               io_sets: &'a [(BigUint,HashMap<Expr, BigUint>)],
               expr_width: u32)
               -> TemplateSearch<'a> {
        TemplateSearch {
            args: args,
            io_sets: io_sets,
            expr_width: expr_width
        }
    }

    /// Execute with an I/O set and return true if it matches the
    /// result
    fn execute_once(&self,
                    e: &Expr,
                    io_set: &HashMap<Expr, BigUint>)
                    -> Result<BigUint,()>
    {
        let state = State::borrow(io_set);
        Ok(
            try!(execute_expr(&state, e, self.expr_width)).value().clone())
    }

    /// Execute all the I/O sets for an expression and return true if
    /// the expected outputs match with the obtained ones
    fn execute_expected(&self, e: &Expr) -> bool {
        for &(ref expected, ref io_set) in self.io_sets {
            if let Ok(res) = self.execute_once(e, io_set) {
                if &res == expected {
                    continue;
                } else {
                    return false;
                }
            } else {
                return false;
            }
        }
        return true;
    }

    /// Try Template
    fn try_template<T>(&self) -> Vec<Expr>
        where T: Template2
    {
        // Generate all the possible orders of combinations of the arguments
        let mut all = Vec::new();
        for combination in self.args.iter()
            .cloned().combinations_n(T::args() as usize)
        {
            let mut comb = combination.to_vec();
            all.push(comb.clone());
            while (comb).next_permutation() {
                if !all.contains(&comb) {
                    all.push(comb.clone());
                }
            }
        }

        // Now generate the exprs and save only the ones that pass the
        // execution tests
        let mut res = Vec::new();
        for exprs in &all {
            T::exec(exprs, self.expr_width)
                .iter()
                .filter(|&e| self.execute_expected(e))
                .inspect(|&e| res.push(e.clone()))
                .count();
        }
        res
    }

    /// Try finding exprs for the template
    pub fn work(&self) -> Vec<Expr> {
        let mut res = Vec::new();
        macro_rules! exec_for{
            ($res:expr, $($T:ident),*) => (
                $(
                    //println!("template: {}", stringify!($T));
                    $res.append(&mut self.try_template::<$T>());
                )*
            )
        }
        exec_for!(res,
                  ParityTemplate, CarryTemplate, SignTemplate, ZeroTemplate);
        res
    }
}
