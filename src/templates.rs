use expr::{Expr, ExprType, ExprBuilder};
use op::{OpArith, OpLogic, OpUnary, OpBool, OpCast};

use num::bigint::{BigUint, ToBigUint};
use num::traits::{One, Zero, ToPrimitive};
use std::iter::Iterator;

/// Trait required for becoming a Template
pub trait Template where Self: Iterator<Item=Expr> {
    fn new(base: &[&Expr]) -> Self;
}

/// FlagTemplate needs a previous result to inherit a flag from it
struct FlagTemplate {
    exprs: Vec<Expr>,
}

impl FlagTemplate {

    /// Mod(Add(Bit(0), Add(Bit(1), Add(Bit....))), 2)
    pub fn parity_flag(&self, e: &Expr) -> Expr {
        let width = e.get_width().unwrap();
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

    /* Carry flag
    ITE(EQ(MSB(p1), MSB(r1)), 0, 1)
    ITE(EQ(MSB(p1),
    if eq < p1 || eq < p2
     */
    fn exprs_to_replace(&self, width: u32) -> Vec<Expr> {
        // Carry
        // Overflow
        // Parity
        vec![]
    }
}

impl Iterator for FlagTemplate {
    type Item = Expr;
    fn next(&mut self) -> Option<Expr> {
        Some(Expr::Reg("EAX".to_owned(), 32))
    }
}

impl Template for FlagTemplate {
    fn new(exprs: &[&Expr]) -> Self {
        FlagTemplate {
            exprs: exprs.into_iter().map(|&e| e.clone()).collect()
        }
    }
}

#[cfg(test)]
mod test {
    use expr::{Expr};
    use templates::FlagTemplate;
    use templates::Template;

    #[test]
    fn test_parity_flag() {
        let e1 = Expr::Reg("EAX".to_owned(), 32);
        let e2 = Expr::Reg("EAX".to_owned(), 32);
        let f_t = FlagTemplate::new(&[&e1, &e2]);
        let parity = f_t.parity_flag(&e1);
        println!("parity: {:?}", parity);

    }
}
