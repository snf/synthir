#![feature(test)]
#![allow(non_snake_case)]
extern crate test;
extern crate synthir;
extern crate num;

use num::bigint::ToBigUint;
use std::collections::HashMap;
use synthir::emulator::{State, execute_expr};
use synthir::expr::{Expr, ExprType};
use synthir::op::OpArith;
use synthir::sample::{RandExprSampler};
use test::{Bencher, black_box};

// Emulator
#[bench]
fn bench_Mul(b: &mut Bencher) {
    let v1 = 0x8000_0000u32.to_biguint().unwrap();
    let v2 = 0x10.to_biguint().unwrap();
    let map = HashMap::new();
    let state = State::borrow(&map);
    let e = Expr::ArithOp(OpArith::Mul,
                          Box::new(Expr::Int(v1)),
                          Box::new(Expr::Int(v2)),
                          ExprType::Int(64));

    b.iter(|| {
        black_box(execute_expr(&state, &e, 64).ok());
    });
}

#[bench]
fn bench_Sub_ovf(b: &mut Bencher) {
    let v1 = 1.to_biguint().unwrap();
    let v2 = 2.to_biguint().unwrap();
    let map = HashMap::new();
    let state = State::borrow(&map);
    let e = Expr::ArithOp(OpArith::Sub,
                          Box::new(Expr::Int(v1)),
                          Box::new(Expr::Int(v2)),
                          ExprType::Int(32));

    b.iter(|| {
        black_box(execute_expr(&state, &e, 32).ok());
    });
}

#[bench]
fn bench_Add_ovf_trim(b: &mut Bencher) {
    let v1 = 0x8000_0000u32.to_biguint().unwrap();
    let v2 = 0x8000_0000u32.to_biguint().unwrap();
    let map = HashMap::new();
    let state = State::borrow(&map);
    let e = Expr::ArithOp(OpArith::Add,
                          Box::new(Expr::Int(v1)),
                          Box::new(Expr::Int(v2)),
                          ExprType::Int(32));

    b.iter(|| {
        black_box(execute_expr(&state, &e, 32).ok());
    });
}

// Sampler
#[bench]
fn bench_sample1(b: &mut Bencher) {
    let r1 = Expr::Reg("EAX".to_owned(), 32);
    let r2 = Expr::Reg("EBX".to_owned(), 32);
    let mut sample = RandExprSampler::new(&[r1, r2]);
    b.iter(|| {
        black_box(sample.sample_expr_w(Some(32)));
    });
}
