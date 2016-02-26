#![feature(test)]
#![allow(unused_variables)]
#![allow(dead_code)]
extern crate num;
extern crate rand;
extern crate capstone;
extern crate bit_vec;
extern crate bit_set;
extern crate libc;
extern crate time;
extern crate llvm_assemble;
extern crate z3;
extern crate itertools;
extern crate permutohedron;
extern crate synthir_execute as execute;

extern crate test;

#[macro_use]
pub mod utils;
pub mod op;
pub mod expr;
pub mod stmt;
pub mod emulator;
pub mod execution;
pub mod definitions;
pub mod native;
pub mod disassembler;
pub mod assembler;
//pub mod synth;
pub mod sample;
pub mod stochastic;
pub mod templates;
pub mod x86_64;
pub mod verify;
pub mod work;

mod test_stochastic {
    use num::bigint::ToBigUint;
    use std::collections::HashMap;

    use expr::Expr;
    use stochastic::{Stochastic};
    use verify::equal_or_counter;

    pub fn test_popcnt() {
        let eax = Expr::Reg("EAX".to_owned(), 32);
        let args = vec![eax.clone()];
        let mut io_sets = vec![];
        for &(res, dep) in &[(2, 0x11), (13, 0xfff001),
                             (2, 0x1002), (4, 0x10101010),
                             (32, 0xffff_ffffu32), (16, 0xaaAAaaAAu32),
                             (16, 0x55555555u32)]
        {
            let mut io_name = HashMap::new();
            io_name.insert(eax.clone(), dep.to_biguint().unwrap());
            io_sets.push((res.to_biguint().unwrap(), io_name));
        }
        let mut stoc = Stochastic::new(&args, &io_sets, 32);
        stoc.work();
    }

    pub fn test_sub_eax_1() {
        let eax = Expr::Reg("EAX".to_owned(), 32);
        let args = vec![eax.clone()];
        let mut io_sets = vec![];
        for &(res, dep) in &[(0x10, 0x11), (0xfff000, 0xfff001),
                             (0x1001, 0x1002), (0xffff, 0x10000),
                             (0xffFF_ffFFu32, 0x0)]
        {
            let mut io_name = HashMap::new();
            io_name.insert(eax.clone(), dep.to_biguint().unwrap());
            io_sets.push((res.to_biguint().unwrap(), io_name));
        }
        let mut res = Vec::new();
        for i in 0 .. 2 {
            let mut stoc = Stochastic::new(&args, &io_sets, 32);
            stoc.work();
            res.push(stoc.get_expr());
        }
        println!("res: {:#?}", res);
        let counter = equal_or_counter(&res[0], &res[1], 32);
        println!("counter: {:#?}", counter);
    }

    pub fn test_equal_expr() {
        use expr::{ExprType};
        use op::OpArith;
        let a = Expr::ArithOp(OpArith::SDiv,
                              Box::new(Expr::Int(10.to_biguint().unwrap())),
                              Box::new(Expr::Int(10.to_biguint().unwrap())),
                              ExprType::Int(1));
        let b = a.clone();
        let counter = equal_or_counter(&a, &b, 1);
        println!("counter: {:#?}", counter);
    }
}

// When you don't trust your programs, you verify them exhaustively
mod test_sub_eax_1 {
    use num::bigint::ToBigUint;
    use num::traits::ToPrimitive;
    use std::collections::HashMap;

    use op::OpLogic::*;
    use op::OpUnary::*;
    use expr::Expr::*;
    use emulator::{State, execute_expr};

    // LogicOp(Xor, Reg("EAX", 32), LogicOp(Xor, UnOp(Neg, Reg("EAX",
    // 32)), UnOp(Not, Reg("EAX", 32))))
    pub fn test_xor_neg_not() {
        let raw_eax = Reg("EAX".to_owned(), 32);
        let eax = Box::new(raw_eax.clone());
        let e = LogicOp(Xor,
                        eax.clone(),
                        Box::new(LogicOp(Xor,
                                         Box::new(UnOp(Neg, eax.clone(), 32)),
                                         Box::new(UnOp(Not, eax.clone(), 32)),
                                         32)),
                        32);
        for i in 0 .. 0xffFFffFFu32 {
            let mut map = HashMap::new();
            map.insert(raw_eax.clone(), i.to_biguint().unwrap());
            let state = State::borrow(&map);
            let res = execute_expr(&state, &e, 32);
            if let Ok(res) = res {
                if res.value().to_u32().unwrap() != (i - 1) {
                    println!("not eq: {}, expected: {}", res.value(), (i - 1));
                    break;
                }
            } else {
                println!("err");
                break;
            }
            if i % 0x10000 == 0 {
                println!("report: {:x}", i);
            }
        }
        println!("finished!");
    }
}

mod test_work {
    use disassembler::Disassemble;
    use work::{Work};
    use x86_64::X86_64;

    pub fn new_work() {
        Work::new(&X86_64);
    }

    pub fn inc_al() {
        // inc al
        // This is the code that facilitates the deps resolution
        // let ins = X86_64::disassemble(&[0xFE, 0xC0], 0x1000).unwrap();
        // println!("ins: {:?}", ins);

        // let mut deps = HashMap::new();
        // // deps.insert(Dep::new("OF").bit_width(1),
        // //             vec![Dep::new("AL").bit_width(8)]);
        // deps.insert(Dep::new("AL").bit_width(8),
        //             vec![Dep::new("AL").bit_width(8)]);

        // let x86_64 = X86_64;
        // let work = Work::new(&x86_64);

        // let res = work.get_io_sets(&ins, &deps).unwrap();

        // for (dep, ioset) in &res {
        //     work.gen_expr_from_io_set(&ins, dep, ioset);
        // }

        if let Some(ins) = X86_64::disassemble(&[0xFE, 0xC0], 0x1000)
        // inc rax
        //if let Some(ins) = X86_64::disassemble(&[0x48, 0xFF, 0xC0], 0x1000)
        // sub al
        //if let Some(ins) = X86_64::disassemble(&[0xFE, 0xC8], 0x1000)
        {
            println!("ins: {:?}", ins);

            let x86_64 = X86_64;
            let work = Work::new(&x86_64);
            let res = work.work_instruction(&ins);
            println!("res: {:?}", res);

        } else {
            println!("failed disassembling");
            return;
        }

    }

    pub fn add_eax_ebx() {
        if let Some(ins) = X86_64::disassemble(&[0x01, 0xD8], 0x1000)
        {
            println!("ins: {:?}", ins);

            let x86_64 = X86_64;
            let work = Work::new(&x86_64);
            let res = work.work_instruction(&ins);
            println!("res: {:?}", res);

        } else {
            println!("failed disassembling");
            return;
        }

    }

    // Compute the bitwise AND NOT of packed single-precision (32-bit)
    // floating-point elements in a and b, and store the results in
    // dst.
    pub fn vandnps() {
        if let Some(ins) =
            X86_64::disassemble(&[0xC5, 0xF4, 0x55, 0xC2], 0x1000)
        {
            println!("ins: {:?}", ins);

            let x86_64 = X86_64;
            let work = Work::new(&x86_64);
            let res = work.work_instruction(&ins);
            println!("res: {:?}", res);

        } else {
            println!("failed disassembling");
            return;
        }
    }

    // Add packed double-precision (64-bit) floating-point elements in
    // a and b, and store the results in dst.
    pub fn vxorpd() {
        if let Some(ins) =
            X86_64::disassemble(&[0xC5, 0xF5, 0x57, 0xC2], 0x1000)
        {
            println!("ins: {:?}", ins);

            let x86_64 = X86_64;
            let work = Work::new(&x86_64);
            let res = work.work_instruction(&ins);
            println!("res: {:?}", res);

        } else {
            println!("failed disassembling");
            return;
        }
    }

    // Add packed double-precision (64-bit) floating-point elements in
    // a and b, and store the results in dst.
    pub fn w_vaddps() {

        if let Some(ins) =
            X86_64::disassemble(&[0xC5, 0xF4, 0x58, 0xC2], 0x1000)
        {
            println!("ins: {:?}", ins);

            let x86_64 = X86_64;
            let work = Work::new(&x86_64);
            let res = work.work_instruction(&ins);
            println!("res: {:?}", res);

        } else {
            println!("failed disassembling");
            return;
        }
    }

    pub fn push_rax() {

        if let Some(ins) =
            X86_64::disassemble(&[0x50], 0x1000)
        {
            println!("ins: {:?}", ins);

            let x86_64 = X86_64;
            let work = Work::new(&x86_64);
            let res = work.work_instruction(&ins);
            println!("res: {:?}", res);

        } else {
            println!("failed disassembling");
            return;
        }
    }

    pub fn pop_rax() {

        if let Some(ins) =
            X86_64::disassemble(&[0x58], 0x1000)
        {
            println!("ins: {:?}", ins);

            let x86_64 = X86_64;
            let work = Work::new(&x86_64);
            let res = work.work_instruction(&ins);
            println!("res: {:?}", res);

        } else {
            println!("failed disassembling");
            return;
        }
    }

    pub fn cmp_rax_rbx() {

        if let Some(ins) =
            X86_64::disassemble(&[0x48, 0x39, 0xD8], 0x1000)
        {
            println!("ins: {:?}", ins);

            let x86_64 = X86_64;
            let work = Work::new(&x86_64);
            let res = work.work_instruction(&ins);
            println!("res: {:?}", res);

        } else {
            println!("failed disassembling");
            return;
        }
    }

    pub fn mul_rcx() {

        if let Some(ins) =
            X86_64::disassemble(&[0x48, 0xF7, 0xE1], 0x1000)
        {
            println!("ins: {:?}", ins);

            let x86_64 = X86_64;
            let work = Work::new(&x86_64);
            let res = work.work_instruction(&ins);
            println!("res: {:?}", res);

        } else {
            println!("failed disassembling");
            return;
        }
    }

    pub fn popcnt_rax_rcx() {

        if let Some(ins) =
            X86_64::disassemble(&[0xF3, 0x48, 0x0F, 0xB8, 0xC1], 0x1000)
        {
            println!("ins: {:?}", ins);

            let x86_64 = X86_64;
            let work = Work::new(&x86_64);
            let res = work.work_instruction(&ins);
            println!("res: {:?}", res);

        } else {
            println!("failed disassembling");
            return;
        }
    }

}

fn main() {
    println!("Hello, world!");
    //test_work::new_work();
    //test_work::inc_al();
    //test_work::add_eax_ebx();
    //test_work::vandnps();
    //test_work::vxorpd();
    //test_work::vaddps();
    //test_sub_eax_1::test_xor_neg_not();
    // for i in 0..2 {
    //     test_stochastic::test_sub_eax_1();
    // }
    //test_stochastic::test_sub_eax_1();
    //test_stochastic::test_equal_expr();
    //test_stochastic::test_popcnt();
    //test_work::push_rax();
    //test_work::pop_rax();
    //test_work::popcnt_rax_rcx();
    test_work::mul_rcx();
    //test_work::cmp_rax_rbx();
    //test_stochastic::mul_rcx();
    //test_work::w_vaddps();
}
