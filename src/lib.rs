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
extern crate crossbeam;
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
pub mod sample;
pub mod stochastic;
pub mod templates;
pub mod x86_64;
pub mod verify;
pub mod work;
