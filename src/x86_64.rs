use assembler::Assemble;
use disassembler::Disassemble;
use native::{Arch, Instruction};
//use definitions::{RegDefinition, SubRegDefinition, Definition, GenDefinition};

//use std::collections::HashMap;

#[allow(non_camel_case_types)]
pub struct X86_64;

impl X86_64 {
    fn get_opnd_width(opnd: &str) -> u32 {
        let definition = Self::gen_definition();
        if definition.has_reg(opnd) {
            definition.get_reg_width(opnd)
        } else {
            // Parse it, it's (probably?) a memory pointer
            if opnd.starts_with("dword ptr") {
                32
            } else if opnd.starts_with("qword ptr") {
                64
            } else if opnd.starts_with("byte ptr") {
                8
            } else {
                panic!(format!("opnd: `{}` not supported", opnd))
            }
        }
    }
}

impl Assemble for X86_64 {
    fn assemble(text: &str) -> Option<Vec<u8>> {
        Self::assemble_text(Arch::X86_64, text)
    }
}

impl Disassemble for X86_64 {
    // XXX_ this could be improved, making Instruction a better
    // representation of the operands in a generic way so it's parsed
    // once by `disassemble` instead of every time it's needed by some
    // other function.
    fn get_registers(ins: &Instruction) -> Vec<&'static str> {
        let mut res = Vec::new();
        let def = Self::gen_definition();
        for opnd in &ins.opnds {
            let t = opnd.text().to_uppercase();
            // Try to match Reg or SubReg
            for &reg_name in def.regs.keys().chain(def.sub_regs.keys()) {
                // XXX_ contains can match name parts (AX in EAX)
                // if t.contains(reg_name) {
                if t == reg_name {
                    res.push(reg_name);
                }
            }
        }
        res
    }

    fn disassemble(bytes: &[u8], address: u64) -> Option<Instruction> {
        let (mnemonic, op_str) =
            Self::disassemble_arch(Arch::X86_64, bytes, address).unwrap();

        let opnds: Vec<(&str, u32)> =
            if op_str.trim().is_empty() {
                Vec::new()
            } else {
                op_str.split(',')
                    .map(|s| s.trim())
                    .map(|s| (s, Self::get_opnd_width(s)))
                    .collect()
            };
        Some(Instruction::new(&mnemonic,
                              &opnds))

    }
}

include!("x86_64_gen.rs");

#[test]
fn test_disassemble() {
    let out = X86_64::disassemble(&[0xff, 0x25, 0x42, 0x7e, 0x21, 0x00], 0x1000);
    println!("{:?}", out);
    let out = X86_64::disassemble(&[0x48, 0x89, 0xE0], 0x1000);
    println!("{:?}", out);
    let out = X86_64::disassemble(&[0x89, 0xC0], 0x1000);
    println!("{:?}", out);
    let out = X86_64::disassemble(&[0xAA], 0x1000);
    println!("{:?}", out);
    let out = X86_64::disassemble(&[0x48, 0xA3, 0x00, 0x90, 0x9A,
                                    0x93, 0xE6, 0x7F, 0x00, 0x00], 0x1000);
    println!("{:?}", out);

}

#[test]
fn test_assemble() {
    let out = X86_64::assemble("mov eax, 0x10");
    //println!("{:?}", out);
    assert_eq!(true, out.is_some());
    // Test the pre defined code
    let def = X86_64::gen_definition();
    // Epilogue
    let epilogue = def.epilogue;
    let parsed_epilogue = epilogue.replace("%0", "0x40ffff");
    let asm_epilogue = X86_64::assemble(&parsed_epilogue);
    //println!("{:?}", asm_epilogue);
    assert_eq!(true, asm_epilogue.is_some());
    // Prologue
    let prologue = def.prologue;
    let parsed_prologue = prologue.replace("%0", "0x40ffff");
    let asm_prologue = X86_64::assemble(&parsed_prologue);
    //println!("{:?}", asm_prologue);
    assert_eq!(true, asm_prologue.is_some());
}
