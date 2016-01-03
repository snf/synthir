use std::collections::HashMap;
use std::fmt::{Display, Formatter, Error};
use std::slice::SliceConcatExt;

use assembler::Assemble;
use disassembler::Disassemble;

use definitions::{GenDefinition, MemoryDefinition, Definition};
//use execute::{Execute, PtraceExecute};

#[allow(non_camel_case_types)]
#[derive(Clone,Copy,Debug)]
pub enum Arch {
    X86,
    X86_64,
    Arm,
    Arm64,
    Thumb,
    Mips
}

/// The trait that has to be implemented for running the synthetizer
pub trait Native: GenDefinition + Assemble + Disassemble { }
impl<T> Native for T where T: GenDefinition + Assemble + Disassemble {}

#[derive(Clone, Debug)]
pub struct Opnd {
    pub len: u32,
    pub text: String
}

#[derive(Clone, Debug)]
pub struct Instruction {
    pub mnemonic: String,
    pub opnds: Vec<Opnd>,
}

impl Opnd {
    pub fn new(text: &str, len: u32) -> Opnd {
        Opnd { text: text.to_owned(), len: len }
    }
    pub fn text(&self) -> &str {
        &self.text
    }
}

impl Instruction {
    pub fn new(mnemonic: &str, opnds: &[(&str, u32)]) -> Instruction
    {
        Instruction {
            mnemonic: mnemonic.to_owned(),
            opnds: opnds.into_iter()
                .map(
                    |&(s, len)|
                    Opnd::new(s, len))
                .collect()
        }
    }

    pub fn to_text(&self) -> String {
        let opnds: String = self.opnds.iter()
            .map(|s| s.text())
            .collect::<Vec<&str>>()
            .join(",");
        format!("{} {}", self.mnemonic, opnds)
    }

    pub fn assemble<T: Assemble>(&self) -> Option<Vec<u8>> {
        let s = self.to_text();
        T::assemble(&s)
    }

}

impl Display for Instruction {
    fn fmt(&self, formatter: &mut Formatter) -> Result<(), Error> {
        let _ = formatter.write_str(&self.mnemonic);
        let _ = formatter.write_str(" ");
        for opnd in &self.opnds {
            let _ = formatter.write_str(opnd.text());
        }
        Ok(())
    }
}


// fn get_dependencies() {
//     let deps = X86::gen_definition();
//     let regs_mem = MemoryDefinition::new_with_size(deps.regs_size);

//     //let dep_regs: Vec<&str> = Vec::new();
//     let dep_regs: HashMap<&str, Vec<&str>> = HashMap::new();
//     let mod_regs: Vec<&str> = Vec::new();
//     // The algorithm is:

//     // Change one register at a time and check if the output changes.
//     // If it changes, I add it to the possible list of sources. Also
//     // check for the modifications and diff at the bit granularity.

//     //
// }
