use std::fmt::{Display, Formatter, Error};

use assembler::Assemble;
use disassembler::Disassemble;

use definitions::{GenDefinition};

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

#[derive(Clone,Copy,Debug,PartialEq)]
pub enum OpndType {
    Reg,
    Label,
    Constant
}

#[derive(Clone,Debug,PartialEq)]
pub struct Opnd {
    len: Option<u32>,
    text: String,
    ty: OpndType
}

#[derive(Clone,Debug,PartialEq)]
pub struct Instruction {
    pub mnemonic: String,
    pub opnds: Vec<Opnd>,
}

impl Opnd {
    pub fn new(text: &str, len: Option<u32>, ty: OpndType) -> Opnd {
        Opnd { text: text.to_owned(), len: len, ty: ty }
    }
    pub fn get_text(&self) -> &str { &self.text }
    pub fn get_width(&self) -> Option<u32> { self.len }
    pub fn get_type(&self) -> OpndType { self.ty }
}

impl Instruction {
    pub fn new(mnemonic: &str, opnds: &[(&str, Option<u32>, OpndType)])
               -> Instruction
    {
        Instruction {
            mnemonic: mnemonic.to_owned(),
            opnds: opnds.into_iter()
                .map(
                    |&(s, len, ty)|
                    Opnd::new(s, len, ty))
                .collect()
        }
    }

    pub fn to_text(&self) -> String {
        let opnds: String = self.opnds.iter()
            .map(|s| s.get_text())
            .collect::<Vec<&str>>()
            .join(",");
        format!("{} {}", self.mnemonic, opnds)
    }

    pub fn assemble<T: Assemble>(&self) -> Option<Vec<u8>> {
        let s = self.to_text();
        T::assemble(&s)
    }

    pub fn get_opnds(&self) -> &[Opnd] { &self.opnds }
    pub fn get_mnemonic(&self) -> &str { &self.mnemonic }
}

impl Display for Instruction {
    fn fmt(&self, formatter: &mut Formatter) -> Result<(), Error> {
        let _ = formatter.write_str(&self.mnemonic);
        let _ = formatter.write_str(" ");
        for opnd in &self.opnds {
            let _ = formatter.write_str(opnd.get_text());
        }
        Ok(())
    }
}
