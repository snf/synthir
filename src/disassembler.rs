use native::{Arch, Instruction};
use capstone as cs;

pub trait Disassemble {
    fn disassemble(bytes: &[u8], address: u64) -> Option<Instruction>;

    fn get_registers(ins: &Instruction) -> Vec<&'static str>;

    /// Disassemble the bytes as if they were at the specified address
    /// for some arch.
    /// Return (mnemonic, opnd_str).
    fn disassemble_arch(arch: Arch, bytes: &[u8], address: u64)
                        -> Option<(String, String)>
    {
        let cs_arch = match arch {
            Arch::X86 | Arch::X86_64 => cs::Arch::X86,
            Arch::Arm | Arch::Arm64 => cs::Arch::Arm,
            _ => panic!("not supported")
        };
        let cs_mode = match arch {
            Arch::X86 | Arch::Arm => cs::MODE_32,
            Arch::X86_64 | Arch::Arm64 => cs::MODE_64,
            _ => panic!("not supported")
        };
        match cs::Engine::new(cs_arch, cs_mode) {
            Ok(e) => {
                match e.disasm(bytes, address, 1) {
                    Ok(mut insns) => {
                        if insns.len() != 1 {
                            panic!("Should be only one instruction at a time");
                        }
                        let insn = insns.remove(0);
                        Some((insn.mnemonic, insn.op_str))
                        // let opnds: Vec<&str> =
                        //     insn.op_str.split(',')
                        //     .collect();
                        // Some(Instruction::new(Arch::X86_64,
                        //                       &insn.mnemonic,
                        //                       opnds)
                        //     )
                    },
                    Err(err) => {
                        //panic!("# Engine::disasm failed: {:?} {:?}", err.code, err.desc);
                        None
                    }
                }},
            Err(err) => {
                panic!("#Engine::new failed: {:?} {:?}", err.code, err.desc);
            }
        }
    }
}
