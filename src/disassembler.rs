use native::{Arch, Instruction};
use capstone as cs;

pub trait Disassemble {
    fn disassemble(bytes: &[u8], address: u64) -> Result<Instruction, ()>;

    fn get_registers(ins: &Instruction) -> Vec<&'static str>;

    /// Disassemble the bytes as if they were at the specified address
    /// for some arch.
    /// Return (mnemonic, opnd_str).
    fn disassemble_arch(arch: Arch, bytes: &[u8], address: u64)
                        -> Result<(String, String), ()>
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
        let e = try!(cs::Engine::new(cs_arch, cs_mode).map_err(|e| ()));
        let mut insns = try!(e.disasm(bytes, address, 1).map_err(|e| ()));
        if insns.len() != 1 {
            panic!("Should be only one instruction at a time");
        }
        let insn = insns.remove(0);
        Ok((insn.mnemonic, insn.op_str))
    }
}

pub fn disassemble<T: Disassemble>(_ignore: &T, bytes: &[u8], address: u64)
                                   -> Result<Instruction, ()>
{
        T::disassemble(bytes, address)
}
