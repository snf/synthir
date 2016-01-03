use native::{Arch, Instruction};
use llvm_assemble as l_a;

pub trait Assemble {
    fn assemble(text: &str) -> Option<Vec<u8>>;

    /// Assemble text to byte arrays
    fn assemble_text(arch: Arch, text: &str) -> Option<Vec<u8>> {
        let n_arch = match arch {
            Arch::X86_64 => l_a::Arch::X86_64,
            Arch::X86 => l_a::Arch::X86,
            Arch::Arm => l_a::Arch::Arm,
            Arch::Arm64 => l_a::Arch::Arm64,
            Arch::Mips => l_a::Arch::Mips,
            _ => panic!("not supported")
        };
        l_a::assemble(n_arch, text)
    }
}

#[test]
fn int3() {
    use llvm_assemble::{Arch, assemble};
    assert_eq!(assemble(Arch::X86, "int3").unwrap(), [0xcc]);
}

#[test]
fn nop_nop() {
    use llvm_assemble::{Arch, assemble};
    assert_eq!(assemble(Arch::X86_64, "nop; nop").unwrap(), [0x90, 0x90]);
}
