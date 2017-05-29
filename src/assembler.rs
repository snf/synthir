use native::{Arch};
use keystone as k_a;

pub trait Assemble {
    fn assemble(text: &str) -> Option<Vec<u8>>;

    /// Assemble text to byte arrays
    fn assemble_text(arch: Arch, text: &str) -> Option<Vec<u8>> {
        let (n_arch, n_bits) = match arch {
            Arch::X86_64 => (k_a::Arch::X86, k_a::MODE_64),
            Arch::X86 => (k_a::Arch::X86, k_a::MODE_32),
            Arch::Arm => (k_a::Arch::ARM, k_a::MODE_32),
            Arch::Arm64 => (k_a::Arch::ARM64, k_a::MODE_64),
            Arch::Mips => (k_a::Arch::MIPS, k_a::MODE_32),
            _ => panic!("not supported")
        };
        let engine = k_a::Keystone::new(n_arch, n_bits)
            .expect("Could not initialize Keystone engine");
        engine.option(k_a::OptionType::SYNTAX, k_a::OPT_SYNTAX_INTEL)
            .expect("Could not set option to nasm syntax");
        let res = engine.asm(text.to_string(), 0);
        match res {
            Ok(asm) => Some(asm.bytes),
            Err(_) => None
        }
    }
}

#[test]
fn int3() {
    use x86_64::X86_64;
    use assembler::Assemble;
    assert_eq!(X86_64::assemble("int3").unwrap(), [0xcc]);
}

#[test]
fn nop_nop() {
    use x86_64::X86_64;
    use assembler::Assemble;
    assert_eq!(X86_64::assemble("nop; nop").unwrap(), [0x90, 0x90]);
}
