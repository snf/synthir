use std::collections::{HashMap, HashSet};
use std::borrow::Borrow;
use num::traits::{One};
use num::bigint::{BigUint};

use bit_set::BitSet;

use native::Instruction;

#[derive(Clone)]
pub struct MemoryDefinition {
    memory: Vec<u8>
}

#[derive(Clone)]
pub struct RegDefinition {
    // The offset in the memory definition
    pub offset: u32,
    // The bit width (32bits: width=32)
    pub width: u32,
    pub vector: bool,
    pub sp: bool,
    pub fp: bool,
    pub ip: bool,
    pub segment: bool,
    // If it's a flag, only analyze sub regs
    pub flags: bool,
    pub sub_regs: Vec<&'static str>
}

impl RegDefinition {
    pub fn bit_off(&self) -> u32 {
        self.offset * 8
    }

    // XXX_ check this function
    pub fn bit_off_inside(&self, bit_off: u32) -> bool {
        if bit_off >= self.bit_off() &&
           bit_off <  self.bit_off() + self.width {
               true
           } else {
               false
           }
    }
    pub fn bit_range(&self) -> (u32, u32) {
        (self.bit_off(), self.bit_off() + self.width)
    }

    pub fn byte_width(&self) -> u32 { self.width / 8 }
    pub fn bit_width(&self) -> u32 { self.width }
    pub fn offset(&self) -> u32 { self.offset }
    pub fn is_vector(&self) -> bool { self.vector }
    pub fn is_fp(&self) -> bool { self.fp }
    pub fn is_ip(&self) -> bool { self.ip }
    pub fn is_sp(&self) -> bool { self.sp }
    pub fn is_segment(&self) -> bool { self.segment }
    pub fn is_flags(&self) -> bool { self.flags }
}

#[derive(Clone)]
pub struct SubRegDefinition {
    pub parent: &'static str,
    pub width: u32,
    // Least significant bit, ex: EAX in RAX: from_bit = 0, width = 32
    pub from_bit: u32,
}

impl SubRegDefinition {
    pub fn bit_off_in(&self, bit_off: u32, parent: &RegDefinition) -> bool {
        // Return if doesn't belong to this parent
        if !parent.bit_off_inside(bit_off) {
            return false;
        }

        // Substract from parent off
        let bit_off = bit_off - parent.bit_off();

        // XXX_ hack for x86, move to Definition and use self.endian
        // Convert bit_off to little endian
        let parent_width = parent.width;
        let parent_width_bytes = parent.width / 8;

        // 16 width:
        // 2 bwidth:
        // (7)  00000001 00000000 => 0
        // (8)  00000000 10000000 => 15
        // (15) 00000000 00000001 => 8

        let mut real_off = 0;
        for i in 0 .. parent_width_bytes {
            let start_width = i * 8;
            let end_width = i * 8 + 7;
            if  bit_off >= start_width &&
                bit_off <= end_width
            {
                // 0 to 8 = 0 to 8
                real_off = start_width + end_width - bit_off;
                //debugln!("bit_off: {}, real: {}", bit_off, real_off);
            }
        }

        // debugln!("parent: off: {}", parent.bit_off());
        // debugln!("child: off: {}", self.from_bit);

        if real_off >= self.from_bit &&
           real_off <  self.from_bit + self.width {
               //debugln!("found!!!");
               true
           } else {
               false
           }
    }
    pub fn bit_width(&self) -> u32 { self.width }
    pub fn from_bit(&self) -> u32 { self.from_bit }
    pub fn get_parent(&self) -> &'static str { self.parent }
}

#[derive(Clone)]
pub enum Endianness {
    Little,
    Big
}

#[derive(Clone)]
pub struct Definition {
    pub regs_size: u32,
    pub base_reg: &'static str,
    pub regs: HashMap<&'static str, RegDefinition>,
    pub sub_regs: HashMap<&'static str, SubRegDefinition>,
    pub epilogue: &'static str,
    pub epilogue_steps: u32,
    pub prologue: &'static str,
    pub prologue_steps: u32,
    pub endian: Endianness
}

impl MemoryDefinition {
    pub fn new_with_size(size: u32) -> MemoryDefinition {
        MemoryDefinition {
            memory: vec![0; size as usize] }
    }

    pub fn adopt_vec(memory: Vec<u8>) -> MemoryDefinition {
        MemoryDefinition {
            memory: memory }
    }

    pub fn from_slice(memory: &[u8]) -> MemoryDefinition {
        MemoryDefinition {
            memory: memory.to_owned() }
    }

    pub fn get_reg_value(&self, reg: &RegDefinition) -> &[u8] {
        let start = reg.offset as usize;
        let end = (reg.offset + reg.byte_width()) as usize;
        &self.memory[start..end]
    }

    pub fn get_memory(&self) -> &[u8] {
        &self.memory[..]
    }

    pub fn get_memory_mut(&mut self) -> &mut [u8] {
        &mut self.memory[..]
    }

    /// Get byte offsets of the modifications
    pub fn get_byte_diff(&self, other: &MemoryDefinition) -> Vec<u32> {
        let mut v = Vec::new();
        for i in 0 .. self.memory.len() {
            //debugln!("[{}] = {}", i, other.memory[i]);
            if self.memory[i] != other.memory[i] {
                v.push(i as u32)
            }
        }
        v
    }

    pub fn get_bit_diff(&self, other: &MemoryDefinition) -> Vec<u32> {
        let mut v = Vec::new();
        let mut orig_set = BitSet::from_bytes(&self.memory[..]);
        let other_set = BitSet::from_bytes(&other.memory[..]);
        // This method sets the bits that are different
        orig_set.symmetric_difference_with(&other_set);
        // Save the bit's being set
        for off in &orig_set {
            v.push(off as u32);
        }
        v
    }

    /// Memset
    pub fn fill_with(&mut self, val: u8) {
        for i in 0 .. self.memory.len() {
            self.memory[i] = val;
        }
    }

    /// Memcpy
    pub fn memcpy(&mut self, mem: &[u8]) {
        for (i, v) in mem.iter().enumerate() {
            self.memory[i] = *v;
        }
    }
}

impl Definition {
    /// Get regs size
    pub fn get_regs_size(&self) -> u32 { self.regs_size }

    /// Get the regs' HashMap
    pub fn get_regs(&self) -> &HashMap<&'static str, RegDefinition> {
        &self.regs
    }

    /// Get the subregs' HashMap
    pub fn get_sub_regs(&self)
                        -> &HashMap<&'static str, SubRegDefinition>
    {
        &self.sub_regs
    }

    // XXX_ implement me in Definition
    // pub fn sys_width(&self) -> u32 { 64 }

    // XXX_ ugly hack for avoiding annotating Dep with lifetime params
    /// Get an &'static str from an &str if it's inside known regs
    pub fn regname_to_regname(&self, name: &str) -> &'static str {
        self.regs.keys()
            .chain(self.sub_regs.keys())
            .find(|&n| n == &name)
            .unwrap()
    }

    /// Get the max width of a register
    pub fn max_reg_width(&self) -> u32 {
        self.regs.values()
            .map(|v| v.bit_width())
            .max()
            .unwrap()
    }

    /// Get the width of the instruction pointer
    pub fn get_mem_width(&self) -> u32 {
        self.regs.values()
            .find(|v| v.ip)
            .unwrap()
            .bit_width()
    }

    /// Is Parent reg?
    pub fn is_parent(&self, child: &str, parent: &str) -> bool {
        if !self.regs.contains_key(parent) {
            false
        } else if self.regs[parent].sub_regs.contains(&child)
            && self.regs[parent].flags == false {
            true
        } else {
            false
        }
    }

    /// Is Super reg? (does this reg contains the other one)
    pub fn is_super_reg(&self, reg: &str, other: &str) -> bool {
        let super_regs = self.get_super_regs(other);
        if super_regs.contains(&reg) {
            true
        } else {
            false
        }
    }

    /// Get all super regs (regs that contains this reg)
    pub fn get_super_regs(&self, child: &str) -> Vec<&'static str> {
        let mut res = Vec::new();
        if let Some(parent) = self.get_parent_reg(child) {
            res.push(parent);
            let child_def = &self.sub_regs[child];
            for reg in &self.regs[parent].sub_regs {
                let reg_def = &self.sub_regs[reg];
                if child_def.from_bit() <= reg_def.from_bit() &&
                    reg_def.bit_width() + reg_def.from_bit() >
                    child_def.bit_width() + child_def.from_bit()
                {
                    res.push(reg);
                }
            }
        }
        res
    }

    /// Is Instruction Pointer?
    pub fn is_ip(&self, name: &str) -> bool {
        self.regs.contains_key(name) && self.regs[name].ip
    }

    /// Is flags (parent)
    pub fn is_flags(&self, name: &str) -> bool {
        if self.regs.contains_key(name) {
            self.regs[name].flags
        } else {
            false
        }
    }

    /// Is flag (parent or sub)
    pub fn is_flag(&self, name: &str) -> bool {
        if self.regs.contains_key(name) {
            self.regs[name].flags
        } else if self.sub_regs.contains_key(name) {
            self.regs[self.get_parent_reg(name).unwrap()]
                .flags
        } else {
            false
        }
    }

    /// Get RegDefinition for reg name, if this is not a parent reg or
    /// it doesn't exist, it will panic.
    pub fn get_reg(&self, name: &str) -> &RegDefinition {
        &self.regs[name]
    }

    pub fn get_parent_reg(&self, name: &str) -> Option<&'static str> {
        if self.regs.contains_key(name) {
            None
        } else if self.sub_regs.contains_key(name) {
            let parent = self.sub_regs[name].parent;
            if self.regs[parent].flags {
                None
            } else {
                Some(self.sub_regs[name].parent)
            }
        } else {
            panic!("always check the register exists")
        }
    }

    pub fn has_reg(&self, name: &str) -> bool {
        let upname = name.to_uppercase();
        let upn: &str = &upname;
        self.regs.contains_key(upn) || self.sub_regs.contains_key(upn)
    }

    pub fn get_reg_width(&self, name: &str) -> u32 {
        let upname = name.to_uppercase();
        let upn: &str = &upname;
        if self.regs.contains_key(upn) {
            self.regs[upn].width
        } else if self.sub_regs.contains_key(upn) {
            self.sub_regs[upn].width
        } else {
            panic!(format!("reg: {} is not here, you should have checked", name))
        }
    }

    pub fn get_reg_w_width(&self, width: u32) -> HashSet<&str> {
        let mut regs = HashSet::new();
        for (name, reg) in &self.regs {
            if reg.width == width {
                regs.insert(*name);
            }
        }
        for (name, reg) in &self.sub_regs {
            if reg.width == width {
                regs.insert(*name);
            }
        }
        regs
        //panic!(format!("couldn't find a reg of width: {}", width));
    }

    fn get_reg_for_bits(&self, bit_off: u32) -> Vec<&str> {
        let mut regs: Vec<&str> = Vec::new();
        let reg_name = self.find_reg_for_bits(bit_off);
        let reg = &self.regs[reg_name];
        let reg_bit_off = reg.bit_off();
        regs.push(reg_name);
        let _ = reg.sub_regs
            .iter()
            .map(|r| (r, &self.sub_regs[r]))
            .filter(|&(name, r)| r.bit_off_in(bit_off, reg))
            .inspect(|&(name, r)| regs.push(name));
        regs
    }

    fn get_bits_for_reg(&self, reg: &str) -> (u32, u32) {
        self.regs[reg].bit_range()
    }

    fn get_regs_modified(&self, diff: u64) -> Vec<&str> {
        // XXX
        let mut res = Vec::new();
        for (name, reg) in &self.regs {
            res.push(name.borrow());
        }
        res
    }

    // XXX_ return Option<&str>?
    pub fn find_reg_for_bits(&self, bit_off: u32) -> &'static str {
        for (name, reg) in &self.regs {
            if reg.bit_off_inside(bit_off) {
                return name;
            }
        }
        // XXX_ too bad, panic!
        //""
        panic!("bit outside this definition");
    }

    /// Find Sub Regs that match bit offset in the register's memory map
    pub fn find_sregs_for_bits(&self, bit_off: u32) -> Vec<&'static str> {
        let mut res = Vec::new();
        for (name, reg_def) in &self.sub_regs {
            let parent = &self.regs[reg_def.parent];
            let parent_bit_off = parent.bit_off();
            // debugln!("parent: name: {}", reg_def.parent);
            // debugln!("child: name: {}", name);
            if reg_def.bit_off_in(bit_off, parent) {
                res.push(&name[..]);
            }
        }
        res
    }

    pub fn bytes_to_biguint(&self, bytes: &[u8]) -> BigUint {
        match self.endian {
            Endianness::Little => BigUint::from_bytes_le(bytes),
            Endianness::Big => BigUint::from_bytes_be(bytes)
        }
    }

    pub fn biguint_to_bytes(&self, val: &BigUint, byte_width: u32) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(byte_width as usize);
        let val =
            if val.bits() * 8 > (byte_width as usize) {
                val & ((BigUint::one() << ((byte_width as usize) * 8))
                       - BigUint::one())
            } else {
                val.clone()
            };
        match self.endian {
            Endianness::Little => {
                let mut val_bytes = val.to_bytes_le();
                bytes.append(&mut val_bytes);
                let to_fill = (byte_width as usize) - bytes.len();
                for b in 0 .. to_fill {
                    bytes.push(0);
                }
            },
            Endianness::Big    => {
                let mut val_bytes = val.to_bytes_be();
                let to_fill = (byte_width as usize) - bytes.len();
                for b in 0 .. to_fill {
                    bytes.push(0);
                }
                bytes.append(&mut val_bytes);
            }
        }
        bytes
    }

    pub fn get_reg_value(&self, reg: &str, mem: &MemoryDefinition) -> BigUint {
        let bytes = mem.get_memory();
        if self.regs.contains_key(reg) {
            let reg_def = &self.regs[reg];
            // Always assume parent regs are 8bit multiple
            assert!(reg_def.bit_width() % 8 == 0);
            let value = self.bytes_to_biguint(
                &bytes[reg_def.offset() as usize
                      ..
                      (reg_def.offset()+reg_def.byte_width()) as usize]);
            value
        } else if self.sub_regs.contains_key(reg) {
            let sub_reg = &self.sub_regs[reg];
            let par_reg = &self.regs[sub_reg.parent];
            let par_reg_value = self.get_reg_value(sub_reg.parent, mem);
            (par_reg_value >> (sub_reg.from_bit() as usize)) &
                ((BigUint::one() << (sub_reg.bit_width() as usize))
                 - BigUint::one())
        } else {
            panic!(format!("This reg is not registered in this definition: {:?}", reg))
        }
    }

    pub fn set_reg_value(&self, reg: &str, val: &BigUint,
                         mem: &mut MemoryDefinition)
    {
        if self.regs.contains_key(reg) {
            let reg = &self.regs[reg];
            let bytes = self.biguint_to_bytes(val, reg.byte_width());
            let mut memory = mem.get_memory_mut();
            for (i, &byte) in bytes.iter().enumerate() {
                let mem_offset = (reg.offset() as usize)+ i;
                memory[mem_offset] = byte;
            }
        } else if self.sub_regs.contains_key(reg) {
            let sub_reg = &self.sub_regs[reg];
            let val =
                if val.bits() > (sub_reg.bit_width() as usize) {
                    val & ((BigUint::one() << (sub_reg.bit_width() as usize))
                           - BigUint::one())
                } else {
                    val.clone()
                };
            let par_reg = &self.regs[sub_reg.parent];
            let reg_val = self.get_reg_value(sub_reg.parent, mem);
            let parent_bits = (BigUint::one() << (par_reg.bit_width() as usize))
                - BigUint::one();
            let masked_bits = ((BigUint::one() << (sub_reg.bit_width() as usize))
                               - BigUint::one()) << (sub_reg.from_bit() as usize);
            let masked_parent = reg_val & (parent_bits ^ masked_bits);

            let sub_reg_val = val << (sub_reg.from_bit() as usize);
            let parent_val = masked_parent | sub_reg_val;
            self.set_reg_value(sub_reg.parent, &parent_val, mem);
        } else {
            panic!("This reg is not registered in this definiiton");
        }
    }

    // XXX_ check this function :D
    /// Get the minimal length of bytes (starting from 0) modified in
    /// a memory region
    pub fn get_min_modified_len_bytes(&self, mem_before: &MemoryDefinition,
                                      mem_after: &MemoryDefinition,
                                      len: u32) -> u32
    {
        let bytes_before = &mem_before.get_memory()[0..len as usize];
        let bytes_after = &mem_after.get_memory()[0..len as usize];
        assert_eq!(bytes_before.len(), bytes_after.len());
        let mut diff = Vec::new();
        let zip_bytes = bytes_before.iter().zip(bytes_after.iter());
        for (i, (byte1, byte2)) in zip_bytes.enumerate() {
            if byte1 != byte2 {
                diff.push(i as u32);
            }
        }
        diff.into_iter()
            .min()
            .unwrap_or(len)
    }

    /// Return a new vector containing only the main regs avoiding the
    /// children
    // XXX_ make this more efficient, this sucks
    pub fn filter_children_regs(&self, regs: &[&'static str])
                                -> Vec<&'static str>
    {
        let mut final_vec = regs.to_owned();
        loop {
            let mut vec = Vec::new();
            let mut changed = false;
            for r in &final_vec {
                let srs = self.get_super_regs(r);
                // Compiler complains that it's uninitialized
                // 2hard4rust
                let mut insert_reg = *r;
                if srs.is_empty() {
                    insert_reg = *r;
                } else {
                    for sr in srs {
                        if final_vec.contains(&sr) {
                            changed = true;
                            insert_reg = sr;
                            break;
                        }
                    }
                    if changed == false {
                        insert_reg = r;
                    }
                }
                if !vec.contains(&insert_reg) {
                    vec.push(insert_reg);
                }
            }
            final_vec = vec.to_owned();
            if changed == false {
                break;
            }
        }
        final_vec
    }

    /// Walk the bits getting which are the minimal modified registers
    pub fn get_min_modified_regs(&self, regs_before: &MemoryDefinition,
                                 regs_after: &MemoryDefinition)
                                 -> Vec<&'static str>
    {
        let mut res = Vec::new();
        let bit_diff = regs_before.get_bit_diff(regs_after);
        for diff in bit_diff {
            let parent_reg = self.find_reg_for_bits(diff);
            let sub_regs = self.find_sregs_for_bits(diff);
            let to_insert =
                &if sub_regs.is_empty() {
                    parent_reg
                } else {
                    // Find which is the minimium modified register
                    sub_regs
                        .into_iter()
                        .min_by_key(|&name| self.sub_regs[name].bit_width())
                        .unwrap()
                };
            if !res.contains(to_insert) {
                res.push(to_insert);
            }
        }
        self.filter_children_regs(&res)
    }

    pub fn inst_to_text_code(&self, ins: &Instruction,
                             regs_before: u64, regs_after: u64) -> String
    {
        // Unify it with the epilogue and prologue
        let mut s = "".to_owned();
        s = s + &self.prologue.replace("%0", &regs_before.to_string());
        s = s + ";";
        s = s + &ins.to_text();
        s = s + ";";
        s = s + &self.epilogue.replace("%0", &regs_after.to_string());
        s
    }
}

// This is the definition for each platform
pub trait GenDefinition {
    fn gen_definition() -> Definition;
}

// Tests
#[test]
fn test_def_reg_value_little() {
    use num::bigint::ToBigUint;
    let mut regs = HashMap::new();
    let mut sub_regs = HashMap::new();
    regs.insert("EAX",
                RegDefinition {
                    offset: 0,
                    width: 32,
                    vector: false,
                    fp: false,
                    ip: false,
                    segment: false,
                    flags: false,
                    sub_regs: vec!["AX"]
                });
    sub_regs.insert("AX",
                    SubRegDefinition {
                        parent: "EAX",
                        width: 16,
                        from_bit: 0
                    });
    regs.insert("EFLAGS",
                RegDefinition {
                    offset: 4,
                    width: 32,
                    vector: false,
                    fp: false,
                    ip: false,
                    segment: false,
                    flags: true,
                    sub_regs: vec!["PF"]
                });
    sub_regs.insert("PF",
                    SubRegDefinition {
                        parent: "EFLAGS",
                        width: 2,
                        from_bit: 1
                    });
    let def = Definition {
        regs_size: 64,
        base_reg: "EAX",
        regs: regs,
        sub_regs: sub_regs,
        epilogue: "",
        epilogue_steps: 0,
        prologue: "",
        prologue_steps: 0,
        endian: Endianness::Little
    };

    let mut mem = MemoryDefinition::adopt_vec(vec![0x01, 0x02, 0x03, 0x04,
                                                   0x06, 0x00, 0x00, 0x00]);

    // Test from vec
    let eax = def.get_reg_value("EAX", &mem);
    assert_eq!(eax, 0x04030201.to_biguint().unwrap());

    let ax = def.get_reg_value("AX", &mem);
    assert_eq!(ax, 0x0201.to_biguint().unwrap());

    let eflags = def.get_reg_value("EFLAGS", &mem);
    assert_eq!(eflags, 0x06.to_biguint().unwrap());

    let pf = def.get_reg_value("PF", &mem);
    assert_eq!(pf, 0x03.to_biguint().unwrap());

    // Test get_reg after set_reg with main Regs
    def.set_reg_value("EAX",
                      &0xAABBCCDDu32.to_biguint().unwrap(),
                      &mut mem);
    let eax = def.get_reg_value("EAX", &mem);
    assert_eq!(eax, 0xAABBCCDDu32.to_biguint().unwrap());

    def.set_reg_value("EAX",
                      &0xAABBu32.to_biguint().unwrap(),
                      &mut mem);
    let eax = def.get_reg_value("EAX", &mem);
    assert_eq!(eax, 0xAABBu32.to_biguint().unwrap());

    def.set_reg_value("EAX",
                      &0xCCDD0000u32.to_biguint().unwrap(),
                      &mut mem);
    let eax = def.get_reg_value("EAX", &mem);
    assert_eq!(eax, 0xCCDD0000u32.to_biguint().unwrap());

    // Test get_reg after set_reg with SubReg
    def.set_reg_value("EAX",
                      &0xBBBBBBBBu32.to_biguint().unwrap(),
                      &mut mem);
    def.set_reg_value("AX",
                      &0xAA.to_biguint().unwrap(),
                      &mut mem);
    let eax = def.get_reg_value("EAX", &mem);
    assert_eq!(eax, 0xBBBB00AAu32.to_biguint().unwrap());

    def.set_reg_value("PF",
                      &1.to_biguint().unwrap(),
                      &mut mem);
    let eflags = def.get_reg_value("EFLAGS", &mem);
    assert_eq!(eflags, 0x2.to_biguint().unwrap());

    def.set_reg_value("PF",
                      &3.to_biguint().unwrap(),
                      &mut mem);
    let eflags = def.get_reg_value("EFLAGS", &mem);
    assert_eq!(eflags, 0x6.to_biguint().unwrap());
}
