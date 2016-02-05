use num::bigint::BigUint;
use num::bigint::ToBigUint;
use std::collections::HashMap;
use std::cmp::Ordering;
use std::fmt::{Debug, Formatter, Error};
use std::hash::{Hash, Hasher};

use definitions::{Definition, MemoryDefinition};
use execute::{Execute, PtraceExecute};

/// Defines dependencies
#[derive(Clone)]
pub struct Dep {
    base: &'static str,
    width: u32,
    mem: bool,
    mem_ip: bool,
    offset: Option<i32>
}
impl Dep {
    pub fn new(base: &'static str) -> Dep {
        Dep {
            base: base,
            width: 0,
            mem: false,
            mem_ip: false,
            offset: None
        }
    }
    pub fn bit_width(&mut self, width: u32) -> &mut Dep {
        self.width = width;
        self
    }
    pub fn byte_width(&mut self, width: u32) -> &mut Dep {
        self.width = width * 8;
        self
    }
    pub fn mem(&mut self, is: bool) -> &mut Dep {
        self.mem = is;
        self
    }
    pub fn mem_ip(&mut self, is: bool) -> &mut Dep {
        self.mem_ip = is;
        self
    }
    pub fn offset(&mut self, off: i32) -> &mut Dep {
        self.offset = Some(off);
        self
    }
    pub fn get_bit_width(&self) -> u32 { self.width }
    pub fn get_byte_width(&self) -> u32 { self.width / 8 }
    pub fn is_mem(&self) -> bool { self.mem }
    pub fn get_reg(&self) -> &'static str { self.base }
    pub fn get_mem(&self) -> u64 { panic!("deprecated!"); }
    pub fn get_offset(&self) -> i32 { self.offset.unwrap() }
    pub fn get_mem_ip(&self) -> bool { self.mem_ip }
}

impl Debug for Dep {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        if self.is_mem() {
            write!(f, "MEM[{}+{}]:{}", self.get_reg(),
                   self.get_offset(), self.get_bit_width())
        } else {
            write!(f, "{}:{}", self.get_reg(), self.get_bit_width())
        }
    }
}
impl PartialOrd for Dep {
    fn partial_cmp(&self, other: &Dep) -> Option<Ordering> {
        self.base.partial_cmp(other.base)
    }
}
impl Ord for Dep {
    fn cmp(&self, other: &Self) -> Ordering {
        self.base.cmp(other.base)
    }
}
impl PartialEq for Dep {
    fn eq(&self, other: &Dep) -> bool { self.base == other.base }
}
impl Eq for Dep {}
impl Hash for Dep {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write(self.base.as_bytes());
        let mem_u64 = if self.is_mem() {1} else {0};
        state.write_u64(mem_u64);
        state.finish();
    }
}

#[derive(Clone,PartialEq,Eq)]
pub struct Section {
    addr: u64,
    size: u32
}

impl Section {
    pub fn new(addr: u64, size: u32) -> Section {
        Section { addr: addr, size: size }
    }
    pub fn addr(&self) -> u64 { self.addr }
    pub fn size(&self) -> u32 { self.size }
}

/// Holds the execution results
#[derive(Clone)]
pub struct ExecutionRes<'a> {
    def: &'a Definition,
    ip: u64,
    regs_after: MemoryDefinition,
    others: Vec<(Section, MemoryDefinition)>,
    mem_deps: HashMap<Dep, Section>
}

impl<'a> ExecutionRes<'a> {
    pub fn new(def: &'a Definition, ip: u64,
               regs_after: MemoryDefinition,
               others: Vec<(Section, MemoryDefinition)>,
               mem_deps: &HashMap<Dep, Section>)
               -> ExecutionRes<'a>
    {
        ExecutionRes {
            def: def,
            ip: ip,
            regs_after: regs_after,
            others: others,
            mem_deps: mem_deps.clone()
        }
    }

    /// Get address of the pointer
    fn get_ptr_addr(&self, dep: &Dep) -> u64 {
        let section = &self.mem_deps[dep];
        section.addr
    }

    /// Get mem from Dep pointer of bit_width forward and backwards
    fn get_ptr_mem(&self, dep: &Dep) -> &MemoryDefinition {
        let section = &self.mem_deps[dep];
        &self.others
            .iter()
            .find(|&&(ref s, _)| s == section)
            .unwrap()
            .1
    }

    /// Get dep memory directly
    pub fn get_dep_memory(&self, dep: &Dep) -> &MemoryDefinition {
        assert!(dep.is_mem());
        self.get_ptr_mem(dep)
    }

    // XXX_ don't assume the 0x1000,
    // pub fn get_memory(&self, addr: u64) -> &MemoryDefinition {
    //     let mem = self.others.iter()
    //         .find(|a|
    //               a.0.addr <= addr &&
    //               (a.0.addr + 0x1000u64) > addr)
    //         .unwrap();
    //     &mem.1
    // }

    pub fn get_dep_value(&self, dep: &Dep) -> BigUint {
        if dep.is_mem() {
            self.def.bytes_to_biguint(self.get_dep_memory(dep).get_memory())
        } else {
            self.def.get_reg_value(dep.get_reg(), self.get_regs_after())
        }
    }

    pub fn get_regs_after(&self) -> &MemoryDefinition { &self.regs_after }
    pub fn get_ip(&self) -> u64 { self.ip }
}

/// Abstract Execution
#[derive(Clone)]
pub struct Execution<'a> {
    def: &'a Definition,
    regs_after: Section,
    regs_before: Section,
    regs_before_mem: MemoryDefinition,
    others_pair: Vec<(Section, MemoryDefinition)>,
    mem_deps: HashMap<Dep, Section>,
    code: Vec<(Section, MemoryDefinition)>,
    entry: u64
}

// impl<'a> Clone for Execution<'a> {
//     fn clone(&self) -> Execution<'a> {
//         Execution {
//             def: self.def,
//             regs_after: self.regs_after.clone(),
//             regs_before: self.regs_before.clone(),
//             regs_before_mem: self.regs_before_mem.clone(),
//             others_pair: self.others_pair.clone(),
//             code: self.code.clone(),
//             entry: self.entry
//         }
//     }
// }

impl<'a> Execution<'a> {
    pub fn new(def: &'a Definition) -> Execution<'a> {
        let regs_size = def.get_regs_size();
        let regs_before = Self::create_section(regs_size);
        let regs_after  = Self::create_section(regs_size);

        Execution {
            def: def,
            regs_after: regs_after,
            regs_before: regs_before,
            regs_before_mem: MemoryDefinition::new_with_size(regs_size),
            others_pair: Vec::new(),
            mem_deps: HashMap::new(),
            entry: 0,
            code: Vec::new()
        }
    }

    /// Set Dep as pointer creating an internal map to memory
    pub fn setup_ptr(&mut self, dep: &Dep) {
        let mem_size = dep.get_byte_width();
        let section = Self::create_section(mem_size);
        let mem_def = MemoryDefinition::new_with_size(mem_size);
        self.mem_deps.insert(dep.clone(), section.clone());
        self.others_pair.push((section, mem_def));
    }

    /// Does Dep has a pointer
    fn dep_has_ptr(&self, dep: &Dep) -> bool {
        self.mem_deps.contains_key(dep)
    }

    /// Get address of the pointer
    fn get_ptr_addr(&self, dep: &Dep) -> u64 {
        let section = &self.mem_deps[dep];
        section.addr
    }

    /// Get mem from Dep pointer of bit_width forward and backwards
    fn get_ptr_mem(&self, dep: &Dep) -> &MemoryDefinition {
        let section = &self.mem_deps[dep];
        &self.others_pair
            .iter()
            .find(|&&(ref s, _)| s == section)
            .unwrap()
            .1
    }

    /// Get mut mem from Dep pointer of bit_width forward and
    /// backwards
    fn get_mut_ptr_mem(&mut self, dep: &Dep) -> &mut MemoryDefinition {
        let section = &self.mem_deps[dep];
        &mut
            self.others_pair
            .iter_mut()
            .find(|&&mut(ref s, _)| s == section)
            .unwrap()
            .1
    }

    /// Get dep memory directly
    pub fn get_dep_memory(&self, dep: &Dep) -> &MemoryDefinition {
        assert!(dep.is_mem());
        self.get_ptr_mem(dep)
    }

    /// Get dep value
    pub fn get_dep_value(&self, dep: &Dep) -> BigUint {
        if dep.is_mem() {
            self.def.bytes_to_biguint(
                &self.get_dep_memory(dep)
                    .get_memory()[0 .. (dep.get_byte_width() as usize)])
        } else {
            self.def.get_reg_value(dep.get_reg(),
                                   self.get_regs_before_mem())
        }
    }

    /// Set register value
    pub fn set_reg_value(&mut self, reg: &str, val: &BigUint) {
        self.def.set_reg_value(reg, val, &mut self.regs_before_mem);
    }

    pub fn set_dep(&mut self, dep: &Dep, val: &BigUint) {
        if dep.is_mem() {
            // Set reg to pointer address
            let ptr_addr = if self.dep_has_ptr(dep) {
                self.get_ptr_addr(dep)
            } else {
                self.setup_ptr(dep);
                self.get_ptr_addr(dep)
            };
            let off = dep.get_offset();
            let ptr_addr = if off >= 0 {
                ptr_addr - (off as u64)
            } else {
                ptr_addr + (off.abs() as u64)
            }.to_biguint().unwrap();
            self.set_reg_value(dep.get_reg(), &ptr_addr);
            // Set pointer contents
            let bytes = self.def.biguint_to_bytes(val,
                                                  dep.get_byte_width());
            self.get_mut_ptr_mem(dep).memcpy(&bytes);
        } else {
            self.set_reg_value(dep.get_reg(), val);
        }
    }

    pub fn get_regs_before_mem(&self) -> &MemoryDefinition {
        &self.regs_before_mem
    }

    pub fn with_code(&mut self, code: &[u8]) {
        let code_mem = Self::create_section(code.len() as u32);
        self.entry = code_mem.addr();
        self.code.push((code_mem, MemoryDefinition::from_slice(code)));
    }

    pub fn new_section_with(&mut self, data: &[u8]) -> u64 {
        let data_mem = Self::create_section(data.len() as u32);
        let addr = data_mem.addr();
        let mut mem_def = MemoryDefinition::new_with_size(data.len() as u32);
        mem_def.memcpy(data);
        self.others_pair.push((data_mem, mem_def));
        addr
    }

    pub fn new_isolated_section(&mut self, len: u32) -> u64 {
        let data_mem = Self::create_isolated_section(len);
        let addr = data_mem.addr();
        let mem_def = MemoryDefinition::new_with_size(len);
        self.others_pair.push((data_mem, mem_def));
        addr
    }

    // pub fn get_mut_mem_for_addr(&mut self, addr: u64, len: u32) -> &mut [u8] {
    //     let mut section = self.others_pair.iter_mut()
    //         .find(|&&mut (ref s, _)|
    //               s.addr() <= addr &&
    //               (s.addr() + s.size() as u64) > addr)
    //         .unwrap();
    //     if (addr + len as u64) > (section.0.addr() + section.0.size() as u64) {
    //         panic!("address range too big");
    //     }
    //     let start = (addr - section.0.addr()) as usize;
    //     &mut section.1.get_memory_mut()[start .. start + len as usize]
    // }

    pub fn write_section_with(&mut self, addr: u64, data: &[u8]) {
        let mut section = self.others_pair.iter_mut()
            .find(|&&mut (ref s, _)|
                  s.addr() <= addr &&
                  (s.addr() + s.size() as u64) > addr)
            .unwrap();
        if data.len() > (section.0.size() as usize) {
            panic!("data.len() bigger than section.size()");
        }
        section.1.memcpy(data);
    }

    pub fn write_mem_with(&mut self, addr: u64, data: &[u8]) {
        self.write_section_with(addr, data);
    }

    pub fn regs_before_addr(&self) -> u64 {
        self.regs_before.addr()
    }

    pub fn regs_after_addr(&self) -> u64 {
        self.regs_after.addr()
    }

    fn create_section(size: u32) -> Section {
        let addr = PtraceExecute::create_section_rwx(size);
        Section::new(addr, size)
    }

    fn create_isolated_section(size: u32) -> Section {
        let addr = PtraceExecute::create_isolated_section_rwx(size);
        Section::new(addr, size)
    }

    pub fn execute<'b>(&self, def: &'b Definition) -> Option<ExecutionRes<'b>> {
        PtraceExecute::write_memory(self.regs_before.addr(),
                                    self.regs_before_mem.get_memory());

        // Write sections (including code sections)
        for &(ref section, ref mem_def) in
            self.others_pair.iter().chain(self.code.iter()) {
                PtraceExecute::write_memory(section.addr(),
                                            mem_def.get_memory());
        }

        // println!("executing");
        let res = PtraceExecute::execute_with_steps(self.entry,
                                                    self.def.prologue_steps,
                                                    self.def.epilogue_steps);
        // println!("finished");
        if res.is_ok() {
            let after = &self.regs_after;
            let mut registers_read = vec![0; after.size() as usize];
            PtraceExecute::read_child_memory(after.addr(),
                                             &mut registers_read,
                                             after.size());
            let memdef_after = MemoryDefinition::adopt_vec(registers_read);
            let mut others = Vec::new();
            for &(ref section, _) in self.others_pair.iter() {
                let mut new_other = vec![0; section.size() as usize];
                PtraceExecute::read_child_memory(section.addr(),
                                                   &mut new_other,
                                                   section.size());
                others.push((section.clone(),
                              MemoryDefinition::adopt_vec(new_other)));
            }
            PtraceExecute::dispose_execution();
            Some(ExecutionRes::new(def,
                                   res.unwrap(), memdef_after, others,
                                   &self.mem_deps))
        } else {
            PtraceExecute::dispose_execution();
            // println!("err: {:?}", res.err());
            None
        }
    }

}

/// Free all the sections on free
impl<'a> Drop for Execution<'a> {
    fn drop(&mut self) {
        // XXX_ how to manage this?, we have to keep the memory
        // allocated in case of clone() or reserve the memory every
        // time before executing
        // We are currently leaking here
        return;
        // for region in &self.code {
        //     PtraceExecute::dispose_section(region.addr(), region.size());
        // }
        // for region in &self.others {
        //     PtraceExecute::dispose_section(region.addr(), region.size());
        // }
        // PtraceExecute::dispose_section(self.regs_after.addr(),
        //                                self.regs_after.size());
        // PtraceExecute::dispose_section(self.regs_before.addr(),
        //                                self.regs_before.size());
    }
}
