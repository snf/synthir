use std::collections::{HashSet, HashMap};
use std::sync::{Mutex};
use num::traits::{One, Zero, ToPrimitive};
use num::bigint::{ToBigUint, BigUint};
use crossbeam;

use emulator::{State, execute_expr};
use expr::Expr;
use sample::{DepValueSampler};
use native::{Native, Opnd, Instruction};
use definitions::{Definition, GenDefinition};
use execution::{Execution, ExecutionRes, Dep};
use stochastic::Stochastic;
use templates::TemplateSearch;
use utils::{LastCache};
use verify::{equal_or_counter};

// XXX_ this file is probably competing with definition.rs for worst
// code ever (ever).

/// This will start with an assembler instruction and will try getting
/// to the IR transformation
pub struct Work<'a, T: 'a+Native> {
    arch: &'a T,
    def: Definition,
}

pub type IOSet<D, Val> = Vec<(HashMap<D, Val>, Val)>;
pub type IOSets<D, Val> = HashMap<D, IOSet<D, Val>>;

impl<'a, T: Native> Work<'a, T> {
    pub fn new(arch: &T) -> Work<T> {
        Work {
            arch: arch,
            def: T::gen_definition()
        }
    }

    /// Takes an instruction and from the arch definition, replaces
    /// every opnd that is not a register for a register of the same
    /// width as that opnd.
    fn replace_opnds_for_regs(&self, ins: &Instruction) -> Instruction {
        let mut used_ops: Vec<String> = Vec::new();
        let n_opnds = ins.opnds.iter().map(|opnd| {
            let opnd_text = opnd.text.to_uppercase();
            if self.def.has_reg(&opnd_text) {
                if used_ops.contains(&opnd_text) {
                    let avail_regs = self.def.get_reg_w_width(opnd.len);
                    let r = avail_regs.iter()
                        .filter(|r1| !used_ops.iter().any(|r2| &r2 == r1))
                        .nth(0).unwrap().clone();
                    used_ops.push(r.to_owned());
                    Opnd::new(r, opnd.len)
                } else {
                    used_ops.push(opnd_text);
                    opnd.clone()
                }
            } else {
                let avail_regs = self.def.get_reg_w_width(opnd.len);
                let r = avail_regs.iter()
                    .filter(|r1| !used_ops.iter().any(|r2| &r2 == r1))
                    .nth(0).unwrap().clone();
                used_ops.push(r.to_owned());
                Opnd::new(r, opnd.len)
            }
        }).collect();
        Instruction {
            mnemonic: ins.mnemonic.clone(), opnds: n_opnds
        }
    }

    /// Get all the registers that should be tested
    fn get_test_regs(&self, ins: &Instruction) -> Vec<&'static str> {
        let syntactic_deps = T::get_registers(ins);

        let mut regs_to_test: Vec<&str> = self.def
            .regs.iter()
            .filter(|&(k, v)| !v.flags)
            .filter(|&(k, v)|
                    !syntactic_deps.iter().any(|r| self.def.is_super_reg(k, r)))
            .map(|(&k, v)| k)
            .collect();
        let mut subregs_to_test: Vec<&str> = self.def
            .sub_regs.iter()
            .filter(|&(k, v)|
                    syntactic_deps.contains(k))
            .map(|(&k, v)| k)
            .collect();
        let mut flags_to_test: Vec<&str> = self.def
            .sub_regs.iter()
            .filter(|&(k, v)| self.def.get_reg(v.parent).flags)
            .map(|(&k, v)| k)
            .collect();
        let mut all_test_regs: Vec<&str> = Vec::new();
        all_test_regs.append(&mut regs_to_test);
        all_test_regs.append(&mut subregs_to_test);
        all_test_regs.append(&mut flags_to_test);
        all_test_regs.sort();
        all_test_regs
    }

    /// Get all the registers that are used as memory pointers
    fn get_mem_regs(&self, ins: &Instruction,
                    test_regs: &[&'static str])
                    -> Result<Vec<Dep>,()>
    {
        // Constants
        let max_mem_size: u32 = self.def.max_reg_width() * 10;

        // Run with all the registers pointer to unmapped memory and
        // try to force a fault. If this happens, it means that one or
        // more registers are being dereferenced so from now on, the
        // next steps will be used as Deref(Reg) instead of Reg.
        let mut exec = Execution::new(&self.def);
        let text_code = self.def.inst_to_text_code(ins, exec.regs_before_addr(),
                                                   exec.regs_after_addr());
        let bin_code = T::assemble(&text_code).unwrap();

        // Check if this instruction depends on memory (not null just in case)
        let unmapped = BigUint::one();
        exec.with_code(&bin_code);

        for reg in test_regs {
            exec.set_reg_value(reg, &unmapped);
        }
        let exec_res = exec.execute(&self.def);


        // If execution finished well, it doesn't depend on memory, or
        // not at least with the current configuration, otherwise map
        // all the regs and start unmapping one at a time until we get
        // which need to be mapped.
        let mut mem_regs: Vec<Dep> = Vec::new();

        if exec_res.is_none() {
            // Map a page
            let mapped = exec
                .new_isolated_section(0x1000)
                .to_biguint().unwrap();
            // But test with the regs pointint to the middle of it
            let mapped_start = &mapped + 0x800.to_biguint().unwrap();

            for reg in test_regs {
                exec.set_reg_value(reg, &mapped_start);
            }

            let exec_res = exec.execute(&self.def);
            if exec_res.is_none() {
                // XXX_ this is probably a `jmp reg` and should be
                // handled
                println!("here should only fail on jmp MEM[REG]");
                return Err(());
            }
            // Start unmapping one at a time
            for reg in test_regs {
                // Change reg and execute
                exec.set_reg_value(reg, &unmapped);
                if exec.execute(&self.def).is_none() {
                    let mut mem_reg = Dep::new(reg);
                    mem_reg.mem(true)
                        .bit_width(max_mem_size * 2)
                        .offset(-(max_mem_size as i32) * 10 / 8);
                    mem_regs.push(mem_reg);
                }
                // Restore reg
                exec.set_reg_value(reg, &mapped_start);
            }

            // Now check the bounds using the page boundaries for
            // checking it's size
            for mem in mem_regs.iter_mut() {
                let mut upper_bound = max_mem_size as i32;
                let mut lower_bound = max_mem_size as i32;

                for width in 0 .. upper_bound {
                    let mem_start = &mapped + (width).to_biguint().unwrap();
                    exec.set_reg_value(mem.get_reg(), &mem_start);
                    if exec.execute(&self.def).is_some() {
                        upper_bound = width;
                        break;
                    }
                }

                for width in 0 .. lower_bound {
                    let mem_start = &mapped + (0x1000 - width).to_biguint().unwrap();
                    exec.set_reg_value(mem.get_reg(), &mem_start);
                    if exec.execute(&self.def).is_some() {
                        lower_bound = width;
                        break;
                    } else {
                        //println!("adjusting width: {}", width);
                    }
                }

                let offset = -upper_bound.to_i32().unwrap();
                let len = (upper_bound + lower_bound).to_u32().unwrap();

                mem.offset(offset);
                mem.byte_width(len);
            }

        }
        mem_regs.sort();
        println!("mem regs: {:?}", mem_regs);
        Ok(mem_regs)
    }

    /// Get an Execution where the Dep result was different from the
    /// one described.
    fn get_exe_not_dep_res(&self,
                           cache: &mut LastCache<(Execution<'a>, ExecutionRes)>,
                           dep: &Dep, val: &BigUint)
                           -> Option<Execution>
    {
        cache
            .iter()
            .find(|&&(_, ref er)| &er.get_dep_value(dep) != val)
            .map(|&(ref e, _)| e.clone())
    }

    /// Get the register/memory dependencies of an instruction, it
    /// will try to inherit them, executing the instructions using
    /// different values for the registers.
    fn get_dependencies(&self, ins: &Instruction) ->
        Result<HashMap<Dep, Vec<Dep>>, ()>
    {

        let mut res: HashMap<Dep, Vec<Dep>> = HashMap::new();

        let all_test_regs = self.get_test_regs(ins);
        let mem_regs = try!(self.get_mem_regs(ins, &all_test_regs));

        // Create execution environment
        let mut exec = Execution::new(&self.def);
        let text_code = self.def.inst_to_text_code(ins, exec.regs_before_addr(),
                                                   exec.regs_after_addr());
        let bin_code = T::assemble(&text_code).unwrap();
        exec.with_code(&bin_code);

        // Initialize a cache to store the last N executions so we can
        // search them when we need values that modified a register in
        // a special way
        let mut cache: LastCache<(Execution, ExecutionRes)> = LastCache::new(2000);

        // ### Setup all the memory registers here #####
        let mem_regs_str: Vec<&'static str> = mem_regs.iter()
            .map(|d| d.get_reg()).collect();
        let mut all_test: Vec<Dep> = Vec::new();
        for reg in all_test_regs {
            // Avoid modifying it if it's a pointer
            if !mem_regs_str.contains(&reg) {
                let mut mem_reg = Dep::new(reg);
                mem_reg.bit_width(self.def.get_reg_width(reg));
                all_test.push(mem_reg);
            }
        }

        for mem_dep in mem_regs {
            exec.setup_ptr(&mem_dep);
            all_test.push(mem_dep);
        }
        // _____________________________________________

        println!("Test regs: {:?}", all_test);

        // Start changing the registers one at a time and getting
        // which registers are modified
        let iter_sampler = DepValueSampler::new(all_test.len());
        for mut round in iter_sampler {
            // Save before making any mod
            let saved_exec = exec.clone();

            // Modify regs
            for dep in &all_test
            {
                let width = dep.get_bit_width();
                let value = round.get_value(width);
                exec.set_dep(dep, &value);
            }

            // Execute
            let exe_res = exec.execute(&self.def);
            if exe_res.is_none() {
                // XXX_ handle errors here?, hmm, we failed
                println!("unhandled error");
                continue;
                return Err(());
            }
            let exe_res = exe_res.unwrap();
            let regs_after = exe_res.get_regs_after();
            let tmp_mod_regs = self.def.get_min_modified_regs(
                &exec.get_regs_before_mem(),
                &regs_after);

            // Extend the cache with this new case
            cache.push((exec.clone(), exe_res.clone()));

            // If a parent of this register has been already
            // identified, then merge it with it's parent.
            let mut mod_regs: Vec<Dep> = Vec::new();
            for m_reg in tmp_mod_regs.iter().cloned() {
                // Filter IP and parent Flags (we are only interested
                // in individual flags)
                if !self.def.is_ip(m_reg) &&
                    !self.def.is_flags(m_reg)
                {
                    let reg_to_add =
                    {
                        let mut curr = m_reg;
                        for r in res.keys() {
                            let r = r.get_reg();
                            if self.def.is_super_reg(r, curr) {
                                curr = r;
                            }
                        }
                        curr
                    };
                    let mut dep_to_add = Dep::new(reg_to_add);
                    dep_to_add.bit_width(self.def.get_reg_width(reg_to_add));
                    if !mod_regs.contains(&dep_to_add) {
                        mod_regs.push(dep_to_add);
                    }
                }
            }

            // Get memory modified
            let m_mems = all_test.iter().filter(|&d| d.is_mem());
            for mem in m_mems {
                let new_mem = exe_res.get_dep_value(&mem);
                let old_mem = exec.get_dep_value(&mem);
                if new_mem != old_mem {
                    mod_regs.push(mem.clone());
                }
                // println!("mem byte width: {:?}", mem.get_byte_width());
                // let mod_bytes = self.def.get_min_modified_len_bytes(
                //     new_mem,
                //     old_mem,
                //     mem.get_byte_width());
                // Get if there was a memory of more width already
                // saved before and use that one instead.
                // let mut found = false;
                // for t_mem in res.keys() {
                //     if t_mem.get_reg() == mem.get_reg() {
                //         let to_insert =
                //             if t_mem.get_byte_width() > mod_bytes {
                //                 t_mem.clone()
                //             } else {
                //                 let mut n_mem = mem.clone();
                //                 n_mem.byte_width(mod_bytes);
                //                 n_mem
                //             };
                //         found = true;
                //         mod_regs.push(to_insert);
                //     }
                // }
                // XXX_
                //assert_eq!(found, true);
            }

            println!("tmp_mod_regs: {:?}", tmp_mod_regs);
            println!("mod_regs: {:?}", mod_regs);

            // let mut rax = Dep::new("RAX");
            // let rax = rax.bit_width(64);
            // println!("RAX: 0x{:x}", exec.get_dep_value(rax).to_u64().unwrap());
            // println!("RAX after: 0x{:x}", exe_res.get_dep(rax).to_u64().unwrap());
            // let mut rsp_m = Dep::new("RSP");
            // let rsp_m = rsp_m.bit_width(64).mem(true);
            // println!("[RSP]: 0x{:x}", exec.get_dep_value(rsp_m).to_u64().unwrap());

            // For each modified value, start modifying one by one to
            // detect which ones modifies this register. If no
            // register modified it, flip the bits of the modified
            // register and repeat the process, if no deps are found
            // again, it means it only depends on itself.
            let saved_exe_res = &exe_res;
            for dep in &mod_regs {
                //println!("working with: {:?}", dep);
                let prev_val = saved_exe_res.get_dep_value(dep);
                let diff_exec = {
                    if let Some(exec) = self.get_exe_not_dep_res(
                        &mut cache,
                        dep,
                        &exe_res.get_dep_value(dep)) {
                        exec
                    } else {
                        println!("not found: {:?}", dep);
                        println!("dep value: {:?}", exe_res.get_dep_value(dep));
                        cache.last().0.clone()
                        //continue;
                    }
                };

                let mut found = false;

                // Save round and start again but with same values
                round.push();
                round.reset();
                // Modify one
                for m_dep in &all_test
                {
                    // Start with the last reg value and start
                    // modifying each one to match the previous state.
                    let mut curr_exec = exec.clone();

                    // println!("dependant: {:?}, modifying: {:?}",
                    //         reg, m_reg);
                    let value = diff_exec.get_dep_value(m_dep);
                    curr_exec.set_dep(m_dep, &value);

                    // Execute
                    let exe_res = curr_exec.execute(&self.def);

                    // XXX_ handle errors here?
                    if exe_res.is_none() {
                        println!("errrrrr");
                        continue;
                        return Err(());
                    }
                    let exe_res = exe_res.unwrap();
                    let curr_regs_after = exe_res.get_regs_after();
                    // Now compare reg value with the previous one
                    let curr_val = exe_res.get_dep_value(dep);
                    // println!("curr_val: {:?}, prev_val: {:?}",
                    //          curr_val, prev_val);
                    if curr_val != prev_val {
                        found = true;
                        // Add this register to the dependency graph
                        if !res.contains_key(dep) {
                            res.insert(dep.clone(), Vec::new());
                        }
                        println!("dep[{:?}].insert({:?})", dep, m_dep);
                        let this_dep = res.get_mut(dep).unwrap();
                        if !this_dep.contains(m_dep) {
                            this_dep.push(m_dep.clone());
                        }
                    }
                }
                // Restore round
                round.pop();

            }
            //let new_regs = self.def.normalize_regs(&regs_after);
            //println!("mod regs: {:?}", &mod_regs);
        }
        println!("res: {:?}", res);
        Ok(res)
    }

    /// Get I/O sets for the specified instruction that will be used
    /// to approximate the semantics
    fn get_io_sets(&self, ins: &Instruction,
                   deps: &HashMap<Dep, Vec<Dep>>)
                   -> Result<IOSets<Dep, BigUint>, ()>
    {
        // Initialize result store
        let mut io_sets: IOSets<Dep, BigUint> = HashMap::new();
        // Initialize execution environment
        let mut exec = Execution::new(&self.def);
        let text_code = self.def.inst_to_text_code(ins, exec.regs_before_addr(),
                                                   exec.regs_after_addr());
        let bin_code = T::assemble(&text_code).unwrap();
        exec.with_code(&bin_code);

        // Collect the I/O sets
        for (res, deps) in deps {
            // Prepare result Dep if it's a pointer
            if res.is_mem() {
                exec.setup_ptr(res);
                exec.set_dep(res, &BigUint::zero());
            }
            // Prepare result holders and value iterator
            let mut new_results = Vec::new();
            let deps_len = deps.len();
            let value_sampler = DepValueSampler::new(deps_len);
            for mut round in value_sampler {
                // Set the dependencies and store the input set
                let mut new_input_set = HashMap::new();
                for dep in deps {
                    let width = dep.get_bit_width();
                    let value = round.get_value(width);
                    // println!("dep: {:?}, value: {:?}", dep, value);
                    exec.set_dep(dep, &value);
                    new_input_set.insert(dep.clone(), value);
                }
                // Execute
                let exec_res = exec.execute(&self.def);
                // It can fail here due to other errors: alignment,
                // floating point, etc
                // It shouldn't fail here but fail gracefully :)
                if exec_res.is_none() {
                    // XXX_
                    println!("somehow failed here");
                    continue;
                    return Err(())
                }
                // Get the result
                let dep_res = {
                    let exec_res = exec_res.unwrap();
                    exec_res.get_dep_value(res)
                };
                // Store the result and add output set if it's unique
                let new = !new_results
                    .iter()
                    .filter(|&&(_, ref b)| b == &dep_res)
                    .map(|&(ref h, _)| h)
                    .any(|h: &HashMap<Dep, BigUint>| {
                        h
                            .iter()
                            .any(|(d, v)| new_input_set.get(d).unwrap() == v)
                    });
                if new {
                    new_results.push((new_input_set, dep_res));
                }
            }
            // Store the I/O sets
            io_sets.insert(res.clone(), new_results);
        }
        // XXX_ check if all the outputs were altered by the rounds
        // before returning
        Ok(io_sets)
    }

    /// Convert Dep to Expr
    fn dep_to_expr(&self, dep: &Dep) -> Expr {
        if dep.is_mem() {
            let mut s = String::new();
            s.push_str("M_");
            s.push_str(dep.get_reg());
            s.push_str("_");
            s.push_str(&dep.get_offset().to_string());
            Expr::Reg(s, dep.get_bit_width())
        } else {
            Expr::Reg(dep.get_reg().to_owned(), dep.get_bit_width())
        }
    }

    /// Convert Expr back to Dep
    fn expr_to_dep(&self, e: &Expr) -> Dep {
        if !e.is_reg() {
            panic!("Expr must be a Reg");
        }
        let s_t = e.get_reg_name();
        let w = e.get_width().unwrap();
        let mem = s_t.starts_with("M_");
        if mem {
            let mut splitted = s_t.split("_");
            let reg_name_t = splitted.nth(1).unwrap();
            let reg_name = self.def.regname_to_regname(reg_name_t);
            let offset = splitted.nth(0).unwrap().parse::<i32>().unwrap();
            let mut dep = Dep::new(reg_name);
            dep.mem(true).offset(offset).bit_width(w);
            dep
        } else {
            let s = self.def.regname_to_regname(s_t);
            let mut dep = Dep::new(s);
            dep.bit_width(w);
            dep
        }
    }

    /// Get all the Expr from an IOSet
    fn get_expr_ioset(&self, io_set: &IOSet<Dep, BigUint>)
                      -> Vec<Expr>
    {
        io_set
            .iter()
            .flat_map(
                |x| x.0
                    .keys()
                    .map(|k| self.dep_to_expr(k)))
            .collect()
    }

    /// Order the io_set by a tuple of (result, HashMap<var, value>)
    fn ioset_to_res_var_val(&self, io_set: &IOSet<Dep, BigUint>)
                            -> Vec<(BigUint, HashMap<Expr, BigUint>)>
    {
        io_set
            .iter()
            .map(|&(ref h, ref v)| {
                let mut hm = HashMap::new();
                let _: Vec<()> =
                    h.iter()
                    .map(|(d, dv)| {
                        hm.insert(self.dep_to_expr(&d), dv.clone());
                        ()
                    })
                    .collect();
                (v.clone(), hm)
            })
            .collect()
    }

    /// Stochastic search of the expression
    fn get_expr_stochastic(&self, ins: &Instruction,
                           dep: &Dep,
                           io_set: &IOSet<Dep, BigUint>)
                           -> Vec<Expr>
    {
        let res = Mutex::new(Vec::new());
        let expr_inits = self.get_expr_ioset(io_set);
        let io_set_e = self.ioset_to_res_var_val(io_set);
        //println!("io_set_e: {:?}", io_set_e);

        const MIN_EXPRS: usize = 1;
        crossbeam::scope(|scope| {
            for i in 0 .. MIN_EXPRS {
                scope.spawn(|| {
                    let mut stoc = Stochastic::new(&expr_inits,
                                                   &io_set_e,
                                                   dep.get_bit_width());
                    stoc.set_max_secs(600.0);
                    stoc.work();
                    if stoc.get_cost() == 0.0 {
                        let val = stoc.get_expr();
                        res.lock().unwrap().push(val);
                    }
                });
            }
        });
        res.into_inner().unwrap()
    }

    /// Template search of the expression
    fn get_expr_template(&self,
                         ins: &Instruction,
                         dep: &Dep,
                         io_set: &IOSet<Dep, BigUint>,
                         others: &[&Expr])
                         -> Vec<Expr>
    {
        let expr_inits_v = self.get_expr_ioset(io_set);
        let mut expr_inits: Vec<&Expr> =
            expr_inits_v.iter().collect();
        expr_inits.extend_from_slice(others);

        let io_set_e = self.ioset_to_res_var_val(io_set);

        let template = TemplateSearch::new(&expr_inits,
                                           &io_set_e,
                                           dep.get_bit_width());
        template.work()
    }

    /// Emulate all the I/O sets for an expression and return true if
    /// the expected outputs match with the obtained ones
    fn emulate_expected(&self,
                        e: &Expr,
                        width: u32,
                        io_set: &IOSet<Dep, BigUint>,
                        others: &[&Expr]) -> bool
    {
        let io_set_e = self.ioset_to_res_var_val(io_set);
        for (ref expected, ref io_set) in io_set_e {
            if let Ok(res) = execute_expr(&State::borrow(io_set), e, width) {
                if res.value() == expected {
                    continue;
                } else {
                    return false;
                }
            } else {
                return false;
            }
        }
        return true;
    }

    /// Emulate the Expr once and return the result
    fn emulate_once(&self,
                    e: &Expr,
                    width: u32,
                    inputs: &HashMap<Expr, BigUint>)
                    -> Result<BigUint, ()>
    {
        if let Ok(res) = execute_expr(&State::borrow(inputs), e, width) {
            Ok(res.value().clone())
        } else {
            Err(())
        }
    }

    /// Execute the Instruction once and return the result of the
    /// dependency needed here
    fn execute_once(&self, ins: &Instruction,
                    dep_res: &Dep,
                    inputs: &HashMap<Dep, BigUint>)
                    -> Result<BigUint, ()>
    {
        // Initialize execution environment
        let mut exec = Execution::new(&self.def);
        let text_code = self.def.inst_to_text_code(ins, exec.regs_before_addr(),
                                                   exec.regs_after_addr());
        let bin_code = T::assemble(&text_code).unwrap();
        exec.with_code(&bin_code);

        // Set the inputs
        for (dep, val) in inputs {
            if dep.is_mem() {
                exec.setup_ptr(dep);
                exec.set_dep(dep, &BigUint::zero());
            }
            exec.set_dep(dep, &val);
        }

        // Execute
        let exec_res = exec.execute(&self.def);
        if exec_res.is_none() {
            return Err(())
        }

        // Get the result
        let val_res = {
            let exec_res = exec_res.unwrap();
            exec_res.get_dep_value(dep_res)
        };
        Ok(val_res)
    }

    // fn emulate_execute_match(&self, ins: &Instruction,
    //                          dep: &Dep

    /// Verify and/or create new test cases
    /// 1) Get in pairs and verify
    /// 2) If equal, discard the larger
    /// 3) Execute and discard the one not corresponding
    /// 4) Add new I/O to ioset
    /// 5) Go to 1 until only one is left
    fn verify_contrast_reduce(&self, ins: &Instruction,
                              dep: &Dep,
                              io_set: &IOSet<Dep, BigUint>,
                              exprs: &[&Expr])
                              -> (Option<Expr>, IOSet<Dep, BigUint>)
    {
        let mut exprs = exprs.to_vec();
        let mut io_set = io_set.clone();

        while exprs.len() > 1 {
            let fst = exprs.remove(0);
            let snd = exprs.remove(0);
            println!("Verify new round\nfst: {:?}\nsnd: {:?}", fst, snd);

            if let Some(counter) =
                equal_or_counter(fst, snd, dep.get_bit_width())
            {
                println!("Different\nCounter: {:?}", counter);
                let mut counter_m: HashMap<Dep, BigUint> = HashMap::new();
                let _ = counter.iter()
                    .inspect(|&(k, v)| {
                        counter_m.insert(self.expr_to_dep(k), v.clone());
                    })
                    .count();
                if let Ok(ex_res) = self.execute_once(ins, dep, &counter_m) {
                    io_set.push((counter_m, ex_res.clone()));
                    let mut count = 0;
                    println!("real_res: {:?}", ex_res);
                    if let Ok(em_res) = self.emulate_once(
                        fst, dep.get_bit_width(), &counter)
                    {
                        println!("fst_res: {:?}", em_res);
                        if em_res == ex_res {
                            count += 1;
                            exprs.push(fst);
                        }
                    } else { panic!("couldn't emulate expr") }
                    if let Ok(em_res) = self.emulate_once(
                        snd, dep.get_bit_width(), &counter)
                    {
                        println!("snd_res: {:?}", em_res);
                        if em_res == ex_res {
                            count += 1;
                            exprs.push(snd);
                        }
                    } else { panic!("couldn't emulate expr") }
                    if count > 1 { panic!("solved failed us :(") }
                } else {
                    panic!(
                        "verify_contrast_reduce couldn't execute counter example");
                }
            } else {
                println!("Equal, removing larger");
                let good = if fst.get_size() < snd.get_size() { fst } else { snd };
                exprs.push(good);
            }
        }
        if exprs.is_empty() {
            (None, io_set.clone())
        } else {
            println!("verify_contrast_reduce res: {:?}", exprs.get(0));
            (Some(exprs.remove(0).clone()), io_set.clone())
        }
    }

    /// Synthetize the Expr store in the Expr
    /// 1) Use the templates
    /// 2) Use the stochastic approach with the cost function
    /// 3) Use the mixed template + stochastic
    fn synthetize(&self, ins: &Instruction,
                      dep: &Dep,
                      io_set: &IOSet<Dep, BigUint>,
                      others: &[&Expr])
                      -> Expr
    {
        let mut exprs: Vec<Expr> = Vec::new();

        let mut from_template = self.get_expr_template(ins, dep, io_set, others);
        println!("from template: {:?}", from_template);
        exprs.append(&mut from_template);

        let mut from_stochastic = self.get_expr_stochastic(ins, dep, io_set);
        println!("from stochastic: {:?}", from_stochastic);
        exprs.append(&mut from_stochastic);

        {
            let r_exprs: Vec<&Expr> = exprs.iter().map(|k| k).collect();
            let a = self.verify_contrast_reduce(ins, dep, io_set, &r_exprs);
        }
        if exprs.is_empty() {
            Expr::NoOp
        } else {
            exprs.remove(0)
        }
    }

    pub fn work_instruction(&self, ins: &Instruction)
                            -> Result<HashMap<Expr,Expr>, ()>
    {
        // Take the instruction and assemble it with proper arguments
        let n_ins = self.replace_opnds_for_regs(ins);
        println!("Instruction: {:?}", n_ins);

        // Get deps
        let deps = try!(self.get_dependencies(&n_ins));
        println!("Dependencies: {:?}", deps);

        // Get I/O sets
        let io_sets = try!(self.get_io_sets(&n_ins, &deps));
        // Pretty print, or try at least
        let _ = io_sets.iter()
            .inspect(|&(k, v)| {
                print!("{:?}: ", k);
                let _ = v.iter()
                    .inspect(|&&(ref ks, ref res)| {
                        println!("Res: {} ", res);
                        let _ = ks.iter()
                            .inspect(|&(k, v)| {
                                println!("\t{:?}: {}", k, v);
                            }).count();
                    }).count();

            }).count();

        // First sort the pairs by expressions that are bigger so
        // flags are calculated at the end
        let mut io_sets_vec: Vec<(&Dep, &IOSet<Dep, BigUint>)> =
            io_sets.iter().map(|(a, b)| (a, b)).collect();
         io_sets_vec.sort_by(|a, b| a.0.get_bit_width()
                            .cmp(&b.0.get_bit_width()).reverse());

        // Synthetize the expressions for each dep
        let mut progs = HashMap::new();
        for (dep, io_set) in io_sets_vec {
            println!("[+] Working Expr for {:?}", dep);
            let exprs = {
                let others: Vec<&Expr> = progs.values().collect();
                self.synthetize(&n_ins, &dep, &io_set, &others)
            };
            progs.insert(self.dep_to_expr(&dep), exprs);
        }
        println!("Programs: {:?}", progs);

        // XXX_ try to create a relation between the arguments and the
        // Expr involved

        println!("================######=============");

        Ok(progs)

    }

}
#[test]
fn test_get_io_sets() {
    use x86_64::X86_64;
    use disassembler::Disassemble;

    let arch = X86_64;
    let work = Work::new(&arch);

    // inc al
    let ins = X86_64::disassemble(&[0xFE, 0xC0], 0x1000).unwrap();
    println!("ins: {:?}", ins);

    let mut deps = HashMap::new();
    deps.insert(Dep::new("AL").bit_width(8).clone(),
                vec![Dep::new("AL").bit_width(8).clone()]);
    deps.insert(Dep::new("OF").bit_width(1).clone(),
                vec![Dep::new("AL").bit_width(8).clone()]);
    deps.insert(Dep::new("PF").bit_width(1).clone(),
                vec![Dep::new("AL").bit_width(8).clone()]);
    deps.insert(Dep::new("AF").bit_width(1).clone(),
                vec![Dep::new("AL").bit_width(8).clone()]);
    deps.insert(Dep::new("ZF").bit_width(1).clone(),
                vec![Dep::new("AL").bit_width(8).clone()]);

    let res = work.get_io_sets(&ins, &deps);

    //println!("res: {:?}", res);
    assert!(res.is_ok());
}

#[test]
fn test_get_expr_inc_al() {
    use x86_64::X86_64;
    use disassembler::Disassemble;

    let arch = X86_64;
    let work = Work::new(&arch);

    // inc al
    let ins = X86_64::disassemble(&[0xFE, 0xC0], 0x1000).unwrap();
    println!("ins: {:?}", ins);

    let mut deps = HashMap::new();
    // deps.insert(Dep::new("OF").bit_width(1),
    //             vec![Dep::new("AL").bit_width(8)]);
    deps.insert(Dep::new("AL").bit_width(8).clone(),
                vec![Dep::new("AL").bit_width(8).clone()]);
    let res = work.get_io_sets(&ins, &deps).unwrap();

    for (dep, ioset) in &res {
        // XXX_ enable me back
        // work.gen_expr_from_io_set(&ins, dep, ioset);
    }
    //println!("res: {:?}", res);
}

// Re-enable these anoying long tests
// #[test]
// fn test_mem_dependencies() {
//     use x86_64::X86_64;
//     use disassembler::Disassemble;

//     let arch = X86_64;
//     let work = Work::new(&arch);

//     // stosb = MOV byte ptr [RDI], AX
//     let ins = X86_64::disassemble(&[0xAA], 0x1000).unwrap();

//     println!("ins: {:?}", ins);
//     let res = work.get_dependencies(&ins);
// }
// #[test]
// fn test_normal_dependencies() {
//     use x86_64::X86_64;
//     use disassembler::Disassemble;

//     let arch = X86_64;
//     let work = Work::new(&arch);

//     // inc al
//     let ins = X86_64::disassemble(&[0xFE, 0xC0], 0x1000).unwrap();

//     println!("ins: {:?}", ins);
//     let res = work.get_dependencies(&ins);

//     // mov eax, eax
//     let ins = X86_64::disassemble(&[0x89, 0xC0], 0x1000).unwrap();

//     println!("ins: {:?}", ins);
//     let res = work.get_dependencies(&ins);
// }

#[test]
fn test_work() {
    use x86_64::X86_64;
    use disassembler::Disassemble;

    let arch = X86_64;
    let work = Work::new(&arch);
    let opnd1 = Opnd::new("eax", 32);
    let opnd2 = Opnd::new("eax", 32);
    let ins = Instruction {
        mnemonic: "mov".to_owned(),
        opnds: [opnd1, opnd2].to_owned()
    };

    // inc al
    let ins = X86_64::disassemble(&[0xFE, 0xC0], 0x1000).unwrap();

    // dec al
    let ins = X86_64::disassemble(&[0xFE, 0xC8], 0x1000).unwrap();

    println!("ins: {:?}", ins);
    // XXX_ enable me back
    // work.work_instruction(&ins);
}
