use emulator::{State, execute_expr};
use expr::{Expr, EPoints, traverse_get_points};
use sample::{RandExprSampler};
use utils::{Hamming};

use num::bigint::BigUint;
use num::traits::{Float, ToPrimitive};
use rand::{Rng, ThreadRng, thread_rng};
use rand::distributions::{Sample, Weighted, WeightedChoice};
use std::collections::HashMap;
use time::precise_time_s;

/// Different transformations for applying to the Expr
#[derive(Debug,Clone,Copy,PartialEq,Eq,Hash)]
enum Mov {
    /// Insert in the leftmost, rightmost or midlevel of the tree a
    /// new expression
    Insert,
    /// Replace one operation in the tree with one of the same type
    ReplaceOp,
    /// Replace the type of the expression
    ReplaceType,
    /// Replace one expression in the tree with one of the same type
    ReplaceExpr,
    /// Swap an argument expression in an argument with left or right
    Swap,
    /// Remove one expression of the middle
    Remove
}

const MAX_TRIES: u64 = 0x10000000;
const MAX_SECS: f64 = 6000.0;
const REPORT_SECS: f64 = 30.0;
const REPORT_SECS_U64: u64 = REPORT_SECS as u64;
const REPORT_TRIES: u64 = 500;

/// Stochastic algorithm to try to find the operation
pub struct Stochastic {
    io_sets: Vec<(BigUint,HashMap<Expr, BigUint>)>,
    beta: f64,
    curr_cost: f64,
    prev_expr: Expr,
    curr_expr: Expr,
    rng: ThreadRng,
    expr_sampler: RandExprSampler,
    moves_weight: HashMap<Mov, u32>,
    expr_width: u32,
    max_secs: f64,
    max_tries: u64,
}

impl Stochastic {
    pub fn new(args: &[Expr],
               io_sets: &[(BigUint,HashMap<Expr, BigUint>)],
               expr_width: u32)
               -> Stochastic
    {
        let mut moves = HashMap::new();
        moves.insert(Mov::Insert, 1);
        moves.insert(Mov::ReplaceOp, 4);
        moves.insert(Mov::ReplaceType, 6);
        moves.insert(Mov::ReplaceExpr, 1);
        moves.insert(Mov::Swap, 5);
        moves.insert(Mov::Remove, 4);
        Stochastic {
            io_sets: io_sets.to_vec(),
            beta: 0.2,
            curr_cost: 10000.0,
            prev_expr: Expr::NoOp,
            curr_expr: Expr::NoOp,
            rng: thread_rng(),
            expr_sampler: RandExprSampler::new(args),
            moves_weight: moves,
            expr_width: expr_width,
            max_secs: MAX_SECS,
            max_tries: MAX_TRIES
        }
    }

    /// Get cost
    pub fn get_cost(&self) -> f64 { self.curr_cost }

    /// Adjust the max amount of seconds it must run
    pub fn set_max_secs(&mut self, max_secs: f64) {
        self.max_secs = max_secs;
    }

    /// Calculate the new cost of the expression
    fn calc_cost(&self, e: &Expr) -> f64 {
        let cost = Cost::new(&self.io_sets, e, self.expr_width);
        cost.cost()
    }

    /// Get next move to make, always return `ReplaceExpr` if we are
    /// still uninitialized (NoOp)
    fn choose_mov(&mut self) -> Mov {
        if self.curr_expr == Expr::NoOp {
            return Mov::ReplaceExpr;
        }
        let mut weighted: Vec<Weighted<Mov>> =
            self.moves_weight
            .iter()
            .map(|(k, v)| Weighted { item: *k, weight: *v })
            .collect();
        let mut weighted_choice = WeightedChoice::new(&mut weighted);
        weighted_choice.sample(&mut self.rng)
    }

    /// Get possible points to make each modification
    fn get_rand_mov_point(&mut self, mov: Mov) -> Vec<u8> {
        let mut epoints = EPoints::new();
        traverse_get_points(&mut epoints, &self.curr_expr);
        // XXX_ implement me
        //println!("poi: {:?}", epoints);
        let possible_points = match mov {
            Mov::Insert => epoints.get_expr(),
            Mov::ReplaceOp => epoints.get_op(),
            Mov::ReplaceType => epoints.get_ty(),
            Mov::ReplaceExpr => epoints.get_expr(),
            Mov::Swap => epoints.get_bin(),
            Mov::Remove => epoints.get_expr()
        };
        //println!("pos_poi: {:?}", possible_points);
        if let Some(point) = self.rng.choose(&possible_points) {
            point.to_vec()
        } else {
            Vec::new()
        }
    }

    /// Transform the expr one mov already sampled
    fn transform_expr(&mut self, mov: Mov) -> Expr {
        let point = self.get_rand_mov_point(mov);

        let mut transform = Transform::new(
            &mut self.expr_sampler,
            mov,
            &point,
            self.expr_width,
            &mut self.rng);

        transform.run(&self.curr_expr)
    }

    #[inline]
    fn report(&self, tries: u64) {
        println!("Current cost: {}", self.curr_cost);
        println!("Current expr: {:?}", self.curr_expr);
        println!("Tries since last report: {}", tries);
        println!("Exprs/min: {}", (tries * 60/ REPORT_SECS_U64));
    }

    /// All the work
    pub fn work(&mut self) {
        let start = precise_time_s();
        let mut last_report = start;
        let mut tries = 0;
        let mut last_tries = 0;
        loop {
            // One more try
            tries += 1;
            // One more step and update if this is good
            if let Some((e, cost)) = self.work_step() {
                // self.prev_expr = self.curr_expr.clone();
                self.curr_expr = e;
                self.curr_cost = cost;
            }
            // Break if we reached minimal cost
            if self.curr_cost == 0.0 {
                self.report(tries - last_tries);
                break;
            }
            if tries % REPORT_TRIES == 0 {
                let now = precise_time_s();
                // Report every REPORT_SECS
                if (now - last_report) >= REPORT_SECS {
                    let diff = now - last_report - REPORT_SECS;
                    if diff > 5.0 {
                        println!("Something is bad, can't take more than 5 secs to hit the report, diff: {}", diff);
                    }
                    self.report(tries - last_tries);
                    last_report = now;
                    last_tries = tries;
                }
                // Break if max tries
                if tries > self.max_tries {
                    break;
                }
                // Break if we reached the limit
                if now > (self.max_secs + start) {
                    break;
                }
            }
        }
        if self.curr_cost == 0.0 {
            // Try to clean the Expr while not changing the cost
            for i in 0 .. 100000 {
                let e = self.transform_expr(Mov::Remove);
                if self.calc_cost(&e) == 0.0 &&
                    e.get_size() < self.curr_expr.get_size()
                {
                    self.curr_expr = e;
                }
            }
            println!("cleaned: {:?}", self.curr_expr);
        }
    }

    /// Advance one step and return the new expr with the new cost in
    /// case it progressed
    pub fn work_step(&mut self) -> Option<(Expr, f64)> {
        let mov = self.choose_mov();
        let new_expr = self.transform_expr(mov);
        if new_expr == self.curr_expr {
            return None
        }
        //println!("new expr: {:?}", new_expr);

        // cost() - ln(p)/b
        let p = self.rng.gen_range::<f64>(0.0, 1.0);
        let beta = self.beta;
        let max = self.curr_cost - (p.ln() / beta);
        //println!("max: {}", max);

        let new_cost = self.calc_cost(&new_expr);
        //println!("new cost: {}", new_cost);

        if new_cost > max {
            None
        } else {
            //println!("accepting new cost!");
            Some((new_expr, new_cost))
        }
    }

    /// Get the current expression
    pub fn get_expr(&self) -> Expr { self.curr_expr.clone() }
}

const MISALIGN_PENALTY: f64 = 1f64;

/// A temporal object that will be used to calculate the different
/// costs of the new expression.
struct Cost<'a>{
    io_sets: &'a [(BigUint,HashMap<Expr, BigUint>)],
    expr: &'a  Expr,
    width: u32
}

impl<'a> Cost<'a> {
    fn new(io_sets: &'a [(BigUint,HashMap<Expr, BigUint>)],
           expr: &'a Expr,
           width: u32)
           -> Cost<'a>
    {
        Cost {
            io_sets: io_sets,
            expr: expr,
            width: width
        }
    }

    /// Are we using all the dependencies?, if not, increase the cost
    fn deps_cost(&self) -> u32 {
        let mut cost = 0;
        let regs = self.expr.get_regs();
        let derefs = self.expr.get_derefs();
        let deps: Vec<Expr> = self.io_sets
            .iter()
            .flat_map(|&(_, ref hm)|
                 hm
                      .iter()
                      .map(|(e, _)| e.clone()))
            .collect();
        for dep in &deps {
            if !regs.contains(&dep) && !derefs.contains(&dep) {
                cost += 2;
            }
        }
        cost
    }

    /// How close are we to the desired result (matching bits)
    fn hamming_distance(&self, a: &BigUint, b: &BigUint) -> u32 {
        let a_h = a.hamming_weight();
        let b_h = b.hamming_weight();
        if a_h > b_h {
            a_h - b_h
        } else {
            b_h - a_h
        }
        //let diff = a ^ b;
        //diff.hamming_weight()
    }

    fn expr_len(&self) -> u32 {
        self.expr.get_size()
    }

    // XXX_ decide
    /// Get the distance from one value to another
    fn int_distance(&self, a: &BigUint, b: &BigUint) -> f64 {
        let diff =
            if a == b {
                0.0
            } else {
                let diff =
                    if a > b {
                        a - b
                    } else {
                        b - a
                    };
                if let Some(diff) = diff.to_f64() {
                    // println!("diff: {}", diff);
                    // println!("ln_diff: {}", diff.ln());
                    diff
                } else {
                    100.0
                }
            };
        Float::min(4.0, diff)
    }

    fn calc_res_arith_float_distance(&self) -> f64 {
        0.0
    }

    /// Return a high cost for expressions that crash, we should not
    /// accept them
    fn crash_cost(&self) -> f64 {
        10000.0
    }

    /// Execute with an I/O set and calculate the cost
    fn execute_once(&self,
                    io_set: &HashMap<Expr, BigUint>)
                    -> Result<BigUint,()>
    {
        let state = State::borrow(io_set);
        Ok(
            // XXX_ self.width * 2 just for testing
            try!(execute_expr(&state, self.expr, self.width * 2)).value().clone())
    }

    /// Default cost for hamming
    fn default_hamming(&self) -> f64 {
        // XXX_ play with this const
        4f64 * ((self.width / 8) as f64)
    }

    /// Calculate cost of this expression
    fn cost(&self) -> f64 {
        let mut cost: f64 = 0.0;
        cost += self.deps_cost().to_f64().unwrap();
        for &(ref res, ref io_set) in self.io_sets {
            let io_res = self.execute_once(io_set);
            if let Ok(io_res) = io_res {
                cost += Float::min(
                    self.default_hamming(),
                    self.hamming_distance(res, &io_res).to_f64().unwrap()
                );
                if res != &io_res {
                    cost += MISALIGN_PENALTY;
                }
                cost += self.int_distance(res, &io_res);
            } else {
                return self.crash_cost();
            }
        }
        // Size penalty
        let size = self.expr.get_size();
        if size > 70 {
            cost += (size - 70).to_f64().unwrap();
        }
        cost
    }
}

/// Helper to transform one expression into another with specific moves
struct Transform<'a> {
    sampler: &'a mut RandExprSampler,
    trans: Mov,
    pos: &'a [u8],
    width: u32,
    rng: &'a mut ThreadRng
}

impl<'a> Transform<'a> {
    /// Create new Transform
    fn new(sampler: &'a mut RandExprSampler,
           trans: Mov, pos: &'a[u8], width: u32,
           rng: &'a mut ThreadRng)
           -> Transform<'a>
    {
        Transform {
            sampler: sampler,
            trans: trans,
            pos: pos,
            width: width,
            rng: rng
        }
    }

    /// Run the Transform once and return result
    fn run(&mut self, e: &Expr) -> Expr {
        // println!("mov: {:?}", self.trans);
        // println!("pos: {:?}", self.pos);
        // println!("expr: {:?}", e);
        // If mov is not possible, return the same expr
        if self.pos.is_empty() {
            e.clone()
        } else {
            let mut trans_e = e.clone();
            self.transform_op3(&mut trans_e, 0);
            trans_e
        }
    }

    fn transform_insert(&mut self, e: &mut Expr) {
        *e = {
            // Rust complains about `values differ in mutability` if a
            // Vec from e is created directly.
            let i_e: &Expr = e;
            let mut leaf = vec![i_e];
            self.sampler.sample_expr_w_with_leafs(&mut leaf, None)
        };
    }

    fn transform_replace_expr(&mut self, e: &mut Expr) {
        *e = {
            let mut leafs = e.get_leafs();
            self.sampler.sample_expr_w_with_leafs(&mut leafs, None)
        };
    }

    fn transform_remove(&mut self, e: &mut Expr) {
        let leaf = {
            let leafs = e.get_leafs();
            if leafs.is_empty() {
                self.sampler.sample_expr_w(e.get_width())
            } else {
                let choice = self.rng.gen_range(0, leafs.len());
                leafs[choice].clone()
            }
        };
        *e = leaf;
    }

    fn transform_swap(&mut self, e: &mut Expr) {
        use expr::Expr::*;
        match *e {
            ArithOp(_, ref mut e1, ref mut e2, _) |
            LogicOp(_, ref mut e1, ref mut e2, _) |
            BoolOp(_, ref mut e1, ref mut e2, _) => {
                let e1_t = e1.clone();
                *e1 = e2.clone();
                *e2 = e1_t;
            }
            ITE(_, ref mut e2, ref mut e3) => {
                    let e2_t = e2.clone();
                    *e2 = e3.clone();
                    *e3 = e2_t;
            }
            _ => panic!("swap not supported: {:?}", e)
        };
    }

    fn transform_replace_type(&mut self, e: &mut Expr) {
        use expr::Expr::*;
        match *e {
            ArithOp(_, _, _, ref mut ty) =>
                *ty = self.sampler.sample_ty(None),
            LogicOp(_, _, _, ref mut w) | BoolOp(_, _, _, ref mut w) |
            UnOp(_, _, ref mut w) =>
                *w = self.sampler.sample_width(None),
            Bits(ref mut b1, ref mut b2, _) => {
                let (bit1, bit2) = self.sampler.sample_bits(None, None);
                *b1 = bit1;
                *b2 = bit2;
            }
            Bit(ref mut b, _) =>
                *b = self.sampler.sample_bit(None),
            Cast(_, _, ref mut ty) =>
                *ty = self.sampler.sample_ty(None),
            _ => panic!(format!("replace type supported: {:?}", e))
        };
    }

    fn transform_replace_op(&mut self, e: &mut Expr) {
        use expr::Expr::*;
        match *e {
            ArithOp(ref mut o, ref e1, ref e2, ref mut ty) =>
                *o = self.sampler.sample_arithop(),
            LogicOp(ref mut o, ref e1, ref e2, _) =>
                *o = self.sampler.sample_logicop(),
            BoolOp(ref mut o, ref e1, ref e2, _) =>
                *o = self.sampler.sample_boolop(),
            UnOp(ref mut o, ref e1, _) =>
                *o = self.sampler.sample_unop(),
            Cast(ref mut o, _, _) =>
                *o = self.sampler.sample_castop(),
            _ => panic!(format!("replace op supported: {:?}", e))
        };
    }

    fn transform_boolop(&mut self, e: &mut Expr) {
        use self::Mov::*;

        match self.trans {
            Remove | Insert | ReplaceExpr
                => *e = self.sampler.sample_boolexpr(),
            Swap => self.transform_swap(e),
            ReplaceOp => self.transform_replace_op(e),
            ReplaceType => self.transform_replace_type(e),
        }
    }

    fn transform_op3(&mut self, e: &mut Expr, curr: usize)
    {
        use expr::Expr::*;
        use self::Mov::*;

        let (zero, curr) = (self.pos.get(curr).unwrap(), curr+1);
        let (p, this) =
            if let Some(p) = self.pos.get(curr) {
                (*p, false)
            } else {
                (0, true)
            };
        let e_width = e.get_width();

        if !this {
            match *e {
                ArithOp(_, ref mut e1, ref mut e2, _) |
                LogicOp(_, ref mut e1, ref mut e2, _) |
                BoolOp(_, ref mut e1, ref mut e2, _) => {
                    if p == 1 {
                        self.transform_op3(e1, curr+1);
                    } else if p == 2 {
                        self.transform_op3(e2, curr+1);
                    } else {
                        panic!(format!(
                            "binary operator and p({})!=[1;2]", p));
                    }
                }
                ITE(ref mut e1, ref mut e2, ref mut e3) => {
                    if p == 1 {
                        self.transform_op3(e1, curr+1);
                    } else if p == 2 {
                        self.transform_op3(e2, curr+1);
                    } else if p == 3 {
                        self.transform_op3(e3, curr+1);
                    } else {
                        panic!(format!(
                            "binary operator and p({})!=[1;2]", p));
                    }
                },
                UnOp(_, ref mut e1, _) | Cast(_, ref mut e1, _) |
                Bit(_, ref mut e1) | Bits(_, _, ref mut e1) =>
                {
                    self.transform_op3(e1, curr+1);
                }
                _ => panic!(format!("not supported/implement me: {:?}", e))
            }
        }
        else if this && e.is_boolop() {
            self.transform_boolop(e);
        }
        else if this && self.trans == Remove {
            self.transform_remove(e);
        }
        else if this && self.trans == Insert {
            self.transform_insert(e);
        }
        else if this && self.trans == ReplaceExpr {
            self.transform_replace_expr(e);
        }
        else if this && self.trans == Swap {
            self.transform_swap(e);
        }
        else if this && self.trans == ReplaceOp {
            self.transform_replace_op(e);
        }
        else if this && self.trans == ReplaceType {
            self.transform_replace_type(e);
        }
        else {
            panic!("trans: {:?}, this: {:?}, expr: {:?}", self.trans, this, e);
        }
    }
}

#[cfg(test)]
mod test {
    use num::bigint::ToBigUint;
    use std::collections::HashMap;
    use test::{Bencher, black_box};

    use expr::Expr;
    use stochastic::{Stochastic, Transform};

    // Helper
    fn get_inc_eax_stochastic() -> Stochastic {
        let eax = Expr::Reg("EAX".to_owned(), 32);
        let args = vec![eax.clone()];
        let mut io_sets = vec![];
        for &(res, dep) in &[(0x10, 0x9), (0xfff000, 0xfff001),
                             (0x1001, 0x1002), (0xffff, 0x10000)]
        {
            let mut io_name = HashMap::new();
            io_name.insert(eax.clone(), dep.to_biguint().unwrap());
            io_sets.push((res.to_biguint().unwrap(), io_name));
        }
        Stochastic::new(&args, &io_sets, 32)
    }

    #[test]
    fn test_inc_eax() {
        let mut stoc = get_inc_eax_stochastic();
        //stoc.work();
    }

    #[bench]
    fn bench_inc_eax(b: &mut Bencher) {
        let mut stoc = get_inc_eax_stochastic();
        b.iter(|| {
            //black_box(stoc.work());
        });
    }
}
