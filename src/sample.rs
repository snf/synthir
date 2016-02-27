use op::{OpArith, OpLogic, OpUnary, OpBool, OpCast,
         OPARITH, OPLOGIC, OPUNARY, OPBOOL, OPCAST};
use expr::{Expr, ExprType, ILiteral};
use utils::CloneWeightedChoice;

use num::{BigUint};
use num::traits::{Zero};
use num::bigint::ToBigUint;
use rand;
use rand::distributions::{Weighted, IndependentSample};
use rand::{Rng, ThreadRng};
use std::iter::Iterator;

mod constants {
    use num::{BigUint};
    use num::traits::{One, Zero};
    use num::bigint::ToBigUint;
    use utils::AsRaw;

    /// Generates the 0b1010.. or 0b0101... representations
    pub fn gen_alternated_bit(bits: u32, shift: bool) -> BigUint {
        if bits == 1 {
            return
                if shift {
                    BigUint::one()
                } else {
                    BigUint::zero()
                };
        }
        let (mut start, end) = {
            if shift == true {
                (BigUint::zero(), bits)
            } else {
                (BigUint::one() << 1, bits - 3)
            }
        };
        for i in 0 .. end {
            start = start << 1;
            if i % 2 == 0 {
                start = start + BigUint::one();
            }
        }
        start
    }

    /// Generates 0xff00 or 0x00ff representations
    pub fn gen_half(bits: u32, upper: bool) -> BigUint {
        let half_bits = bits / 2;
        let mut start = BigUint::zero();
        for i in 0 .. half_bits {
            start = start | BigUint::one() << i as usize;
        }
        if upper {
            start = start << half_bits as usize;
        }
        start
    }

    /// Max value for width
    pub fn max_value(width: u32) -> BigUint {
        (BigUint::one() << (width as usize)) - BigUint::one()
    }

    /// Max positive value for width
    pub fn max_pos_value(width: u32) -> BigUint {
        max_value(width) / 2.to_biguint().unwrap()
    }

    // XXX_ 0 and -0 are very simple in floating point
    pub fn gen_alternated_float(which: u32) -> BigUint {
        use std::f32::INFINITY;
        if which % 3 == 0 {
            0.0f32.as_raw().to_biguint().unwrap()
        } else if which % 3 == 1 {
            // Infinity
            INFINITY.as_raw().to_biguint().unwrap()
        } else {
            (-0.0f32).as_raw().to_biguint().unwrap()
        }
    }

    pub fn gen_float(which: u32) -> BigUint {
        BigUint::one()
    }

    pub fn to_vector(value: &BigUint, orig_width: u32, times: u32)
                     -> BigUint
    {
        assert!(orig_width < 32);

        value.clone()
    }
}

/// Self state tracking object for sampling values
pub trait ValueSampler
{
    // Methods
    fn get_value(&mut self, width: u32) -> BigUint;
    fn reset(&mut self);
    fn push(&mut self);
    fn pop(&mut self);
    fn increase(&mut self);
    fn current(&self) -> usize;
    // Ctor
    fn new(round: usize) -> Self where Self: Sized;
}

/// Contains the static methods that generate the values
trait ValueSamplerStatic {
    fn get_value_static(round: usize, curr: usize, width: u32) -> BigUint;
    fn rounds() -> usize;
}

/// Implement the basic functions for samplers iterators when they
/// have curr and stack in the struct
macro_rules! impl_basic_I{
    () => (
        fn reset(&mut self) { self.curr = 0; }
        fn push(&mut self) { self.stack.push(self.curr);  }
        fn pop(&mut self) { self.curr = self.stack.pop().unwrap(); }
        fn current(&self) -> usize { self.curr }
        fn increase(&mut self) { self.curr += 1 }
        fn get_value(&mut self, width: u32) -> BigUint {
            let value = Self::get_value_static(self.round, self.curr, width);
            self.increase();
            value
        }
        )
}

/// Some simple constants
pub struct ConstSampler {
    round: usize,
    curr: usize,
    stack: Vec<usize>
}
impl ValueSamplerStatic for ConstSampler {
    fn rounds() -> usize { 28 }
    fn get_value_static(round: usize, curr: usize, width: u32) -> BigUint {
        // println!("reting value, round: {}, curr: {}, width: {}",
        //         round, curr, width);
        match round {
            0 ... 1 => {
                let curr = curr % 2 == 0;
                constants::gen_alternated_bit(width, curr)
            },
            1 ... 2 => {
                let curr = !curr % 2 == 0;
                constants::gen_alternated_bit(width, curr)
            },
            3 ... 4 => {
                let curr = curr % 2 == 0;
                constants::gen_half(width, curr)
            },
            5 ... 6 => {
                let curr = !curr % 2 == 0;
                constants::gen_half(width, curr)
            },
            7 ... 8 => {
                let curr = curr % 3 == 0;
                constants::gen_alternated_bit(width, curr)
            },
            9 ... 10 => {
                let curr = (curr + 1) % 3 == 0;
                constants::gen_alternated_bit(width, curr)
            },
            11 ... 12 => {
                let curr = curr % 3 == 0;
                constants::gen_half(width, curr)
            },
            13 ... 14 => {
                let curr = (curr + 1) % 3 == 0;
                constants::gen_half(width, curr)
            },
            15 ... 16 => {
                if curr % 2 == 1 { 3.to_biguint().unwrap() }
                else { 2.to_biguint().unwrap() }
            },
            17 ... 18 => {
                if curr % 2 == 1 { 4.to_biguint().unwrap() }
                else { 3.to_biguint().unwrap() }
            },
            19 ... 20 => {
                if curr % 2 == 1 { 7.to_biguint().unwrap() }
                else { 2.to_biguint().unwrap() }
            },
            21 ... 22 => {
                if curr % 2 == 1 { 29.to_biguint().unwrap() }
                else { 4.to_biguint().unwrap() }
            },
            23 ... 24 => {
                if curr % 2 == 1 { 53.to_biguint().unwrap() }
                else { 8.to_biguint().unwrap() }
            },
            25 ... 26 => {
                if curr % 2 == 1 { 2510.to_biguint().unwrap() }
                else { 16.to_biguint().unwrap() }
            },
            27 => BigUint::zero(),
            _ => panic!(format!("unreachable: {}", curr))
        }
    }
}

impl ValueSampler for ConstSampler {
    impl_basic_I!();

    fn new(round: usize) -> ConstSampler {
        ConstSampler {
            round: round,
            curr: 0,
            stack: Vec::new()
        }
    }
}

/// Try making overflows/underflows
pub struct OverflowSampler {
    round: usize,
    curr: usize,
    stack: Vec<usize>
}
impl ValueSamplerStatic for OverflowSampler {
    fn rounds() -> usize { 13 }
    fn get_value_static(round: usize, curr: usize, width: u32) -> BigUint {
        match round {
            0 => constants::max_value(width),
            1 => {
                if curr % 2 == 0 {
                    BigUint::zero()
                } else {
                    constants::max_value(width)
                }
            },
            2 => {
                if curr % 3 == 0 {
                    BigUint::zero()
                } else {
                    constants::max_value(width)
                }
            },
            3 => {
                if (curr + 1) % 2 == 0 {
                    BigUint::zero()
                } else {
                    constants::max_value(width)
                }
            },
            4 => {
                if (curr + 1) % 3 == 0 {
                    BigUint::zero()
                } else {
                    constants::max_value(width)
                }
            },
            5 => {
                if curr % 2 == 0 {
                    BigUint::zero()
                } else {
                    constants::max_pos_value(width)
                }
            },
            6 => {
                if curr % 3 == 0 {
                    BigUint::zero()
                } else {
                    constants::max_pos_value(width)
                }
            },
            7 => {
                if curr % 2 == 0 {
                    constants::max_value(width)
                } else {
                    constants::max_pos_value(width)
                }
            },
            8 => {
                if (curr + 1) % 2 == 0 {
                    BigUint::zero()
                } else {
                    constants::max_pos_value(width)
                }
            },
            9 => {
                if (curr + 1) % 3 == 0 {
                    BigUint::zero()
                } else {
                    constants::max_pos_value(width)
                }
            },
            10 => {
                if (curr + 1) % 2 == 0 {
                    constants::max_value(width)
                } else {
                    constants::max_pos_value(width)
                }
            },
            11 => {
                if (curr + 1) % 3 == 0 {
                    constants::max_value(width)
                } else {
                    constants::max_pos_value(width)
                }
            },
            12 => constants::max_pos_value(width),
            _ => unreachable!()
        }
    }
}
impl ValueSampler for OverflowSampler {
    impl_basic_I!();

    fn new(round: usize) -> OverflowSampler {
        OverflowSampler {
            round: round,
            curr: 0,
            stack: Vec::new()
        }
    }
}

pub struct DepValueSampler {
    i: usize,
}

impl DepValueSampler {
    pub fn new(count: usize) -> DepValueSampler {
        DepValueSampler {
            i: 0
        }
    }
}

// macro_rules! return_sampler{
//     ($($e:ident), *) => (
//         let total = $ ( $e::rounds() ) , *
//         )
// }
impl Iterator for DepValueSampler  {
    type Item = Box<ValueSampler>;

    fn next(&mut self) -> Option<Box<ValueSampler>> {
        // return_sampler![ConstSampler, OverflowSampler];
        let total = ConstSampler::rounds() + OverflowSampler::rounds();
        if self.i >= total {
            None
        } else {
            let res: Option<Box<ValueSampler>> = Some(
                if self.i < ConstSampler::rounds() {
                    Box::new(ConstSampler::new(self.i))
                } else if self.i < ConstSampler::rounds() + OverflowSampler::rounds() {
                    Box::new(OverflowSampler::new(self.i - ConstSampler::rounds()))
                } else {
                    unreachable!();
                }
            );
            self.i += 1;
            res
        }
    }
}

/// Algorithms to detect dependencies (these have not been tested nor
/// proved):
///
/// a) gen_alternated_bits * 2
/// b) gen_half * 2
/// c) zero
/// d) full
/// e) values: 7, 29, 53, 2510
///
/// Creates iter number  of num strategies * reg_size
///
/// Exceptions: if width == 1, we alternate between 1 and 0
// enum_and_list!(DepStrategy, DEPSTRATEGY,
//                Alt1, Alt2, Half1, Half2, Zero, Full, Val1, Val2, Val3, Val4,
//                Const7, Const29, Const53, Const2510);

// pub struct DepValue {
//     curr: usize,
//     // XXX_ unused:
//     count: usize,
//     strategy: DepStrategy,
//     stack: Vec<usize>
// }

/// Sample random Expr
pub struct RandExprSampler{
    base: Vec<Expr>,
    rng: ThreadRng,
    sample_choice: CloneWeightedChoice<EEWS>
}

#[derive(Clone,Debug)]
enum EEWS {
    Bit,
    Bits,
    Arith,
    Logic,
    Un,
    ITE,
    Bool,
    Cast,
    Ex,
    IInt
}

impl RandExprSampler {
    pub fn new(base: &[Expr]) -> RandExprSampler{
        let rng = rand::thread_rng();
        RandExprSampler {
            base: base.to_vec(),
            rng: rng,
            sample_choice: CloneWeightedChoice::new(&[
                // Weighted{item: EEWS::Bit, weight: 1},
                // Weighted{item: EEWS::Bits, weight: 1},
                // Weighted{item: EEWS::Arith, weight: 1},
                // Weighted{item: EEWS::Logic, weight: 1},
                // Weighted{item: EEWS::Un, weight: 1},
                // Weighted{item: EEWS::ITE, weight: 1},
                // //Weighted{item: EEWS::Bool, weight: 0},
                // Weighted{item: EEWS::Cast, weight: 1},
                // Weighted{item: EEWS::Ex, weight: 1},
                // Weighted{item: EEWS::IInt, weight: 1}
                Weighted{item: EEWS::Bit, weight: 3},
                Weighted{item: EEWS::Bits, weight: 3},
                Weighted{item: EEWS::Arith, weight: 10},
                Weighted{item: EEWS::Logic, weight: 10},
                Weighted{item: EEWS::Un, weight: 5},
                Weighted{item: EEWS::ITE, weight: 1},
                //Weighted{item: EEWS::Bool, weight: 0},
                Weighted{item: EEWS::Cast, weight: 3},
                Weighted{item: EEWS::Ex, weight: 1},
                Weighted{item: EEWS::IInt, weight: 1}
                ]
                                                    )
        }
    }

    fn sample_constant_shift(&mut self, width: u32) -> i16 {
        let width = width as i16;
        self.rng.gen_range(-width, width)
    }


    fn sample_constant(&mut self) -> i16 {
        self.rng.gen_range(-16, 16)
    }
    fn sample_literal(&mut self) -> ILiteral {
        self.sample_constant()
    }
    fn sample_base(&mut self) -> Expr {
        if self.base.is_empty() {
            Expr::Int(self.sample_const().to_biguint().unwrap())
        } else {
            let which = self.rng.gen_range(0, self.base.len());
            self.base[which].clone()
        }
    }
    fn sample_from_arr<T: Clone>(&mut self, arr: &[T]) -> T {
        let len = arr.len();
        let which = self.rng.gen_range(0, len);
        arr[which].clone()
    }
    fn sample_from_arr_remove<T: Clone>(&mut self, arr: &mut Vec<T>)
                                        -> Option<T>
    {
        if arr.is_empty() {
            None
        } else {
            let len = arr.len();
            let which = self.rng.gen_range(0, len);
            Some(arr.remove(which))
        }
    }

    fn sample_ex(&mut self) -> Expr {
        if self.rng.gen::<bool>() {
            self.sample_base()
        } else {
            Expr::Int(self.sample_const().to_biguint().unwrap())
        }
    }

    pub fn sample_arithop(&mut self) -> OpArith {
        self.sample_from_arr(&OPARITH)
    }
    pub fn sample_logicop(&mut self) -> OpLogic {
        self.sample_from_arr(&OPLOGIC)
    }
    pub fn sample_unop(&mut self) -> OpUnary {
        self.sample_from_arr(&OPUNARY)
    }
    pub fn sample_boolop(&mut self) -> OpBool {
        self.sample_from_arr(&OPBOOL)
    }
    pub fn sample_castop(&mut self) -> OpCast {
        self.sample_from_arr(&OPCAST)
    }
    pub fn sample_bit(&mut self, max: Option<u32>) -> u32 {
        let max =
            if max.is_none() {
                self.sample_const()
            } else {
                max.unwrap()
            };
        self.rng.gen_range(0, max)
    }
    pub fn sample_bits(&mut self, range: Option<u32>, max: Option<u32>)
                   -> (u32, u32)
    {
        let (range, max) =
            if range.is_none() || max.is_none() {
                let range = self.sample_const();
                let max = self.sample_const();
                (range, max)
            } else {
                (range.unwrap(), max.unwrap())
            };
        let val = self.rng.gen_range(0, max);
        (val, val+range)
    }
    fn sample_const(&mut self) -> u32 {
        let values = [1, 4, 8, 16, 32, 64, 128, 256];
        self.sample_from_arr(&values)
    }
    pub fn sample_ty(&mut self, width: Option<u32>) -> ExprType {
        // XXX_ add float
        ExprType::Int(
            if let Some(w) = width {
                w
            } else {
                self.sample_const()
            })
    }

    pub fn sample_width(&mut self, width: Option<u32>) -> u32 {
        if let Some(w) = width {
            if self.rng.gen_range(0, 10) > 7 {
                self.sample_const()
            } else {
                w
            }
        } else {
            self.sample_const()
        }
    }

    pub fn sample_expr_w_with_leafs(&mut self, leafs: &mut Vec<&Expr>,
                                    width: Option<u32>)
                                   -> Expr
    {
        use expr::Expr::*;

        // Replace with an Expr sampled from the array if avail
        macro_rules! leaf_or_e(
            ($e:expr, $leafs:expr) => ({
                if let Some(n_e) = self.sample_from_arr_remove($leafs) {
                    *$e = Box::new(n_e.clone())
                }
        }));

        let mut new_e;

        // Sample until getting a non-ultimate expr where we can plug
        // our leaf
        loop {
            new_e = self.sample_expr_w(width);
            if !new_e.is_last() {
                break;
            }
        }

        match new_e {
            ArithOp(_, ref mut e1, ref mut e2, _) |
            LogicOp(_,ref mut e1, ref mut e2, _) |
            BoolOp(_, ref mut e1, ref mut e2, _) => {
                if self.rng.gen::<bool>() {
                    leaf_or_e!(e1, leafs);
                    leaf_or_e!(e2, leafs);
                } else {
                    leaf_or_e!(e2, leafs);
                    leaf_or_e!(e1, leafs);
                }
            },
            UnOp(_, ref mut e1, _) | Cast(_, ref mut e1, _) |
            Bit(_, ref mut e1) | Bits(_, _, ref mut e1) => {
                leaf_or_e!(e1, leafs);
            },
            ITE(_, ref mut e2, ref mut e3) => {
                if self.rng.gen::<bool>() {
                    leaf_or_e!(e2, leafs);
                    leaf_or_e!(e3, leafs);
                } else {
                    leaf_or_e!(e3, leafs);
                    leaf_or_e!(e2, leafs);
                }
            },
            _ => ()
        };
        new_e
    }

    pub fn sample_boolexpr(&mut self) -> Expr {
        Expr::BoolOp(self.sample_boolop(),
                     Box::new(self.sample_ex()),
                     Box::new(self.sample_ex()),
                     self.sample_width(None))
    }

    pub fn sample_expr_w(&mut self, width: Option<u32>) -> Expr {
        let n_width = self.sample_width(width);
        match self.sample_choice.ind_sample(&mut self.rng) {
            EEWS::Ex => self.sample_ex(),
            EEWS::IInt => Expr::IInt(self.sample_literal()),
            EEWS::Bits => {
                let e = self.sample_base();
                let w_e = e.get_width();
                let (bit1, bit2) = self.sample_bits(width, w_e);
                Expr::Bits(bit1, bit2, Box::new(e))
            }
            EEWS::Bit => {
                let e = self.sample_base();
                let w = e.get_width();
                let bit = self.sample_bit(w);
                Expr::Bit(bit, Box::new(e))
            }
            EEWS::Arith => Expr::ArithOp(self.sample_arithop(),
                               Box::new(self.sample_ex()),
                               Box::new(self.sample_ex()),
                               self.sample_ty(width)),
            EEWS::Logic => Expr::LogicOp(self.sample_logicop(),
                                         Box::new(self.sample_ex()),
                                         Box::new(self.sample_ex()),
                                         // XXX_ fixme
                                         n_width),
            EEWS::Un => Expr::UnOp(self.sample_unop(),
                                   Box::new(self.sample_ex()),
                                   // XXX_ fixme
                                   n_width),
            EEWS::ITE => Expr::ITE(Box::new(self.sample_boolexpr()),
                                   Box::new(self.sample_ex()),
                                   Box::new(self.sample_ex())),
            EEWS::Cast => Expr::Cast(self.sample_castop(),
                                     Box::new(self.sample_ex()),
                                     self.sample_ty(width)),
            //_ => unreachable!()
            wc => panic!(format!("not supported: {:?}", wc))

        }
    }

}

// Always start with the expression with the same expression of the
// StoreStmt, for example if it's StoreReg(Reg("EAX"), RESULT), start
// with: Reg("EAX").
fn start() {

}

fn sample_dep() {
}
fn sample_binop() {

}
// Sample randomly now
fn sample() -> Expr {
    Expr::Reg("EAX".to_owned(), 32)
}

#[cfg(bench)]
mod bench {
    use expr::Expr;
    use num::bigint::ToBigUint;
    use sample::{constants, RandExprSampler};
    use test::{Bencher, black_box};

    #[bench]
    fn bench_sample1(b: &mut Bencher) {
        let r1 = Expr::Reg("EAX".to_owned(), 32);
        let r2 = Expr::Reg("EBX".to_owned(), 32);
        let mut sample = RandExprSampler::new(&[r1, r2]);
        b.iter(|| {
            black_box(sample.sample_expr_w(Some(32)));
        });
    }

}

#[cfg(test)]
mod test {
    use expr::Expr;
    use num::bigint::ToBigUint;
    use sample::{constants, RandExprSampler};
    use test::{Bencher, black_box};

    #[test]
    fn test_gen_alternated() {
        let bu = constants::gen_alternated_bit(8, false);
        assert_eq!(bu, 0x55.to_biguint().unwrap());

        let bu = constants::gen_alternated_bit(8, true);
        assert_eq!(bu, 0xaa.to_biguint().unwrap());

        let bu = constants::gen_alternated_bit(32, false);
        assert_eq!(bu, 0x55555555u32.to_biguint().unwrap());

        let bu = constants::gen_alternated_bit(32, true);
        assert_eq!(bu, 0xaaAAaaAAu32.to_biguint().unwrap());
    }

    #[test]
    fn test_gen_half() {
        let bu = constants::gen_half(8, false);
        assert_eq!(bu, 0xf.to_biguint().unwrap());

        let bu = constants::gen_half(8, true);
        assert_eq!(bu, 0xf0.to_biguint().unwrap());
    }
}
