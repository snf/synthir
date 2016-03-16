use num::bigint::{ToBigUint, BigUint};
use num::traits::{One, ToPrimitive};

use std::num::ParseIntError;
use std::mem;

pub trait Max {
    fn max(width: u32) -> Self;
}
impl Max for BigUint {
    fn max(width: u32) -> BigUint {
        (BigUint::one() << width as usize) - BigUint::one()
    }
}

pub trait ParseNum {
    fn parse_num(num: &str) -> Result<Self, ParseIntError> where Self: Sized;
}

impl ParseNum for u64 {
    fn parse_num(num: &str) -> Result<Self, ParseIntError> {
        if num.len() > 2 && &num[0..2] == "0x" {
            u64::from_str_radix(&num[2..], 16)
        } else {
            u64::from_str_radix(num, 10)
        }
    }
}

trait AsFixed<T> {
    fn as_fixed(&self) -> T;
}
impl AsFixed<u64> for BigUint {
    #[inline]
    fn as_fixed(&self) -> u64 {
        (self & 0xffFFffFF_ffFFffFFu64.to_biguint().unwrap())
            .to_u64()
            .unwrap()
    }
}
impl AsFixed<f64> for BigUint {
    #[inline]
    fn as_fixed(&self) -> f64 {
        let imm: u64 = self.as_fixed();
        unsafe { mem::transmute(imm) }
    }
}
impl AsFixed<f32> for BigUint {
    #[inline]
    fn as_fixed(&self) -> f32 {
        let imm: u64 = self.as_fixed();
        unsafe { mem::transmute(imm as u32) }
    }
}

/// Implement this trait for Floats to get the raw representation
pub trait AsRaw<T> {
    /// Returns the raw representation of the floating point number
    fn as_raw(&self) -> T;
}

impl AsRaw<u32> for f32 {
    fn as_raw(&self) -> u32 {
        unsafe { mem::transmute(*self) }
    }
}
impl AsRaw<u64> for f64 {
    fn as_raw(&self) -> u64 {
        unsafe { mem::transmute(*self) }
    }
}

use std::collections::VecDeque;
use std::collections::vec_deque;

// Objects
pub struct LastCache<T> {
    stack: VecDeque<T>,
    max: usize
}

impl<T> LastCache<T> {
    pub fn new(max: usize) -> LastCache<T> {
        LastCache {
            stack: VecDeque::with_capacity(max),
            max: max
        }
    }

    pub fn push(&mut self, item: T) {
        if self.stack.len() >= self.max {
            self.stack.pop_back();
            self.stack.remove(self.max - 1);
        }
        self.stack.push_front(item);
    }

    pub fn iter<'a>(&'a mut self) -> vec_deque::Iter<'a, T> {
        self.stack.iter()
    }

    pub fn last<'a>(&'a self) -> &T {
        self.stack.front().unwrap()
    }
}

impl<'a, T> IntoIterator for &'a LastCache<T> {
    type Item = &'a T;
    type IntoIter = vec_deque::Iter<'a, T>;

    fn into_iter(self) -> vec_deque::Iter<'a, T> {
        self.stack.iter()
    }
}

// Arithmetic/binary utils
/// Hamming weight
pub trait Hamming {
    fn hamming_weight(&self) -> u32;
}

impl Hamming for u64 {
    fn hamming_weight(&self) -> u32 {
        // Optimized by rust's intrinsics
        self.count_ones()
    }
}

impl Hamming for BigUint {
    fn hamming_weight(&self) -> u32 {
        let bits = self.bits();
        if bits <= 64 {
            let in64 = self.to_u64().unwrap();
            in64.hamming_weight()
        } else {
            let mut res = 0;
            for i in 0..bits {
                res += ((self >> i) & BigUint::one()).to_u32().unwrap();
            }
            res
        }
    }
}

pub fn flip_bits(num: &BigUint, width: u32) -> BigUint {
    let bits = num.bits();
    ((BigUint::one() << width as usize) - BigUint::one()) ^ num
}

// CloneWeightedChoice
use rand::Rng;
use rand::distributions::{Sample, Weighted, Range, IndependentSample};

pub struct CloneWeightedChoice<T> {
    items: Vec<Weighted<T>>,
    weight_range: Range<u32>
}

impl<T: Clone> CloneWeightedChoice<T> {
    /// Create a new `WeightedChoice`.
    ///
    /// Panics if:
    /// - `v` is empty
    /// - the total weight is 0
    /// - the total weight is larger than a `u32` can contain.
    pub fn new(items: &[Weighted<T>]) -> CloneWeightedChoice<T> {
        // strictly speaking, this is subsumed by the total weight == 0 case
        assert!(!items.is_empty(), "WeightedChoice::new called with no items");
        let mut items = items.to_vec();

        let mut running_total: u32 = 0;

        // we convert the list from individual weights to cumulative
        // weights so we can binary search. This *could* drop elements
        // with weight == 0 as an optimisation.
        for item in items.iter_mut() {
            running_total = match running_total.checked_add(item.weight) {
                Some(n) => n,
                None => panic!("WeightedChoice::new called with a total weight \
                               larger than a u32 can contain")
            };

            item.weight = running_total;
        }
        assert!(running_total != 0, "WeightedChoice::new called with a total weight of 0");

        CloneWeightedChoice {
            items: items,
            // we're likely to be generating numbers in this range
            // relatively often, so might as well cache it
            weight_range: Range::new(0, running_total)
        }
    }
}

impl<T: Clone> Sample<T> for CloneWeightedChoice<T> {
    fn sample<R: Rng>(&mut self, rng: &mut R) -> T { self.ind_sample(rng) }
}

impl<T: Clone> IndependentSample<T> for CloneWeightedChoice<T> {
    fn ind_sample<R: Rng>(&self, rng: &mut R) -> T {
        // we want to find the first element that has cumulative
        // weight > sample_weight, which we do by binary since the
        // cumulative weights of self.items are sorted.

        // choose a weight in [0, total_weight)
        let sample_weight = self.weight_range.ind_sample(rng);

        // short circuit when it's the first item
        if sample_weight < self.items[0].weight {
            return self.items[0].item.clone();
        }

        let mut idx = 0;
        let mut modifier = self.items.len();

        // now we know that every possibility has an element to the
        // left, so we can just search for the last element that has
        // cumulative weight <= sample_weight, then the next one will
        // be "it". (Note that this greatest element will never be the
        // last element of the vector, since sample_weight is chosen
        // in [0, total_weight) and the cumulative weight of the last
        // one is exactly the total weight.)
        while modifier > 1 {
            let i = idx + modifier / 2;
            if self.items[i].weight <= sample_weight {
                // we're small, so look to the right, but allow this
                // exact element still.
                idx = i;
                // we need the `/ 2` to round up otherwise we'll drop
                // the trailing elements when `modifier` is odd.
                modifier += 1;
            } else {
                // otherwise we're too big, so go left. (i.e. do
                // nothing)
            }
            modifier /= 2;
        }
        return self.items[idx + 1].item.clone();
    }
}

// Macro utils
macro_rules! for_one {
    ($x:ident) => (1)
}
macro_rules! enum_and_list{
    ($e_name:ident, $c_name:ident, $($m:ident),+) => {
        #[derive(Copy,Clone,Debug,PartialEq,Eq,Hash,PartialOrd,Ord)]
        pub enum $e_name {
            $(
                $m,
            )+
        }
        pub const $c_name: [$e_name; 0 $( + for_one!($m) )+ ] =
            [ $($e_name::$m,)+ ];
        impl Rand for $e_name {
            fn rand<R: Rng>(rng: &mut R) -> Self {
                let val = rng.gen_range(0, $c_name.len());
                $c_name[val]
            }
        }
    }
}

macro_rules! debugln{
    ($($arg:tt)*) => (
        {
            use std::io::Write;
            match writeln!(&mut ::std::io::stderr(), $($arg)* ) {
                Ok(_) => {},
                Err(x) => panic!("Unable to write to stderr: {}", x),
            }
        }
     )
}

macro_rules! debug{
    ($($arg:tt)*) => (
        {
            use std::io::Write;
            match write!(&mut ::std::io::stderr(), $($arg)* ) {
                Ok(_) => {},
                Err(x) => panic!("Unable to write to stderr: {}", x),
            }
        }
     )
}

#[cfg(test)]
mod test_util {
    use num::bigint::{BigUint, ToBigUint};
    use num::traits::{One, Zero};
    use utils::AsRaw;
    use utils::Hamming;
    use utils::LastCache;

    #[test]
    fn test_hamming() {
        let full_128_bu = {
            let mut res = BigUint::zero();
            for i in 0..128 {
                res = res | BigUint::one() << i;
            }
            res
        };
        assert_eq!(full_128_bu.hamming_weight(), 128);

        let full_64_u64 = 0xffFF_ffFF_ffFF_ffFFu64;
        assert_eq!(full_64_u64.hamming_weight(), 64);

        let full_64_bu = 0xffFF_ffFF_ffFF_ffFFu64.to_biguint().unwrap();
        assert_eq!(full_64_bu.hamming_weight(), 64);
    }

    #[test]
    fn test_biguint() {
        assert_eq!(0xaabbccddu32.to_biguint().unwrap().to_string(),
                   "2864434397");
    }

    #[test]
    fn test_lastcache() {
        let mut lc = LastCache::new(10);
        for i in 0 .. 10 {
            lc.push(1);
        }
        assert_eq!(lc.iter().all(|i| *i == 1), true);
        lc.push(2);
        lc.push(2);
        let mut i = 0;
        for item in &lc {
            if i == 0 || i == 1 {
                assert_eq!(*item, 2);
            } else {
                assert_eq!(*item, 1);
            }
            i += 1;
        }
    }

    #[test]
    fn test_as_raw_f32() {
        use std::f32::INFINITY;
        assert_eq!(2.8411367E-29f32.as_raw(), 0x10101010);
        assert_eq!((-4.99999998E11f32).as_raw(), 0xd2e8d4a5);
        assert_eq!(100.0f32.as_raw(), 0x42c80000);
        assert_eq!(0.0f32.as_raw(), 0);
        assert_eq!((-0.0f32).as_raw(), 0x80000000);
        assert_eq!(INFINITY.as_raw(), 0x7f800000);
    }

    #[test]
    fn test_as_raw_f64() {
        use std::f64::INFINITY;
        assert_eq!(2.8411367E-29f64.as_raw(), 0x3A0202020351C16B);
        assert_eq!((-4.99999998E11f64).as_raw(), 0xC25D1A94A00C0000);
        assert_eq!(100.0f64.as_raw(), 0x4059000000000000);
        assert_eq!(0.0f64.as_raw(), 0);
        assert_eq!((-0.0f64).as_raw(), 0x8000000000000000);
        assert_eq!(INFINITY.as_raw(), 0x7ff0000000000000);
    }

}
