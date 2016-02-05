//use rand::distributions::Sample;
use rand::{Rand, Rng};

enum_and_list!(OpArith, OPARITH,
               Add, Sub, Mul, Div, SDiv, Mod, SMod, ARShift, ALShift);

enum_and_list!(OpLogic, OPLOGIC,
               And, Xor, Or, LLShift, LRShift);

enum_and_list!(OpUnary, OPUNARY,
               Neg, Not);

enum_and_list!(OpBool, OPBOOL,
               LT, LE, SLT, SLE, EQ, NEQ);

enum_and_list!(OpCast, OPCAST,
               CastLow, CastHigh, CastSigned);

pub enum AnyOp {
    Arith(OpArith),
    Logic(OpLogic),

}

#[test]
fn oparith_sample() {
    let mut rng = thread_rng();
    let val = OpArith::rand(&mut rng);
}
