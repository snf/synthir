use rand::{Rand, Rng};

enum_and_list!(OpArith, OPARITH,
               Add, Sub, Mul, Div, SDiv, URem, SRem, ARShift, ALShift);

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
