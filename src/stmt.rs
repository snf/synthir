use expr::Expr;

#[derive(Clone,Debug)]
pub enum Stmt {
    StoreReg(Expr, Expr),
    StoreMem(Expr, Expr),
    Jmp(Expr),
    CJmp(Expr, Expr),
    // Eval first expr and if it's true, keep evaluating stmt, can be
    // formed with CJmp
    //RepeatWhile(Expr, Box<Stmt>),
    // Semantics too complicated for this tool
    Special(String)
}
