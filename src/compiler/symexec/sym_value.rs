/// Symbolic value — tracks the origin and computation history of a register/memory value.
#[derive(Debug, Clone)]
pub enum SymValue {
    /// Function parameter (nth argument per System V ABI: xmm0=0, xmm1=1, ...).
    Param(usize),
    /// Load from memory: `*base[index]`.
    Load {
        base: Box<SymValue>,
        index: Box<SymValue>,
    },
    /// Floating-point constant.
    Const(f64),
    /// Addition.
    Add(Box<SymValue>, Box<SymValue>),
    /// Subtraction.
    Sub(Box<SymValue>, Box<SymValue>),
    /// Multiplication.
    Mul(Box<SymValue>, Box<SymValue>),
    /// Division.
    Div(Box<SymValue>, Box<SymValue>),
    /// Fused multiply-add: a * b + c.
    Fma(Box<SymValue>, Box<SymValue>, Box<SymValue>),
    /// Negation.
    Neg(Box<SymValue>),
    /// Absolute value.
    Abs(Box<SymValue>),
    /// Maximum.
    Max(Box<SymValue>, Box<SymValue>),
    /// Minimum.
    Min(Box<SymValue>, Box<SymValue>),
    /// Square root.
    Sqrt(Box<SymValue>),
    /// Reciprocal (1/x).
    Recip(Box<SymValue>),
    /// Reciprocal square root (1/sqrt(x)).
    Rsqrt(Box<SymValue>),
    /// libm function call.
    Call(LibmFn, Vec<SymValue>),
    /// Unknown / untrackable value.
    Unknown(String),
}

/// Recognized libm functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LibmFn {
    Expf,
    Sqrtf,
    Tanhf,
    Logf,
    Fabsf,
}

impl std::fmt::Display for LibmFn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LibmFn::Expf => write!(f, "expf"),
            LibmFn::Sqrtf => write!(f, "sqrtf"),
            LibmFn::Tanhf => write!(f, "tanhf"),
            LibmFn::Logf => write!(f, "logf"),
            LibmFn::Fabsf => write!(f, "fabsf"),
        }
    }
}

impl std::fmt::Display for SymValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SymValue::Param(n) => write!(f, "param({n})"),
            SymValue::Load { base, .. } => write!(f, "load({base})"),
            SymValue::Const(v) => write!(f, "c:{v:?}"),
            SymValue::Add(a, b) => write!(f, "({a} + {b})"),
            SymValue::Sub(a, b) => write!(f, "({a} - {b})"),
            SymValue::Mul(a, b) => write!(f, "({a} * {b})"),
            SymValue::Div(a, b) => write!(f, "({a} / {b})"),
            SymValue::Fma(a, b, c) => write!(f, "fma({a}, {b}, {c})"),
            SymValue::Neg(a) => write!(f, "(-{a})"),
            SymValue::Abs(a) => write!(f, "abs({a})"),
            SymValue::Max(a, b) => write!(f, "max({a}, {b})"),
            SymValue::Min(a, b) => write!(f, "min({a}, {b})"),
            SymValue::Sqrt(a) => write!(f, "sqrt({a})"),
            SymValue::Recip(a) => write!(f, "recip({a})"),
            SymValue::Rsqrt(a) => write!(f, "rsqrt({a})"),
            SymValue::Call(func, args) => {
                let args_str: Vec<String> = args.iter().map(|a| format!("{a}")).collect();
                write!(f, "{func}({})", args_str.join(", "))
            }
            SymValue::Unknown(s) => write!(f, "?({s})"),
        }
    }
}

// ---------------------------------------------------------------------------
// Algebraic simplification
// ---------------------------------------------------------------------------

impl SymValue {
    /// Recursively simplify: constant folding + identity elimination.
    pub fn simplify(&self) -> SymValue {
        match self {
            SymValue::Param(_) | SymValue::Const(_) | SymValue::Unknown(_) => self.clone(),

            SymValue::Neg(a) => {
                let a = a.simplify();
                match &a {
                    SymValue::Const(v) => SymValue::Const(-v),
                    SymValue::Neg(inner) => *inner.clone(),
                    _ => SymValue::Neg(Box::new(a)),
                }
            }

            SymValue::Abs(a) => {
                let a = a.simplify();
                match &a {
                    SymValue::Const(v) => SymValue::Const(v.abs()),
                    _ => SymValue::Abs(Box::new(a)),
                }
            }

            SymValue::Add(a, b) => {
                let a = a.simplify();
                let b = b.simplify();
                if let (SymValue::Const(va), SymValue::Const(vb)) = (&a, &b) {
                    return SymValue::Const(va + vb);
                }
                if is_zero(&b) { return a; }
                if is_zero(&a) { return b; }
                SymValue::Add(Box::new(a), Box::new(b))
            }

            SymValue::Sub(a, b) => {
                let a = a.simplify();
                let b = b.simplify();
                if let (SymValue::Const(va), SymValue::Const(vb)) = (&a, &b) {
                    return SymValue::Const(va - vb);
                }
                if is_zero(&b) { return a; }
                if sym_eq(&a, &b) { return SymValue::Const(0.0); }
                SymValue::Sub(Box::new(a), Box::new(b))
            }

            SymValue::Mul(a, b) => {
                let a = a.simplify();
                let b = b.simplify();
                if let (SymValue::Const(va), SymValue::Const(vb)) = (&a, &b) {
                    return SymValue::Const(va * vb);
                }
                if is_zero(&a) || is_zero(&b) { return SymValue::Const(0.0); }
                if is_one(&b) { return a; }
                if is_one(&a) { return b; }
                SymValue::Mul(Box::new(a), Box::new(b))
            }

            SymValue::Div(a, b) => {
                let a = a.simplify();
                let b = b.simplify();
                if let (SymValue::Const(va), SymValue::Const(vb)) = (&a, &b) {
                    if *vb != 0.0 {
                        return SymValue::Const(va / vb);
                    }
                }
                if is_one(&b) { return a; }
                if sym_eq(&a, &b) { return SymValue::Const(1.0); }
                SymValue::Div(Box::new(a), Box::new(b))
            }

            SymValue::Fma(a, b, c) => {
                let a = a.simplify();
                let b = b.simplify();
                let c = c.simplify();
                if let (SymValue::Const(va), SymValue::Const(vb), SymValue::Const(vc)) =
                    (&a, &b, &c)
                {
                    return SymValue::Const(va.mul_add(*vb, *vc));
                }
                SymValue::Fma(Box::new(a), Box::new(b), Box::new(c))
            }

            SymValue::Max(a, b) => {
                let a = a.simplify();
                let b = b.simplify();
                if let (SymValue::Const(va), SymValue::Const(vb)) = (&a, &b) {
                    return SymValue::Const(va.max(*vb));
                }
                SymValue::Max(Box::new(a), Box::new(b))
            }

            SymValue::Min(a, b) => {
                let a = a.simplify();
                let b = b.simplify();
                if let (SymValue::Const(va), SymValue::Const(vb)) = (&a, &b) {
                    return SymValue::Const(va.min(*vb));
                }
                SymValue::Min(Box::new(a), Box::new(b))
            }

            SymValue::Sqrt(a) => {
                let a = a.simplify();
                if let SymValue::Const(v) = &a {
                    return SymValue::Const(v.sqrt());
                }
                SymValue::Sqrt(Box::new(a))
            }

            SymValue::Recip(a) => {
                let a = a.simplify();
                if let SymValue::Const(v) = &a {
                    if *v != 0.0 {
                        return SymValue::Const(1.0 / v);
                    }
                }
                SymValue::Recip(Box::new(a))
            }

            SymValue::Rsqrt(a) => {
                let a = a.simplify();
                if let SymValue::Const(v) = &a {
                    if *v > 0.0 {
                        return SymValue::Const(1.0 / v.sqrt());
                    }
                }
                SymValue::Rsqrt(Box::new(a))
            }

            SymValue::Call(f, args) => {
                let args: Vec<_> = args.iter().map(|a| a.simplify()).collect();
                if args.len() == 1 {
                    if let SymValue::Const(v) = &args[0] {
                        let result = match f {
                            LibmFn::Expf => Some(v.exp()),
                            LibmFn::Sqrtf => Some(v.sqrt()),
                            LibmFn::Tanhf => Some(v.tanh()),
                            LibmFn::Logf => Some(v.ln()),
                            LibmFn::Fabsf => Some(v.abs()),
                        };
                        if let Some(r) = result {
                            return SymValue::Const(r);
                        }
                    }
                }
                SymValue::Call(*f, args)
            }

            SymValue::Load { base, index } => SymValue::Load {
                base: Box::new(base.simplify()),
                index: Box::new(index.simplify()),
            },
        }
    }
}

fn is_zero(v: &SymValue) -> bool {
    matches!(v, SymValue::Const(x) if *x == 0.0)
}

fn is_one(v: &SymValue) -> bool {
    matches!(v, SymValue::Const(x) if *x == 1.0)
}

/// Structural equality check (used for x - x → 0, x / x → 1).
fn sym_eq(a: &SymValue, b: &SymValue) -> bool {
    // Use Display representation as a conservative structural check.
    format!("{a}") == format!("{b}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simplify_add_zero() {
        let v = SymValue::Add(
            Box::new(SymValue::Param(0)),
            Box::new(SymValue::Const(0.0)),
        );
        assert!(matches!(v.simplify(), SymValue::Param(0)));
    }

    #[test]
    fn simplify_zero_add() {
        let v = SymValue::Add(
            Box::new(SymValue::Const(0.0)),
            Box::new(SymValue::Param(0)),
        );
        assert!(matches!(v.simplify(), SymValue::Param(0)));
    }

    #[test]
    fn simplify_mul_one() {
        let v = SymValue::Mul(
            Box::new(SymValue::Param(0)),
            Box::new(SymValue::Const(1.0)),
        );
        assert!(matches!(v.simplify(), SymValue::Param(0)));
    }

    #[test]
    fn simplify_mul_zero() {
        let v = SymValue::Mul(
            Box::new(SymValue::Param(0)),
            Box::new(SymValue::Const(0.0)),
        );
        assert!(matches!(v.simplify(), SymValue::Const(x) if x == 0.0));
    }

    #[test]
    fn simplify_sub_self() {
        let v = SymValue::Sub(
            Box::new(SymValue::Param(0)),
            Box::new(SymValue::Param(0)),
        );
        assert!(matches!(v.simplify(), SymValue::Const(x) if x == 0.0));
    }

    #[test]
    fn simplify_div_self() {
        let v = SymValue::Div(
            Box::new(SymValue::Param(0)),
            Box::new(SymValue::Param(0)),
        );
        assert!(matches!(v.simplify(), SymValue::Const(x) if x == 1.0));
    }

    #[test]
    fn simplify_constant_fold() {
        let v = SymValue::Add(
            Box::new(SymValue::Const(2.0)),
            Box::new(SymValue::Const(3.0)),
        );
        assert!(matches!(v.simplify(), SymValue::Const(x) if x == 5.0));
    }

    #[test]
    fn simplify_nested() {
        // (param(0) + 0) * 1 → param(0)
        let v = SymValue::Mul(
            Box::new(SymValue::Add(
                Box::new(SymValue::Param(0)),
                Box::new(SymValue::Const(0.0)),
            )),
            Box::new(SymValue::Const(1.0)),
        );
        assert!(matches!(v.simplify(), SymValue::Param(0)));
    }

    #[test]
    fn simplify_double_neg() {
        let v = SymValue::Neg(Box::new(SymValue::Neg(Box::new(SymValue::Param(0)))));
        assert!(matches!(v.simplify(), SymValue::Param(0)));
    }

    #[test]
    fn simplify_fma_const() {
        let v = SymValue::Fma(
            Box::new(SymValue::Const(2.0)),
            Box::new(SymValue::Const(3.0)),
            Box::new(SymValue::Const(4.0)),
        );
        assert!(matches!(v.simplify(), SymValue::Const(x) if x == 10.0));
    }
}
