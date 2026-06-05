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
    /// Conditional select: `Select(kind, cond_lhs, cond_rhs, true_val, false_val)`.
    /// Represents `if cond_lhs <kind> cond_rhs { true_val } else { false_val }`.
    Select {
        kind: SelectKind,
        cond_lhs: Box<SymValue>,
        cond_rhs: Box<SymValue>,
        true_val: Box<SymValue>,
        false_val: Box<SymValue>,
    },
    /// Unknown / untrackable value.
    Unknown(String),
}

/// Comparison kind for Select nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectKind {
    /// `>` (unsigned above for floats)
    Gt,
    /// `>=`
    Ge,
    /// `<` (unsigned below for floats)
    Lt,
    /// `<=`
    Le,
    /// `==`
    Eq,
    /// `!=`
    Ne,
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

impl std::fmt::Display for SelectKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SelectKind::Gt => write!(f, ">"),
            SelectKind::Ge => write!(f, ">="),
            SelectKind::Lt => write!(f, "<"),
            SelectKind::Le => write!(f, "<="),
            SelectKind::Eq => write!(f, "=="),
            SelectKind::Ne => write!(f, "!="),
        }
    }
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
            SymValue::Select { kind, cond_lhs, cond_rhs, true_val, false_val } => {
                write!(f, "select({cond_lhs} {kind} {cond_rhs}, {true_val}, {false_val})")
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

            SymValue::Select { kind, cond_lhs, cond_rhs, true_val, false_val } => {
                let cl = cond_lhs.simplify();
                let cr = cond_rhs.simplify();
                let tv = true_val.simplify();
                let fv = false_val.simplify();

                // Pattern: Select(a > b, a, b) → Max(a, b)
                //          Select(a >= b, a, b) → Max(a, b)
                //          Select(a < b, a, b) → Min(a, b)
                //          Select(a <= b, a, b) → Min(a, b)
                // Also handle the swapped case:
                //          Select(a > b, b, a) → Min(a, b)
                //          Select(a < b, b, a) → Max(a, b)
                let cl_s = format!("{cl}");
                let cr_s = format!("{cr}");
                let tv_s = format!("{tv}");
                let fv_s = format!("{fv}");

                // true_val == cond_lhs && false_val == cond_rhs
                if tv_s == cl_s && fv_s == cr_s {
                    match kind {
                        SelectKind::Gt | SelectKind::Ge => {
                            return SymValue::Max(Box::new(cl), Box::new(cr));
                        }
                        SelectKind::Lt | SelectKind::Le => {
                            return SymValue::Min(Box::new(cl), Box::new(cr));
                        }
                        _ => {}
                    }
                }
                // true_val == cond_rhs && false_val == cond_lhs (swapped)
                if tv_s == cr_s && fv_s == cl_s {
                    match kind {
                        SelectKind::Gt | SelectKind::Ge => {
                            return SymValue::Min(Box::new(cl), Box::new(cr));
                        }
                        SelectKind::Lt | SelectKind::Le => {
                            return SymValue::Max(Box::new(cl), Box::new(cr));
                        }
                        _ => {}
                    }
                }

                // Constant condition folding
                if let (SymValue::Const(lv), SymValue::Const(rv)) = (&cl, &cr) {
                    let cond = match kind {
                        SelectKind::Gt => lv > rv,
                        SelectKind::Ge => lv >= rv,
                        SelectKind::Lt => (lv) < rv,
                        SelectKind::Le => lv <= rv,
                        SelectKind::Eq => (lv - rv).abs() < f64::EPSILON,
                        SelectKind::Ne => (lv - rv).abs() >= f64::EPSILON,
                    };
                    return if cond { tv } else { fv };
                }

                // Same true/false → collapse
                if tv_s == fv_s {
                    return tv;
                }

                SymValue::Select {
                    kind: *kind,
                    cond_lhs: Box::new(cl),
                    cond_rhs: Box::new(cr),
                    true_val: Box::new(tv),
                    false_val: Box::new(fv),
                }
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

    // --- New tests (wave-12kaa) ---

    #[test]
    fn display_param_format() {
        // Arrange
        let v = SymValue::Param(3);
        // Act
        let s = format!("{v}");
        // Assert
        assert_eq!(s, "param(3)");
    }

    #[test]
    fn display_select_kind_all_variants() {
        // Arrange & Act & Assert
        assert_eq!(format!("{}", SelectKind::Gt), ">");
        assert_eq!(format!("{}", SelectKind::Ge), ">=");
        assert_eq!(format!("{}", SelectKind::Lt), "<");
        assert_eq!(format!("{}", SelectKind::Le), "<=");
        assert_eq!(format!("{}", SelectKind::Eq), "==");
        assert_eq!(format!("{}", SelectKind::Ne), "!=");
    }

    #[test]
    fn display_libm_fn_all_variants() {
        // Arrange & Act & Assert
        assert_eq!(format!("{}", LibmFn::Expf), "expf");
        assert_eq!(format!("{}", LibmFn::Sqrtf), "sqrtf");
        assert_eq!(format!("{}", LibmFn::Tanhf), "tanhf");
        assert_eq!(format!("{}", LibmFn::Logf), "logf");
        assert_eq!(format!("{}", LibmFn::Fabsf), "fabsf");
    }

    #[test]
    fn select_kind_partial_eq_copy() {
        // Arrange
        let a = SelectKind::Gt;
        let b = a; // Copy
        // Act & Assert
        assert_eq!(a, b);
        assert_ne!(a, SelectKind::Lt);
    }

    #[test]
    fn libm_fn_partial_eq_copy() {
        // Arrange
        let a = LibmFn::Tanhf;
        let b = a; // Copy
        // Act & Assert
        assert_eq!(a, b);
        assert_ne!(a, LibmFn::Expf);
    }

    #[test]
    fn simplify_div_by_one() {
        // Arrange: param(0) / 1.0 → param(0)
        let v = SymValue::Div(
            Box::new(SymValue::Param(0)),
            Box::new(SymValue::Const(1.0)),
        );
        // Act
        let result = v.simplify();
        // Assert
        assert!(matches!(result, SymValue::Param(0)));
    }

    #[test]
    fn simplify_sub_zero_rhs() {
        // Arrange: param(0) - 0.0 → param(0)
        let v = SymValue::Sub(
            Box::new(SymValue::Param(0)),
            Box::new(SymValue::Const(0.0)),
        );
        // Act
        let result = v.simplify();
        // Assert
        assert!(matches!(result, SymValue::Param(0)));
    }

    #[test]
    fn simplify_sqrt_const() {
        // Arrange: sqrt(9.0) → 3.0
        let v = SymValue::Sqrt(Box::new(SymValue::Const(9.0)));
        // Act
        let result = v.simplify();
        // Assert
        assert!(matches!(result, SymValue::Const(x) if (x - 3.0).abs() < 1e-10));
    }

    #[test]
    fn simplify_recip_const() {
        // Arrange: recip(4.0) → 0.25
        let v = SymValue::Recip(Box::new(SymValue::Const(4.0)));
        // Act
        let result = v.simplify();
        // Assert
        assert!(matches!(result, SymValue::Const(x) if (x - 0.25).abs() < 1e-10));
    }

    #[test]
    fn simplify_abs_const() {
        // Arrange: abs(-7.0) → 7.0
        let v = SymValue::Abs(Box::new(SymValue::Const(-7.0)));
        // Act
        let result = v.simplify();
        // Assert
        assert!(matches!(result, SymValue::Const(x) if x == 7.0));
    }

    #[test]
    fn simplify_select_const_condition_true() {
        // Arrange: select(5.0 > 3.0, 10.0, 20.0) → 10.0
        let v = SymValue::Select {
            kind: SelectKind::Gt,
            cond_lhs: Box::new(SymValue::Const(5.0)),
            cond_rhs: Box::new(SymValue::Const(3.0)),
            true_val: Box::new(SymValue::Const(10.0)),
            false_val: Box::new(SymValue::Const(20.0)),
        };
        // Act
        let result = v.simplify();
        // Assert
        assert!(matches!(result, SymValue::Const(x) if x == 10.0));
    }

    #[test]
    fn simplify_select_same_true_false() {
        // Arrange: select(param(0) > param(1), 42.0, 42.0) → 42.0
        let v = SymValue::Select {
            kind: SelectKind::Gt,
            cond_lhs: Box::new(SymValue::Param(0)),
            cond_rhs: Box::new(SymValue::Param(1)),
            true_val: Box::new(SymValue::Const(42.0)),
            false_val: Box::new(SymValue::Const(42.0)),
        };
        // Act
        let result = v.simplify();
        // Assert
        assert!(matches!(result, SymValue::Const(x) if x == 42.0));
    }

    #[test]
    fn simplify_select_to_max() {
        // Arrange: select(param(0) > param(1), param(0), param(1)) → Max(param(0), param(1))
        let v = SymValue::Select {
            kind: SelectKind::Gt,
            cond_lhs: Box::new(SymValue::Param(0)),
            cond_rhs: Box::new(SymValue::Param(1)),
            true_val: Box::new(SymValue::Param(0)),
            false_val: Box::new(SymValue::Param(1)),
        };
        // Act
        let result = v.simplify();
        // Assert
        assert!(matches!(result, SymValue::Max(_, _)));
    }

    #[test]
    fn simplify_call_expf_const() {
        // Arrange: expf(0.0) → 1.0
        let v = SymValue::Call(LibmFn::Expf, vec![SymValue::Const(0.0)]);
        // Act
        let result = v.simplify();
        // Assert
        assert!(matches!(result, SymValue::Const(x) if (x - 1.0).abs() < 1e-10));
    }

    #[test]
    fn clone_preserves_structure() {
        // Arrange
        let original = SymValue::Add(
            Box::new(SymValue::Mul(
                Box::new(SymValue::Param(0)),
                Box::new(SymValue::Const(2.0)),
            )),
            Box::new(SymValue::Const(1.0)),
        );
        // Act
        let cloned = original.clone();
        // Assert: Display output is identical (structural equality proxy)
        assert_eq!(format!("{original}"), format!("{cloned}"));
    }

    #[test]
    fn display_unknown_format() {
        // Arrange
        let v = SymValue::Unknown("extern".to_string());
        // Act
        let s = format!("{v}");
        // Assert
        assert_eq!(s, "?(extern)");
    }

    // ── Additional tests (wave-12kab) ──────────────────────────────────────

    #[test]
    fn display_const_format() {
        // Arrange
        let v = SymValue::Const(3.14);
        // Act
        let s = format!("{v}");
        // Assert
        assert_eq!(s, "c:3.14");
    }

    #[test]
    fn display_add_format() {
        // Arrange
        let v = SymValue::Add(
            Box::new(SymValue::Const(1.0)),
            Box::new(SymValue::Const(2.0)),
        );
        // Act
        let s = format!("{v}");
        // Assert
        assert_eq!(s, "(c:1.0 + c:2.0)");
    }

    #[test]
    fn display_sub_format() {
        let v = SymValue::Sub(
            Box::new(SymValue::Param(0)),
            Box::new(SymValue::Param(1)),
        );
        assert_eq!(format!("{v}"), "(param(0) - param(1))");
    }

    #[test]
    fn display_mul_format() {
        let v = SymValue::Mul(
            Box::new(SymValue::Param(0)),
            Box::new(SymValue::Const(2.0)),
        );
        assert_eq!(format!("{v}"), "(param(0) * c:2.0)");
    }

    #[test]
    fn display_div_format() {
        let v = SymValue::Div(
            Box::new(SymValue::Param(0)),
            Box::new(SymValue::Const(4.0)),
        );
        assert_eq!(format!("{v}"), "(param(0) / c:4.0)");
    }

    #[test]
    fn display_fma_format() {
        let v = SymValue::Fma(
            Box::new(SymValue::Param(0)),
            Box::new(SymValue::Param(1)),
            Box::new(SymValue::Const(1.0)),
        );
        let s = format!("{v}");
        assert!(s.starts_with("fma("));
        assert!(s.contains("param(0)"));
        assert!(s.contains("param(1)"));
    }

    #[test]
    fn display_neg_format() {
        let v = SymValue::Neg(Box::new(SymValue::Param(0)));
        assert_eq!(format!("{v}"), "(-param(0))");
    }

    #[test]
    fn display_abs_format() {
        let v = SymValue::Abs(Box::new(SymValue::Param(2)));
        assert_eq!(format!("{v}"), "abs(param(2))");
    }

    #[test]
    fn display_max_format() {
        let v = SymValue::Max(
            Box::new(SymValue::Param(0)),
            Box::new(SymValue::Param(1)),
        );
        assert_eq!(format!("{v}"), "max(param(0), param(1))");
    }

    #[test]
    fn display_min_format() {
        let v = SymValue::Min(
            Box::new(SymValue::Const(0.0)),
            Box::new(SymValue::Param(0)),
        );
        assert_eq!(format!("{v}"), "min(c:0.0, param(0))");
    }

    #[test]
    fn display_sqrt_format() {
        let v = SymValue::Sqrt(Box::new(SymValue::Param(0)));
        assert_eq!(format!("{v}"), "sqrt(param(0))");
    }

    #[test]
    fn display_recip_format() {
        let v = SymValue::Recip(Box::new(SymValue::Const(2.0)));
        assert_eq!(format!("{v}"), "recip(c:2.0)");
    }

    #[test]
    fn display_rsqrt_format() {
        let v = SymValue::Rsqrt(Box::new(SymValue::Const(4.0)));
        assert_eq!(format!("{v}"), "rsqrt(c:4.0)");
    }

    #[test]
    fn display_call_format() {
        let v = SymValue::Call(LibmFn::Tanhf, vec![SymValue::Param(0)]);
        assert_eq!(format!("{v}"), "tanhf(param(0))");
    }

    #[test]
    fn display_load_format() {
        let v = SymValue::Load {
            base: Box::new(SymValue::Param(0)),
            index: Box::new(SymValue::Const(1.0)),
        };
        let s = format!("{v}");
        assert!(s.starts_with("load("));
        assert!(s.contains("param(0)"));
    }

    #[test]
    fn simplify_neg_const() {
        let v = SymValue::Neg(Box::new(SymValue::Const(5.0)));
        assert!(matches!(v.simplify(), SymValue::Const(x) if x == -5.0));
    }

    #[test]
    fn simplify_rsqrt_const_positive() {
        let v = SymValue::Rsqrt(Box::new(SymValue::Const(16.0)));
        let result = v.simplify();
        match result {
            SymValue::Const(x) => assert!((x - 0.25).abs() < 1e-10, "1/sqrt(16) = 0.25"),
            other => panic!("expected Const, got {:?}", other),
        }
    }

    #[test]
    fn simplify_rsqrt_const_zero_stays_symbolic() {
        // rsqrt(0) is undefined, should not fold
        let v = SymValue::Rsqrt(Box::new(SymValue::Const(0.0)));
        assert!(matches!(v.simplify(), SymValue::Rsqrt(_)));
    }

    #[test]
    fn simplify_rsqrt_const_negative_stays_symbolic() {
        // rsqrt(-1) is undefined (sqrt of negative), should not fold
        let v = SymValue::Rsqrt(Box::new(SymValue::Const(-1.0)));
        assert!(matches!(v.simplify(), SymValue::Rsqrt(_)));
    }

    #[test]
    fn simplify_recip_zero_stays_symbolic() {
        // recip(0) is division by zero, should not fold
        let v = SymValue::Recip(Box::new(SymValue::Const(0.0)));
        assert!(matches!(v.simplify(), SymValue::Recip(_)));
    }

    #[test]
    fn simplify_call_sqrtf_const() {
        let v = SymValue::Call(LibmFn::Sqrtf, vec![SymValue::Const(25.0)]);
        assert!(matches!(v.simplify(), SymValue::Const(x) if (x - 5.0).abs() < 1e-10));
    }

    #[test]
    fn simplify_call_logf_const() {
        let v = SymValue::Call(LibmFn::Logf, vec![SymValue::Const(1.0)]);
        assert!(matches!(v.simplify(), SymValue::Const(x) if x.abs() < 1e-10));
    }

    #[test]
    fn simplify_call_fabsf_const() {
        let v = SymValue::Call(LibmFn::Fabsf, vec![SymValue::Const(-42.0)]);
        assert!(matches!(v.simplify(), SymValue::Const(x) if x == 42.0));
    }

    #[test]
    fn simplify_select_const_condition_false() {
        // select(3.0 > 5.0, 10.0, 20.0) → 20.0
        let v = SymValue::Select {
            kind: SelectKind::Gt,
            cond_lhs: Box::new(SymValue::Const(3.0)),
            cond_rhs: Box::new(SymValue::Const(5.0)),
            true_val: Box::new(SymValue::Const(10.0)),
            false_val: Box::new(SymValue::Const(20.0)),
        };
        assert!(matches!(v.simplify(), SymValue::Const(x) if x == 20.0));
    }

    #[test]
    fn simplify_select_to_min() {
        // select(param(0) < param(1), param(0), param(1)) → Min(param(0), param(1))
        let v = SymValue::Select {
            kind: SelectKind::Lt,
            cond_lhs: Box::new(SymValue::Param(0)),
            cond_rhs: Box::new(SymValue::Param(1)),
            true_val: Box::new(SymValue::Param(0)),
            false_val: Box::new(SymValue::Param(1)),
        };
        assert!(matches!(v.simplify(), SymValue::Min(_, _)));
    }

    #[test]
    fn simplify_select_ge_to_max() {
        // select(param(0) >= param(1), param(0), param(1)) → Max(param(0), param(1))
        let v = SymValue::Select {
            kind: SelectKind::Ge,
            cond_lhs: Box::new(SymValue::Param(0)),
            cond_rhs: Box::new(SymValue::Param(1)),
            true_val: Box::new(SymValue::Param(0)),
            false_val: Box::new(SymValue::Param(1)),
        };
        assert!(matches!(v.simplify(), SymValue::Max(_, _)));
    }

    #[test]
    fn simplify_select_swapped_gt_to_min() {
        // select(param(0) > param(1), param(1), param(0)) → Min(param(0), param(1))
        let v = SymValue::Select {
            kind: SelectKind::Gt,
            cond_lhs: Box::new(SymValue::Param(0)),
            cond_rhs: Box::new(SymValue::Param(1)),
            true_val: Box::new(SymValue::Param(1)),
            false_val: Box::new(SymValue::Param(0)),
        };
        assert!(matches!(v.simplify(), SymValue::Min(_, _)));
    }

    #[test]
    fn simplify_max_const() {
        let v = SymValue::Max(
            Box::new(SymValue::Const(3.0)),
            Box::new(SymValue::Const(7.0)),
        );
        assert!(matches!(v.simplify(), SymValue::Const(x) if x == 7.0));
    }

    #[test]
    fn simplify_min_const() {
        let v = SymValue::Min(
            Box::new(SymValue::Const(3.0)),
            Box::new(SymValue::Const(7.0)),
        );
        assert!(matches!(v.simplify(), SymValue::Const(x) if x == 3.0));
    }

    #[test]
    fn simplify_mul_by_one_lhs() {
        // 1.0 * param(0) → param(0)
        let v = SymValue::Mul(
            Box::new(SymValue::Const(1.0)),
            Box::new(SymValue::Param(0)),
        );
        assert!(matches!(v.simplify(), SymValue::Param(0)));
    }

    #[test]
    fn simplify_div_const_by_const() {
        let v = SymValue::Div(
            Box::new(SymValue::Const(10.0)),
            Box::new(SymValue::Const(4.0)),
        );
        assert!(matches!(v.simplify(), SymValue::Const(x) if (x - 2.5).abs() < 1e-10));
    }

    #[test]
    fn simplify_sub_const_fold() {
        let v = SymValue::Sub(
            Box::new(SymValue::Const(10.0)),
            Box::new(SymValue::Const(3.0)),
        );
        assert!(matches!(v.simplify(), SymValue::Const(x) if x == 7.0));
    }

    #[test]
    fn simplify_call_with_non_const_stays() {
        let v = SymValue::Call(LibmFn::Expf, vec![SymValue::Param(0)]);
        assert!(matches!(v.simplify(), SymValue::Call(LibmFn::Expf, _)));
    }

    #[test]
    fn simplify_load_propagates() {
        let v = SymValue::Load {
            base: Box::new(SymValue::Add(
                Box::new(SymValue::Const(1.0)),
                Box::new(SymValue::Const(2.0)),
            )),
            index: Box::new(SymValue::Const(0.0)),
        };
        let result = v.simplify();
        // base should be folded to Const(3.0)
        match result {
            SymValue::Load { base, .. } => {
                assert!(matches!(*base, SymValue::Const(3.0)),
                    "base should be folded to Const(3.0)");
            }
            other => panic!("expected Load, got {:?}", other),
        }
    }

    #[test]
    fn simplify_div_by_zero_stays_symbolic() {
        let v = SymValue::Div(
            Box::new(SymValue::Const(1.0)),
            Box::new(SymValue::Const(0.0)),
        );
        // Should not fold (division by zero)
        assert!(matches!(v.simplify(), SymValue::Div(_, _)));
    }
}
