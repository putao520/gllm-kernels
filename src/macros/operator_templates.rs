/// Defines element-wise operations (add, mul, silu, etc.)
#[macro_export]
macro_rules! define_element_wise_ops {
    ($isa:ident, $elem:ident) => {
        #[inline(always)]
        pub fn add(a: &[$elem], b: &[$elem], out: &mut [$elem]) {
            const LANES: usize = $crate::simd_primitive!($isa, $elem, lanes);
            let len = a.len();
            assert!(b.len() == len && out.len() == len);

            let mut i = 0;
            while i + LANES <= len {
                #[allow(unused_unsafe)]
                unsafe {
                    let va = $crate::simd_primitive!($isa, $elem, load, a.as_ptr().add(i));
                    let vb = $crate::simd_primitive!($isa, $elem, load, b.as_ptr().add(i));
                    let res = $crate::simd_primitive!($isa, $elem, add, va, vb);
                    $crate::simd_primitive!($isa, $elem, store, out.as_mut_ptr().add(i), res);
                }
                i += LANES;
            }

            // Handle remainder
            #[allow(clippy::manual_memcpy)]
            #[allow(unused_unsafe)]
            while i < len {
                out[i] = a[i] + b[i]; 
                i += 1;
            }
        }

        #[inline(always)]
        pub fn mul(a: &[$elem], b: &[$elem], out: &mut [$elem]) {
            const LANES: usize = $crate::simd_primitive!($isa, $elem, lanes);
            let len = a.len();
            assert!(b.len() == len && out.len() == len);

            let mut i = 0;
            while i + LANES <= len {
                #[allow(unused_unsafe)]
                unsafe {
                    let va = $crate::simd_primitive!($isa, $elem, load, a.as_ptr().add(i));
                    let vb = $crate::simd_primitive!($isa, $elem, load, b.as_ptr().add(i));
                    let res = $crate::simd_primitive!($isa, $elem, mul, va, vb);
                    $crate::simd_primitive!($isa, $elem, store, out.as_mut_ptr().add(i), res);
                }
                i += LANES;
            }

            // Handle remainder
            #[allow(unused_unsafe)]
            while i < len {
                out[i] = a[i] * b[i];
                i += 1;
            }
        }

        #[inline(always)]
        pub fn silu(a: &[$elem], out: &mut [$elem]) {
            const LANES: usize = $crate::simd_primitive!($isa, $elem, lanes);
            let len = a.len();
            assert!(out.len() == len);

            let mut i = 0;
            while i + LANES <= len {
                #[allow(unused_unsafe)]
                unsafe {
                    let va = $crate::simd_primitive!($isa, $elem, load, a.as_ptr().add(i));
                    let neg_va = $crate::simd_primitive!($isa, $elem, neg, va);
                    let exp_neg = $crate::simd_primitive!($isa, $elem, exp, neg_va);
                    let one = $crate::simd_primitive!($isa, $elem, splat, <$elem as Element>::ONE);
                    let one_plus_exp = $crate::simd_primitive!($isa, $elem, add, one, exp_neg);
                    let sigmoid = $crate::simd_primitive!($isa, $elem, recip, one_plus_exp);
                    let res = $crate::simd_primitive!($isa, $elem, mul, va, sigmoid);
                    $crate::simd_primitive!($isa, $elem, store, out.as_mut_ptr().add(i), res);
                }
                i += LANES;
            }

             // Handle remainder for scalar fallback or small arrays
            while i < len {
                let val_f = a[i].to_f32();
                let sigmoid = 1.0f32 / (1.0f32 + (-val_f).exp());
                out[i] = <$elem as Element>::from_f32(val_f * sigmoid);
                i += 1;
            }
        }
    };
}

/// Defines additional BLAS-1 vector operations (sub, div, scale, axpy, sum, max, min, sum_squares)
#[macro_export]
macro_rules! define_blas1_ops {
    ($isa:ident, $elem:ident) => {
        #[inline(always)]
        pub fn sub(a: &[$elem], b: &[$elem], out: &mut [$elem]) {
            const LANES: usize = $crate::simd_primitive!($isa, $elem, lanes);
            let len = a.len();
            assert!(b.len() == len && out.len() == len);
            let mut i = 0;
            while i + LANES <= len {
                #[allow(unused_unsafe)]
                unsafe {
                    let va = $crate::simd_primitive!($isa, $elem, load, a.as_ptr().add(i));
                    let vb = $crate::simd_primitive!($isa, $elem, load, b.as_ptr().add(i));
                    let res = $crate::simd_primitive!($isa, $elem, sub, va, vb);
                    $crate::simd_primitive!($isa, $elem, store, out.as_mut_ptr().add(i), res);
                }
                i += LANES;
            }
            while i < len { out[i] = a[i] - b[i]; i += 1; }
        }

        #[inline(always)]
        pub fn div(a: &[$elem], b: &[$elem], out: &mut [$elem]) {
            const LANES: usize = $crate::simd_primitive!($isa, $elem, lanes);
            let len = a.len();
            assert!(b.len() == len && out.len() == len);
            let mut i = 0;
            while i + LANES <= len {
                #[allow(unused_unsafe)]
                unsafe {
                    let va = $crate::simd_primitive!($isa, $elem, load, a.as_ptr().add(i));
                    let vb = $crate::simd_primitive!($isa, $elem, load, b.as_ptr().add(i));
                    let res = $crate::simd_primitive!($isa, $elem, div, va, vb);
                    $crate::simd_primitive!($isa, $elem, store, out.as_mut_ptr().add(i), res);
                }
                i += LANES;
            }
            while i < len { out[i] = a[i] / b[i]; i += 1; }
        }

        /// scale: out[i] = a[i] * scalar
        #[inline(always)]
        pub fn scale(a: &[$elem], scalar: $elem, out: &mut [$elem]) {
            const LANES: usize = $crate::simd_primitive!($isa, $elem, lanes);
            let len = a.len();
            assert!(out.len() == len);
            let mut i = 0;
            #[allow(unused_unsafe)]
            let vs = unsafe { $crate::simd_primitive!($isa, $elem, splat, scalar) };
            while i + LANES <= len {
                #[allow(unused_unsafe)]
                unsafe {
                    let va = $crate::simd_primitive!($isa, $elem, load, a.as_ptr().add(i));
                    let res = $crate::simd_primitive!($isa, $elem, mul, va, vs);
                    $crate::simd_primitive!($isa, $elem, store, out.as_mut_ptr().add(i), res);
                }
                i += LANES;
            }
            while i < len { out[i] = a[i] * scalar; i += 1; }
        }

        /// axpy: y[i] += alpha * x[i]
        #[inline(always)]
        pub fn axpy(alpha: $elem, x: &[$elem], y: &mut [$elem]) {
            const LANES: usize = $crate::simd_primitive!($isa, $elem, lanes);
            let len = x.len();
            assert!(y.len() == len);
            let mut i = 0;
            #[allow(unused_unsafe)]
            let va = unsafe { $crate::simd_primitive!($isa, $elem, splat, alpha) };
            while i + LANES <= len {
                #[allow(unused_unsafe)]
                unsafe {
                    let vx = $crate::simd_primitive!($isa, $elem, load, x.as_ptr().add(i));
                    let vy = $crate::simd_primitive!($isa, $elem, load, y.as_ptr().add(i));
                    let res = $crate::simd_primitive!($isa, $elem, fma, va, vx, vy);
                    $crate::simd_primitive!($isa, $elem, store, y.as_mut_ptr().add(i), res);
                }
                i += LANES;
            }
            while i < len { y[i] += alpha * x[i]; i += 1; }
        }

        /// sum: returns sum of all elements
        #[inline(always)]
        pub fn sum(a: &[$elem]) -> $elem {
            const LANES: usize = $crate::simd_primitive!($isa, $elem, lanes);
            let len = a.len();
            let mut i = 0;
            #[allow(unused_unsafe)]
            let mut acc = unsafe { $crate::simd_primitive!($isa, $elem, zero) };
            while i + LANES <= len {
                #[allow(unused_unsafe)]
                unsafe {
                    let va = $crate::simd_primitive!($isa, $elem, load, a.as_ptr().add(i));
                    acc = $crate::simd_primitive!($isa, $elem, add, acc, va);
                }
                i += LANES;
            }
            #[allow(unused_unsafe)]
            let mut result: f32 = unsafe { $crate::simd_primitive!($isa, $elem, reduce_sum, acc) };
            while i < len { result += a[i].to_f32(); i += 1; }
            <$elem as Element>::from_f32(result)
        }

        /// max_val: returns maximum element
        #[inline(always)]
        pub fn max_val(a: &[$elem]) -> $elem {
            const LANES: usize = $crate::simd_primitive!($isa, $elem, lanes);
            let len = a.len();
            assert!(len > 0);
            let mut i = 0;
            #[allow(unused_unsafe)]
            let mut acc = unsafe { $crate::simd_primitive!($isa, $elem, splat, <$elem as Element>::from_f32(f32::NEG_INFINITY)) };
            while i + LANES <= len {
                #[allow(unused_unsafe)]
                unsafe {
                    let va = $crate::simd_primitive!($isa, $elem, load, a.as_ptr().add(i));
                    acc = $crate::simd_primitive!($isa, $elem, max, acc, va);
                }
                i += LANES;
            }
            #[allow(unused_unsafe)]
            let mut result: f32 = unsafe { $crate::simd_primitive!($isa, $elem, reduce_max, acc) };
            while i < len { let v = a[i].to_f32(); if v > result { result = v; } i += 1; }
            <$elem as Element>::from_f32(result)
        }

        /// sum_squares: returns sum of x[i]^2
        #[inline(always)]
        pub fn sum_squares(a: &[$elem]) -> $elem {
            const LANES: usize = $crate::simd_primitive!($isa, $elem, lanes);
            let len = a.len();
            let mut i = 0;
            #[allow(unused_unsafe)]
            let mut acc = unsafe { $crate::simd_primitive!($isa, $elem, zero) };
            while i + LANES <= len {
                #[allow(unused_unsafe)]
                unsafe {
                    let va = $crate::simd_primitive!($isa, $elem, load, a.as_ptr().add(i));
                    acc = $crate::simd_primitive!($isa, $elem, fma, va, va, acc);
                }
                i += LANES;
            }
            #[allow(unused_unsafe)]
            let mut result: f32 = unsafe { $crate::simd_primitive!($isa, $elem, reduce_sum, acc) };
            while i < len { let v = a[i].to_f32(); result += v * v; i += 1; }
            <$elem as Element>::from_f32(result)
        }

        /// dot: returns dot product of a and b (single pass FMA + reduce)
        #[inline(always)]
        pub fn dot(a: &[$elem], b: &[$elem]) -> $elem {
            const LANES: usize = $crate::simd_primitive!($isa, $elem, lanes);
            let len = a.len();
            debug_assert_eq!(len, b.len());
            let mut i = 0;
            #[allow(unused_unsafe)]
            let mut acc = unsafe { $crate::simd_primitive!($isa, $elem, zero) };
            while i + LANES <= len {
                #[allow(unused_unsafe)]
                unsafe {
                    let va = $crate::simd_primitive!($isa, $elem, load, a.as_ptr().add(i));
                    let vb = $crate::simd_primitive!($isa, $elem, load, b.as_ptr().add(i));
                    acc = $crate::simd_primitive!($isa, $elem, fma, va, vb, acc);
                }
                i += LANES;
            }
            #[allow(unused_unsafe)]
            let mut result: f32 = unsafe { $crate::simd_primitive!($isa, $elem, reduce_sum, acc) };
            while i < len { result += a[i].to_f32() * b[i].to_f32(); i += 1; }
            <$elem as Element>::from_f32(result)
        }

        /// bias_rows: c[row] += bias for each row (fused SIMD, single dispatch)
        /// c: [m, n] row-major, bias: [n]
        #[inline(always)]
        pub fn bias_rows(c: &mut [$elem], bias: &[$elem], m: usize, n: usize) {
            const LANES: usize = $crate::simd_primitive!($isa, $elem, lanes);
            debug_assert_eq!(c.len(), m * n);
            debug_assert_eq!(bias.len(), n);
            for row in 0..m {
                let off = row * n;
                let mut j = 0;
                while j + LANES <= n {
                    #[allow(unused_unsafe)]
                    unsafe {
                        let vc = $crate::simd_primitive!($isa, $elem, load, c.as_ptr().add(off + j));
                        let vb = $crate::simd_primitive!($isa, $elem, load, bias.as_ptr().add(j));
                        let res = $crate::simd_primitive!($isa, $elem, add, vc, vb);
                        $crate::simd_primitive!($isa, $elem, store, c.as_mut_ptr().add(off + j), res);
                    }
                    j += LANES;
                }
                while j < n { c[off + j] = c[off + j] + bias[j]; j += 1; }
            }
        }
    };
}

/// Defines activation functions (relu, gelu, tanh, softmax, swiglu, exp)
#[macro_export]
macro_rules! define_activation_ops {
    ($isa:ident, $elem:ident) => {
        /// relu: out[i] = max(0, a[i])
        #[inline(always)]
        pub fn relu(a: &[$elem], out: &mut [$elem]) {
            const LANES: usize = $crate::simd_primitive!($isa, $elem, lanes);
            let len = a.len();
            assert!(out.len() == len);
            let mut i = 0;
            #[allow(unused_unsafe)]
            let vz = unsafe { $crate::simd_primitive!($isa, $elem, zero) };
            while i + LANES <= len {
                #[allow(unused_unsafe)]
                unsafe {
                    let va = $crate::simd_primitive!($isa, $elem, load, a.as_ptr().add(i));
                    let res = $crate::simd_primitive!($isa, $elem, max, va, vz);
                    $crate::simd_primitive!($isa, $elem, store, out.as_mut_ptr().add(i), res);
                }
                i += LANES;
            }
            while i < len { out[i] = <$elem as Element>::max(a[i], <$elem as Element>::ZERO); i += 1; }
        }

        /// gelu: out[i] = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        #[inline(always)]
        pub fn gelu(a: &[$elem], out: &mut [$elem]) {
            const LANES: usize = $crate::simd_primitive!($isa, $elem, lanes);
            let len = a.len();
            assert!(out.len() == len);

            let mut i = 0;
            while i + LANES <= len {
                #[allow(unused_unsafe)]
                unsafe {
                    let vx = $crate::simd_primitive!($isa, $elem, load, a.as_ptr().add(i));
                    let half = $crate::simd_primitive!($isa, $elem, splat, <$elem as Element>::from_f32(0.5));
                    let one = $crate::simd_primitive!($isa, $elem, splat, <$elem as Element>::ONE);
                    let vc = $crate::simd_primitive!($isa, $elem, splat, <$elem as Element>::from_f32(0.044715));
                    let vs = $crate::simd_primitive!($isa, $elem, splat, <$elem as Element>::from_f32(0.7978845608));

                    // x^3
                    let x2 = $crate::simd_primitive!($isa, $elem, mul, vx, vx);
                    let x3 = $crate::simd_primitive!($isa, $elem, mul, x2, vx);
                    // x + 0.044715 * x^3
                    let inner = $crate::simd_primitive!($isa, $elem, fma, vc, x3, vx);
                    // sqrt(2/pi) * inner
                    let scaled = $crate::simd_primitive!($isa, $elem, mul, vs, inner);
                    // tanh via (exp(2x)-1)/(exp(2x)+1)
                    let two = $crate::simd_primitive!($isa, $elem, splat, <$elem as Element>::from_f32(2.0));
                    let two_x = $crate::simd_primitive!($isa, $elem, mul, two, scaled);
                    let exp_2x = $crate::simd_primitive!($isa, $elem, exp, two_x);
                    let num = $crate::simd_primitive!($isa, $elem, sub, exp_2x, one);
                    let den = $crate::simd_primitive!($isa, $elem, add, exp_2x, one);
                    let tanh_val = $crate::simd_primitive!($isa, $elem, div, num, den);
                    // 0.5 * x * (1 + tanh)
                    let one_plus_tanh = $crate::simd_primitive!($isa, $elem, add, one, tanh_val);
                    let half_x = $crate::simd_primitive!($isa, $elem, mul, half, vx);
                    let res = $crate::simd_primitive!($isa, $elem, mul, half_x, one_plus_tanh);
                    $crate::simd_primitive!($isa, $elem, store, out.as_mut_ptr().add(i), res);
                }
                i += LANES;
            }
            // Scalar remainder (f32 arithmetic)
            while i < len {
                let x = a[i].to_f32();
                let inner = 0.7978845608f32 * (x + 0.044715f32 * x * x * x);
                let e2x = (2.0f32 * inner).exp();
                let tanh_val = (e2x - 1.0f32) / (e2x + 1.0f32);
                out[i] = <$elem as Element>::from_f32(0.5f32 * x * (1.0f32 + tanh_val));
                i += 1;
            }
        }

        /// tanh: out[i] = tanh(a[i]) = (exp(2x)-1)/(exp(2x)+1)
        #[inline(always)]
        pub fn tanh(a: &[$elem], out: &mut [$elem]) {
            const LANES: usize = $crate::simd_primitive!($isa, $elem, lanes);
            let len = a.len();
            assert!(out.len() == len);
            let mut i = 0;
            while i + LANES <= len {
                #[allow(unused_unsafe)]
                unsafe {
                    let vx = $crate::simd_primitive!($isa, $elem, load, a.as_ptr().add(i));
                    let one = $crate::simd_primitive!($isa, $elem, splat, <$elem as Element>::ONE);
                    let two = $crate::simd_primitive!($isa, $elem, splat, <$elem as Element>::from_f32(2.0));
                    let two_x = $crate::simd_primitive!($isa, $elem, mul, two, vx);
                    let exp_2x = $crate::simd_primitive!($isa, $elem, exp, two_x);
                    let num = $crate::simd_primitive!($isa, $elem, sub, exp_2x, one);
                    let den = $crate::simd_primitive!($isa, $elem, add, exp_2x, one);
                    let res = $crate::simd_primitive!($isa, $elem, div, num, den);
                    $crate::simd_primitive!($isa, $elem, store, out.as_mut_ptr().add(i), res);
                }
                i += LANES;
            }
            while i < len {
                let x = a[i].to_f32();
                let e2x = (2.0f32 * x).exp();
                out[i] = <$elem as Element>::from_f32((e2x - 1.0f32) / (e2x + 1.0f32));
                i += 1;
            }
        }

        /// exp: out[i] = exp(a[i])
        #[inline(always)]
        pub fn exp(a: &[$elem], out: &mut [$elem]) {
            const LANES: usize = $crate::simd_primitive!($isa, $elem, lanes);
            let len = a.len();
            assert!(out.len() == len);
            let mut i = 0;
            while i + LANES <= len {
                #[allow(unused_unsafe)]
                unsafe {
                    let va = $crate::simd_primitive!($isa, $elem, load, a.as_ptr().add(i));
                    let res = $crate::simd_primitive!($isa, $elem, exp, va);
                    $crate::simd_primitive!($isa, $elem, store, out.as_mut_ptr().add(i), res);
                }
                i += LANES;
            }
            while i < len { out[i] = <$elem as Element>::from_f32(a[i].to_f32().exp()); i += 1; }
        }

        /// softmax: out = softmax(a) = exp(a - max(a)) / sum(exp(a - max(a)))
        /// 3-pass: (1) reduce_max, (2) exp + sum, (3) normalize.
        /// In-place safe (a == out is ok)
        #[inline(always)]
        pub fn softmax(a: &[$elem], out: &mut [$elem]) {
            const LANES: usize = $crate::simd_primitive!($isa, $elem, lanes);
            let len = a.len();
            assert!(out.len() == len);

            // Pass 1: find max for numerical stability
            let mut i = 0;
            #[allow(unused_unsafe)]
            let mut max_acc = unsafe { $crate::simd_primitive!($isa, $elem, splat, <$elem as Element>::from_f32(f32::NEG_INFINITY)) };
            while i + LANES <= len {
                #[allow(unused_unsafe)]
                unsafe {
                    let va = $crate::simd_primitive!($isa, $elem, load, a.as_ptr().add(i));
                    max_acc = $crate::simd_primitive!($isa, $elem, max, max_acc, va);
                }
                i += LANES;
            }
            #[allow(unused_unsafe)]
            let mut max_val: f32 = unsafe { $crate::simd_primitive!($isa, $elem, reduce_max, max_acc) };
            while i < len { let v = a[i].to_f32(); if v > max_val { max_val = v; } i += 1; }

            // Pass 2: exp(x - max) and accumulate sum
            i = 0;
            #[allow(unused_unsafe)]
            let vmax = unsafe { $crate::simd_primitive!($isa, $elem, splat, <$elem as Element>::from_f32(max_val)) };
            #[allow(unused_unsafe)]
            let mut sum_acc = unsafe { $crate::simd_primitive!($isa, $elem, zero) };
            while i + LANES <= len {
                #[allow(unused_unsafe)]
                unsafe {
                    let va = $crate::simd_primitive!($isa, $elem, load, a.as_ptr().add(i));
                    let shifted = $crate::simd_primitive!($isa, $elem, sub, va, vmax);
                    let e = $crate::simd_primitive!($isa, $elem, exp, shifted);
                    $crate::simd_primitive!($isa, $elem, store, out.as_mut_ptr().add(i), e);
                    sum_acc = $crate::simd_primitive!($isa, $elem, add, sum_acc, e);
                }
                i += LANES;
            }
            #[allow(unused_unsafe)]
            let mut sum_val: f32 = unsafe { $crate::simd_primitive!($isa, $elem, reduce_sum, sum_acc) };
            while i < len {
                let e = (a[i].to_f32() - max_val).exp();
                out[i] = <$elem as Element>::from_f32(e);
                sum_val += e;
                i += 1;
            }

            // Pass 3: normalize by 1/sum
            i = 0;
            let inv_sum = 1.0f32 / sum_val;
            #[allow(unused_unsafe)]
            let vinv = unsafe { $crate::simd_primitive!($isa, $elem, splat, <$elem as Element>::from_f32(inv_sum)) };
            while i + LANES <= len {
                #[allow(unused_unsafe)]
                unsafe {
                    let va = $crate::simd_primitive!($isa, $elem, load, out.as_ptr().add(i));
                    let res = $crate::simd_primitive!($isa, $elem, mul, va, vinv);
                    $crate::simd_primitive!($isa, $elem, store, out.as_mut_ptr().add(i), res);
                }
                i += LANES;
            }
            while i < len { out[i] = <$elem as Element>::from_f32(out[i].to_f32() * inv_sum); i += 1; }
        }

        /// swiglu: out[i] = silu(gate[i]) * up[i] (single pass fusion)
        #[inline(always)]
        pub fn swiglu(gate: &[$elem], up: &[$elem], out: &mut [$elem]) {
            const LANES: usize = $crate::simd_primitive!($isa, $elem, lanes);
            let len = gate.len();
            debug_assert_eq!(up.len(), len);
            debug_assert_eq!(out.len(), len);

            let mut i = 0;
            while i + LANES <= len {
                #[allow(unused_unsafe)]
                unsafe {
                    let vg = $crate::simd_primitive!($isa, $elem, load, gate.as_ptr().add(i));
                    let vu = $crate::simd_primitive!($isa, $elem, load, up.as_ptr().add(i));
                    let neg_g = $crate::simd_primitive!($isa, $elem, neg, vg);
                    let exp_neg = $crate::simd_primitive!($isa, $elem, exp, neg_g);
                    let one = $crate::simd_primitive!($isa, $elem, splat, <$elem as Element>::ONE);
                    let denom = $crate::simd_primitive!($isa, $elem, add, one, exp_neg);
                    let sigmoid = $crate::simd_primitive!($isa, $elem, recip, denom);
                    // silu(gate) * up = gate * sigmoid(gate) * up
                    let silu = $crate::simd_primitive!($isa, $elem, mul, vg, sigmoid);
                    let res = $crate::simd_primitive!($isa, $elem, mul, silu, vu);
                    $crate::simd_primitive!($isa, $elem, store, out.as_mut_ptr().add(i), res);
                }
                i += LANES;
            }
            while i < len {
                let g = gate[i].to_f32();
                let sigmoid = 1.0f32 / (1.0f32 + (-g).exp());
                out[i] = <$elem as Element>::from_f32(g * sigmoid * up[i].to_f32());
                i += 1;
            }
        }
    };
}

/// Defines normalization operations (rms_norm, layer_norm)
#[macro_export]
macro_rules! define_norm_ops {
    ($isa:ident, $elem:ident) => {
        /// rms_norm: out[i] = (a[i] / rms) * weight[i]
        /// where rms = sqrt(mean(a^2) + eps)
        #[inline(always)]
        pub fn rms_norm(a: &[$elem], weight: &[$elem], out: &mut [$elem], eps: $elem) {
            const LANES: usize = $crate::simd_primitive!($isa, $elem, lanes);
            let len = a.len();
            assert!(weight.len() == len && out.len() == len);

            // Pass 1: sum of squares
            let mut i = 0;
            #[allow(unused_unsafe)]
            let mut ss_acc = unsafe { $crate::simd_primitive!($isa, $elem, zero) };
            while i + LANES <= len {
                #[allow(unused_unsafe)]
                unsafe {
                    let va = $crate::simd_primitive!($isa, $elem, load, a.as_ptr().add(i));
                    ss_acc = $crate::simd_primitive!($isa, $elem, fma, va, va, ss_acc);
                }
                i += LANES;
            }
            #[allow(unused_unsafe)]
            let mut ss: f32 = unsafe { $crate::simd_primitive!($isa, $elem, reduce_sum, ss_acc) };
            while i < len { let v = a[i].to_f32(); ss += v * v; i += 1; }

            // rms = 1/sqrt(mean + eps)
            let eps_f = eps.to_f32();
            let inv_rms_f = 1.0f32 / (ss / (len as f32) + eps_f).sqrt();
            let inv_rms = <$elem as Element>::from_f32(inv_rms_f);

            // Pass 2: normalize and scale
            i = 0;
            #[allow(unused_unsafe)]
            let v_inv = unsafe { $crate::simd_primitive!($isa, $elem, splat, inv_rms) };
            while i + LANES <= len {
                #[allow(unused_unsafe)]
                unsafe {
                    let vw = $crate::simd_primitive!($isa, $elem, load, weight.as_ptr().add(i));
                    let vscale = $crate::simd_primitive!($isa, $elem, mul, v_inv, vw);
                    let va = $crate::simd_primitive!($isa, $elem, load, a.as_ptr().add(i));
                    let res = $crate::simd_primitive!($isa, $elem, mul, va, vscale);
                    $crate::simd_primitive!($isa, $elem, store, out.as_mut_ptr().add(i), res);
                }
                i += LANES;
            }
            while i < len {
                out[i] = <$elem as Element>::from_f32(a[i].to_f32() * inv_rms_f * weight[i].to_f32());
                i += 1;
            }
        }

        /// layer_norm: out[i] = ((a[i] - mean) / sqrt(var + eps)) * weight[i] + bias[i]
        /// 2-pass: fused sum + sum_of_squares, then normalize+scale+bias.
        /// Uses var = E[x²] - E[x]² formula (f32 accumulation for numerical stability).
        #[inline(always)]
        pub fn layer_norm(a: &[$elem], weight: &[$elem], bias: &[$elem], out: &mut [$elem], eps: $elem) {
            const LANES: usize = $crate::simd_primitive!($isa, $elem, lanes);
            let len = a.len();
            assert!(weight.len() == len && bias.len() == len && out.len() == len);
            let n = len as f32;

            // Pass 1 (fused): sum and sum-of-squares in a single traversal
            let mut i = 0;
            #[allow(unused_unsafe)]
            let mut sum_acc = unsafe { $crate::simd_primitive!($isa, $elem, zero) };
            #[allow(unused_unsafe)]
            let mut sq_acc = unsafe { $crate::simd_primitive!($isa, $elem, zero) };
            while i + LANES <= len {
                #[allow(unused_unsafe)]
                unsafe {
                    let va = $crate::simd_primitive!($isa, $elem, load, a.as_ptr().add(i));
                    sum_acc = $crate::simd_primitive!($isa, $elem, add, sum_acc, va);
                    sq_acc = $crate::simd_primitive!($isa, $elem, fma, va, va, sq_acc);
                }
                i += LANES;
            }
            #[allow(unused_unsafe)]
            let mut sum_val: f32 = unsafe { $crate::simd_primitive!($isa, $elem, reduce_sum, sum_acc) };
            #[allow(unused_unsafe)]
            let mut sq_val: f32 = unsafe { $crate::simd_primitive!($isa, $elem, reduce_sum, sq_acc) };
            while i < len {
                let v = a[i].to_f32();
                sum_val += v;
                sq_val += v * v;
                i += 1;
            }

            let mean_f = sum_val / n;
            // var = E[x²] - E[x]² = (sum_sq / n) - mean²
            let var_f = sq_val / n - mean_f * mean_f;
            let eps_f = eps.to_f32();
            let inv_std_f = 1.0f32 / (var_f + eps_f).sqrt();
            let mean = <$elem as Element>::from_f32(mean_f);
            let inv_std = <$elem as Element>::from_f32(inv_std_f);

            // Pass 2: normalize, scale, bias
            i = 0;
            #[allow(unused_unsafe)]
            let vmean = unsafe { $crate::simd_primitive!($isa, $elem, splat, mean) };
            #[allow(unused_unsafe)]
            let vinv = unsafe { $crate::simd_primitive!($isa, $elem, splat, inv_std) };
            while i + LANES <= len {
                #[allow(unused_unsafe)]
                unsafe {
                    let va = $crate::simd_primitive!($isa, $elem, load, a.as_ptr().add(i));
                    let vw = $crate::simd_primitive!($isa, $elem, load, weight.as_ptr().add(i));
                    let vb = $crate::simd_primitive!($isa, $elem, load, bias.as_ptr().add(i));
                    let diff = $crate::simd_primitive!($isa, $elem, sub, va, vmean);
                    let normed = $crate::simd_primitive!($isa, $elem, mul, diff, vinv);
                    let scaled = $crate::simd_primitive!($isa, $elem, fma, normed, vw, vb);
                    $crate::simd_primitive!($isa, $elem, store, out.as_mut_ptr().add(i), scaled);
                }
                i += LANES;
            }
            while i < len {
                out[i] = <$elem as Element>::from_f32(
                    (a[i].to_f32() - mean_f) * inv_std_f * weight[i].to_f32() + bias[i].to_f32()
                );
                i += 1;
            }
        }
    };
}

/// Defines RoPE and Embedding operations
#[macro_export]
macro_rules! define_position_ops {
    ($isa:ident, $elem:ident) => {
        /// rope: Apply Rotary Positional Embeddings (SIMD optimized)
        /// data: [seq_len, head_dim], cos: [seq_len, half], sin: [seq_len, half]
        /// For each position, the first half and second half form complex pairs:
        ///   data[i]'        = data[i] * cos[i] - data[i+half] * sin[i]
        ///   data[i+half]'   = data[i] * sin[i] + data[i+half] * cos[i]
        #[inline(always)]
        pub fn rope(data: &mut [$elem], cos: &[$elem], sin: &[$elem], head_dim: usize) {
            const LANES: usize = $crate::simd_primitive!($isa, $elem, lanes);
            let half = head_dim / 2;
            let seq_len = data.len() / head_dim;
            let elem_size = std::mem::size_of::<$elem>();
            let elems_per_cl = 64 / elem_size;
            for pos in 0..seq_len {
                let base = pos * head_dim;
                let cs_base = pos * half;

                // Prefetch next position's cos/sin
                if pos + 1 < seq_len {
                    #[allow(unused_variables)]
                    let next_cs_base = (pos + 1) * half;
                    let mut cl = 0;
                    while cl * elems_per_cl < half {
                        #[allow(unused_unsafe)]
                        unsafe {
                            $crate::simd_primitive!($isa, $elem, prefetch, cos.as_ptr().add(next_cs_base + cl * elems_per_cl) as *const i8, 0);
                            $crate::simd_primitive!($isa, $elem, prefetch, sin.as_ptr().add(next_cs_base + cl * elems_per_cl) as *const i8, 0);
                        }
                        cl += 1;
                    }
                }
                let mut i = 0;
                // SIMD main loop over the half dimension
                while i + LANES <= half {
                    #[allow(unused_unsafe)]
                    unsafe {
                        let vx0 = $crate::simd_primitive!($isa, $elem, load, data.as_ptr().add(base + i));
                        let vx1 = $crate::simd_primitive!($isa, $elem, load, data.as_ptr().add(base + i + half));
                        let vc  = $crate::simd_primitive!($isa, $elem, load, cos.as_ptr().add(cs_base + i));
                        let vs  = $crate::simd_primitive!($isa, $elem, load, sin.as_ptr().add(cs_base + i));
                        // x0' = x0 * cos - x1 * sin  →  fnmadd(x1, sin, x0*cos)
                        let t0 = $crate::simd_primitive!($isa, $elem, mul, vx0, vc);
                        let r0 = $crate::simd_primitive!($isa, $elem, fnmadd, vx1, vs, t0);
                        // x1' = x0 * sin + x1 * cos  →  fma(x1, cos, x0*sin)
                        let t2 = $crate::simd_primitive!($isa, $elem, mul, vx0, vs);
                        let r1 = $crate::simd_primitive!($isa, $elem, fma, vx1, vc, t2);
                        $crate::simd_primitive!($isa, $elem, store, data.as_mut_ptr().add(base + i), r0);
                        $crate::simd_primitive!($isa, $elem, store, data.as_mut_ptr().add(base + i + half), r1);
                    }
                    i += LANES;
                }
                // Scalar remainder
                while i < half {
                    let x0 = data[base + i].to_f32();
                    let x1 = data[base + i + half].to_f32();
                    let c = cos[cs_base + i].to_f32();
                    let s = sin[cs_base + i].to_f32();
                    data[base + i]        = <$elem as Element>::from_f32(x0 * c - x1 * s);
                    data[base + i + half] = <$elem as Element>::from_f32(x0 * s + x1 * c);
                    i += 1;
                }
            }
        }

        /// rope_with_pos: RoPE with explicit position offset (SIMD optimized)
        /// Same as rope but cos/sin are indexed by (pos + position) instead of pos.
        #[inline(always)]
        pub fn rope_with_pos(data: &mut [$elem], cos: &[$elem], sin: &[$elem], head_dim: usize, position: usize) {
            const LANES: usize = $crate::simd_primitive!($isa, $elem, lanes);
            let half = head_dim / 2;
            let seq_len = data.len() / head_dim;
            let elem_size = std::mem::size_of::<$elem>();
            let elems_per_cl = 64 / elem_size;
            for pos in 0..seq_len {
                let base = pos * head_dim;
                let actual_pos = pos + position;
                let cs_base = actual_pos * half;

                // Prefetch next position's cos/sin
                if pos + 1 < seq_len {
                    #[allow(unused_variables)]
                    let next_cs_base = (pos + 1 + position) * half;
                    let mut cl = 0;
                    while cl * elems_per_cl < half {
                        #[allow(unused_unsafe)]
                        unsafe {
                            $crate::simd_primitive!($isa, $elem, prefetch, cos.as_ptr().add(next_cs_base + cl * elems_per_cl) as *const i8, 0);
                            $crate::simd_primitive!($isa, $elem, prefetch, sin.as_ptr().add(next_cs_base + cl * elems_per_cl) as *const i8, 0);
                        }
                        cl += 1;
                    }
                }

                let mut i = 0;
                while i + LANES <= half {
                    #[allow(unused_unsafe)]
                    unsafe {
                        let vx0 = $crate::simd_primitive!($isa, $elem, load, data.as_ptr().add(base + i));
                        let vx1 = $crate::simd_primitive!($isa, $elem, load, data.as_ptr().add(base + i + half));
                        let vc  = $crate::simd_primitive!($isa, $elem, load, cos.as_ptr().add(cs_base + i));
                        let vs  = $crate::simd_primitive!($isa, $elem, load, sin.as_ptr().add(cs_base + i));
                        let t0 = $crate::simd_primitive!($isa, $elem, mul, vx0, vc);
                        let r0 = $crate::simd_primitive!($isa, $elem, fnmadd, vx1, vs, t0);
                        let t2 = $crate::simd_primitive!($isa, $elem, mul, vx0, vs);
                        let r1 = $crate::simd_primitive!($isa, $elem, fma, vx1, vc, t2);
                        $crate::simd_primitive!($isa, $elem, store, data.as_mut_ptr().add(base + i), r0);
                        $crate::simd_primitive!($isa, $elem, store, data.as_mut_ptr().add(base + i + half), r1);
                    }
                    i += LANES;
                }
                while i < half {
                    let x0 = data[base + i].to_f32();
                    let x1 = data[base + i + half].to_f32();
                    let c = cos[cs_base + i].to_f32();
                    let s = sin[cs_base + i].to_f32();
                    data[base + i]        = <$elem as Element>::from_f32(x0 * c - x1 * s);
                    data[base + i + half] = <$elem as Element>::from_f32(x0 * s + x1 * c);
                    i += 1;
                }
            }
        }

        /// embedding_lookup: out = embedding_table[token_ids]
        /// table: [vocab_size, dim], ids: [seq_len], out: [seq_len, dim]
        /// Software prefetch: while copying row i, prefetch row i+1 to hide DRAM latency.
        #[inline(always)]
        pub fn embedding_lookup(table: &[$elem], ids: &[u32], out: &mut [$elem], dim: usize) {
            const LANES: usize = $crate::simd_primitive!($isa, $elem, lanes);
            let n = ids.len();
            // Prefetch distance in cache lines (64 bytes each)
            let elem_size = std::mem::size_of::<$elem>();
            let elems_per_cl = 64 / elem_size; // elements per cache line

            for i in 0..n {
                let id = ids[i] as usize;
                let src = &table[id * dim..(id + 1) * dim];
                let dst = &mut out[i * dim..(i + 1) * dim];

                // Prefetch next row while copying current
                if i + 1 < n {
                    let next_id = ids[i + 1] as usize;
                    #[allow(unused_variables)]
                    let next_base = table.as_ptr() as usize + next_id * dim * elem_size;
                    // Prefetch multiple cache lines covering the row
                    let mut cl = 0;
                    while cl * elems_per_cl < dim {
                        #[allow(unused_unsafe)]
                        unsafe {
                            $crate::simd_primitive!($isa, $elem, prefetch, (next_base + cl * 64) as *const i8, 0);
                        }
                        cl += 1;
                    }
                }

                // SIMD copy for the current row
                let mut j = 0;
                while j + LANES <= dim {
                    #[allow(unused_unsafe)]
                    unsafe {
                        let v = $crate::simd_primitive!($isa, $elem, load, src.as_ptr().add(j));
                        $crate::simd_primitive!($isa, $elem, store, dst.as_mut_ptr().add(j), v);
                    }
                    j += LANES;
                }
                // Scalar remainder
                while j < dim {
                    dst[j] = src[j];
                    j += 1;
                }
            }
        }
    };
}

/// Defines GEMV (General Matrix-Vector Multiply)
#[macro_export]
macro_rules! define_gemv_op {
    ($isa:ident, $elem:ident) => {
        /// gemv: y = A * x + y (M x K) * (K) -> (M)
        /// 2-row processing to reuse x vector loads + 4 accumulators per row for ILP.
        #[inline(always)]
        pub fn gemv(a: &[$elem], x: &[$elem], y: &mut [$elem], m: usize, k: usize) {
            const LANES: usize = $crate::simd_primitive!($isa, $elem, lanes);
            assert!(a.len() >= m * k && x.len() >= k && y.len() >= m);

            // Process 2 rows at a time to reuse x loads
            let mut row = 0;
            while row + 2 <= m {
                let row0_ptr = &a[row * k..];
                let row1_ptr = &a[(row + 1) * k..];

                let mut i = 0;
                #[allow(unused_unsafe)]
                let mut r0_acc0 = unsafe { $crate::simd_primitive!($isa, $elem, zero) };
                #[allow(unused_unsafe)]
                let mut r0_acc1 = unsafe { $crate::simd_primitive!($isa, $elem, zero) };
                #[allow(unused_unsafe)]
                let mut r1_acc0 = unsafe { $crate::simd_primitive!($isa, $elem, zero) };
                #[allow(unused_unsafe)]
                let mut r1_acc1 = unsafe { $crate::simd_primitive!($isa, $elem, zero) };

                while i + LANES * 2 <= k {
                    #[allow(unused_unsafe)]
                    unsafe {
                        let vx0 = $crate::simd_primitive!($isa, $elem, load, x.as_ptr().add(i));
                        let vx1 = $crate::simd_primitive!($isa, $elem, load, x.as_ptr().add(i + LANES));

                        let va00 = $crate::simd_primitive!($isa, $elem, load, row0_ptr.as_ptr().add(i));
                        let va01 = $crate::simd_primitive!($isa, $elem, load, row0_ptr.as_ptr().add(i + LANES));
                        r0_acc0 = $crate::simd_primitive!($isa, $elem, fma, va00, vx0, r0_acc0);
                        r0_acc1 = $crate::simd_primitive!($isa, $elem, fma, va01, vx1, r0_acc1);

                        let va10 = $crate::simd_primitive!($isa, $elem, load, row1_ptr.as_ptr().add(i));
                        let va11 = $crate::simd_primitive!($isa, $elem, load, row1_ptr.as_ptr().add(i + LANES));
                        r1_acc0 = $crate::simd_primitive!($isa, $elem, fma, va10, vx0, r1_acc0);
                        r1_acc1 = $crate::simd_primitive!($isa, $elem, fma, va11, vx1, r1_acc1);
                    }
                    i += LANES * 2;
                }

                // Drain remaining full vectors
                while i + LANES <= k {
                    #[allow(unused_unsafe)]
                    unsafe {
                        let vx = $crate::simd_primitive!($isa, $elem, load, x.as_ptr().add(i));
                        let va0 = $crate::simd_primitive!($isa, $elem, load, row0_ptr.as_ptr().add(i));
                        let va1 = $crate::simd_primitive!($isa, $elem, load, row1_ptr.as_ptr().add(i));
                        r0_acc0 = $crate::simd_primitive!($isa, $elem, fma, va0, vx, r0_acc0);
                        r1_acc0 = $crate::simd_primitive!($isa, $elem, fma, va1, vx, r1_acc0);
                    }
                    i += LANES;
                }

                // Reduce accumulators
                #[allow(unused_unsafe)]
                let r0_sum = unsafe { $crate::simd_primitive!($isa, $elem, add, r0_acc0, r0_acc1) };
                #[allow(unused_unsafe)]
                let mut dot0: f32 = unsafe { $crate::simd_primitive!($isa, $elem, reduce_sum, r0_sum) };
                #[allow(unused_unsafe)]
                let r1_sum = unsafe { $crate::simd_primitive!($isa, $elem, add, r1_acc0, r1_acc1) };
                #[allow(unused_unsafe)]
                let mut dot1: f32 = unsafe { $crate::simd_primitive!($isa, $elem, reduce_sum, r1_sum) };

                // Scalar tail
                while i < k {
                    let xv = x[i].to_f32();
                    dot0 += row0_ptr[i].to_f32() * xv;
                    dot1 += row1_ptr[i].to_f32() * xv;
                    i += 1;
                }
                y[row]     = <$elem as Element>::from_f32(y[row].to_f32() + dot0);
                y[row + 1] = <$elem as Element>::from_f32(y[row + 1].to_f32() + dot1);
                row += 2;
            }

            // Handle odd remaining row
            if row < m {
                let row_ptr = &a[row * k..];
                let mut i = 0;
                #[allow(unused_unsafe)]
                let mut acc0 = unsafe { $crate::simd_primitive!($isa, $elem, zero) };
                #[allow(unused_unsafe)]
                let mut acc1 = unsafe { $crate::simd_primitive!($isa, $elem, zero) };

                while i + LANES * 2 <= k {
                    #[allow(unused_unsafe)]
                    unsafe {
                        let vx0 = $crate::simd_primitive!($isa, $elem, load, x.as_ptr().add(i));
                        let vx1 = $crate::simd_primitive!($isa, $elem, load, x.as_ptr().add(i + LANES));
                        let va0 = $crate::simd_primitive!($isa, $elem, load, row_ptr.as_ptr().add(i));
                        let va1 = $crate::simd_primitive!($isa, $elem, load, row_ptr.as_ptr().add(i + LANES));
                        acc0 = $crate::simd_primitive!($isa, $elem, fma, va0, vx0, acc0);
                        acc1 = $crate::simd_primitive!($isa, $elem, fma, va1, vx1, acc1);
                    }
                    i += LANES * 2;
                }
                while i + LANES <= k {
                    #[allow(unused_unsafe)]
                    unsafe {
                        let vx = $crate::simd_primitive!($isa, $elem, load, x.as_ptr().add(i));
                        let va = $crate::simd_primitive!($isa, $elem, load, row_ptr.as_ptr().add(i));
                        acc0 = $crate::simd_primitive!($isa, $elem, fma, va, vx, acc0);
                    }
                    i += LANES;
                }

                #[allow(unused_unsafe)]
                let acc_sum = unsafe { $crate::simd_primitive!($isa, $elem, add, acc0, acc1) };
                #[allow(unused_unsafe)]
                let mut dot: f32 = unsafe { $crate::simd_primitive!($isa, $elem, reduce_sum, acc_sum) };
                while i < k { dot += row_ptr[i].to_f32() * x[i].to_f32(); i += 1; }
                y[row] = <$elem as Element>::from_f32(y[row].to_f32() + dot);
            }
        }
    };
}



/// Defines Matrix Multiplication (GEMM) — thin dispatcher.
/// Routes to `define_matmul_x86!` (AVX-512 / AVX2), `define_matmul_neon!`, or scalar fallback.
#[macro_export]
macro_rules! define_matmul_op {
    (avx512, $elem:ident) => {
        $crate::define_matmul_x86!(avx512, $elem, 14, 16, 2, "avx512f");
    };
    (avx2, $elem:ident) => {
        $crate::define_matmul_x86!(avx2, $elem, 6, 8, 2, "avx2", "fma");
    };
    (neon, $elem:ident) => {
        $crate::define_matmul_neon!($elem);
    };
    ($isa:ident, $elem:ident) => {
        // Default / Scalar Implementation
        #[inline(always)]
        pub fn matmul(a: &[$elem], b: &[$elem], c: &mut [$elem], m_size: usize, n_size: usize, k_size: usize) {
             assert!(a.len() >= m_size * k_size);
            assert!(b.len() >= n_size * k_size);
            assert!(c.len() >= m_size * n_size);

            for m in 0..m_size {
                for n in 0..n_size {
                    let mut sum: $elem = <$elem as Element>::ZERO;
                    for k in 0..k_size {
                        let val_a = a[m * k_size + k];
                        let val_b = b[k * n_size + n];
                        sum = <$elem as Element>::mul_add(sum, val_a, val_b);
                    }
                    c[m * n_size + n] = sum;
                }
            }
        }

        /// matmul_bias: C = A * B + bias (scalar fallback)
        #[inline(always)]
        pub fn matmul_bias(a: &[$elem], b: &[$elem], bias: &[$elem], c: &mut [$elem], m_size: usize, n_size: usize, k_size: usize) {
            assert!(a.len() >= m_size * k_size);
            assert!(b.len() >= n_size * k_size);
            assert!(c.len() >= m_size * n_size);
            assert!(bias.len() >= n_size);

            for m in 0..m_size {
                for n in 0..n_size {
                    let mut sum: $elem = <$elem as Element>::ZERO;
                    for k in 0..k_size {
                        let val_a = a[m * k_size + k];
                        let val_b = b[k * n_size + n];
                        sum = <$elem as Element>::mul_add(sum, val_a, val_b);
                    }
                    c[m * n_size + n] = sum + bias[n];
                }
            }
        }

        /// Pre-pack B matrix (scalar: just copies B in row-major order).
        pub fn pack_b(b: &[$elem], n_size: usize, k_size: usize) -> Vec<$elem> {
            assert!(b.len() >= k_size * n_size);
            b[..k_size * n_size].to_vec()
        }

        /// Matmul using pre-packed B (scalar fallback — same as regular matmul).
        #[inline(always)]
        pub fn matmul_prepacked(a: &[$elem], packed_b: &[$elem], c: &mut [$elem], m_size: usize, n_size: usize, k_size: usize) {
            assert!(a.len() >= m_size * k_size);
            assert!(c.len() >= m_size * n_size);

            for m in 0..m_size {
                for n in 0..n_size {
                    let mut sum: $elem = <$elem as Element>::ZERO;
                    for k in 0..k_size {
                        sum = <$elem as Element>::mul_add(sum, a[m * k_size + k], packed_b[k * n_size + n]);
                    }
                    c[m * n_size + n] = sum;
                }
            }
        }

        /// Matmul+bias using pre-packed B (scalar fallback).
        #[inline(always)]
        pub fn matmul_bias_prepacked(a: &[$elem], packed_b: &[$elem], bias: &[$elem], c: &mut [$elem], m_size: usize, n_size: usize, k_size: usize) {
            assert!(a.len() >= m_size * k_size);
            assert!(c.len() >= m_size * n_size);
            assert!(bias.len() >= n_size);

            for m in 0..m_size {
                for n in 0..n_size {
                    let mut sum: $elem = <$elem as Element>::ZERO;
                    for k in 0..k_size {
                        sum = <$elem as Element>::mul_add(sum, a[m * k_size + k], packed_b[k * n_size + n]);
                    }
                    c[m * n_size + n] = sum + bias[n];
                }
            }
        }
    };
}
///
/// All internal computation is done in f32 space regardless of $elem type.
/// For f32: zero-copy via as_f32_slice(). For f16/bf16: one-time conversion.
/// SIMD operations use simd_primitive!($isa, f32, ...) for the hot loops.
#[macro_export]
macro_rules! define_flash_attention_ops {
    ($isa:ident, $elem:ident) => {
        /// Dot product of two f32 slices, using simd_primitive! for SIMD ops.
        /// Double-accumulator for better ILP.
        #[inline(always)]
        fn attn_dot_f32(a: *const f32, b: *const f32, len: usize) -> f32 {
            const LANES: usize = $crate::simd_primitive!($isa, f32, lanes);
            let mut d = 0usize;
            #[allow(unused_unsafe)]
            let mut acc0 = unsafe { $crate::simd_primitive!($isa, f32, zero) };
            #[allow(unused_unsafe)]
            let mut acc1 = unsafe { $crate::simd_primitive!($isa, f32, zero) };
            while d + LANES * 2 <= len {
                #[allow(unused_unsafe)]
                unsafe {
                    let va0 = $crate::simd_primitive!($isa, f32, load, a.add(d));
                    let vb0 = $crate::simd_primitive!($isa, f32, load, b.add(d));
                    acc0 = $crate::simd_primitive!($isa, f32, fma, va0, vb0, acc0);
                    let va1 = $crate::simd_primitive!($isa, f32, load, a.add(d + LANES));
                    let vb1 = $crate::simd_primitive!($isa, f32, load, b.add(d + LANES));
                    acc1 = $crate::simd_primitive!($isa, f32, fma, va1, vb1, acc1);
                }
                d += LANES * 2;
            }
            if d + LANES <= len {
                #[allow(unused_unsafe)]
                unsafe {
                    let va = $crate::simd_primitive!($isa, f32, load, a.add(d));
                    let vb = $crate::simd_primitive!($isa, f32, load, b.add(d));
                    acc0 = $crate::simd_primitive!($isa, f32, fma, va, vb, acc0);
                }
                d += LANES;
            }
            #[allow(unused_unsafe)]
            let combined = unsafe { $crate::simd_primitive!($isa, f32, add, acc0, acc1) };
            #[allow(unused_unsafe)]
            let mut sum: f32 = unsafe { $crate::simd_primitive!($isa, f32, reduce_sum, combined) };
            while d < len { unsafe { sum += *a.add(d) * *b.add(d); } d += 1; }
            sum
        }

        /// FMA accumulate: acc[0..len] += alpha * src[0..len], using simd_primitive!.
        /// Double-width unrolling for better throughput.
        #[inline(always)]
        fn attn_fma_f32(acc: *mut f32, src: *const f32, alpha: f32, len: usize) {
            const LANES: usize = $crate::simd_primitive!($isa, f32, lanes);
            #[allow(unused_unsafe)]
            let va = unsafe { $crate::simd_primitive!($isa, f32, splat, alpha) };
            let mut d = 0usize;
            while d + LANES * 2 <= len {
                #[allow(unused_unsafe)]
                unsafe {
                    let cur0 = $crate::simd_primitive!($isa, f32, load, acc.add(d));
                    let vs0 = $crate::simd_primitive!($isa, f32, load, src.add(d));
                    let res0 = $crate::simd_primitive!($isa, f32, fma, va, vs0, cur0);
                    $crate::simd_primitive!($isa, f32, store, acc.add(d), res0);
                    let cur1 = $crate::simd_primitive!($isa, f32, load, acc.add(d + LANES));
                    let vs1 = $crate::simd_primitive!($isa, f32, load, src.add(d + LANES));
                    let res1 = $crate::simd_primitive!($isa, f32, fma, va, vs1, cur1);
                    $crate::simd_primitive!($isa, f32, store, acc.add(d + LANES), res1);
                }
                d += LANES * 2;
            }
            if d + LANES <= len {
                #[allow(unused_unsafe)]
                unsafe {
                    let cur = $crate::simd_primitive!($isa, f32, load, acc.add(d));
                    let vs = $crate::simd_primitive!($isa, f32, load, src.add(d));
                    let res = $crate::simd_primitive!($isa, f32, fma, va, vs, cur);
                    $crate::simd_primitive!($isa, f32, store, acc.add(d), res);
                }
                d += LANES;
            }
            while d < len { unsafe { *acc.add(d) += alpha * *src.add(d); } d += 1; }
        }

        /// SIMD scale: acc[0..len] *= scale
        #[inline(always)]
        fn attn_scale_f32(acc: *mut f32, scale: f32, len: usize) {
            const LANES: usize = $crate::simd_primitive!($isa, f32, lanes);
            #[allow(unused_unsafe)]
            let vs = unsafe { $crate::simd_primitive!($isa, f32, splat, scale) };
            let mut d = 0usize;
            while d + LANES <= len {
                #[allow(unused_unsafe)]
                unsafe {
                    let cur = $crate::simd_primitive!($isa, f32, load, acc.add(d));
                    let res = $crate::simd_primitive!($isa, f32, mul, cur, vs);
                    $crate::simd_primitive!($isa, f32, store, acc.add(d), res);
                }
                d += LANES;
            }
            while d < len { unsafe { *acc.add(d) *= scale; } d += 1; }
        }

        /// Standard multi-head attention with optional causal masking.
        /// Online softmax: single pass over K/V, no scores buffer.
        /// Maintains running (max, sum_exp, acc) and rescales on new max.
        #[inline(always)]
        pub fn flash_attention(
            q: &[$elem], k: &[$elem], v: &[$elem], output: &mut [$elem],
            seq_len: usize, num_heads: usize, head_dim: usize,
            scale: f32, causal: bool,
        ) {
            let mut acc = vec![0.0f32; head_dim];

            // For f32 elements we use zero-copy slices; for f16/bf16 we convert
            // only the head_dim-sized vector we need into a reusable buffer,
            // avoiding a full-tensor upfront allocation.
            let is_f32 = <$elem as Element>::as_f32_slice(q).is_some();
            let mut q_buf = vec![0.0f32; if is_f32 { 0 } else { head_dim }];
            let mut k_buf = vec![0.0f32; if is_f32 { 0 } else { head_dim }];
            let mut v_buf = vec![0.0f32; if is_f32 { 0 } else { head_dim }];

            #[inline(always)]
            fn convert_slice<E: Element>(src: &[E], buf: &mut [f32], off: usize, len: usize) -> *const f32 {
                if let Some(f) = E::as_f32_slice(src) {
                    unsafe { f.as_ptr().add(off) }
                } else {
                    for d in 0..len {
                        unsafe { *buf.get_unchecked_mut(d) = src.get_unchecked(off + d).to_f32(); }
                    }
                    buf.as_ptr()
                }
            }

            for h in 0..num_heads {
                for i in 0..seq_len {
                    let q_off = h * seq_len * head_dim + i * head_dim;
                    let o_off = q_off;
                    let max_j = if causal { i + 1 } else { seq_len };

                    let q_ptr = convert_slice(q, &mut q_buf, q_off, head_dim);
                    let mut running_max = f32::NEG_INFINITY;
                    let mut running_sum = 0.0f32;
                    acc[..head_dim].fill(0.0);

                    for j in 0..max_j {
                        let kv_off = h * seq_len * head_dim + j * head_dim;

                        let k_ptr = convert_slice(k, &mut k_buf, kv_off, head_dim);

                        // Prefetch next K/V source data
                        if j + 1 < max_j {
                            let next_off = kv_off + head_dim;
                            if is_f32 {
                                if let Some(kf) = <$elem as Element>::as_f32_slice(k) {
                                    let next_k = unsafe { kf.as_ptr().add(next_off) };
                                    let next_v = unsafe { <$elem as Element>::as_f32_slice(v).unwrap_unchecked().as_ptr().add(next_off) };
                                    #[cfg(target_arch = "x86_64")]
                                    unsafe {
                                        std::arch::x86_64::_mm_prefetch(next_k as *const i8, std::arch::x86_64::_MM_HINT_T0);
                                        std::arch::x86_64::_mm_prefetch(next_v as *const i8, std::arch::x86_64::_MM_HINT_T0);
                                    }
                                    #[cfg(target_arch = "aarch64")]
                                    unsafe {
                                        std::arch::aarch64::_prefetch(next_k as *const i8, 0, 3);
                                        std::arch::aarch64::_prefetch(next_v as *const i8, 0, 3);
                                    }
                                }
                            } else {
                                // For f16/bf16, prefetch the source element data
                                let next_k_src = unsafe { k.as_ptr().add(next_off) };
                                let next_v_src = unsafe { v.as_ptr().add(next_off) };
                                #[cfg(target_arch = "x86_64")]
                                unsafe {
                                    std::arch::x86_64::_mm_prefetch(next_k_src as *const i8, std::arch::x86_64::_MM_HINT_T0);
                                    std::arch::x86_64::_mm_prefetch(next_v_src as *const i8, std::arch::x86_64::_MM_HINT_T0);
                                }
                                #[cfg(target_arch = "aarch64")]
                                unsafe {
                                    std::arch::aarch64::_prefetch(next_k_src as *const i8, 0, 3);
                                    std::arch::aarch64::_prefetch(next_v_src as *const i8, 0, 3);
                                }
                            }
                        }

                        let s = attn_dot_f32(q_ptr, k_ptr, head_dim) * scale;

                        if s > running_max {
                            // Rescale existing accumulator and sum
                            let correction = (running_max - s).exp();
                            attn_scale_f32(acc.as_mut_ptr(), correction, head_dim);
                            running_sum *= correction;
                            running_max = s;
                        }

                        let w = (s - running_max).exp();
                        running_sum += w;

                        let v_ptr = convert_slice(v, &mut v_buf, kv_off, head_dim);
                        attn_fma_f32(acc.as_mut_ptr(), v_ptr, w, head_dim);
                    }

                    // Normalize: acc /= running_sum
                    let inv_sum = 1.0 / running_sum;
                    attn_scale_f32(acc.as_mut_ptr(), inv_sum, head_dim);

                    if let Some(of) = <$elem as Element>::as_f32_slice_mut(output) {
                        unsafe {
                            std::ptr::copy_nonoverlapping(acc.as_ptr(), of.as_mut_ptr().add(o_off), head_dim);
                        }
                    } else {
                        for d in 0..head_dim {
                            unsafe {
                                *output.get_unchecked_mut(o_off + d) = <$elem as Element>::from_f32(*acc.get_unchecked(d));
                            }
                        }
                    }
                }
            }
        }

        /// Paged KV-cache attention with GQA support.
        /// Online softmax: single pass, no scores buffer.
        #[inline(always)]
        pub fn flash_attention_paged(
            q: &[$elem], k_cache: &[$elem], v_cache: &[$elem],
            page_table: &[usize], output: &mut [$elem],
            seq_len: usize, cache_len: usize,
            num_heads: usize, num_kv_heads: usize, head_dim: usize,
            page_size: usize, scale: f32,
        ) {
            let heads_per_kv = num_heads / num_kv_heads;
            let pages_per_kv = (cache_len + page_size - 1) / page_size;

            let mut acc = vec![0.0f32; head_dim];

            let is_f32 = <$elem as Element>::as_f32_slice(q).is_some();
            let mut q_buf = vec![0.0f32; if is_f32 { 0 } else { head_dim }];
            let mut k_buf = vec![0.0f32; if is_f32 { 0 } else { head_dim }];
            let mut v_buf = vec![0.0f32; if is_f32 { 0 } else { head_dim }];

            #[inline(always)]
            fn convert_slice<E: Element>(src: &[E], buf: &mut [f32], off: usize, len: usize) -> *const f32 {
                if let Some(f) = E::as_f32_slice(src) {
                    unsafe { f.as_ptr().add(off) }
                } else {
                    for d in 0..len {
                        unsafe { *buf.get_unchecked_mut(d) = src.get_unchecked(off + d).to_f32(); }
                    }
                    buf.as_ptr()
                }
            }

            for h in 0..num_heads {
                let kv_h = h / heads_per_kv;
                for i in 0..seq_len {
                    let q_off = h * seq_len * head_dim + i * head_dim;
                    let o_off = q_off;

                    let q_ptr = convert_slice(q, &mut q_buf, q_off, head_dim);
                    let mut running_max = f32::NEG_INFINITY;
                    let mut running_sum = 0.0f32;
                    acc[..head_dim].fill(0.0);

                    for j in 0..cache_len {
                        let page_idx = j / page_size;
                        let page_off = j % page_size;
                        let phys_page = unsafe { *page_table.get_unchecked(kv_h * pages_per_kv + page_idx) };
                        let kv_off = phys_page * page_size * head_dim + page_off * head_dim;

                        let k_ptr = convert_slice(k_cache, &mut k_buf, kv_off, head_dim);

                        // Prefetch next K/V source data
                        if j + 1 < cache_len {
                            let np_idx = (j + 1) / page_size;
                            let np_off = (j + 1) % page_size;
                            let np_phys = unsafe { *page_table.get_unchecked(kv_h * pages_per_kv + np_idx) };
                            let next_off = np_phys * page_size * head_dim + np_off * head_dim;
                            if is_f32 {
                                if let Some(kf) = <$elem as Element>::as_f32_slice(k_cache) {
                                    let next_k = unsafe { kf.as_ptr().add(next_off) };
                                    let next_v = unsafe { <$elem as Element>::as_f32_slice(v_cache).unwrap_unchecked().as_ptr().add(next_off) };
                                    #[cfg(target_arch = "x86_64")]
                                    unsafe {
                                        std::arch::x86_64::_mm_prefetch(next_k as *const i8, std::arch::x86_64::_MM_HINT_T0);
                                        std::arch::x86_64::_mm_prefetch(next_v as *const i8, std::arch::x86_64::_MM_HINT_T0);
                                    }
                                    #[cfg(target_arch = "aarch64")]
                                    unsafe {
                                        std::arch::aarch64::_prefetch(next_k as *const i8, 0, 3);
                                        std::arch::aarch64::_prefetch(next_v as *const i8, 0, 3);
                                    }
                                }
                            } else {
                                let next_k_src = unsafe { k_cache.as_ptr().add(next_off) };
                                let next_v_src = unsafe { v_cache.as_ptr().add(next_off) };
                                #[cfg(target_arch = "x86_64")]
                                unsafe {
                                    std::arch::x86_64::_mm_prefetch(next_k_src as *const i8, std::arch::x86_64::_MM_HINT_T0);
                                    std::arch::x86_64::_mm_prefetch(next_v_src as *const i8, std::arch::x86_64::_MM_HINT_T0);
                                }
                                #[cfg(target_arch = "aarch64")]
                                unsafe {
                                    std::arch::aarch64::_prefetch(next_k_src as *const i8, 0, 3);
                                    std::arch::aarch64::_prefetch(next_v_src as *const i8, 0, 3);
                                }
                            }
                        }

                        let s = attn_dot_f32(q_ptr, k_ptr, head_dim) * scale;

                        if s > running_max {
                            let correction = (running_max - s).exp();
                            attn_scale_f32(acc.as_mut_ptr(), correction, head_dim);
                            running_sum *= correction;
                            running_max = s;
                        }

                        let w = (s - running_max).exp();
                        running_sum += w;

                        let v_ptr = convert_slice(v_cache, &mut v_buf, kv_off, head_dim);
                        attn_fma_f32(acc.as_mut_ptr(), v_ptr, w, head_dim);
                    }

                    let inv_sum = 1.0 / running_sum;
                    attn_scale_f32(acc.as_mut_ptr(), inv_sum, head_dim);

                    if let Some(of) = <$elem as Element>::as_f32_slice_mut(output) {
                        unsafe {
                            std::ptr::copy_nonoverlapping(acc.as_ptr(), of.as_mut_ptr().add(o_off), head_dim);
                        }
                    } else {
                        for d in 0..head_dim {
                            unsafe {
                                *output.get_unchecked_mut(o_off + d) = <$elem as Element>::from_f32(*acc.get_unchecked(d));
                            }
                        }
                    }
                }
            }
        }
    };
}
