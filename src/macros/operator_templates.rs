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



/// Defines Matrix Multiplication (GEMM)
#[macro_export]
macro_rules! define_matmul_op {
    (avx512, $elem:ident) => {
        // AVX-512 Optimized Implementation: 14×32 microkernel
        // 28 zmm accumulator registers (14 rows × 2 vectors of 16 lanes)
        // Loop order: chunk(KC) → m → n — B stays hot in cache
        #[target_feature(enable = "avx512f")]
        pub unsafe fn matmul_avx512(a: &[$elem], b: &[$elem], c: &mut [$elem], m_size: usize, n_size: usize, k_size: usize) {
            const TILE_M: usize = 14;
            const TILE_N_VECS: usize = 2;
            const LANES: usize = 16;
            const TILE_N: usize = TILE_N_VECS * LANES; // 32
            const KC: usize = 256;

            assert!(a.len() >= m_size * k_size);
            assert!(b.len() >= n_size * k_size);
            assert!(c.len() >= m_size * n_size);

            let n_strips = (n_size + TILE_N - 1) / TILE_N;
            let n_chunks = (k_size + KC - 1) / KC;
            let chunk_size = n_strips * KC * TILE_N;

            // Pack all of B upfront (same layout as pack_b)
            let total_packed = n_chunks * chunk_size;
            thread_local! { static WS: std::cell::Cell<Vec<$elem>> = std::cell::Cell::new(Vec::new()); }
            let mut packed_b = WS.with(|c| c.take());
            if packed_b.capacity() < total_packed { packed_b.reserve(total_packed - packed_b.len()); }
            unsafe { packed_b.set_len(total_packed); std::ptr::write_bytes(packed_b.as_mut_ptr(), 0, total_packed); }

            {
                let mut k_start = 0usize;
                let mut chunk = 0usize;
                while k_start < k_size {
                    let kc = KC.min(k_size - k_start);
                    let base = chunk * chunk_size;
                    for i in 0..n_strips {
                        let n_start = i * TILE_N;
                        let actual_n = TILE_N.min(n_size.saturating_sub(n_start));
                        for k in 0..kc {
                            let dst = base + i * KC * TILE_N + k * TILE_N;
                            let src_row = (k_start + k) * n_size + n_start;
                            std::ptr::copy_nonoverlapping(
                                b.as_ptr().add(src_row),
                                packed_b.as_mut_ptr().add(dst),
                                actual_n,
                            );
                        }
                    }
                    k_start += KC;
                    chunk += 1;
                }
            }

            // Zero C matrix — accumulation across chunks
            for i in 0..c.len().min(m_size * n_size) { c[i] = <$elem as Element>::ZERO; }
            let c_ptr = c.as_mut_ptr();

            // Main loop: chunk → m → n (B stays hot in L1/L2)
            let mut k_start = 0usize;
            let mut chunk = 0usize;
            while k_start < k_size {
                let kc = KC.min(k_size - k_start);

                let mut m = 0;
                while m + TILE_M <= m_size {
                    let mut n = 0;
                    let mut strip_idx = 0;
                    while n + TILE_N <= n_size {
                        unsafe {
                            // Load C tile from memory (accumulate across chunks)
                            macro_rules! load_row {
                                ($row:expr) => {(
                                    $crate::simd_primitive!(avx512, $elem, loadu, c_ptr.add((m + $row) * n_size + n)),
                                    $crate::simd_primitive!(avx512, $elem, loadu, c_ptr.add((m + $row) * n_size + n + LANES)),
                                )};
                            }
                            let (mut c_0_0, mut c_0_1) = load_row!(0);
                            let (mut c_1_0, mut c_1_1) = load_row!(1);
                            let (mut c_2_0, mut c_2_1) = load_row!(2);
                            let (mut c_3_0, mut c_3_1) = load_row!(3);
                            let (mut c_4_0, mut c_4_1) = load_row!(4);
                            let (mut c_5_0, mut c_5_1) = load_row!(5);
                            let (mut c_6_0, mut c_6_1) = load_row!(6);
                            let (mut c_7_0, mut c_7_1) = load_row!(7);
                            let (mut c_8_0, mut c_8_1) = load_row!(8);
                            let (mut c_9_0, mut c_9_1) = load_row!(9);
                            let (mut c_10_0, mut c_10_1) = load_row!(10);
                            let (mut c_11_0, mut c_11_1) = load_row!(11);
                            let (mut c_12_0, mut c_12_1) = load_row!(12);
                            let (mut c_13_0, mut c_13_1) = load_row!(13);

                            macro_rules! fma_row {
                                ($a_base:expr, $vb0:ident, $vb1:ident, $row:expr, $c0:ident, $c1:ident) => {
                                    let va = $crate::simd_primitive!(avx512, $elem, splat, *$a_base.add($row));
                                    $c0 = $crate::simd_primitive!(avx512, $elem, fma, va, $vb0, $c0);
                                    $c1 = $crate::simd_primitive!(avx512, $elem, fma, va, $vb1, $c1);
                                };
                            }
                            macro_rules! fma_all_rows {
                                ($a_base:expr, $vb0:ident, $vb1:ident) => {
                                    fma_row!($a_base, $vb0, $vb1, 0, c_0_0, c_0_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size, c_1_0, c_1_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size * 2, c_2_0, c_2_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size * 3, c_3_0, c_3_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size * 4, c_4_0, c_4_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size * 5, c_5_0, c_5_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size * 6, c_6_0, c_6_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size * 7, c_7_0, c_7_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size * 8, c_8_0, c_8_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size * 9, c_9_0, c_9_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size * 10, c_10_0, c_10_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size * 11, c_11_0, c_11_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size * 12, c_12_0, c_12_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size * 13, c_13_0, c_13_1);
                                };
                            }

                            let mut a_col = a.as_ptr().add(m * k_size + k_start);
                            let mut b_ptr = packed_b.as_ptr().add(chunk * chunk_size + strip_idx * KC * TILE_N);

                            let mut _k = 0usize;
                            let k_unroll_end = kc & !7;
                            while _k < k_unroll_end {
                                $crate::simd_primitive!(avx512, $elem, prefetch, b_ptr.add(TILE_N * 16) as *const i8, 0);

                                let vb0_0 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr);
                                let vb0_1 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr.add(LANES));
                                fma_all_rows!(a_col, vb0_0, vb0_1); a_col = a_col.add(1);

                                let vb1_0 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr.add(TILE_N * 1));
                                let vb1_1 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr.add(TILE_N * 1 + LANES));
                                fma_all_rows!(a_col, vb1_0, vb1_1); a_col = a_col.add(1);

                                let vb2_0 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr.add(TILE_N * 2));
                                let vb2_1 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr.add(TILE_N * 2 + LANES));
                                fma_all_rows!(a_col, vb2_0, vb2_1); a_col = a_col.add(1);

                                let vb3_0 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr.add(TILE_N * 3));
                                let vb3_1 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr.add(TILE_N * 3 + LANES));
                                fma_all_rows!(a_col, vb3_0, vb3_1); a_col = a_col.add(1);

                                let vb4_0 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr.add(TILE_N * 4));
                                let vb4_1 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr.add(TILE_N * 4 + LANES));
                                fma_all_rows!(a_col, vb4_0, vb4_1); a_col = a_col.add(1);

                                let vb5_0 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr.add(TILE_N * 5));
                                let vb5_1 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr.add(TILE_N * 5 + LANES));
                                fma_all_rows!(a_col, vb5_0, vb5_1); a_col = a_col.add(1);

                                let vb6_0 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr.add(TILE_N * 6));
                                let vb6_1 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr.add(TILE_N * 6 + LANES));
                                fma_all_rows!(a_col, vb6_0, vb6_1); a_col = a_col.add(1);

                                let vb7_0 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr.add(TILE_N * 7));
                                let vb7_1 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr.add(TILE_N * 7 + LANES));
                                fma_all_rows!(a_col, vb7_0, vb7_1); a_col = a_col.add(1);

                                b_ptr = b_ptr.add(TILE_N * 8);
                                _k += 8;
                            }
                            while _k < kc {
                                let vb_0 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr);
                                let vb_1 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr.add(LANES));
                                fma_all_rows!(a_col, vb_0, vb_1); a_col = a_col.add(1);
                                b_ptr = b_ptr.add(TILE_N);
                                _k += 1;
                            }

                            macro_rules! store_row {
                                ($row:expr, $c0:expr, $c1:expr) => {
                                    $crate::simd_primitive!(avx512, $elem, storeu, c_ptr.add((m + $row) * n_size + n), $c0);
                                    $crate::simd_primitive!(avx512, $elem, storeu, c_ptr.add((m + $row) * n_size + n + LANES), $c1);
                                };
                            }
                            store_row!(0, c_0_0, c_0_1); store_row!(1, c_1_0, c_1_1);
                            store_row!(2, c_2_0, c_2_1); store_row!(3, c_3_0, c_3_1);
                            store_row!(4, c_4_0, c_4_1); store_row!(5, c_5_0, c_5_1);
                            store_row!(6, c_6_0, c_6_1); store_row!(7, c_7_0, c_7_1);
                            store_row!(8, c_8_0, c_8_1); store_row!(9, c_9_0, c_9_1);
                            store_row!(10, c_10_0, c_10_1); store_row!(11, c_11_0, c_11_1);
                            store_row!(12, c_12_0, c_12_1); store_row!(13, c_13_0, c_13_1);
                        }
                        n += TILE_N;
                        strip_idx += 1;
                    }
                    m += TILE_M;
                }

                k_start += KC;
                chunk += 1;
            }

            // Remainder N (scalar) — columns not covered by TILE_N
            let n_main = (n_size / TILE_N) * TILE_N;
            for m in 0..m_size {
                for n in n_main..n_size {
                    let mut sum = <$elem as Element>::ZERO;
                    for k in 0..k_size {
                        sum = <$elem as Element>::mul_add(sum, a[m * k_size + k], b[k * n_size + n]);
                    }
                    c[m * n_size + n] = sum;
                }
            }
            // Remainder M (scalar) — rows not covered by TILE_M
            let m_main = (m_size / TILE_M) * TILE_M;
            for m in m_main..m_size {
                for n in 0..n_main {
                    let mut sum = <$elem as Element>::ZERO;
                    for k in 0..k_size {
                        sum = <$elem as Element>::mul_add(sum, a[m * k_size + k], b[k * n_size + n]);
                    }
                    c[m * n_size + n] = sum;
                }
            }
            WS.with(|c| c.set(packed_b));
        }

        #[inline(always)]
        pub fn matmul(a: &[$elem], b: &[$elem], c: &mut [$elem], m_size: usize, n_size: usize, k_size: usize) {
            unsafe { matmul_avx512(a, b, c, m_size, n_size, k_size); }
        }

        /// matmul_bias: C = A * B + bias (fused — bias loaded into accumulators, no second pass)
        #[target_feature(enable = "avx512f")]
        pub unsafe fn matmul_bias_avx512(a: &[$elem], b: &[$elem], bias: &[$elem], c: &mut [$elem], m_size: usize, n_size: usize, k_size: usize) {
            const TILE_M: usize = 14;
            const TILE_N_VECS: usize = 2;
            const LANES: usize = 16;
            const TILE_N: usize = TILE_N_VECS * LANES;
            const KC: usize = 256;

            assert!(a.len() >= m_size * k_size);
            assert!(b.len() >= n_size * k_size);
            assert!(c.len() >= m_size * n_size);
            assert!(bias.len() >= n_size);

            let n_strips = (n_size + TILE_N - 1) / TILE_N;
            let n_chunks = (k_size + KC - 1) / KC;
            let chunk_size = n_strips * KC * TILE_N;

            // Pack all of B upfront
            let total_packed = n_chunks * chunk_size;
            thread_local! { static WS: std::cell::Cell<Vec<$elem>> = std::cell::Cell::new(Vec::new()); }
            let mut packed_b = WS.with(|c| c.take());
            if packed_b.capacity() < total_packed { packed_b.reserve(total_packed - packed_b.len()); }
            unsafe { packed_b.set_len(total_packed); std::ptr::write_bytes(packed_b.as_mut_ptr(), 0, total_packed); }

            {
                let mut k_start = 0usize;
                let mut chunk = 0usize;
                while k_start < k_size {
                    let kc = KC.min(k_size - k_start);
                    let base = chunk * chunk_size;
                    for i in 0..n_strips {
                        let n_start = i * TILE_N;
                        let actual_n = TILE_N.min(n_size.saturating_sub(n_start));
                        for k in 0..kc {
                            let dst = base + i * KC * TILE_N + k * TILE_N;
                            let src_row = (k_start + k) * n_size + n_start;
                            std::ptr::copy_nonoverlapping(
                                b.as_ptr().add(src_row),
                                packed_b.as_mut_ptr().add(dst),
                                actual_n,
                            );
                        }
                    }
                    k_start += KC;
                    chunk += 1;
                }
            }

            // Init C with bias (each row gets the same bias vector)
            let c_ptr = c.as_mut_ptr();
            for m in 0..m_size {
                unsafe { std::ptr::copy_nonoverlapping(bias.as_ptr(), c_ptr.add(m * n_size), n_size); }
            }

            // Main loop: chunk → m → n (B stays hot in L1/L2)
            let mut k_start = 0usize;
            let mut chunk = 0usize;
            while k_start < k_size {
                let kc = KC.min(k_size - k_start);

                let mut m = 0;
                while m + TILE_M <= m_size {
                    let mut n = 0;
                    let mut strip_idx = 0;
                    while n + TILE_N <= n_size {
                        unsafe {
                            // Load C tile from memory (accumulate across chunks)
                            macro_rules! load_row {
                                ($row:expr) => {(
                                    $crate::simd_primitive!(avx512, $elem, loadu, c_ptr.add((m + $row) * n_size + n)),
                                    $crate::simd_primitive!(avx512, $elem, loadu, c_ptr.add((m + $row) * n_size + n + LANES)),
                                )};
                            }
                            let (mut c_0_0, mut c_0_1) = load_row!(0);
                            let (mut c_1_0, mut c_1_1) = load_row!(1);
                            let (mut c_2_0, mut c_2_1) = load_row!(2);
                            let (mut c_3_0, mut c_3_1) = load_row!(3);
                            let (mut c_4_0, mut c_4_1) = load_row!(4);
                            let (mut c_5_0, mut c_5_1) = load_row!(5);
                            let (mut c_6_0, mut c_6_1) = load_row!(6);
                            let (mut c_7_0, mut c_7_1) = load_row!(7);
                            let (mut c_8_0, mut c_8_1) = load_row!(8);
                            let (mut c_9_0, mut c_9_1) = load_row!(9);
                            let (mut c_10_0, mut c_10_1) = load_row!(10);
                            let (mut c_11_0, mut c_11_1) = load_row!(11);
                            let (mut c_12_0, mut c_12_1) = load_row!(12);
                            let (mut c_13_0, mut c_13_1) = load_row!(13);

                            macro_rules! fma_row {
                                ($a_base:expr, $vb0:ident, $vb1:ident, $row:expr, $c0:ident, $c1:ident) => {
                                    let va = $crate::simd_primitive!(avx512, $elem, splat, *$a_base.add($row));
                                    $c0 = $crate::simd_primitive!(avx512, $elem, fma, va, $vb0, $c0);
                                    $c1 = $crate::simd_primitive!(avx512, $elem, fma, va, $vb1, $c1);
                                };
                            }
                            macro_rules! fma_all_rows {
                                ($a_base:expr, $vb0:ident, $vb1:ident) => {
                                    fma_row!($a_base, $vb0, $vb1, 0, c_0_0, c_0_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size, c_1_0, c_1_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size * 2, c_2_0, c_2_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size * 3, c_3_0, c_3_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size * 4, c_4_0, c_4_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size * 5, c_5_0, c_5_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size * 6, c_6_0, c_6_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size * 7, c_7_0, c_7_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size * 8, c_8_0, c_8_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size * 9, c_9_0, c_9_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size * 10, c_10_0, c_10_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size * 11, c_11_0, c_11_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size * 12, c_12_0, c_12_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size * 13, c_13_0, c_13_1);
                                };
                            }

                            let mut a_col = a.as_ptr().add(m * k_size + k_start);
                            let mut b_ptr = packed_b.as_ptr().add(chunk * chunk_size + strip_idx * KC * TILE_N);

                            let mut _k = 0usize;
                            let k_unroll_end = kc & !7;
                            while _k < k_unroll_end {
                                $crate::simd_primitive!(avx512, $elem, prefetch, b_ptr.add(TILE_N * 16) as *const i8, 0);

                                let vb0_0 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr);
                                let vb0_1 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr.add(LANES));
                                fma_all_rows!(a_col, vb0_0, vb0_1); a_col = a_col.add(1);

                                let vb1_0 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr.add(TILE_N * 1));
                                let vb1_1 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr.add(TILE_N * 1 + LANES));
                                fma_all_rows!(a_col, vb1_0, vb1_1); a_col = a_col.add(1);

                                let vb2_0 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr.add(TILE_N * 2));
                                let vb2_1 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr.add(TILE_N * 2 + LANES));
                                fma_all_rows!(a_col, vb2_0, vb2_1); a_col = a_col.add(1);

                                let vb3_0 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr.add(TILE_N * 3));
                                let vb3_1 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr.add(TILE_N * 3 + LANES));
                                fma_all_rows!(a_col, vb3_0, vb3_1); a_col = a_col.add(1);

                                let vb4_0 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr.add(TILE_N * 4));
                                let vb4_1 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr.add(TILE_N * 4 + LANES));
                                fma_all_rows!(a_col, vb4_0, vb4_1); a_col = a_col.add(1);

                                let vb5_0 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr.add(TILE_N * 5));
                                let vb5_1 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr.add(TILE_N * 5 + LANES));
                                fma_all_rows!(a_col, vb5_0, vb5_1); a_col = a_col.add(1);

                                let vb6_0 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr.add(TILE_N * 6));
                                let vb6_1 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr.add(TILE_N * 6 + LANES));
                                fma_all_rows!(a_col, vb6_0, vb6_1); a_col = a_col.add(1);

                                let vb7_0 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr.add(TILE_N * 7));
                                let vb7_1 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr.add(TILE_N * 7 + LANES));
                                fma_all_rows!(a_col, vb7_0, vb7_1); a_col = a_col.add(1);

                                b_ptr = b_ptr.add(TILE_N * 8);
                                _k += 8;
                            }
                            while _k < kc {
                                let vb_0 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr);
                                let vb_1 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr.add(LANES));
                                fma_all_rows!(a_col, vb_0, vb_1); a_col = a_col.add(1);
                                b_ptr = b_ptr.add(TILE_N);
                                _k += 1;
                            }

                            macro_rules! store_row {
                                ($row:expr, $c0:expr, $c1:expr) => {
                                    $crate::simd_primitive!(avx512, $elem, storeu, c_ptr.add((m + $row) * n_size + n), $c0);
                                    $crate::simd_primitive!(avx512, $elem, storeu, c_ptr.add((m + $row) * n_size + n + LANES), $c1);
                                };
                            }
                            store_row!(0, c_0_0, c_0_1); store_row!(1, c_1_0, c_1_1);
                            store_row!(2, c_2_0, c_2_1); store_row!(3, c_3_0, c_3_1);
                            store_row!(4, c_4_0, c_4_1); store_row!(5, c_5_0, c_5_1);
                            store_row!(6, c_6_0, c_6_1); store_row!(7, c_7_0, c_7_1);
                            store_row!(8, c_8_0, c_8_1); store_row!(9, c_9_0, c_9_1);
                            store_row!(10, c_10_0, c_10_1); store_row!(11, c_11_0, c_11_1);
                            store_row!(12, c_12_0, c_12_1); store_row!(13, c_13_0, c_13_1);
                        }
                        n += TILE_N;
                        strip_idx += 1;
                    }
                    m += TILE_M;
                }

                k_start += KC;
                chunk += 1;
            }

            // Remainder N (scalar) — columns not covered by TILE_N
            let n_main = (n_size / TILE_N) * TILE_N;
            for m in 0..m_size {
                for n in n_main..n_size {
                    let mut sum = bias[n];
                    for k in 0..k_size {
                        sum = <$elem as Element>::mul_add(sum, a[m * k_size + k], b[k * n_size + n]);
                    }
                    c[m * n_size + n] = sum;
                }
            }
            // Remainder M (scalar) — rows not covered by TILE_M
            let m_main = (m_size / TILE_M) * TILE_M;
            for m in m_main..m_size {
                for n in 0..n_main {
                    let mut sum = bias[n];
                    for k in 0..k_size {
                        sum = <$elem as Element>::mul_add(sum, a[m * k_size + k], b[k * n_size + n]);
                    }
                    c[m * n_size + n] = sum;
                }
            }
            WS.with(|c| c.set(packed_b));
        }

        #[inline(always)]
        pub fn matmul_bias(a: &[$elem], b: &[$elem], bias: &[$elem], c: &mut [$elem], m_size: usize, n_size: usize, k_size: usize) {
            unsafe { matmul_bias_avx512(a, b, bias, c, m_size, n_size, k_size); }
        }

        /// Pre-pack B matrix for repeated matmul calls with the same weights.
        /// Returns (packed_data, n_size, k_size) — pass to matmul_prepacked / matmul_bias_prepacked.
        pub fn pack_b(b: &[$elem], n_size: usize, k_size: usize) -> Vec<$elem> {
            const TILE_N_VECS: usize = 2;
            const LANES: usize = 16;
            const TILE_N: usize = TILE_N_VECS * LANES;
            const KC: usize = 256;

            assert!(b.len() >= k_size * n_size);
            let n_strips = (n_size + TILE_N - 1) / TILE_N;
            let n_chunks = (k_size + KC - 1) / KC;
            let chunk_size = n_strips * KC * TILE_N;
            let total = n_chunks * chunk_size;
            let mut packed = vec![<$elem as Element>::ZERO; total];

            let mut k_start = 0usize;
            let mut chunk = 0usize;
            while k_start < k_size {
                let kc = KC.min(k_size - k_start);
                let base = chunk * chunk_size;
                for (i, n_start) in (0..n_size).step_by(TILE_N).enumerate() {
                    let actual_n = TILE_N.min(n_size - n_start);
                    for k in 0..kc {
                        let dst = base + i * KC * TILE_N + k * TILE_N;
                        let src_row = (k_start + k) * n_size + n_start;
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                b.as_ptr().add(src_row),
                                packed.as_mut_ptr().add(dst),
                                actual_n,
                            );
                            // Remainder already zero from vec! init
                        }
                    }
                }
                k_start += KC;
                chunk += 1;
            }
            packed
        }

        /// Matmul using pre-packed B. `packed_b` must come from `pack_b` with matching n_size/k_size.
        /// Loop order: chunk(KC) → m → n — B stays hot in cache
        #[target_feature(enable = "avx512f")]
        unsafe fn matmul_prepacked_avx512(a: &[$elem], packed_b: &[$elem], c: &mut [$elem], m_size: usize, n_size: usize, k_size: usize) {
            const TILE_M: usize = 14;
            const TILE_N_VECS: usize = 2;
            const LANES: usize = 16;
            const TILE_N: usize = TILE_N_VECS * LANES;
            const KC: usize = 256;

            assert!(a.len() >= m_size * k_size);
            assert!(c.len() >= m_size * n_size);

            let n_strips = (n_size + TILE_N - 1) / TILE_N;
            let chunk_size = n_strips * KC * TILE_N;

            // Zero C matrix — accumulation across chunks
            for i in 0..c.len().min(m_size * n_size) { c[i] = <$elem as Element>::ZERO; }
            let c_ptr = c.as_mut_ptr();

            // Main loop: chunk → m → n (B stays hot in L1/L2)
            let mut k_start = 0usize;
            let mut chunk = 0usize;
            while k_start < k_size {
                let kc = KC.min(k_size - k_start);

                let mut m = 0;
                while m + TILE_M <= m_size {
                    let mut n = 0;
                    let mut strip_idx = 0;
                    while n + TILE_N <= n_size {
                        unsafe {
                            // Load C tile from memory (accumulate across chunks)
                            macro_rules! load_row {
                                ($row:expr) => {(
                                    $crate::simd_primitive!(avx512, $elem, loadu, c_ptr.add((m + $row) * n_size + n)),
                                    $crate::simd_primitive!(avx512, $elem, loadu, c_ptr.add((m + $row) * n_size + n + LANES)),
                                )};
                            }
                            let (mut c_0_0, mut c_0_1) = load_row!(0);
                            let (mut c_1_0, mut c_1_1) = load_row!(1);
                            let (mut c_2_0, mut c_2_1) = load_row!(2);
                            let (mut c_3_0, mut c_3_1) = load_row!(3);
                            let (mut c_4_0, mut c_4_1) = load_row!(4);
                            let (mut c_5_0, mut c_5_1) = load_row!(5);
                            let (mut c_6_0, mut c_6_1) = load_row!(6);
                            let (mut c_7_0, mut c_7_1) = load_row!(7);
                            let (mut c_8_0, mut c_8_1) = load_row!(8);
                            let (mut c_9_0, mut c_9_1) = load_row!(9);
                            let (mut c_10_0, mut c_10_1) = load_row!(10);
                            let (mut c_11_0, mut c_11_1) = load_row!(11);
                            let (mut c_12_0, mut c_12_1) = load_row!(12);
                            let (mut c_13_0, mut c_13_1) = load_row!(13);

                            macro_rules! fma_row {
                                ($a_base:expr, $vb0:ident, $vb1:ident, $row:expr, $c0:ident, $c1:ident) => {
                                    let va = $crate::simd_primitive!(avx512, $elem, splat, *$a_base.add($row));
                                    $c0 = $crate::simd_primitive!(avx512, $elem, fma, va, $vb0, $c0);
                                    $c1 = $crate::simd_primitive!(avx512, $elem, fma, va, $vb1, $c1);
                                };
                            }
                            macro_rules! fma_all_rows {
                                ($a_base:expr, $vb0:ident, $vb1:ident) => {
                                    fma_row!($a_base, $vb0, $vb1, 0, c_0_0, c_0_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size, c_1_0, c_1_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size * 2, c_2_0, c_2_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size * 3, c_3_0, c_3_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size * 4, c_4_0, c_4_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size * 5, c_5_0, c_5_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size * 6, c_6_0, c_6_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size * 7, c_7_0, c_7_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size * 8, c_8_0, c_8_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size * 9, c_9_0, c_9_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size * 10, c_10_0, c_10_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size * 11, c_11_0, c_11_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size * 12, c_12_0, c_12_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size * 13, c_13_0, c_13_1);
                                };
                            }

                            let mut a_col = a.as_ptr().add(m * k_size + k_start);
                            let mut b_ptr = packed_b.as_ptr().add(chunk * chunk_size + strip_idx * KC * TILE_N);

                            let mut _k = 0usize;
                            let k_unroll_end = kc & !7;
                            while _k < k_unroll_end {
                                $crate::simd_primitive!(avx512, $elem, prefetch, b_ptr.add(TILE_N * 16) as *const i8, 0);

                                let vb0_0 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr);
                                let vb0_1 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr.add(LANES));
                                fma_all_rows!(a_col, vb0_0, vb0_1); a_col = a_col.add(1);

                                let vb1_0 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr.add(TILE_N * 1));
                                let vb1_1 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr.add(TILE_N * 1 + LANES));
                                fma_all_rows!(a_col, vb1_0, vb1_1); a_col = a_col.add(1);

                                let vb2_0 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr.add(TILE_N * 2));
                                let vb2_1 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr.add(TILE_N * 2 + LANES));
                                fma_all_rows!(a_col, vb2_0, vb2_1); a_col = a_col.add(1);

                                let vb3_0 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr.add(TILE_N * 3));
                                let vb3_1 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr.add(TILE_N * 3 + LANES));
                                fma_all_rows!(a_col, vb3_0, vb3_1); a_col = a_col.add(1);

                                let vb4_0 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr.add(TILE_N * 4));
                                let vb4_1 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr.add(TILE_N * 4 + LANES));
                                fma_all_rows!(a_col, vb4_0, vb4_1); a_col = a_col.add(1);

                                let vb5_0 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr.add(TILE_N * 5));
                                let vb5_1 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr.add(TILE_N * 5 + LANES));
                                fma_all_rows!(a_col, vb5_0, vb5_1); a_col = a_col.add(1);

                                let vb6_0 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr.add(TILE_N * 6));
                                let vb6_1 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr.add(TILE_N * 6 + LANES));
                                fma_all_rows!(a_col, vb6_0, vb6_1); a_col = a_col.add(1);

                                let vb7_0 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr.add(TILE_N * 7));
                                let vb7_1 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr.add(TILE_N * 7 + LANES));
                                fma_all_rows!(a_col, vb7_0, vb7_1); a_col = a_col.add(1);

                                b_ptr = b_ptr.add(TILE_N * 8);
                                _k += 8;
                            }
                            while _k < kc {
                                let vb_0 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr);
                                let vb_1 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr.add(LANES));
                                fma_all_rows!(a_col, vb_0, vb_1); a_col = a_col.add(1);
                                b_ptr = b_ptr.add(TILE_N);
                                _k += 1;
                            }

                            macro_rules! store_row {
                                ($row:expr, $c0:expr, $c1:expr) => {
                                    $crate::simd_primitive!(avx512, $elem, storeu, c_ptr.add((m + $row) * n_size + n), $c0);
                                    $crate::simd_primitive!(avx512, $elem, storeu, c_ptr.add((m + $row) * n_size + n + LANES), $c1);
                                };
                            }
                            store_row!(0, c_0_0, c_0_1); store_row!(1, c_1_0, c_1_1);
                            store_row!(2, c_2_0, c_2_1); store_row!(3, c_3_0, c_3_1);
                            store_row!(4, c_4_0, c_4_1); store_row!(5, c_5_0, c_5_1);
                            store_row!(6, c_6_0, c_6_1); store_row!(7, c_7_0, c_7_1);
                            store_row!(8, c_8_0, c_8_1); store_row!(9, c_9_0, c_9_1);
                            store_row!(10, c_10_0, c_10_1); store_row!(11, c_11_0, c_11_1);
                            store_row!(12, c_12_0, c_12_1); store_row!(13, c_13_0, c_13_1);
                        }
                        n += TILE_N;
                        strip_idx += 1;
                    }
                    m += TILE_M;
                }

                k_start += KC;
                chunk += 1;
            }

            // Remainder N (scalar) — columns not covered by TILE_N
            let n_main = (n_size / TILE_N) * TILE_N;
            for m in 0..m_size {
                for n in n_main..n_size {
                    let mut sum = <$elem as Element>::ZERO;
                    for k in 0..k_size {
                        let chunk_idx = k / KC; let k_in_chunk = k % KC;
                        sum = <$elem as Element>::mul_add(sum, a[m * k_size + k], packed_b[chunk_idx * chunk_size + (n / TILE_N) * KC * TILE_N + k_in_chunk * TILE_N + (n % TILE_N)]);
                    }
                    c[m * n_size + n] = sum;
                }
            }
            // Remainder M (scalar) — rows not covered by TILE_M
            let m_main = (m_size / TILE_M) * TILE_M;
            for m in m_main..m_size {
                for n in 0..n_main {
                    let mut sum = <$elem as Element>::ZERO;
                    for k in 0..k_size {
                        let chunk_idx = k / KC; let k_in_chunk = k % KC;
                        sum = <$elem as Element>::mul_add(sum, a[m * k_size + k], packed_b[chunk_idx * chunk_size + (n / TILE_N) * KC * TILE_N + k_in_chunk * TILE_N + (n % TILE_N)]);
                    }
                    c[m * n_size + n] = sum;
                }
            }
        }

        #[inline(always)]
        pub fn matmul_prepacked(a: &[$elem], packed_b: &[$elem], c: &mut [$elem], m_size: usize, n_size: usize, k_size: usize) {
            unsafe { matmul_prepacked_avx512(a, packed_b, c, m_size, n_size, k_size); }
        }

        /// Matmul+bias using pre-packed B. Loop order: chunk(KC) → m → n.
        #[target_feature(enable = "avx512f")]
        unsafe fn matmul_bias_prepacked_avx512(a: &[$elem], packed_b: &[$elem], bias: &[$elem], c: &mut [$elem], m_size: usize, n_size: usize, k_size: usize) {
            const TILE_M: usize = 14;
            const TILE_N_VECS: usize = 2;
            const LANES: usize = 16;
            const TILE_N: usize = TILE_N_VECS * LANES;
            const KC: usize = 256;

            assert!(a.len() >= m_size * k_size);
            assert!(c.len() >= m_size * n_size);
            assert!(bias.len() >= n_size);

            let n_strips = (n_size + TILE_N - 1) / TILE_N;
            let chunk_size = n_strips * KC * TILE_N;

            // Init C with bias (each row gets the same bias vector)
            let c_ptr = c.as_mut_ptr();
            for m in 0..m_size {
                unsafe { std::ptr::copy_nonoverlapping(bias.as_ptr(), c_ptr.add(m * n_size), n_size); }
            }

            // Main loop: chunk → m → n (B stays hot in L1/L2)
            let mut k_start = 0usize;
            let mut chunk = 0usize;
            while k_start < k_size {
                let kc = KC.min(k_size - k_start);

                let mut m = 0;
                while m + TILE_M <= m_size {
                    let mut n = 0;
                    let mut strip_idx = 0;
                    while n + TILE_N <= n_size {
                        unsafe {
                            // Load C tile from memory (accumulate across chunks)
                            macro_rules! load_row {
                                ($row:expr) => {(
                                    $crate::simd_primitive!(avx512, $elem, loadu, c_ptr.add((m + $row) * n_size + n)),
                                    $crate::simd_primitive!(avx512, $elem, loadu, c_ptr.add((m + $row) * n_size + n + LANES)),
                                )};
                            }
                            let (mut c_0_0, mut c_0_1) = load_row!(0);
                            let (mut c_1_0, mut c_1_1) = load_row!(1);
                            let (mut c_2_0, mut c_2_1) = load_row!(2);
                            let (mut c_3_0, mut c_3_1) = load_row!(3);
                            let (mut c_4_0, mut c_4_1) = load_row!(4);
                            let (mut c_5_0, mut c_5_1) = load_row!(5);
                            let (mut c_6_0, mut c_6_1) = load_row!(6);
                            let (mut c_7_0, mut c_7_1) = load_row!(7);
                            let (mut c_8_0, mut c_8_1) = load_row!(8);
                            let (mut c_9_0, mut c_9_1) = load_row!(9);
                            let (mut c_10_0, mut c_10_1) = load_row!(10);
                            let (mut c_11_0, mut c_11_1) = load_row!(11);
                            let (mut c_12_0, mut c_12_1) = load_row!(12);
                            let (mut c_13_0, mut c_13_1) = load_row!(13);

                            macro_rules! fma_row {
                                ($a_base:expr, $vb0:ident, $vb1:ident, $row:expr, $c0:ident, $c1:ident) => {
                                    let va = $crate::simd_primitive!(avx512, $elem, splat, *$a_base.add($row));
                                    $c0 = $crate::simd_primitive!(avx512, $elem, fma, va, $vb0, $c0);
                                    $c1 = $crate::simd_primitive!(avx512, $elem, fma, va, $vb1, $c1);
                                };
                            }
                            macro_rules! fma_all_rows {
                                ($a_base:expr, $vb0:ident, $vb1:ident) => {
                                    fma_row!($a_base, $vb0, $vb1, 0, c_0_0, c_0_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size, c_1_0, c_1_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size * 2, c_2_0, c_2_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size * 3, c_3_0, c_3_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size * 4, c_4_0, c_4_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size * 5, c_5_0, c_5_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size * 6, c_6_0, c_6_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size * 7, c_7_0, c_7_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size * 8, c_8_0, c_8_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size * 9, c_9_0, c_9_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size * 10, c_10_0, c_10_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size * 11, c_11_0, c_11_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size * 12, c_12_0, c_12_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size * 13, c_13_0, c_13_1);
                                };
                            }

                            let mut a_col = a.as_ptr().add(m * k_size + k_start);
                            let mut b_ptr = packed_b.as_ptr().add(chunk * chunk_size + strip_idx * KC * TILE_N);

                            let mut _k = 0usize;
                            let k_unroll_end = kc & !7;
                            while _k < k_unroll_end {
                                $crate::simd_primitive!(avx512, $elem, prefetch, b_ptr.add(TILE_N * 16) as *const i8, 0);

                                let vb0_0 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr);
                                let vb0_1 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr.add(LANES));
                                fma_all_rows!(a_col, vb0_0, vb0_1); a_col = a_col.add(1);

                                let vb1_0 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr.add(TILE_N * 1));
                                let vb1_1 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr.add(TILE_N * 1 + LANES));
                                fma_all_rows!(a_col, vb1_0, vb1_1); a_col = a_col.add(1);

                                let vb2_0 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr.add(TILE_N * 2));
                                let vb2_1 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr.add(TILE_N * 2 + LANES));
                                fma_all_rows!(a_col, vb2_0, vb2_1); a_col = a_col.add(1);

                                let vb3_0 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr.add(TILE_N * 3));
                                let vb3_1 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr.add(TILE_N * 3 + LANES));
                                fma_all_rows!(a_col, vb3_0, vb3_1); a_col = a_col.add(1);

                                let vb4_0 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr.add(TILE_N * 4));
                                let vb4_1 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr.add(TILE_N * 4 + LANES));
                                fma_all_rows!(a_col, vb4_0, vb4_1); a_col = a_col.add(1);

                                let vb5_0 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr.add(TILE_N * 5));
                                let vb5_1 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr.add(TILE_N * 5 + LANES));
                                fma_all_rows!(a_col, vb5_0, vb5_1); a_col = a_col.add(1);

                                let vb6_0 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr.add(TILE_N * 6));
                                let vb6_1 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr.add(TILE_N * 6 + LANES));
                                fma_all_rows!(a_col, vb6_0, vb6_1); a_col = a_col.add(1);

                                let vb7_0 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr.add(TILE_N * 7));
                                let vb7_1 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr.add(TILE_N * 7 + LANES));
                                fma_all_rows!(a_col, vb7_0, vb7_1); a_col = a_col.add(1);

                                b_ptr = b_ptr.add(TILE_N * 8);
                                _k += 8;
                            }
                            while _k < kc {
                                let vb_0 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr);
                                let vb_1 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr.add(LANES));
                                fma_all_rows!(a_col, vb_0, vb_1); a_col = a_col.add(1);
                                b_ptr = b_ptr.add(TILE_N);
                                _k += 1;
                            }

                            macro_rules! store_row {
                                ($row:expr, $c0:expr, $c1:expr) => {
                                    $crate::simd_primitive!(avx512, $elem, storeu, c_ptr.add((m + $row) * n_size + n), $c0);
                                    $crate::simd_primitive!(avx512, $elem, storeu, c_ptr.add((m + $row) * n_size + n + LANES), $c1);
                                };
                            }
                            store_row!(0, c_0_0, c_0_1); store_row!(1, c_1_0, c_1_1);
                            store_row!(2, c_2_0, c_2_1); store_row!(3, c_3_0, c_3_1);
                            store_row!(4, c_4_0, c_4_1); store_row!(5, c_5_0, c_5_1);
                            store_row!(6, c_6_0, c_6_1); store_row!(7, c_7_0, c_7_1);
                            store_row!(8, c_8_0, c_8_1); store_row!(9, c_9_0, c_9_1);
                            store_row!(10, c_10_0, c_10_1); store_row!(11, c_11_0, c_11_1);
                            store_row!(12, c_12_0, c_12_1); store_row!(13, c_13_0, c_13_1);
                        }
                        n += TILE_N;
                        strip_idx += 1;
                    }
                    m += TILE_M;
                }

                k_start += KC;
                chunk += 1;
            }

            // Remainder N (scalar) — columns not covered by TILE_N
            let n_main = (n_size / TILE_N) * TILE_N;
            for m in 0..m_size {
                for n in n_main..n_size {
                    let mut sum = bias[n];
                    for k in 0..k_size {
                        let chunk_idx = k / KC; let k_in_chunk = k % KC;
                        sum = <$elem as Element>::mul_add(sum, a[m * k_size + k], packed_b[chunk_idx * chunk_size + (n / TILE_N) * KC * TILE_N + k_in_chunk * TILE_N + (n % TILE_N)]);
                    }
                    c[m * n_size + n] = sum;
                }
            }
            // Remainder M (scalar) — rows not covered by TILE_M
            let m_main = (m_size / TILE_M) * TILE_M;
            for m in m_main..m_size {
                for n in 0..n_main {
                    let mut sum = bias[n];
                    for k in 0..k_size {
                        let chunk_idx = k / KC; let k_in_chunk = k % KC;
                        sum = <$elem as Element>::mul_add(sum, a[m * k_size + k], packed_b[chunk_idx * chunk_size + (n / TILE_N) * KC * TILE_N + k_in_chunk * TILE_N + (n % TILE_N)]);
                    }
                    c[m * n_size + n] = sum;
                }
            }
        }

        #[inline(always)]
        pub fn matmul_bias_prepacked(a: &[$elem], packed_b: &[$elem], bias: &[$elem], c: &mut [$elem], m_size: usize, n_size: usize, k_size: usize) {
            unsafe { matmul_bias_prepacked_avx512(a, packed_b, bias, c, m_size, n_size, k_size); }
        }
    };

    (avx2, $elem:ident) => {
        // AVX2 Optimized Implementation: 6×16 microkernel
        // Loop order: chunk(KC) → m → n — B stays hot in cache
        #[target_feature(enable = "avx2")]
        pub unsafe fn matmul_avx2(a: &[$elem], b: &[$elem], c: &mut [$elem], m_size: usize, n_size: usize, k_size: usize) {
            const TILE_M: usize = 6;
            const TILE_N_VECS: usize = 2;
            const LANES: usize = 8;
            const TILE_N: usize = TILE_N_VECS * LANES; // 16
            const KC: usize = 256;

            assert!(a.len() >= m_size * k_size);
            assert!(b.len() >= n_size * k_size);
            assert!(c.len() >= m_size * n_size);

            let n_strips = (n_size + TILE_N - 1) / TILE_N;
            let n_chunks = (k_size + KC - 1) / KC;
            let chunk_size = n_strips * KC * TILE_N;

            // Pack all of B upfront
            let total_packed = n_chunks * chunk_size;
            thread_local! { static WS: std::cell::Cell<Vec<$elem>> = std::cell::Cell::new(Vec::new()); }
            let mut packed_b = WS.with(|c| c.take());
            if packed_b.capacity() < total_packed { packed_b.reserve(total_packed - packed_b.len()); }
            unsafe { packed_b.set_len(total_packed); std::ptr::write_bytes(packed_b.as_mut_ptr(), 0, total_packed); }

            {
                let mut k_start = 0usize;
                let mut chunk = 0usize;
                while k_start < k_size {
                    let kc = KC.min(k_size - k_start);
                    let base = chunk * chunk_size;
                    for i in 0..n_strips {
                        let n_start = i * TILE_N;
                        let actual_n = TILE_N.min(n_size.saturating_sub(n_start));
                        for k in 0..kc {
                            let dst = base + i * KC * TILE_N + k * TILE_N;
                            let src_row = (k_start + k) * n_size + n_start;
                            std::ptr::copy_nonoverlapping(
                                b.as_ptr().add(src_row),
                                packed_b.as_mut_ptr().add(dst),
                                actual_n,
                            );
                        }
                    }
                    k_start += KC;
                    chunk += 1;
                }
            }

            // Zero C matrix — accumulation across chunks
            for i in 0..c.len().min(m_size * n_size) { c[i] = <$elem as Element>::ZERO; }

            // Main loop: chunk → m → n (B stays hot in L1/L2)
            let c_ptr = c.as_mut_ptr();
            let mut k_start = 0usize;
            let mut chunk = 0usize;
            while k_start < k_size {
                let kc = KC.min(k_size - k_start);

                let mut m = 0;
                while m + TILE_M <= m_size {
                    let mut n = 0;
                    let mut strip_idx = 0;
                    while n + TILE_N <= n_size {
                        unsafe {
                            // Load C tile from memory (accumulate across chunks)
                            macro_rules! load_row {
                                ($row:expr) => {(
                                    $crate::simd_primitive!(avx2, $elem, loadu, c_ptr.add((m + $row) * n_size + n)),
                                    $crate::simd_primitive!(avx2, $elem, loadu, c_ptr.add((m + $row) * n_size + n + LANES)),
                                )};
                            }
                            let (mut c_0_0, mut c_0_1) = load_row!(0);
                            let (mut c_1_0, mut c_1_1) = load_row!(1);
                            let (mut c_2_0, mut c_2_1) = load_row!(2);
                            let (mut c_3_0, mut c_3_1) = load_row!(3);
                            let (mut c_4_0, mut c_4_1) = load_row!(4);
                            let (mut c_5_0, mut c_5_1) = load_row!(5);

                            macro_rules! fma_row {
                                ($a_base:expr, $vb0:ident, $vb1:ident, $row:expr, $c0:ident, $c1:ident) => {
                                    let va = $crate::simd_primitive!(avx2, $elem, splat, *$a_base.add($row));
                                    $c0 = $crate::simd_primitive!(avx2, $elem, fma, va, $vb0, $c0);
                                    $c1 = $crate::simd_primitive!(avx2, $elem, fma, va, $vb1, $c1);
                                };
                            }
                            macro_rules! fma_all {
                                ($a_base:expr, $vb0:ident, $vb1:ident) => {
                                    fma_row!($a_base, $vb0, $vb1, 0, c_0_0, c_0_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size, c_1_0, c_1_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size * 2, c_2_0, c_2_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size * 3, c_3_0, c_3_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size * 4, c_4_0, c_4_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size * 5, c_5_0, c_5_1);
                                };
                            }

                            let mut a_col = a.as_ptr().add(m * k_size + k_start);
                            let mut b_ptr = packed_b.as_ptr().add(chunk * chunk_size + strip_idx * KC * TILE_N);

                            let mut _k = 0usize;
                            let k_unroll_end = kc & !7;
                            while _k < k_unroll_end {
                                $crate::simd_primitive!(avx2, $elem, prefetch, b_ptr.add(TILE_N * 16) as *const i8, 0);

                                let vb0_0 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr);
                                let vb0_1 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr.add(LANES));
                                fma_all!(a_col, vb0_0, vb0_1); a_col = a_col.add(1);

                                let vb1_0 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr.add(TILE_N));
                                let vb1_1 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr.add(TILE_N + LANES));
                                fma_all!(a_col, vb1_0, vb1_1); a_col = a_col.add(1);

                                let vb2_0 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr.add(TILE_N * 2));
                                let vb2_1 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr.add(TILE_N * 2 + LANES));
                                fma_all!(a_col, vb2_0, vb2_1); a_col = a_col.add(1);

                                let vb3_0 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr.add(TILE_N * 3));
                                let vb3_1 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr.add(TILE_N * 3 + LANES));
                                fma_all!(a_col, vb3_0, vb3_1); a_col = a_col.add(1);

                                let vb4_0 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr.add(TILE_N * 4));
                                let vb4_1 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr.add(TILE_N * 4 + LANES));
                                fma_all!(a_col, vb4_0, vb4_1); a_col = a_col.add(1);

                                let vb5_0 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr.add(TILE_N * 5));
                                let vb5_1 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr.add(TILE_N * 5 + LANES));
                                fma_all!(a_col, vb5_0, vb5_1); a_col = a_col.add(1);

                                let vb6_0 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr.add(TILE_N * 6));
                                let vb6_1 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr.add(TILE_N * 6 + LANES));
                                fma_all!(a_col, vb6_0, vb6_1); a_col = a_col.add(1);

                                let vb7_0 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr.add(TILE_N * 7));
                                let vb7_1 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr.add(TILE_N * 7 + LANES));
                                fma_all!(a_col, vb7_0, vb7_1); a_col = a_col.add(1);

                                b_ptr = b_ptr.add(TILE_N * 8);
                                _k += 8;
                            }
                            while _k < kc {
                                let vb_0 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr);
                                let vb_1 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr.add(LANES));
                                fma_all!(a_col, vb_0, vb_1); a_col = a_col.add(1);
                                b_ptr = b_ptr.add(TILE_N);
                                _k += 1;
                            }

                            // Store C tile back
                            macro_rules! store_row {
                                ($row:expr, $c0:expr, $c1:expr) => {
                                    $crate::simd_primitive!(avx2, $elem, storeu, c_ptr.add((m + $row) * n_size + n), $c0);
                                    $crate::simd_primitive!(avx2, $elem, storeu, c_ptr.add((m + $row) * n_size + n + LANES), $c1);
                                };
                            }
                            store_row!(0, c_0_0, c_0_1); store_row!(1, c_1_0, c_1_1);
                            store_row!(2, c_2_0, c_2_1); store_row!(3, c_3_0, c_3_1);
                            store_row!(4, c_4_0, c_4_1); store_row!(5, c_5_0, c_5_1);
                        }
                        n += TILE_N;
                        strip_idx += 1;
                    }
                    m += TILE_M;
                }

                k_start += KC;
                chunk += 1;
            }

            // Remainder N (scalar) — columns not covered by TILE_N
            let n_main = (n_size / TILE_N) * TILE_N;
            for m in 0..m_size {
                for n in n_main..n_size {
                    let mut sum = <$elem as Element>::ZERO;
                    for k in 0..k_size {
                        sum = <$elem as Element>::mul_add(sum, a[m * k_size + k], b[k * n_size + n]);
                    }
                    c[m * n_size + n] = sum;
                }
            }
            // Remainder M (scalar) — rows not covered by TILE_M
            let m_main = (m_size / TILE_M) * TILE_M;
            for m in m_main..m_size {
                for n in 0..n_main {
                    let mut sum = <$elem as Element>::ZERO;
                    for k in 0..k_size {
                        sum = <$elem as Element>::mul_add(sum, a[m * k_size + k], b[k * n_size + n]);
                    }
                    c[m * n_size + n] = sum;
                }
            }
            WS.with(|c| c.set(packed_b));
        }

        #[inline(always)]
        pub fn matmul(a: &[$elem], b: &[$elem], c: &mut [$elem], m_size: usize, n_size: usize, k_size: usize) {
            unsafe { matmul_avx2(a, b, c, m_size, n_size, k_size); }
        }

        /// matmul_bias: C = A * B + bias (fused — bias loaded into accumulators)
        #[target_feature(enable = "avx2")]
        pub unsafe fn matmul_bias_avx2(a: &[$elem], b: &[$elem], bias: &[$elem], c: &mut [$elem], m_size: usize, n_size: usize, k_size: usize) {
            const TILE_M: usize = 6;
            const TILE_N_VECS: usize = 2;
            const LANES: usize = 8;
            const TILE_N: usize = TILE_N_VECS * LANES;
            const KC: usize = 256;

            assert!(a.len() >= m_size * k_size);
            assert!(b.len() >= n_size * k_size);
            assert!(c.len() >= m_size * n_size);
            assert!(bias.len() >= n_size);

            let n_strips = (n_size + TILE_N - 1) / TILE_N;
            let n_chunks = (k_size + KC - 1) / KC;
            let chunk_size = n_strips * KC * TILE_N;

            let total_packed = n_chunks * chunk_size;
            thread_local! { static WS: std::cell::Cell<Vec<$elem>> = std::cell::Cell::new(Vec::new()); }
            let mut packed_b = WS.with(|c| c.take());
            if packed_b.capacity() < total_packed { packed_b.reserve(total_packed - packed_b.len()); }
            unsafe { packed_b.set_len(total_packed); std::ptr::write_bytes(packed_b.as_mut_ptr(), 0, total_packed); }

            {
                let mut k_start = 0usize;
                let mut chunk = 0usize;
                while k_start < k_size {
                    let kc = KC.min(k_size - k_start);
                    let base = chunk * chunk_size;
                    for i in 0..n_strips {
                        let n_start = i * TILE_N;
                        let actual_n = TILE_N.min(n_size.saturating_sub(n_start));
                        for k in 0..kc {
                            let dst = base + i * KC * TILE_N + k * TILE_N;
                            let src_row = (k_start + k) * n_size + n_start;
                            std::ptr::copy_nonoverlapping(
                                b.as_ptr().add(src_row),
                                packed_b.as_mut_ptr().add(dst),
                                actual_n,
                            );
                        }
                    }
                    k_start += KC;
                    chunk += 1;
                }
            }

            // Init C with bias (each row gets the same bias vector)
            let c_ptr = c.as_mut_ptr();
            for m in 0..m_size {
                unsafe { std::ptr::copy_nonoverlapping(bias.as_ptr(), c_ptr.add(m * n_size), n_size); }
            }

            // Main loop: chunk → m → n (B stays hot in L1/L2)
            let mut k_start = 0usize;
            let mut chunk = 0usize;
            while k_start < k_size {
                let kc = KC.min(k_size - k_start);

                let mut m = 0;
                while m + TILE_M <= m_size {
                    let mut n = 0;
                    let mut strip_idx = 0;
                    while n + TILE_N <= n_size {
                        unsafe {
                            macro_rules! load_row {
                                ($row:expr) => {(
                                    $crate::simd_primitive!(avx2, $elem, loadu, c_ptr.add((m + $row) * n_size + n)),
                                    $crate::simd_primitive!(avx2, $elem, loadu, c_ptr.add((m + $row) * n_size + n + LANES)),
                                )};
                            }
                            let (mut c_0_0, mut c_0_1) = load_row!(0);
                            let (mut c_1_0, mut c_1_1) = load_row!(1);
                            let (mut c_2_0, mut c_2_1) = load_row!(2);
                            let (mut c_3_0, mut c_3_1) = load_row!(3);
                            let (mut c_4_0, mut c_4_1) = load_row!(4);
                            let (mut c_5_0, mut c_5_1) = load_row!(5);

                            macro_rules! fma_row {
                                ($a_base:expr, $vb0:ident, $vb1:ident, $row:expr, $c0:ident, $c1:ident) => {
                                    let va = $crate::simd_primitive!(avx2, $elem, splat, *$a_base.add($row));
                                    $c0 = $crate::simd_primitive!(avx2, $elem, fma, va, $vb0, $c0);
                                    $c1 = $crate::simd_primitive!(avx2, $elem, fma, va, $vb1, $c1);
                                };
                            }
                            macro_rules! fma_all {
                                ($a_base:expr, $vb0:ident, $vb1:ident) => {
                                    fma_row!($a_base, $vb0, $vb1, 0, c_0_0, c_0_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size, c_1_0, c_1_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size * 2, c_2_0, c_2_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size * 3, c_3_0, c_3_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size * 4, c_4_0, c_4_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size * 5, c_5_0, c_5_1);
                                };
                            }

                            let mut a_col = a.as_ptr().add(m * k_size + k_start);
                            let mut b_ptr = packed_b.as_ptr().add(chunk * chunk_size + strip_idx * KC * TILE_N);

                            let mut _k = 0usize;
                            let k_unroll_end = kc & !7;
                            while _k < k_unroll_end {
                                $crate::simd_primitive!(avx2, $elem, prefetch, b_ptr.add(TILE_N * 16) as *const i8, 0);

                                let vb0_0 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr);
                                let vb0_1 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr.add(LANES));
                                fma_all!(a_col, vb0_0, vb0_1); a_col = a_col.add(1);

                                let vb1_0 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr.add(TILE_N));
                                let vb1_1 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr.add(TILE_N + LANES));
                                fma_all!(a_col, vb1_0, vb1_1); a_col = a_col.add(1);

                                let vb2_0 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr.add(TILE_N * 2));
                                let vb2_1 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr.add(TILE_N * 2 + LANES));
                                fma_all!(a_col, vb2_0, vb2_1); a_col = a_col.add(1);

                                let vb3_0 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr.add(TILE_N * 3));
                                let vb3_1 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr.add(TILE_N * 3 + LANES));
                                fma_all!(a_col, vb3_0, vb3_1); a_col = a_col.add(1);

                                let vb4_0 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr.add(TILE_N * 4));
                                let vb4_1 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr.add(TILE_N * 4 + LANES));
                                fma_all!(a_col, vb4_0, vb4_1); a_col = a_col.add(1);

                                let vb5_0 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr.add(TILE_N * 5));
                                let vb5_1 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr.add(TILE_N * 5 + LANES));
                                fma_all!(a_col, vb5_0, vb5_1); a_col = a_col.add(1);

                                let vb6_0 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr.add(TILE_N * 6));
                                let vb6_1 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr.add(TILE_N * 6 + LANES));
                                fma_all!(a_col, vb6_0, vb6_1); a_col = a_col.add(1);

                                let vb7_0 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr.add(TILE_N * 7));
                                let vb7_1 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr.add(TILE_N * 7 + LANES));
                                fma_all!(a_col, vb7_0, vb7_1); a_col = a_col.add(1);

                                b_ptr = b_ptr.add(TILE_N * 8);
                                _k += 8;
                            }
                            while _k < kc {
                                let vb_0 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr);
                                let vb_1 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr.add(LANES));
                                fma_all!(a_col, vb_0, vb_1); a_col = a_col.add(1);
                                b_ptr = b_ptr.add(TILE_N);
                                _k += 1;
                            }

                            macro_rules! store_row {
                                ($row:expr, $c0:expr, $c1:expr) => {
                                    $crate::simd_primitive!(avx2, $elem, storeu, c_ptr.add((m + $row) * n_size + n), $c0);
                                    $crate::simd_primitive!(avx2, $elem, storeu, c_ptr.add((m + $row) * n_size + n + LANES), $c1);
                                };
                            }
                            store_row!(0, c_0_0, c_0_1); store_row!(1, c_1_0, c_1_1);
                            store_row!(2, c_2_0, c_2_1); store_row!(3, c_3_0, c_3_1);
                            store_row!(4, c_4_0, c_4_1); store_row!(5, c_5_0, c_5_1);
                        }
                        n += TILE_N;
                        strip_idx += 1;
                    }
                    m += TILE_M;
                }

                k_start += KC;
                chunk += 1;
            }

            // Remainder N (scalar with bias)
            let n_main = (n_size / TILE_N) * TILE_N;
            for m in 0..m_size {
                for n in n_main..n_size {
                    let mut sum = bias[n];
                    for k in 0..k_size {
                        sum = <$elem as Element>::mul_add(sum, a[m * k_size + k], b[k * n_size + n]);
                    }
                    c[m * n_size + n] = sum;
                }
            }
            // Remainder M (scalar with bias) — only columns already handled by SIMD
            let m_main = (m_size / TILE_M) * TILE_M;
            for m in m_main..m_size {
                for n in 0..n_main {
                    let mut sum = bias[n];
                    for k in 0..k_size {
                        sum = <$elem as Element>::mul_add(sum, a[m * k_size + k], b[k * n_size + n]);
                    }
                    c[m * n_size + n] = sum;
                }
            }
            WS.with(|c| c.set(packed_b));
        }

        #[inline(always)]
        pub fn matmul_bias(a: &[$elem], b: &[$elem], bias: &[$elem], c: &mut [$elem], m_size: usize, n_size: usize, k_size: usize) {
            unsafe { matmul_bias_avx2(a, b, bias, c, m_size, n_size, k_size); }
        }

        /// Pre-pack B matrix for repeated matmul calls with the same weights.
        pub fn pack_b(b: &[$elem], n_size: usize, k_size: usize) -> Vec<$elem> {
            const LANES: usize = 8;
            const TILE_N: usize = 2 * LANES;
            const KC: usize = 256;

            assert!(b.len() >= k_size * n_size);
            let n_strips = (n_size + TILE_N - 1) / TILE_N;
            let n_chunks = (k_size + KC - 1) / KC;
            let chunk_size = n_strips * KC * TILE_N;
            let total = n_chunks * chunk_size;
            let mut packed = vec![<$elem as Element>::ZERO; total];

            let mut k_start = 0usize;
            let mut chunk = 0usize;
            while k_start < k_size {
                let kc = KC.min(k_size - k_start);
                let base = chunk * chunk_size;
                for (i, n_start) in (0..n_size).step_by(TILE_N).enumerate() {
                    let actual_n = TILE_N.min(n_size - n_start);
                    for k in 0..kc {
                        let dst = base + i * KC * TILE_N + k * TILE_N;
                        let src_row = (k_start + k) * n_size + n_start;
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                b.as_ptr().add(src_row),
                                packed.as_mut_ptr().add(dst),
                                actual_n,
                            );
                        }
                    }
                }
                k_start += KC;
                chunk += 1;
            }
            packed
        }

        /// Matmul using pre-packed B (AVX2). Loop order: chunk(KC) → m → n.
        #[target_feature(enable = "avx2", enable = "fma")]
        unsafe fn matmul_prepacked_avx2(a: &[$elem], packed_b: &[$elem], c: &mut [$elem], m_size: usize, n_size: usize, k_size: usize) {
            const TILE_M: usize = 6;
            const LANES: usize = 8;
            const TILE_N: usize = 2 * LANES;
            const KC: usize = 256;

            assert!(a.len() >= m_size * k_size);
            assert!(c.len() >= m_size * n_size);

            let n_strips = (n_size + TILE_N - 1) / TILE_N;
            let chunk_size = n_strips * KC * TILE_N;

            // Zero C matrix — accumulation across chunks
            for i in 0..c.len().min(m_size * n_size) { c[i] = <$elem as Element>::ZERO; }
            let c_ptr = c.as_mut_ptr();

            // Main loop: chunk → m → n (B stays hot in L1/L2)
            let mut k_start = 0usize;
            let mut chunk = 0usize;
            while k_start < k_size {
                let kc = KC.min(k_size - k_start);

                let mut m = 0;
                while m + TILE_M <= m_size {
                    let mut n = 0;
                    let mut strip_idx = 0;
                    while n + TILE_N <= n_size {
                        unsafe {
                            // Load C tile from memory (accumulate across chunks)
                            macro_rules! load_row {
                                ($row:expr) => {(
                                    $crate::simd_primitive!(avx2, $elem, loadu, c_ptr.add((m + $row) * n_size + n)),
                                    $crate::simd_primitive!(avx2, $elem, loadu, c_ptr.add((m + $row) * n_size + n + LANES)),
                                )};
                            }
                            let (mut c_0_0, mut c_0_1) = load_row!(0);
                            let (mut c_1_0, mut c_1_1) = load_row!(1);
                            let (mut c_2_0, mut c_2_1) = load_row!(2);
                            let (mut c_3_0, mut c_3_1) = load_row!(3);
                            let (mut c_4_0, mut c_4_1) = load_row!(4);
                            let (mut c_5_0, mut c_5_1) = load_row!(5);

                            macro_rules! fma_row {
                                ($a_base:expr, $vb0:ident, $vb1:ident, $row:expr, $c0:ident, $c1:ident) => {
                                    let va = $crate::simd_primitive!(avx2, $elem, splat, *$a_base.add($row));
                                    $c0 = $crate::simd_primitive!(avx2, $elem, fma, va, $vb0, $c0);
                                    $c1 = $crate::simd_primitive!(avx2, $elem, fma, va, $vb1, $c1);
                                };
                            }
                            macro_rules! fma_all {
                                ($a_base:expr, $vb0:ident, $vb1:ident) => {
                                    fma_row!($a_base, $vb0, $vb1, 0, c_0_0, c_0_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size, c_1_0, c_1_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size * 2, c_2_0, c_2_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size * 3, c_3_0, c_3_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size * 4, c_4_0, c_4_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size * 5, c_5_0, c_5_1);
                                };
                            }

                            let mut a_col = a.as_ptr().add(m * k_size + k_start);
                            let mut b_ptr = packed_b.as_ptr().add(chunk * chunk_size + strip_idx * KC * TILE_N);

                            let mut _k = 0usize;
                            let k_unroll_end = kc & !7;
                            while _k < k_unroll_end {
                                $crate::simd_primitive!(avx2, $elem, prefetch, b_ptr.add(TILE_N * 16) as *const i8, 0);

                                let vb0_0 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr);
                                let vb0_1 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr.add(LANES));
                                fma_all!(a_col, vb0_0, vb0_1); a_col = a_col.add(1);

                                let vb1_0 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr.add(TILE_N * 1));
                                let vb1_1 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr.add(TILE_N * 1 + LANES));
                                fma_all!(a_col, vb1_0, vb1_1); a_col = a_col.add(1);

                                let vb2_0 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr.add(TILE_N * 2));
                                let vb2_1 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr.add(TILE_N * 2 + LANES));
                                fma_all!(a_col, vb2_0, vb2_1); a_col = a_col.add(1);

                                let vb3_0 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr.add(TILE_N * 3));
                                let vb3_1 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr.add(TILE_N * 3 + LANES));
                                fma_all!(a_col, vb3_0, vb3_1); a_col = a_col.add(1);

                                let vb4_0 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr.add(TILE_N * 4));
                                let vb4_1 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr.add(TILE_N * 4 + LANES));
                                fma_all!(a_col, vb4_0, vb4_1); a_col = a_col.add(1);

                                let vb5_0 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr.add(TILE_N * 5));
                                let vb5_1 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr.add(TILE_N * 5 + LANES));
                                fma_all!(a_col, vb5_0, vb5_1); a_col = a_col.add(1);

                                let vb6_0 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr.add(TILE_N * 6));
                                let vb6_1 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr.add(TILE_N * 6 + LANES));
                                fma_all!(a_col, vb6_0, vb6_1); a_col = a_col.add(1);

                                let vb7_0 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr.add(TILE_N * 7));
                                let vb7_1 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr.add(TILE_N * 7 + LANES));
                                fma_all!(a_col, vb7_0, vb7_1); a_col = a_col.add(1);

                                b_ptr = b_ptr.add(TILE_N * 8);
                                _k += 8;
                            }
                            while _k < kc {
                                let vb_0 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr);
                                let vb_1 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr.add(LANES));
                                fma_all!(a_col, vb_0, vb_1); a_col = a_col.add(1);
                                b_ptr = b_ptr.add(TILE_N);
                                _k += 1;
                            }

                            macro_rules! store_row {
                                ($row:expr, $c0:expr, $c1:expr) => {
                                    $crate::simd_primitive!(avx2, $elem, storeu, c_ptr.add((m + $row) * n_size + n), $c0);
                                    $crate::simd_primitive!(avx2, $elem, storeu, c_ptr.add((m + $row) * n_size + n + LANES), $c1);
                                };
                            }
                            store_row!(0, c_0_0, c_0_1); store_row!(1, c_1_0, c_1_1);
                            store_row!(2, c_2_0, c_2_1); store_row!(3, c_3_0, c_3_1);
                            store_row!(4, c_4_0, c_4_1); store_row!(5, c_5_0, c_5_1);
                        }
                        n += TILE_N;
                        strip_idx += 1;
                    }
                    m += TILE_M;
                }

                k_start += KC;
                chunk += 1;
            }

            // Remainder N (scalar) — columns not covered by TILE_N
            let n_main = (n_size / TILE_N) * TILE_N;
            for m in 0..m_size {
                for n in n_main..n_size {
                    let mut sum = <$elem as Element>::ZERO;
                    for k in 0..k_size {
                        let chunk_idx = k / KC; let k_in_chunk = k % KC;
                        sum = <$elem as Element>::mul_add(sum, a[m * k_size + k], packed_b[chunk_idx * chunk_size + (n / TILE_N) * KC * TILE_N + k_in_chunk * TILE_N + (n % TILE_N)]);
                    }
                    c[m * n_size + n] = sum;
                }
            }
            // Remainder M (scalar) — rows not covered by TILE_M
            let m_main = (m_size / TILE_M) * TILE_M;
            for m in m_main..m_size {
                for n in 0..n_main {
                    let mut sum = <$elem as Element>::ZERO;
                    for k in 0..k_size {
                        let chunk_idx = k / KC; let k_in_chunk = k % KC;
                        sum = <$elem as Element>::mul_add(sum, a[m * k_size + k], packed_b[chunk_idx * chunk_size + (n / TILE_N) * KC * TILE_N + k_in_chunk * TILE_N + (n % TILE_N)]);
                    }
                    c[m * n_size + n] = sum;
                }
            }
        }

        #[inline(always)]
        pub fn matmul_prepacked(a: &[$elem], packed_b: &[$elem], c: &mut [$elem], m_size: usize, n_size: usize, k_size: usize) {
            unsafe { matmul_prepacked_avx2(a, packed_b, c, m_size, n_size, k_size); }
        }

        /// Matmul+bias using pre-packed B (AVX2). Loop order: chunk(KC) → m → n.
        #[target_feature(enable = "avx2", enable = "fma")]
        unsafe fn matmul_bias_prepacked_avx2(a: &[$elem], packed_b: &[$elem], bias: &[$elem], c: &mut [$elem], m_size: usize, n_size: usize, k_size: usize) {
            const TILE_M: usize = 6;
            const LANES: usize = 8;
            const TILE_N: usize = 2 * LANES;
            const KC: usize = 256;

            assert!(a.len() >= m_size * k_size);
            assert!(c.len() >= m_size * n_size);
            assert!(bias.len() >= n_size);

            let n_strips = (n_size + TILE_N - 1) / TILE_N;
            let chunk_size = n_strips * KC * TILE_N;

            // Init C with bias (each row gets the same bias vector)
            let c_ptr = c.as_mut_ptr();
            for m in 0..m_size {
                unsafe { std::ptr::copy_nonoverlapping(bias.as_ptr(), c_ptr.add(m * n_size), n_size); }
            }

            // Main loop: chunk → m → n (B stays hot in L1/L2)
            let mut k_start = 0usize;
            let mut chunk = 0usize;
            while k_start < k_size {
                let kc = KC.min(k_size - k_start);

                let mut m = 0;
                while m + TILE_M <= m_size {
                    let mut n = 0;
                    let mut strip_idx = 0;
                    while n + TILE_N <= n_size {
                        unsafe {
                            // Load C tile from memory (accumulate across chunks)
                            macro_rules! load_row {
                                ($row:expr) => {(
                                    $crate::simd_primitive!(avx2, $elem, loadu, c_ptr.add((m + $row) * n_size + n)),
                                    $crate::simd_primitive!(avx2, $elem, loadu, c_ptr.add((m + $row) * n_size + n + LANES)),
                                )};
                            }
                            let (mut c_0_0, mut c_0_1) = load_row!(0);
                            let (mut c_1_0, mut c_1_1) = load_row!(1);
                            let (mut c_2_0, mut c_2_1) = load_row!(2);
                            let (mut c_3_0, mut c_3_1) = load_row!(3);
                            let (mut c_4_0, mut c_4_1) = load_row!(4);
                            let (mut c_5_0, mut c_5_1) = load_row!(5);

                            macro_rules! fma_row {
                                ($a_base:expr, $vb0:ident, $vb1:ident, $row:expr, $c0:ident, $c1:ident) => {
                                    let va = $crate::simd_primitive!(avx2, $elem, splat, *$a_base.add($row));
                                    $c0 = $crate::simd_primitive!(avx2, $elem, fma, va, $vb0, $c0);
                                    $c1 = $crate::simd_primitive!(avx2, $elem, fma, va, $vb1, $c1);
                                };
                            }
                            macro_rules! fma_all {
                                ($a_base:expr, $vb0:ident, $vb1:ident) => {
                                    fma_row!($a_base, $vb0, $vb1, 0, c_0_0, c_0_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size, c_1_0, c_1_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size * 2, c_2_0, c_2_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size * 3, c_3_0, c_3_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size * 4, c_4_0, c_4_1);
                                    fma_row!($a_base, $vb0, $vb1, k_size * 5, c_5_0, c_5_1);
                                };
                            }

                            let mut a_col = a.as_ptr().add(m * k_size + k_start);
                            let mut b_ptr = packed_b.as_ptr().add(chunk * chunk_size + strip_idx * KC * TILE_N);

                            let mut _k = 0usize;
                            let k_unroll_end = kc & !7;
                            while _k < k_unroll_end {
                                $crate::simd_primitive!(avx2, $elem, prefetch, b_ptr.add(TILE_N * 16) as *const i8, 0);

                                let vb0_0 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr);
                                let vb0_1 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr.add(LANES));
                                fma_all!(a_col, vb0_0, vb0_1); a_col = a_col.add(1);

                                let vb1_0 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr.add(TILE_N * 1));
                                let vb1_1 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr.add(TILE_N * 1 + LANES));
                                fma_all!(a_col, vb1_0, vb1_1); a_col = a_col.add(1);

                                let vb2_0 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr.add(TILE_N * 2));
                                let vb2_1 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr.add(TILE_N * 2 + LANES));
                                fma_all!(a_col, vb2_0, vb2_1); a_col = a_col.add(1);

                                let vb3_0 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr.add(TILE_N * 3));
                                let vb3_1 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr.add(TILE_N * 3 + LANES));
                                fma_all!(a_col, vb3_0, vb3_1); a_col = a_col.add(1);

                                let vb4_0 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr.add(TILE_N * 4));
                                let vb4_1 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr.add(TILE_N * 4 + LANES));
                                fma_all!(a_col, vb4_0, vb4_1); a_col = a_col.add(1);

                                let vb5_0 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr.add(TILE_N * 5));
                                let vb5_1 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr.add(TILE_N * 5 + LANES));
                                fma_all!(a_col, vb5_0, vb5_1); a_col = a_col.add(1);

                                let vb6_0 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr.add(TILE_N * 6));
                                let vb6_1 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr.add(TILE_N * 6 + LANES));
                                fma_all!(a_col, vb6_0, vb6_1); a_col = a_col.add(1);

                                let vb7_0 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr.add(TILE_N * 7));
                                let vb7_1 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr.add(TILE_N * 7 + LANES));
                                fma_all!(a_col, vb7_0, vb7_1); a_col = a_col.add(1);

                                b_ptr = b_ptr.add(TILE_N * 8);
                                _k += 8;
                            }
                            while _k < kc {
                                let vb_0 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr);
                                let vb_1 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr.add(LANES));
                                fma_all!(a_col, vb_0, vb_1); a_col = a_col.add(1);
                                b_ptr = b_ptr.add(TILE_N);
                                _k += 1;
                            }

                            macro_rules! store_row {
                                ($row:expr, $c0:expr, $c1:expr) => {
                                    $crate::simd_primitive!(avx2, $elem, storeu, c_ptr.add((m + $row) * n_size + n), $c0);
                                    $crate::simd_primitive!(avx2, $elem, storeu, c_ptr.add((m + $row) * n_size + n + LANES), $c1);
                                };
                            }
                            store_row!(0, c_0_0, c_0_1); store_row!(1, c_1_0, c_1_1);
                            store_row!(2, c_2_0, c_2_1); store_row!(3, c_3_0, c_3_1);
                            store_row!(4, c_4_0, c_4_1); store_row!(5, c_5_0, c_5_1);
                        }
                        n += TILE_N;
                        strip_idx += 1;
                    }
                    m += TILE_M;
                }

                k_start += KC;
                chunk += 1;
            }

            // Remainder N (scalar) — columns not covered by TILE_N
            let n_main = (n_size / TILE_N) * TILE_N;
            for m in 0..m_size {
                for n in n_main..n_size {
                    let mut sum = bias[n];
                    for k in 0..k_size {
                        let chunk_idx = k / KC; let k_in_chunk = k % KC;
                        sum = <$elem as Element>::mul_add(sum, a[m * k_size + k], packed_b[chunk_idx * chunk_size + (n / TILE_N) * KC * TILE_N + k_in_chunk * TILE_N + (n % TILE_N)]);
                    }
                    c[m * n_size + n] = sum;
                }
            }
            // Remainder M (scalar) — rows not covered by TILE_M
            let m_main = (m_size / TILE_M) * TILE_M;
            for m in m_main..m_size {
                for n in 0..n_main {
                    let mut sum = bias[n];
                    for k in 0..k_size {
                        let chunk_idx = k / KC; let k_in_chunk = k % KC;
                        sum = <$elem as Element>::mul_add(sum, a[m * k_size + k], packed_b[chunk_idx * chunk_size + (n / TILE_N) * KC * TILE_N + k_in_chunk * TILE_N + (n % TILE_N)]);
                    }
                    c[m * n_size + n] = sum;
                }
            }
        }

        #[inline(always)]
        pub fn matmul_bias_prepacked(a: &[$elem], packed_b: &[$elem], bias: &[$elem], c: &mut [$elem], m_size: usize, n_size: usize, k_size: usize) {
            unsafe { matmul_bias_prepacked_avx2(a, packed_b, bias, c, m_size, n_size, k_size); }
        }
    };

    (neon, $elem:ident) => {
        // NEON Optimized Implementation: 8×12 microkernel
        // 24 float32x4_t accumulators (8 rows × 3 vectors of 4 lanes)
        // Register budget: 24 acc + 3 B-vec + 1 A-broadcast + 4 temp = 32 (full NEON file)
        #[inline(always)]
        pub fn matmul(a: &[$elem], b: &[$elem], c: &mut [$elem], m_size: usize, n_size: usize, k_size: usize) {
            const TILE_M: usize = 8;
            const LANES: usize = $crate::simd_primitive!(neon, $elem, lanes);
            const TILE_N_VECS: usize = 3;
            const TILE_N: usize = TILE_N_VECS * LANES; // 12 for f32

            assert!(a.len() >= m_size * k_size);
            assert!(b.len() >= n_size * k_size);
            assert!(c.len() >= m_size * n_size);

            // Pack B into column strips of width TILE_N (thread-local reuse)
            let n_strips = (n_size + TILE_N - 1) / TILE_N;
            let packed_b_size = n_strips * k_size * TILE_N;
            thread_local! { static WS: std::cell::Cell<Vec<$elem>> = std::cell::Cell::new(Vec::new()); }
            let mut packed_b = WS.with(|c| c.take());
            packed_b.clear();
            packed_b.resize(packed_b_size, <$elem as Element>::ZERO);

            for i in 0..n_strips {
                let n_start = i * TILE_N;
                if n_start + TILE_N <= n_size {
                    for k in 0..k_size {
                        let src_off = k * n_size + n_start;
                        let dst_off = i * k_size * TILE_N + k * TILE_N;
                        packed_b[dst_off..dst_off + TILE_N].copy_from_slice(&b[src_off..src_off + TILE_N]);
                    }
                } else {
                    let actual_n = n_size - n_start;
                    for k in 0..k_size {
                        let src_off = k * n_size + n_start;
                        let dst_off = i * k_size * TILE_N + k * TILE_N;
                        packed_b[dst_off..dst_off + actual_n].copy_from_slice(&b[src_off..src_off + actual_n]);
                    }
                }
            }

            // Main tiled loop
            let mut m = 0;
            while m + TILE_M <= m_size {
                let mut n = 0;
                let mut strip_idx = 0;

                while n + TILE_N <= n_size {
                    #[allow(unused_unsafe)]
                    unsafe {
                        // 24 accumulators: 8 rows × 3 vectors
                        let mut c_0_0 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_0_1 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_0_2 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_1_0 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_1_1 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_1_2 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_2_0 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_2_1 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_2_2 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_3_0 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_3_1 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_3_2 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_4_0 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_4_1 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_4_2 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_5_0 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_5_1 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_5_2 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_6_0 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_6_1 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_6_2 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_7_0 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_7_1 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_7_2 = $crate::simd_primitive!(neon, $elem, zero);

                        let mut b_ptr = packed_b.as_ptr().add(strip_idx * k_size * TILE_N);

                        // A row pointers
                        let mut a_ptr_0 = a.as_ptr().add((m + 0) * k_size);
                        let mut a_ptr_1 = a.as_ptr().add((m + 1) * k_size);
                        let mut a_ptr_2 = a.as_ptr().add((m + 2) * k_size);
                        let mut a_ptr_3 = a.as_ptr().add((m + 3) * k_size);
                        let mut a_ptr_4 = a.as_ptr().add((m + 4) * k_size);
                        let mut a_ptr_5 = a.as_ptr().add((m + 5) * k_size);
                        let mut a_ptr_6 = a.as_ptr().add((m + 6) * k_size);
                        let mut a_ptr_7 = a.as_ptr().add((m + 7) * k_size);

                        // 8 rows × 3 FMA each
                        macro_rules! fma_row {
                            ($vb0:ident, $vb1:ident, $vb2:ident, $a_ptr:ident, $c0:ident, $c1:ident, $c2:ident) => {
                                let va = $crate::simd_primitive!(neon, $elem, splat, *$a_ptr);
                                $c0 = $crate::simd_primitive!(neon, $elem, fma, va, $vb0, $c0);
                                $c1 = $crate::simd_primitive!(neon, $elem, fma, va, $vb1, $c1);
                                $c2 = $crate::simd_primitive!(neon, $elem, fma, va, $vb2, $c2);
                            };
                        }
                        macro_rules! fma_all_neon {
                            ($vb0:ident, $vb1:ident, $vb2:ident) => {
                                fma_row!($vb0, $vb1, $vb2, a_ptr_0, c_0_0, c_0_1, c_0_2);
                                fma_row!($vb0, $vb1, $vb2, a_ptr_1, c_1_0, c_1_1, c_1_2);
                                fma_row!($vb0, $vb1, $vb2, a_ptr_2, c_2_0, c_2_1, c_2_2);
                                fma_row!($vb0, $vb1, $vb2, a_ptr_3, c_3_0, c_3_1, c_3_2);
                                fma_row!($vb0, $vb1, $vb2, a_ptr_4, c_4_0, c_4_1, c_4_2);
                                fma_row!($vb0, $vb1, $vb2, a_ptr_5, c_5_0, c_5_1, c_5_2);
                                fma_row!($vb0, $vb1, $vb2, a_ptr_6, c_6_0, c_6_1, c_6_2);
                                fma_row!($vb0, $vb1, $vb2, a_ptr_7, c_7_0, c_7_1, c_7_2);
                            };
                        }
                        macro_rules! advance_a_neon {
                            ($n:expr) => {
                                a_ptr_0 = a_ptr_0.add($n); a_ptr_1 = a_ptr_1.add($n);
                                a_ptr_2 = a_ptr_2.add($n); a_ptr_3 = a_ptr_3.add($n);
                                a_ptr_4 = a_ptr_4.add($n); a_ptr_5 = a_ptr_5.add($n);
                                a_ptr_6 = a_ptr_6.add($n); a_ptr_7 = a_ptr_7.add($n);
                            };
                        }

                        // K-loop unrolled 4×
                        let mut _k = 0usize;
                        let k_unroll_end = k_size & !3;
                        while _k < k_unroll_end {
                            let vb0_0 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr);
                            let vb0_1 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(LANES));
                            let vb0_2 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(LANES * 2));
                            fma_all_neon!(vb0_0, vb0_1, vb0_2);
                            advance_a_neon!(1);

                            let vb1_0 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(TILE_N));
                            let vb1_1 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(TILE_N + LANES));
                            let vb1_2 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(TILE_N + LANES * 2));
                            fma_all_neon!(vb1_0, vb1_1, vb1_2);
                            advance_a_neon!(1);

                            let vb2_0 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(TILE_N * 2));
                            let vb2_1 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(TILE_N * 2 + LANES));
                            let vb2_2 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(TILE_N * 2 + LANES * 2));
                            fma_all_neon!(vb2_0, vb2_1, vb2_2);
                            advance_a_neon!(1);

                            let vb3_0 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(TILE_N * 3));
                            let vb3_1 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(TILE_N * 3 + LANES));
                            let vb3_2 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(TILE_N * 3 + LANES * 2));
                            fma_all_neon!(vb3_0, vb3_1, vb3_2);
                            advance_a_neon!(1);

                            b_ptr = b_ptr.add(TILE_N * 4);
                            _k += 4;
                        }
                        // K remainder
                        while _k < k_size {
                            let vb_0 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr);
                            let vb_1 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(LANES));
                            let vb_2 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(LANES * 2));
                            fma_all_neon!(vb_0, vb_1, vb_2);
                            advance_a_neon!(1);
                            b_ptr = b_ptr.add(TILE_N);
                            _k += 1;
                        }
                        macro_rules! store_row {
                            ($row:expr, $c0:expr, $c1:expr, $c2:expr) => {
                                $crate::simd_primitive!(neon, $elem, storeu, c.as_mut_ptr().add((m + $row) * n_size + n), $c0);
                                $crate::simd_primitive!(neon, $elem, storeu, c.as_mut_ptr().add((m + $row) * n_size + n + LANES), $c1);
                                $crate::simd_primitive!(neon, $elem, storeu, c.as_mut_ptr().add((m + $row) * n_size + n + LANES * 2), $c2);
                            };
                        }
                        store_row!(0, c_0_0, c_0_1, c_0_2);
                        store_row!(1, c_1_0, c_1_1, c_1_2);
                        store_row!(2, c_2_0, c_2_1, c_2_2);
                        store_row!(3, c_3_0, c_3_1, c_3_2);
                        store_row!(4, c_4_0, c_4_1, c_4_2);
                        store_row!(5, c_5_0, c_5_1, c_5_2);
                        store_row!(6, c_6_0, c_6_1, c_6_2);
                        store_row!(7, c_7_0, c_7_1, c_7_2);
                    }
                    n += TILE_N;
                    strip_idx += 1;
                }

                // Remainder N (scalar)
                while n < n_size {
                    for i in 0..TILE_M {
                        let mut sum: $elem = <$elem as Element>::ZERO;
                        for k in 0..k_size {
                            sum = <$elem as Element>::mul_add(sum, a[(m + i) * k_size + k], b[k * n_size + n]);
                        }
                        c[(m + i) * n_size + n] = sum;
                    }
                    n += 1;
                }
                m += TILE_M;
            }

            // Remainder M (scalar)
            while m < m_size {
                for n in 0..n_size {
                    let mut sum: $elem = <$elem as Element>::ZERO;
                    for k in 0..k_size {
                        sum = <$elem as Element>::mul_add(sum, a[m * k_size + k], b[k * n_size + n]);
                    }
                    c[m * n_size + n] = sum;
                }
                m += 1;
            }
            WS.with(|c| c.set(packed_b));
        }

        /// matmul_bias: C = A * B + bias (fused — bias added during store, no second pass)
        #[inline(always)]
        pub fn matmul_bias(a: &[$elem], b: &[$elem], bias: &[$elem], c: &mut [$elem], m_size: usize, n_size: usize, k_size: usize) {
            const TILE_M: usize = 8;
            const LANES: usize = $crate::simd_primitive!(neon, $elem, lanes);
            const TILE_N_VECS: usize = 3;
            const TILE_N: usize = TILE_N_VECS * LANES;

            assert!(a.len() >= m_size * k_size);
            assert!(b.len() >= n_size * k_size);
            assert!(c.len() >= m_size * n_size);
            assert!(bias.len() >= n_size);

            // Pack B (same as matmul, thread-local reuse)
            let n_strips = (n_size + TILE_N - 1) / TILE_N;
            let packed_b_size = n_strips * k_size * TILE_N;
            thread_local! { static WS: std::cell::Cell<Vec<$elem>> = std::cell::Cell::new(Vec::new()); }
            let mut packed_b = WS.with(|c| c.take());
            packed_b.clear();
            packed_b.resize(packed_b_size, <$elem as Element>::ZERO);
            for i in 0..n_strips {
                let n_start = i * TILE_N;
                if n_start + TILE_N <= n_size {
                    for k in 0..k_size {
                        let src_off = k * n_size + n_start;
                        let dst_off = i * k_size * TILE_N + k * TILE_N;
                        packed_b[dst_off..dst_off + TILE_N].copy_from_slice(&b[src_off..src_off + TILE_N]);
                    }
                } else {
                    let actual_n = n_size - n_start;
                    for k in 0..k_size {
                        let src_off = k * n_size + n_start;
                        let dst_off = i * k_size * TILE_N + k * TILE_N;
                        packed_b[dst_off..dst_off + actual_n].copy_from_slice(&b[src_off..src_off + actual_n]);
                    }
                }
            }

            let mut m = 0;
            while m + TILE_M <= m_size {
                let mut n = 0;
                let mut strip_idx = 0;
                while n + TILE_N <= n_size {
                    #[allow(unused_unsafe)]
                    unsafe {
                        let mut c_0_0 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_0_1 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_0_2 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_1_0 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_1_1 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_1_2 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_2_0 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_2_1 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_2_2 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_3_0 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_3_1 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_3_2 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_4_0 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_4_1 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_4_2 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_5_0 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_5_1 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_5_2 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_6_0 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_6_1 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_6_2 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_7_0 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_7_1 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_7_2 = $crate::simd_primitive!(neon, $elem, zero);

                        let mut b_ptr = packed_b.as_ptr().add(strip_idx * k_size * TILE_N);
                        let mut a_ptr_0 = a.as_ptr().add((m + 0) * k_size);
                        let mut a_ptr_1 = a.as_ptr().add((m + 1) * k_size);
                        let mut a_ptr_2 = a.as_ptr().add((m + 2) * k_size);
                        let mut a_ptr_3 = a.as_ptr().add((m + 3) * k_size);
                        let mut a_ptr_4 = a.as_ptr().add((m + 4) * k_size);
                        let mut a_ptr_5 = a.as_ptr().add((m + 5) * k_size);
                        let mut a_ptr_6 = a.as_ptr().add((m + 6) * k_size);
                        let mut a_ptr_7 = a.as_ptr().add((m + 7) * k_size);

                        macro_rules! fma_row_nb {
                            ($vb0:ident, $vb1:ident, $vb2:ident, $a_ptr:ident, $c0:ident, $c1:ident, $c2:ident) => {
                                let va = $crate::simd_primitive!(neon, $elem, splat, *$a_ptr);
                                $c0 = $crate::simd_primitive!(neon, $elem, fma, va, $vb0, $c0);
                                $c1 = $crate::simd_primitive!(neon, $elem, fma, va, $vb1, $c1);
                                $c2 = $crate::simd_primitive!(neon, $elem, fma, va, $vb2, $c2);
                            };
                        }
                        macro_rules! fma_all_nb {
                            ($vb0:ident, $vb1:ident, $vb2:ident) => {
                                fma_row_nb!($vb0, $vb1, $vb2, a_ptr_0, c_0_0, c_0_1, c_0_2);
                                fma_row_nb!($vb0, $vb1, $vb2, a_ptr_1, c_1_0, c_1_1, c_1_2);
                                fma_row_nb!($vb0, $vb1, $vb2, a_ptr_2, c_2_0, c_2_1, c_2_2);
                                fma_row_nb!($vb0, $vb1, $vb2, a_ptr_3, c_3_0, c_3_1, c_3_2);
                                fma_row_nb!($vb0, $vb1, $vb2, a_ptr_4, c_4_0, c_4_1, c_4_2);
                                fma_row_nb!($vb0, $vb1, $vb2, a_ptr_5, c_5_0, c_5_1, c_5_2);
                                fma_row_nb!($vb0, $vb1, $vb2, a_ptr_6, c_6_0, c_6_1, c_6_2);
                                fma_row_nb!($vb0, $vb1, $vb2, a_ptr_7, c_7_0, c_7_1, c_7_2);
                            };
                        }
                        macro_rules! advance_a_nb {
                            ($n:expr) => {
                                a_ptr_0 = a_ptr_0.add($n); a_ptr_1 = a_ptr_1.add($n);
                                a_ptr_2 = a_ptr_2.add($n); a_ptr_3 = a_ptr_3.add($n);
                                a_ptr_4 = a_ptr_4.add($n); a_ptr_5 = a_ptr_5.add($n);
                                a_ptr_6 = a_ptr_6.add($n); a_ptr_7 = a_ptr_7.add($n);
                            };
                        }

                        let mut _k = 0usize;
                        let k_unroll_end = k_size & !3;
                        while _k < k_unroll_end {
                            let vb0_0 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr);
                            let vb0_1 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(LANES));
                            let vb0_2 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(LANES * 2));
                            fma_all_nb!(vb0_0, vb0_1, vb0_2);
                            advance_a_nb!(1);

                            let vb1_0 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(TILE_N));
                            let vb1_1 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(TILE_N + LANES));
                            let vb1_2 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(TILE_N + LANES * 2));
                            fma_all_nb!(vb1_0, vb1_1, vb1_2);
                            advance_a_nb!(1);

                            let vb2_0 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(TILE_N * 2));
                            let vb2_1 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(TILE_N * 2 + LANES));
                            let vb2_2 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(TILE_N * 2 + LANES * 2));
                            fma_all_nb!(vb2_0, vb2_1, vb2_2);
                            advance_a_nb!(1);

                            let vb3_0 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(TILE_N * 3));
                            let vb3_1 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(TILE_N * 3 + LANES));
                            let vb3_2 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(TILE_N * 3 + LANES * 2));
                            fma_all_nb!(vb3_0, vb3_1, vb3_2);
                            advance_a_nb!(1);

                            b_ptr = b_ptr.add(TILE_N * 4);
                            _k += 4;
                        }
                        while _k < k_size {
                            let vb_0 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr);
                            let vb_1 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(LANES));
                            let vb_2 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(LANES * 2));
                            fma_all_nb!(vb_0, vb_1, vb_2);
                            advance_a_nb!(1);
                            b_ptr = b_ptr.add(TILE_N);
                            _k += 1;
                        }
                        let vbias_0 = $crate::simd_primitive!(neon, $elem, loadu, bias.as_ptr().add(n));
                        let vbias_1 = $crate::simd_primitive!(neon, $elem, loadu, bias.as_ptr().add(n + LANES));
                        let vbias_2 = $crate::simd_primitive!(neon, $elem, loadu, bias.as_ptr().add(n + LANES * 2));

                        // Store with fused bias add
                        macro_rules! store_row_bias {
                            ($row:expr, $c0:expr, $c1:expr, $c2:expr) => {
                                let r0 = $crate::simd_primitive!(neon, $elem, add, $c0, vbias_0);
                                let r1 = $crate::simd_primitive!(neon, $elem, add, $c1, vbias_1);
                                let r2 = $crate::simd_primitive!(neon, $elem, add, $c2, vbias_2);
                                $crate::simd_primitive!(neon, $elem, storeu, c.as_mut_ptr().add((m + $row) * n_size + n), r0);
                                $crate::simd_primitive!(neon, $elem, storeu, c.as_mut_ptr().add((m + $row) * n_size + n + LANES), r1);
                                $crate::simd_primitive!(neon, $elem, storeu, c.as_mut_ptr().add((m + $row) * n_size + n + LANES * 2), r2);
                            };
                        }
                        store_row_bias!(0, c_0_0, c_0_1, c_0_2);
                        store_row_bias!(1, c_1_0, c_1_1, c_1_2);
                        store_row_bias!(2, c_2_0, c_2_1, c_2_2);
                        store_row_bias!(3, c_3_0, c_3_1, c_3_2);
                        store_row_bias!(4, c_4_0, c_4_1, c_4_2);
                        store_row_bias!(5, c_5_0, c_5_1, c_5_2);
                        store_row_bias!(6, c_6_0, c_6_1, c_6_2);
                        store_row_bias!(7, c_7_0, c_7_1, c_7_2);
                    }
                    n += TILE_N;
                    strip_idx += 1;
                }
                // Remainder N (scalar with bias)
                while n < n_size {
                    for i in 0..TILE_M {
                        let mut sum: $elem = <$elem as Element>::ZERO;
                        for k in 0..k_size {
                            sum = <$elem as Element>::mul_add(sum, a[(m + i) * k_size + k], b[k * n_size + n]);
                        }
                        c[(m + i) * n_size + n] = sum + bias[n];
                    }
                    n += 1;
                }
                m += TILE_M;
            }
            // Remainder M (scalar with bias)
            while m < m_size {
                for n in 0..n_size {
                    let mut sum: $elem = <$elem as Element>::ZERO;
                    for k in 0..k_size {
                        sum = <$elem as Element>::mul_add(sum, a[m * k_size + k], b[k * n_size + n]);
                    }
                    c[m * n_size + n] = sum + bias[n];
                }
                m += 1;
            }
            WS.with(|c| c.set(packed_b));
        }

        /// Pre-pack B matrix for repeated matmul calls (NEON).
        pub fn pack_b(b: &[$elem], n_size: usize, k_size: usize) -> Vec<$elem> {
            const LANES: usize = $crate::simd_primitive!(neon, $elem, lanes);
            const TILE_N: usize = 3 * LANES;
            const KC: usize = 256;

            assert!(b.len() >= k_size * n_size);
            let n_strips = (n_size + TILE_N - 1) / TILE_N;
            let n_chunks = (k_size + KC - 1) / KC;
            let chunk_size = n_strips * KC * TILE_N;
            let total = n_chunks * chunk_size;
            let mut packed = vec![<$elem as Element>::ZERO; total];

            let mut k_start = 0usize;
            let mut chunk = 0usize;
            while k_start < k_size {
                let kc = KC.min(k_size - k_start);
                let base = chunk * chunk_size;
                for (i, n_start) in (0..n_size).step_by(TILE_N).enumerate() {
                    let actual_n = TILE_N.min(n_size - n_start);
                    for k in 0..kc {
                        let dst = base + i * KC * TILE_N + k * TILE_N;
                        let src_row = (k_start + k) * n_size + n_start;
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                b.as_ptr().add(src_row),
                                packed.as_mut_ptr().add(dst),
                                actual_n,
                            );
                        }
                    }
                }
                k_start += KC;
                chunk += 1;
            }
            packed
        }

        /// Matmul using pre-packed B (NEON SIMD 8×12 microkernel). Loop order: chunk(KC) → m → n.
        #[inline(always)]
        pub fn matmul_prepacked(a: &[$elem], packed_b: &[$elem], c: &mut [$elem], m_size: usize, n_size: usize, k_size: usize) {
            const TILE_M: usize = 8;
            const LANES: usize = $crate::simd_primitive!(neon, $elem, lanes);
            const TILE_N_VECS: usize = 3;
            const TILE_N: usize = TILE_N_VECS * LANES;
            const KC: usize = 256;

            assert!(a.len() >= m_size * k_size);
            assert!(c.len() >= m_size * n_size);

            let n_strips = (n_size + TILE_N - 1) / TILE_N;
            let chunk_size = n_strips * KC * TILE_N;

            let mut m = 0;
            while m + TILE_M <= m_size {
                let mut n = 0;
                let mut strip_idx = 0;

                while n + TILE_N <= n_size {
                    #[allow(unused_unsafe)]
                    unsafe {
                        let mut c_0_0 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_0_1 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_0_2 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_1_0 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_1_1 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_1_2 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_2_0 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_2_1 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_2_2 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_3_0 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_3_1 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_3_2 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_4_0 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_4_1 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_4_2 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_5_0 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_5_1 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_5_2 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_6_0 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_6_1 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_6_2 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_7_0 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_7_1 = $crate::simd_primitive!(neon, $elem, zero);
                        let mut c_7_2 = $crate::simd_primitive!(neon, $elem, zero);

                        macro_rules! fma_row {
                            ($vb0:ident, $vb1:ident, $vb2:ident, $a_ptr:ident, $c0:ident, $c1:ident, $c2:ident) => {
                                let va = $crate::simd_primitive!(neon, $elem, splat, *$a_ptr);
                                $c0 = $crate::simd_primitive!(neon, $elem, fma, va, $vb0, $c0);
                                $c1 = $crate::simd_primitive!(neon, $elem, fma, va, $vb1, $c1);
                                $c2 = $crate::simd_primitive!(neon, $elem, fma, va, $vb2, $c2);
                            };
                        }
                        macro_rules! fma_all_neon {
                            ($vb0:ident, $vb1:ident, $vb2:ident) => {
                                fma_row!($vb0, $vb1, $vb2, a_ptr_0, c_0_0, c_0_1, c_0_2);
                                fma_row!($vb0, $vb1, $vb2, a_ptr_1, c_1_0, c_1_1, c_1_2);
                                fma_row!($vb0, $vb1, $vb2, a_ptr_2, c_2_0, c_2_1, c_2_2);
                                fma_row!($vb0, $vb1, $vb2, a_ptr_3, c_3_0, c_3_1, c_3_2);
                                fma_row!($vb0, $vb1, $vb2, a_ptr_4, c_4_0, c_4_1, c_4_2);
                                fma_row!($vb0, $vb1, $vb2, a_ptr_5, c_5_0, c_5_1, c_5_2);
                                fma_row!($vb0, $vb1, $vb2, a_ptr_6, c_6_0, c_6_1, c_6_2);
                                fma_row!($vb0, $vb1, $vb2, a_ptr_7, c_7_0, c_7_1, c_7_2);
                            };
                        }
                        macro_rules! advance_a_neon {
                            ($n:expr) => {
                                a_ptr_0 = a_ptr_0.add($n); a_ptr_1 = a_ptr_1.add($n);
                                a_ptr_2 = a_ptr_2.add($n); a_ptr_3 = a_ptr_3.add($n);
                                a_ptr_4 = a_ptr_4.add($n); a_ptr_5 = a_ptr_5.add($n);
                                a_ptr_6 = a_ptr_6.add($n); a_ptr_7 = a_ptr_7.add($n);
                            };
                        }

                        let mut k_start = 0usize;
                        let mut chunk = 0usize;
                        while k_start < k_size {
                            let kc = KC.min(k_size - k_start);
                            let mut b_ptr = packed_b.as_ptr().add(chunk * chunk_size + strip_idx * KC * TILE_N);

                            let mut a_ptr_0 = a.as_ptr().add((m + 0) * k_size + k_start);
                            let mut a_ptr_1 = a.as_ptr().add((m + 1) * k_size + k_start);
                            let mut a_ptr_2 = a.as_ptr().add((m + 2) * k_size + k_start);
                            let mut a_ptr_3 = a.as_ptr().add((m + 3) * k_size + k_start);
                            let mut a_ptr_4 = a.as_ptr().add((m + 4) * k_size + k_start);
                            let mut a_ptr_5 = a.as_ptr().add((m + 5) * k_size + k_start);
                            let mut a_ptr_6 = a.as_ptr().add((m + 6) * k_size + k_start);
                            let mut a_ptr_7 = a.as_ptr().add((m + 7) * k_size + k_start);

                            let mut _k = 0usize;
                            let k_unroll_end = kc & !7;
                            while _k < k_unroll_end {
                                let vb0_0 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr);
                                let vb0_1 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(LANES));
                                let vb0_2 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(LANES * 2));
                                fma_all_neon!(vb0_0, vb0_1, vb0_2);
                                advance_a_neon!(1);
                                let vb1_0 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(TILE_N));
                                let vb1_1 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(TILE_N + LANES));
                                let vb1_2 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(TILE_N + LANES * 2));
                                fma_all_neon!(vb1_0, vb1_1, vb1_2);
                                advance_a_neon!(1);
                                let vb2_0 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(TILE_N * 2));
                                let vb2_1 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(TILE_N * 2 + LANES));
                                let vb2_2 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(TILE_N * 2 + LANES * 2));
                                fma_all_neon!(vb2_0, vb2_1, vb2_2);
                                advance_a_neon!(1);
                                let vb3_0 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(TILE_N * 3));
                                let vb3_1 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(TILE_N * 3 + LANES));
                                let vb3_2 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(TILE_N * 3 + LANES * 2));
                                fma_all_neon!(vb3_0, vb3_1, vb3_2);
                                advance_a_neon!(1);
                                let vb4_0 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(TILE_N * 4));
                                let vb4_1 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(TILE_N * 4 + LANES));
                                let vb4_2 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(TILE_N * 4 + LANES * 2));
                                fma_all_neon!(vb4_0, vb4_1, vb4_2);
                                advance_a_neon!(1);
                                let vb5_0 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(TILE_N * 5));
                                let vb5_1 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(TILE_N * 5 + LANES));
                                let vb5_2 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(TILE_N * 5 + LANES * 2));
                                fma_all_neon!(vb5_0, vb5_1, vb5_2);
                                advance_a_neon!(1);
                                let vb6_0 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(TILE_N * 6));
                                let vb6_1 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(TILE_N * 6 + LANES));
                                let vb6_2 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(TILE_N * 6 + LANES * 2));
                                fma_all_neon!(vb6_0, vb6_1, vb6_2);
                                advance_a_neon!(1);
                                let vb7_0 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(TILE_N * 7));
                                let vb7_1 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(TILE_N * 7 + LANES));
                                let vb7_2 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(TILE_N * 7 + LANES * 2));
                                fma_all_neon!(vb7_0, vb7_1, vb7_2);
                                advance_a_neon!(1);
                                b_ptr = b_ptr.add(TILE_N * 8);
                                _k += 8;
                            }
                            while _k < kc {
                                let vb_0 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr);
                                let vb_1 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(LANES));
                                let vb_2 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(LANES * 2));
                                fma_all_neon!(vb_0, vb_1, vb_2);
                                advance_a_neon!(1);
                                b_ptr = b_ptr.add(TILE_N);
                                _k += 1;
                            }
                            k_start += KC;
                            chunk += 1;
                        }

                        macro_rules! store_row {
                            ($row:expr, $c0:expr, $c1:expr, $c2:expr) => {
                                $crate::simd_primitive!(neon, $elem, storeu, c.as_mut_ptr().add((m + $row) * n_size + n), $c0);
                                $crate::simd_primitive!(neon, $elem, storeu, c.as_mut_ptr().add((m + $row) * n_size + n + LANES), $c1);
                                $crate::simd_primitive!(neon, $elem, storeu, c.as_mut_ptr().add((m + $row) * n_size + n + LANES * 2), $c2);
                            };
                        }
                        store_row!(0, c_0_0, c_0_1, c_0_2);
                        store_row!(1, c_1_0, c_1_1, c_1_2);
                        store_row!(2, c_2_0, c_2_1, c_2_2);
                        store_row!(3, c_3_0, c_3_1, c_3_2);
                        store_row!(4, c_4_0, c_4_1, c_4_2);
                        store_row!(5, c_5_0, c_5_1, c_5_2);
                        store_row!(6, c_6_0, c_6_1, c_6_2);
                        store_row!(7, c_7_0, c_7_1, c_7_2);
                    }
                    n += TILE_N;
                    strip_idx += 1;
                }

                // Remainder N (scalar)
                while n < n_size {
                    for i in 0..TILE_M {
                        let mut sum = <$elem as Element>::ZERO;
                        for k in 0..k_size {
                            let ci = k / KC; let ki = k % KC;
                            sum = <$elem as Element>::mul_add(sum, a[(m + i) * k_size + k], packed_b[ci * chunk_size + (n / TILE_N) * KC * TILE_N + ki * TILE_N + (n % TILE_N)]);
                        }
                        c[(m + i) * n_size + n] = sum;
                    }
                    n += 1;
                }
                m += TILE_M;
            }

            // Remainder M (scalar)
            while m < m_size {
                for n in 0..n_size {
                    let mut sum = <$elem as Element>::ZERO;
                    for k in 0..k_size {
                        let ci = k / KC; let ki = k % KC;
                        sum = <$elem as Element>::mul_add(sum, a[m * k_size + k], packed_b[ci * chunk_size + (n / TILE_N) * KC * TILE_N + ki * TILE_N + (n % TILE_N)]);
                    }
                    c[m * n_size + n] = sum;
                }
                m += 1;
            }
        }

        /// Matmul+bias using pre-packed B (NEON SIMD 8×12 microkernel). Loop order: chunk(KC) → m → n.
        #[inline(always)]
        pub fn matmul_bias_prepacked(a: &[$elem], packed_b: &[$elem], bias: &[$elem], c: &mut [$elem], m_size: usize, n_size: usize, k_size: usize) {
            const TILE_M: usize = 8;
            const LANES: usize = $crate::simd_primitive!(neon, $elem, lanes);
            const TILE_N_VECS: usize = 3;
            const TILE_N: usize = TILE_N_VECS * LANES;
            const KC: usize = 256;

            assert!(a.len() >= m_size * k_size);
            assert!(c.len() >= m_size * n_size);
            assert!(bias.len() >= n_size);

            let n_strips = (n_size + TILE_N - 1) / TILE_N;
            let chunk_size = n_strips * KC * TILE_N;

            let mut m = 0;
            while m + TILE_M <= m_size {
                let mut n = 0;
                let mut strip_idx = 0;

                while n + TILE_N <= n_size {
                    #[allow(unused_unsafe)]
                    unsafe {
                        let vbias_0 = $crate::simd_primitive!(neon, $elem, loadu, bias.as_ptr().add(n));
                        let vbias_1 = $crate::simd_primitive!(neon, $elem, loadu, bias.as_ptr().add(n + LANES));
                        let vbias_2 = $crate::simd_primitive!(neon, $elem, loadu, bias.as_ptr().add(n + LANES * 2));
                        let (mut c_0_0, mut c_0_1, mut c_0_2) = (vbias_0, vbias_1, vbias_2);
                        let (mut c_1_0, mut c_1_1, mut c_1_2) = (vbias_0, vbias_1, vbias_2);
                        let (mut c_2_0, mut c_2_1, mut c_2_2) = (vbias_0, vbias_1, vbias_2);
                        let (mut c_3_0, mut c_3_1, mut c_3_2) = (vbias_0, vbias_1, vbias_2);
                        let (mut c_4_0, mut c_4_1, mut c_4_2) = (vbias_0, vbias_1, vbias_2);
                        let (mut c_5_0, mut c_5_1, mut c_5_2) = (vbias_0, vbias_1, vbias_2);
                        let (mut c_6_0, mut c_6_1, mut c_6_2) = (vbias_0, vbias_1, vbias_2);
                        let (mut c_7_0, mut c_7_1, mut c_7_2) = (vbias_0, vbias_1, vbias_2);

                        macro_rules! fma_row {
                            ($vb0:ident, $vb1:ident, $vb2:ident, $a_ptr:ident, $c0:ident, $c1:ident, $c2:ident) => {
                                let va = $crate::simd_primitive!(neon, $elem, splat, *$a_ptr);
                                $c0 = $crate::simd_primitive!(neon, $elem, fma, va, $vb0, $c0);
                                $c1 = $crate::simd_primitive!(neon, $elem, fma, va, $vb1, $c1);
                                $c2 = $crate::simd_primitive!(neon, $elem, fma, va, $vb2, $c2);
                            };
                        }
                        macro_rules! fma_all_neon {
                            ($vb0:ident, $vb1:ident, $vb2:ident) => {
                                fma_row!($vb0, $vb1, $vb2, a_ptr_0, c_0_0, c_0_1, c_0_2);
                                fma_row!($vb0, $vb1, $vb2, a_ptr_1, c_1_0, c_1_1, c_1_2);
                                fma_row!($vb0, $vb1, $vb2, a_ptr_2, c_2_0, c_2_1, c_2_2);
                                fma_row!($vb0, $vb1, $vb2, a_ptr_3, c_3_0, c_3_1, c_3_2);
                                fma_row!($vb0, $vb1, $vb2, a_ptr_4, c_4_0, c_4_1, c_4_2);
                                fma_row!($vb0, $vb1, $vb2, a_ptr_5, c_5_0, c_5_1, c_5_2);
                                fma_row!($vb0, $vb1, $vb2, a_ptr_6, c_6_0, c_6_1, c_6_2);
                                fma_row!($vb0, $vb1, $vb2, a_ptr_7, c_7_0, c_7_1, c_7_2);
                            };
                        }
                        macro_rules! advance_a_neon {
                            ($n:expr) => {
                                a_ptr_0 = a_ptr_0.add($n); a_ptr_1 = a_ptr_1.add($n);
                                a_ptr_2 = a_ptr_2.add($n); a_ptr_3 = a_ptr_3.add($n);
                                a_ptr_4 = a_ptr_4.add($n); a_ptr_5 = a_ptr_5.add($n);
                                a_ptr_6 = a_ptr_6.add($n); a_ptr_7 = a_ptr_7.add($n);
                            };
                        }

                        let mut k_start = 0usize;
                        let mut chunk = 0usize;
                        while k_start < k_size {
                            let kc = KC.min(k_size - k_start);
                            let mut b_ptr = packed_b.as_ptr().add(chunk * chunk_size + strip_idx * KC * TILE_N);

                            let mut a_ptr_0 = a.as_ptr().add((m + 0) * k_size + k_start);
                            let mut a_ptr_1 = a.as_ptr().add((m + 1) * k_size + k_start);
                            let mut a_ptr_2 = a.as_ptr().add((m + 2) * k_size + k_start);
                            let mut a_ptr_3 = a.as_ptr().add((m + 3) * k_size + k_start);
                            let mut a_ptr_4 = a.as_ptr().add((m + 4) * k_size + k_start);
                            let mut a_ptr_5 = a.as_ptr().add((m + 5) * k_size + k_start);
                            let mut a_ptr_6 = a.as_ptr().add((m + 6) * k_size + k_start);
                            let mut a_ptr_7 = a.as_ptr().add((m + 7) * k_size + k_start);

                            let mut _k = 0usize;
                            let k_unroll_end = kc & !7;
                            while _k < k_unroll_end {
                                let vb0_0 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr);
                                let vb0_1 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(LANES));
                                let vb0_2 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(LANES * 2));
                                fma_all_neon!(vb0_0, vb0_1, vb0_2);
                                advance_a_neon!(1);
                                let vb1_0 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(TILE_N));
                                let vb1_1 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(TILE_N + LANES));
                                let vb1_2 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(TILE_N + LANES * 2));
                                fma_all_neon!(vb1_0, vb1_1, vb1_2);
                                advance_a_neon!(1);
                                let vb2_0 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(TILE_N * 2));
                                let vb2_1 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(TILE_N * 2 + LANES));
                                let vb2_2 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(TILE_N * 2 + LANES * 2));
                                fma_all_neon!(vb2_0, vb2_1, vb2_2);
                                advance_a_neon!(1);
                                let vb3_0 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(TILE_N * 3));
                                let vb3_1 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(TILE_N * 3 + LANES));
                                let vb3_2 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(TILE_N * 3 + LANES * 2));
                                fma_all_neon!(vb3_0, vb3_1, vb3_2);
                                advance_a_neon!(1);
                                let vb4_0 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(TILE_N * 4));
                                let vb4_1 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(TILE_N * 4 + LANES));
                                let vb4_2 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(TILE_N * 4 + LANES * 2));
                                fma_all_neon!(vb4_0, vb4_1, vb4_2);
                                advance_a_neon!(1);
                                let vb5_0 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(TILE_N * 5));
                                let vb5_1 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(TILE_N * 5 + LANES));
                                let vb5_2 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(TILE_N * 5 + LANES * 2));
                                fma_all_neon!(vb5_0, vb5_1, vb5_2);
                                advance_a_neon!(1);
                                let vb6_0 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(TILE_N * 6));
                                let vb6_1 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(TILE_N * 6 + LANES));
                                let vb6_2 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(TILE_N * 6 + LANES * 2));
                                fma_all_neon!(vb6_0, vb6_1, vb6_2);
                                advance_a_neon!(1);
                                let vb7_0 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(TILE_N * 7));
                                let vb7_1 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(TILE_N * 7 + LANES));
                                let vb7_2 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(TILE_N * 7 + LANES * 2));
                                fma_all_neon!(vb7_0, vb7_1, vb7_2);
                                advance_a_neon!(1);
                                b_ptr = b_ptr.add(TILE_N * 8);
                                _k += 8;
                            }
                            while _k < kc {
                                let vb_0 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr);
                                let vb_1 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(LANES));
                                let vb_2 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(LANES * 2));
                                fma_all_neon!(vb_0, vb_1, vb_2);
                                advance_a_neon!(1);
                                b_ptr = b_ptr.add(TILE_N);
                                _k += 1;
                            }
                            k_start += KC;
                            chunk += 1;
                        }

                        macro_rules! store_row {
                            ($row:expr, $c0:expr, $c1:expr, $c2:expr) => {
                                $crate::simd_primitive!(neon, $elem, storeu, c.as_mut_ptr().add((m + $row) * n_size + n), $c0);
                                $crate::simd_primitive!(neon, $elem, storeu, c.as_mut_ptr().add((m + $row) * n_size + n + LANES), $c1);
                                $crate::simd_primitive!(neon, $elem, storeu, c.as_mut_ptr().add((m + $row) * n_size + n + LANES * 2), $c2);
                            };
                        }
                        store_row!(0, c_0_0, c_0_1, c_0_2);
                        store_row!(1, c_1_0, c_1_1, c_1_2);
                        store_row!(2, c_2_0, c_2_1, c_2_2);
                        store_row!(3, c_3_0, c_3_1, c_3_2);
                        store_row!(4, c_4_0, c_4_1, c_4_2);
                        store_row!(5, c_5_0, c_5_1, c_5_2);
                        store_row!(6, c_6_0, c_6_1, c_6_2);
                        store_row!(7, c_7_0, c_7_1, c_7_2);
                    }
                    n += TILE_N;
                    strip_idx += 1;
                }

                // Remainder N (scalar)
                while n < n_size {
                    for i in 0..TILE_M {
                        let mut sum = bias[n];
                        for k in 0..k_size {
                            let ci = k / KC; let ki = k % KC;
                            sum = <$elem as Element>::mul_add(sum, a[(m + i) * k_size + k], packed_b[ci * chunk_size + (n / TILE_N) * KC * TILE_N + ki * TILE_N + (n % TILE_N)]);
                        }
                        c[(m + i) * n_size + n] = sum;
                    }
                    n += 1;
                }
                m += TILE_M;
            }

            // Remainder M (scalar)
            while m < m_size {
                for n in 0..n_size {
                    let mut sum = bias[n];
                    for k in 0..k_size {
                        let ci = k / KC; let ki = k % KC;
                        sum = <$elem as Element>::mul_add(sum, a[m * k_size + k], packed_b[ci * chunk_size + (n / TILE_N) * KC * TILE_N + ki * TILE_N + (n % TILE_N)]);
                    }
                    c[m * n_size + n] = sum;
                }
                m += 1;
            }
        }
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
