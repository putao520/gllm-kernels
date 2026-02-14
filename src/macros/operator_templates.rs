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
                    let one = $crate::simd_primitive!($isa, $elem, splat, <$elem as Element>::ONE);
                    let neg_va = $crate::simd_primitive!($isa, $elem, neg, va);
                    let exp_neg = $crate::simd_primitive!($isa, $elem, exp, neg_va);
                    let one_plus_exp = $crate::simd_primitive!($isa, $elem, add, one, exp_neg);
                    let sigmoid = $crate::simd_primitive!($isa, $elem, div, one, one_plus_exp);
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

            // Pass 3: divide by sum
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
                    let va = $crate::simd_primitive!($isa, $elem, load, a.as_ptr().add(i));
                    let vw = $crate::simd_primitive!($isa, $elem, load, weight.as_ptr().add(i));
                    let normed = $crate::simd_primitive!($isa, $elem, mul, va, v_inv);
                    let res = $crate::simd_primitive!($isa, $elem, mul, normed, vw);
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
        #[inline(always)]
        pub fn layer_norm(a: &[$elem], weight: &[$elem], bias: &[$elem], out: &mut [$elem], eps: $elem) {
            const LANES: usize = $crate::simd_primitive!($isa, $elem, lanes);
            let len = a.len();
            assert!(weight.len() == len && bias.len() == len && out.len() == len);
            let n = len as f32;

            // Pass 1: mean
            let mut i = 0;
            #[allow(unused_unsafe)]
            let mut sum_acc = unsafe { $crate::simd_primitive!($isa, $elem, zero) };
            while i + LANES <= len {
                #[allow(unused_unsafe)]
                unsafe {
                    let va = $crate::simd_primitive!($isa, $elem, load, a.as_ptr().add(i));
                    sum_acc = $crate::simd_primitive!($isa, $elem, add, sum_acc, va);
                }
                i += LANES;
            }
            #[allow(unused_unsafe)]
            let mut sum_val: f32 = unsafe { $crate::simd_primitive!($isa, $elem, reduce_sum, sum_acc) };
            while i < len { sum_val += a[i].to_f32(); i += 1; }
            let mean_f = sum_val / n;
            let mean = <$elem as Element>::from_f32(mean_f);

            // Pass 2: variance
            i = 0;
            #[allow(unused_unsafe)]
            let vmean = unsafe { $crate::simd_primitive!($isa, $elem, splat, mean) };
            #[allow(unused_unsafe)]
            let mut var_acc = unsafe { $crate::simd_primitive!($isa, $elem, zero) };
            while i + LANES <= len {
                #[allow(unused_unsafe)]
                unsafe {
                    let va = $crate::simd_primitive!($isa, $elem, load, a.as_ptr().add(i));
                    let diff = $crate::simd_primitive!($isa, $elem, sub, va, vmean);
                    var_acc = $crate::simd_primitive!($isa, $elem, fma, diff, diff, var_acc);
                }
                i += LANES;
            }
            #[allow(unused_unsafe)]
            let mut var_val: f32 = unsafe { $crate::simd_primitive!($isa, $elem, reduce_sum, var_acc) };
            while i < len { let d = a[i].to_f32() - mean_f; var_val += d * d; i += 1; }
            let eps_f = eps.to_f32();
            let inv_std_f = 1.0f32 / (var_val / n + eps_f).sqrt();
            let inv_std = <$elem as Element>::from_f32(inv_std_f);

            // Pass 3: normalize, scale, bias
            i = 0;
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
        /// rope: Apply Rotary Positional Embeddings
        /// q/k are [seq_len, head_dim], cos_sin is [seq_len, head_dim] interleaved cos/sin
        /// For each pair (q[2i], q[2i+1]), apply complex rotation:
        ///   q[2i]'   = q[2i] * cos - q[2i+1] * sin
        ///   q[2i+1]' = q[2i] * sin + q[2i+1] * cos
        #[inline(always)]
        pub fn rope(data: &mut [$elem], cos: &[$elem], sin: &[$elem], head_dim: usize) {
            let half = head_dim / 2;
            let seq_len = data.len() / head_dim;
            for pos in 0..seq_len {
                let base = pos * head_dim;
                for i in 0..half {
                    let x0 = data[base + i];
                    let x1 = data[base + i + half];
                    let c = cos[pos * half + i];
                    let s = sin[pos * half + i];
                    data[base + i]        = x0 * c - x1 * s;
                    data[base + i + half] = x0 * s + x1 * c;
                }
            }
        }

        /// embedding_lookup: out = embedding_table[token_ids]
        /// table: [vocab_size, dim], ids: [seq_len], out: [seq_len, dim]
        #[inline(always)]
        pub fn embedding_lookup(table: &[$elem], ids: &[u32], out: &mut [$elem], dim: usize) {
            for (i, &id) in ids.iter().enumerate() {
                let src = &table[(id as usize) * dim..(id as usize + 1) * dim];
                let dst = &mut out[i * dim..(i + 1) * dim];
                dst.copy_from_slice(src);
            }
        }
    };
}

/// Defines GEMV (General Matrix-Vector Multiply)
#[macro_export]
macro_rules! define_gemv_op {
    ($isa:ident, $elem:ident) => {
        /// gemv: y = A * x + y (M x K) * (K) -> (M)
        #[inline(always)]
        pub fn gemv(a: &[$elem], x: &[$elem], y: &mut [$elem], m: usize, k: usize) {
            const LANES: usize = $crate::simd_primitive!($isa, $elem, lanes);
            assert!(a.len() >= m * k && x.len() >= k && y.len() >= m);
            for row in 0..m {
                let row_ptr = &a[row * k..];
                let mut i = 0;
                #[allow(unused_unsafe)]
                let mut acc = unsafe { $crate::simd_primitive!($isa, $elem, zero) };
                while i + LANES <= k {
                    #[allow(unused_unsafe)]
                    unsafe {
                        let va = $crate::simd_primitive!($isa, $elem, load, row_ptr.as_ptr().add(i));
                        let vx = $crate::simd_primitive!($isa, $elem, load, x.as_ptr().add(i));
                        acc = $crate::simd_primitive!($isa, $elem, fma, va, vx, acc);
                    }
                    i += LANES;
                }
                #[allow(unused_unsafe)]
                let mut dot: f32 = unsafe { $crate::simd_primitive!($isa, $elem, reduce_sum, acc) };
                while i < k { dot += row_ptr[i].to_f32() * x[i].to_f32(); i += 1; }
                y[row] = <$elem as Element>::from_f32(y[row].to_f32() + dot);
            }
        }
    };
}



/// Defines Matrix Multiplication (GEMM)
#[macro_export]
macro_rules! define_matmul_op {
    // NEON f16: 8×12 microkernel with f32 arithmetic (convert on load/store)
    (neon, f16) => {
        #[inline(always)]
        pub fn matmul(a: &[f16], b: &[f16], c: &mut [f16], m_size: usize, n_size: usize, k_size: usize) {
            assert!(a.len() >= m_size * k_size);
            assert!(b.len() >= n_size * k_size);
            assert!(c.len() >= m_size * n_size);

            const TILE_M: usize = 8;
            const TILE_N: usize = 12; // 3 × 4 f32 lanes

            // Pack B into f32 column strips of width TILE_N
            let n_strips = (n_size + TILE_N - 1) / TILE_N;
            let packed_b_size = n_strips * k_size * TILE_N;
            let mut packed_b = vec![0.0f32; packed_b_size];

            for i in 0..n_strips {
                let n_start = i * TILE_N;
                let actual_n = if n_start + TILE_N <= n_size { TILE_N } else { n_size - n_start };
                for k in 0..k_size {
                    let dst_off = i * k_size * TILE_N + k * TILE_N;
                    for x in 0..actual_n {
                        packed_b[dst_off + x] = b[k * n_size + n_start + x].to_f32();
                    }
                }
            }

            let mut m = 0;
            while m + TILE_M <= m_size {
                let mut n = 0;
                let mut strip_idx = 0;
                while n + TILE_N <= n_size {
                    // 24 f32 accumulators
                    let mut acc = [[0.0f32; TILE_N]; TILE_M];
                    let b_base = strip_idx * k_size * TILE_N;
                    for k in 0..k_size {
                        let b_off = b_base + k * TILE_N;
                        for row in 0..TILE_M {
                            let a_val = a[(m + row) * k_size + k].to_f32();
                            for col in 0..TILE_N {
                                acc[row][col] = a_val.mul_add(packed_b[b_off + col], acc[row][col]);
                            }
                        }
                    }
                    for row in 0..TILE_M {
                        for col in 0..TILE_N {
                            c[(m + row) * n_size + n + col] = f16::from_f32(acc[row][col]);
                        }
                    }
                    n += TILE_N;
                    strip_idx += 1;
                }
                while n < n_size {
                    for i in 0..TILE_M {
                        let mut sum = 0.0f32;
                        for k in 0..k_size {
                            sum += a[(m + i) * k_size + k].to_f32() * b[k * n_size + n].to_f32();
                        }
                        c[(m + i) * n_size + n] = f16::from_f32(sum);
                    }
                    n += 1;
                }
                m += TILE_M;
            }
            while m < m_size {
                for n in 0..n_size {
                    let mut sum = 0.0f32;
                    for k in 0..k_size {
                        sum += a[m * k_size + k].to_f32() * b[k * n_size + n].to_f32();
                    }
                    c[m * n_size + n] = f16::from_f32(sum);
                }
                m += 1;
            }
        }
    };
    // NEON bf16: 8×12 microkernel with f32 arithmetic (convert on load/store)
    (neon, bf16) => {
        #[inline(always)]
        pub fn matmul(a: &[bf16], b: &[bf16], c: &mut [bf16], m_size: usize, n_size: usize, k_size: usize) {
            assert!(a.len() >= m_size * k_size);
            assert!(b.len() >= n_size * k_size);
            assert!(c.len() >= m_size * n_size);

            const TILE_M: usize = 8;
            const TILE_N: usize = 12;

            let n_strips = (n_size + TILE_N - 1) / TILE_N;
            let packed_b_size = n_strips * k_size * TILE_N;
            let mut packed_b = vec![0.0f32; packed_b_size];

            for i in 0..n_strips {
                let n_start = i * TILE_N;
                let actual_n = if n_start + TILE_N <= n_size { TILE_N } else { n_size - n_start };
                for k in 0..k_size {
                    let dst_off = i * k_size * TILE_N + k * TILE_N;
                    for x in 0..actual_n {
                        packed_b[dst_off + x] = b[k * n_size + n_start + x].to_f32();
                    }
                }
            }

            let mut m = 0;
            while m + TILE_M <= m_size {
                let mut n = 0;
                let mut strip_idx = 0;
                while n + TILE_N <= n_size {
                    let mut acc = [[0.0f32; TILE_N]; TILE_M];
                    let b_base = strip_idx * k_size * TILE_N;
                    for k in 0..k_size {
                        let b_off = b_base + k * TILE_N;
                        for row in 0..TILE_M {
                            let a_val = a[(m + row) * k_size + k].to_f32();
                            for col in 0..TILE_N {
                                acc[row][col] = a_val.mul_add(packed_b[b_off + col], acc[row][col]);
                            }
                        }
                    }
                    for row in 0..TILE_M {
                        for col in 0..TILE_N {
                            c[(m + row) * n_size + n + col] = bf16::from_f32(acc[row][col]);
                        }
                    }
                    n += TILE_N;
                    strip_idx += 1;
                }
                while n < n_size {
                    for i in 0..TILE_M {
                        let mut sum = 0.0f32;
                        for k in 0..k_size {
                            sum += a[(m + i) * k_size + k].to_f32() * b[k * n_size + n].to_f32();
                        }
                        c[(m + i) * n_size + n] = bf16::from_f32(sum);
                    }
                    n += 1;
                }
                m += TILE_M;
            }
            while m < m_size {
                for n in 0..n_size {
                    let mut sum = 0.0f32;
                    for k in 0..k_size {
                        sum += a[m * k_size + k].to_f32() * b[k * n_size + n].to_f32();
                    }
                    c[m * n_size + n] = bf16::from_f32(sum);
                }
                m += 1;
            }
        }
    };
    // f16/bf16 on any ISA: use scalar Element-based GEMM (SIMD load/store handled by simd_primitive)
    ($isa:ident, f16) => {
        #[inline(always)]
        pub fn matmul(a: &[f16], b: &[f16], c: &mut [f16], m_size: usize, n_size: usize, k_size: usize) {
            assert!(a.len() >= m_size * k_size);
            assert!(b.len() >= n_size * k_size);
            assert!(c.len() >= m_size * n_size);
            for m in 0..m_size {
                for n in 0..n_size {
                    let mut sum = 0.0f32;
                    for k in 0..k_size {
                        sum += a[m * k_size + k].to_f32() * b[k * n_size + n].to_f32();
                    }
                    c[m * n_size + n] = <f16 as Element>::from_f32(sum);
                }
            }
        }
    };
    ($isa:ident, bf16) => {
        #[inline(always)]
        pub fn matmul(a: &[bf16], b: &[bf16], c: &mut [bf16], m_size: usize, n_size: usize, k_size: usize) {
            assert!(a.len() >= m_size * k_size);
            assert!(b.len() >= n_size * k_size);
            assert!(c.len() >= m_size * n_size);
            for m in 0..m_size {
                for n in 0..n_size {
                    let mut sum = 0.0f32;
                    for k in 0..k_size {
                        sum += a[m * k_size + k].to_f32() * b[k * n_size + n].to_f32();
                    }
                    c[m * n_size + n] = <bf16 as Element>::from_f32(sum);
                }
            }
        }
    };
    (avx512, $elem:ident) => {
        // AVX-512 Optimized Implementation: 14×32 microkernel
        // 28 zmm accumulator registers (14 rows × 2 vectors of 16 lanes)
        #[target_feature(enable = "avx512f")]
        pub unsafe fn matmul_avx512(a: &[$elem], b: &[$elem], c: &mut [$elem], m_size: usize, n_size: usize, k_size: usize) {
            const TILE_M: usize = 14;
            const TILE_N_VECS: usize = 2;
            const LANES: usize = 16; // AVX-512 f32
            const TILE_N: usize = TILE_N_VECS * LANES; // 32

            assert!(a.len() >= m_size * k_size);
            assert!(b.len() >= n_size * k_size);
            assert!(c.len() >= m_size * n_size);

            // Pack B into column strips of width TILE_N
            let n_strips = (n_size + TILE_N - 1) / TILE_N;
            let packed_b_size = n_strips * k_size * TILE_N;
            let mut packed_b: Vec<$elem> = Vec::with_capacity(packed_b_size);
            unsafe {
                packed_b.set_len(packed_b_size);
                let packed_ptr = packed_b.as_mut_ptr();
                for i in 0..n_strips {
                    let n_start = i * TILE_N;
                    if n_start + TILE_N <= n_size {
                        for k in 0..k_size {
                            let src = b.as_ptr().add(k * n_size + n_start);
                            let dst = packed_ptr.add(i * k_size * TILE_N + k * TILE_N);
                            std::ptr::copy_nonoverlapping(src, dst, TILE_N);
                        }
                    } else {
                        // Partial strip: zero-pad
                        let actual_n = n_size - n_start;
                        for k in 0..k_size {
                            let dst = packed_ptr.add(i * k_size * TILE_N + k * TILE_N);
                            std::ptr::write_bytes(dst, 0, TILE_N);
                            let src = b.as_ptr().add(k * n_size + n_start);
                            std::ptr::copy_nonoverlapping(src, dst, actual_n);
                        }
                    }
                }
            }

            // Main loop: process TILE_M rows at a time
            let mut m = 0;
            while m + TILE_M <= m_size {
                let mut n = 0;
                let mut strip_idx = 0;
                while n + TILE_N <= n_size {
                    unsafe {
                        // 28 accumulators: 14 rows × 2 vectors
                        let mut c_0_0 = $crate::simd_primitive!(avx512, $elem, zero); let mut c_0_1 = $crate::simd_primitive!(avx512, $elem, zero);
                        let mut c_1_0 = $crate::simd_primitive!(avx512, $elem, zero); let mut c_1_1 = $crate::simd_primitive!(avx512, $elem, zero);
                        let mut c_2_0 = $crate::simd_primitive!(avx512, $elem, zero); let mut c_2_1 = $crate::simd_primitive!(avx512, $elem, zero);
                        let mut c_3_0 = $crate::simd_primitive!(avx512, $elem, zero); let mut c_3_1 = $crate::simd_primitive!(avx512, $elem, zero);
                        let mut c_4_0 = $crate::simd_primitive!(avx512, $elem, zero); let mut c_4_1 = $crate::simd_primitive!(avx512, $elem, zero);
                        let mut c_5_0 = $crate::simd_primitive!(avx512, $elem, zero); let mut c_5_1 = $crate::simd_primitive!(avx512, $elem, zero);
                        let mut c_6_0 = $crate::simd_primitive!(avx512, $elem, zero); let mut c_6_1 = $crate::simd_primitive!(avx512, $elem, zero);
                        let mut c_7_0 = $crate::simd_primitive!(avx512, $elem, zero); let mut c_7_1 = $crate::simd_primitive!(avx512, $elem, zero);
                        let mut c_8_0 = $crate::simd_primitive!(avx512, $elem, zero); let mut c_8_1 = $crate::simd_primitive!(avx512, $elem, zero);
                        let mut c_9_0 = $crate::simd_primitive!(avx512, $elem, zero); let mut c_9_1 = $crate::simd_primitive!(avx512, $elem, zero);
                        let mut c_10_0 = $crate::simd_primitive!(avx512, $elem, zero); let mut c_10_1 = $crate::simd_primitive!(avx512, $elem, zero);
                        let mut c_11_0 = $crate::simd_primitive!(avx512, $elem, zero); let mut c_11_1 = $crate::simd_primitive!(avx512, $elem, zero);
                        let mut c_12_0 = $crate::simd_primitive!(avx512, $elem, zero); let mut c_12_1 = $crate::simd_primitive!(avx512, $elem, zero);
                        let mut c_13_0 = $crate::simd_primitive!(avx512, $elem, zero); let mut c_13_1 = $crate::simd_primitive!(avx512, $elem, zero);

                        let mut b_ptr = packed_b.as_ptr().add(strip_idx * k_size * TILE_N);

                        for _k in 0..k_size {
                            // Load 2 vectors from packed B
                            let vb_0 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr);
                            let vb_1 = $crate::simd_primitive!(avx512, $elem, loadu, b_ptr.add(LANES));

                            // Software prefetch 512B ahead
                            $crate::simd_primitive!(avx512, $elem, prefetch, b_ptr.add(TILE_N * 8) as *const i8, 0);

                            // Broadcast A values and FMA for each of 14 rows
                            macro_rules! fma_row {
                                ($row:expr, $c0:ident, $c1:ident) => {
                                    let va = $crate::simd_primitive!(avx512, $elem, splat, *a.as_ptr().add((m + $row) * k_size + _k));
                                    $c0 = $crate::simd_primitive!(avx512, $elem, fma, va, vb_0, $c0);
                                    $c1 = $crate::simd_primitive!(avx512, $elem, fma, va, vb_1, $c1);
                                };
                            }
                            fma_row!(0, c_0_0, c_0_1);
                            fma_row!(1, c_1_0, c_1_1);
                            fma_row!(2, c_2_0, c_2_1);
                            fma_row!(3, c_3_0, c_3_1);
                            fma_row!(4, c_4_0, c_4_1);
                            fma_row!(5, c_5_0, c_5_1);
                            fma_row!(6, c_6_0, c_6_1);
                            fma_row!(7, c_7_0, c_7_1);
                            fma_row!(8, c_8_0, c_8_1);
                            fma_row!(9, c_9_0, c_9_1);
                            fma_row!(10, c_10_0, c_10_1);
                            fma_row!(11, c_11_0, c_11_1);
                            fma_row!(12, c_12_0, c_12_1);
                            fma_row!(13, c_13_0, c_13_1);

                            b_ptr = b_ptr.add(TILE_N);
                        }

                        // Store results
                        macro_rules! store_row {
                            ($row:expr, $c0:expr, $c1:expr) => {
                                $crate::simd_primitive!(avx512, $elem, storeu, c.as_mut_ptr().add((m + $row) * n_size + n), $c0);
                                $crate::simd_primitive!(avx512, $elem, storeu, c.as_mut_ptr().add((m + $row) * n_size + n + LANES), $c1);
                            };
                        }
                        store_row!(0, c_0_0, c_0_1);
                        store_row!(1, c_1_0, c_1_1);
                        store_row!(2, c_2_0, c_2_1);
                        store_row!(3, c_3_0, c_3_1);
                        store_row!(4, c_4_0, c_4_1);
                        store_row!(5, c_5_0, c_5_1);
                        store_row!(6, c_6_0, c_6_1);
                        store_row!(7, c_7_0, c_7_1);
                        store_row!(8, c_8_0, c_8_1);
                        store_row!(9, c_9_0, c_9_1);
                        store_row!(10, c_10_0, c_10_1);
                        store_row!(11, c_11_0, c_11_1);
                        store_row!(12, c_12_0, c_12_1);
                        store_row!(13, c_13_0, c_13_1);
                    }
                    n += TILE_N;
                    strip_idx += 1;
                }
                // Remainder N (scalar)
                while n < n_size {
                    for i in 0..TILE_M {
                        let mut sum: $elem = <$elem as Element>::ZERO;
                        for k in 0..k_size {
                            sum = a[(m + i) * k_size + k].mul_add(b[k * n_size + n], sum);
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
                        sum = a[m * k_size + k].mul_add(b[k * n_size + n], sum);
                    }
                    c[m * n_size + n] = sum;
                }
                m += 1;
            }
        }

        #[inline(always)]
        pub fn matmul(a: &[$elem], b: &[$elem], c: &mut [$elem], m_size: usize, n_size: usize, k_size: usize) {
            unsafe { matmul_avx512(a, b, c, m_size, n_size, k_size); }
        }
    };

    (avx2, $elem:ident) => {
        // AVX2 Optimized Implementation with Packing
        #[target_feature(enable = "avx2")]
        pub unsafe fn matmul_avx2(a: &[$elem], b: &[$elem], c: &mut [$elem], m_size: usize, n_size: usize, k_size: usize) {
            // Kernel Constants
            const TILE_M: usize = 6;
            const TILE_N_VECS: usize = 2; 
            const LANES: usize = 8; // AVX2 f32
            const TILE_N: usize = TILE_N_VECS * LANES; // 16

            assert!(a.len() >= m_size * k_size);
            assert!(b.len() >= n_size * k_size);
            assert!(c.len() >= m_size * n_size);

            // 1. Pack Matrix B
            // Layout: Column Strips of width TILE_N.
            // Each strip is K * TILE_N elements stored sequentially.
            // This ensures that in the K-loop, we access B sequentially (perfect prefetching).
            let n_strips = (n_size + TILE_N - 1) / TILE_N;
            let packed_b_size = n_strips * k_size * TILE_N;
            let mut packed_b: Vec<$elem> = Vec::with_capacity(packed_b_size);
            
            unsafe {
                packed_b.set_len(packed_b_size);
                let packed_ptr = packed_b.as_mut_ptr();
                
                for i in 0..n_strips {
                    let n_start = i * TILE_N;
                    // Check if full tile
                    if n_start + TILE_N <= n_size {
                         for k in 0..k_size {
                            // Copy row k, cols n_start..n_start+16
                            let src = b.as_ptr().add(k * n_size + n_start);
                            let dst = packed_ptr.add(i * k_size * TILE_N + k * TILE_N);
                            // We can use SIMD load/store for packing too
                            let v0 = $crate::simd_primitive!(avx2, $elem, loadu, src);
                            let v1 = $crate::simd_primitive!(avx2, $elem, loadu, src.add(LANES));
                            $crate::simd_primitive!(avx2, $elem, storeu, dst, v0);
                            $crate::simd_primitive!(avx2, $elem, storeu, dst.add(LANES), v1);
                         }
                    } else {
                        // Partial strip
                         for k in 0..k_size {
                             let dst_base = packed_ptr.add(i * k_size * TILE_N + k * TILE_N);
                             for x in 0..TILE_N {
                                 if n_start + x < n_size {
                                     *dst_base.add(x) = b[k * n_size + n_start + x];
                                 } else {
                                     *dst_base.add(x) = <$elem as Element>::ZERO;
                                 }
                             }
                         }
                    }
                }
            }

            // Tiled Loop
            let mut m = 0;
            while m + TILE_M <= m_size {
                let mut n = 0;
                let mut strip_idx = 0;
                
                while n + TILE_N <= n_size {
                    // Micro-kernel (6x16)
                    unsafe {
                        // Initialize 12 accumulators (6 rows x 2 vectors)
                        let mut c_0_0 = $crate::simd_primitive!(avx2, $elem, zero);
                        let mut c_0_1 = $crate::simd_primitive!(avx2, $elem, zero);
                        let mut c_1_0 = $crate::simd_primitive!(avx2, $elem, zero);
                        let mut c_1_1 = $crate::simd_primitive!(avx2, $elem, zero);
                        let mut c_2_0 = $crate::simd_primitive!(avx2, $elem, zero);
                        let mut c_2_1 = $crate::simd_primitive!(avx2, $elem, zero);
                        let mut c_3_0 = $crate::simd_primitive!(avx2, $elem, zero);
                        let mut c_3_1 = $crate::simd_primitive!(avx2, $elem, zero);
                        let mut c_4_0 = $crate::simd_primitive!(avx2, $elem, zero);
                        let mut c_4_1 = $crate::simd_primitive!(avx2, $elem, zero);
                        let mut c_5_0 = $crate::simd_primitive!(avx2, $elem, zero);
                        let mut c_5_1 = $crate::simd_primitive!(avx2, $elem, zero);

                        let mut k = 0;
                        // Hoist pointers
                        // A ptrs for 6 rows
                        let mut a_ptr_0 = a.as_ptr().add((m + 0) * k_size);
                        let mut a_ptr_1 = a.as_ptr().add((m + 1) * k_size);
                        let mut a_ptr_2 = a.as_ptr().add((m + 2) * k_size);
                        let mut a_ptr_3 = a.as_ptr().add((m + 3) * k_size);
                        let mut a_ptr_4 = a.as_ptr().add((m + 4) * k_size);
                        let mut a_ptr_5 = a.as_ptr().add((m + 5) * k_size);

                        // B ptr from PACKED buffer
                        // strip_idx * (k_size * TILE_N)
                        let mut b_ptr = packed_b.as_ptr().add(strip_idx * k_size * TILE_N);

                        while k < k_size {
                            // Load B vectors (contiguous 16 floats from packed buffer)
                            // Access: b_ptr, b_ptr + LANES
                            // Since packed, they are sequential.
                            let vb_0 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr);
                            let vb_1 = $crate::simd_primitive!(avx2, $elem, loadu, b_ptr.add(LANES));

                            // Broadcast A and FMA
                            // Row 0
                            let val_0 = *a_ptr_0;
                            let va_0 = $crate::simd_primitive!(avx2, $elem, splat, val_0);
                            c_0_0 = $crate::simd_primitive!(avx2, $elem, fma, va_0, vb_0, c_0_0);
                            c_0_1 = $crate::simd_primitive!(avx2, $elem, fma, va_0, vb_1, c_0_1);

                            // Row 1
                            let val_1 = *a_ptr_1;
                            let va_1 = $crate::simd_primitive!(avx2, $elem, splat, val_1);
                            c_1_0 = $crate::simd_primitive!(avx2, $elem, fma, va_1, vb_0, c_1_0);
                            c_1_1 = $crate::simd_primitive!(avx2, $elem, fma, va_1, vb_1, c_1_1);
                            
                            // Row 2
                            let val_2 = *a_ptr_2;
                            let va_2 = $crate::simd_primitive!(avx2, $elem, splat, val_2);
                            c_2_0 = $crate::simd_primitive!(avx2, $elem, fma, va_2, vb_0, c_2_0);
                            c_2_1 = $crate::simd_primitive!(avx2, $elem, fma, va_2, vb_1, c_2_1);

                             // Row 3
                            let val_3 = *a_ptr_3;
                            let va_3 = $crate::simd_primitive!(avx2, $elem, splat, val_3);
                            c_3_0 = $crate::simd_primitive!(avx2, $elem, fma, va_3, vb_0, c_3_0);
                            c_3_1 = $crate::simd_primitive!(avx2, $elem, fma, va_3, vb_1, c_3_1);

                             // Row 4
                            let val_4 = *a_ptr_4;
                            let va_4 = $crate::simd_primitive!(avx2, $elem, splat, val_4);
                            c_4_0 = $crate::simd_primitive!(avx2, $elem, fma, va_4, vb_0, c_4_0);
                            c_4_1 = $crate::simd_primitive!(avx2, $elem, fma, va_4, vb_1, c_4_1);

                             // Row 5
                            let val_5 = *a_ptr_5;
                            let va_5 = $crate::simd_primitive!(avx2, $elem, splat, val_5);
                            c_5_0 = $crate::simd_primitive!(avx2, $elem, fma, va_5, vb_0, c_5_0);
                            c_5_1 = $crate::simd_primitive!(avx2, $elem, fma, va_5, vb_1, c_5_1);

                            k += 1;
                            
                            // Pointer increments
                            a_ptr_0 = a_ptr_0.add(1);
                            a_ptr_1 = a_ptr_1.add(1);
                            a_ptr_2 = a_ptr_2.add(1);
                            a_ptr_3 = a_ptr_3.add(1);
                            a_ptr_4 = a_ptr_4.add(1);
                            a_ptr_5 = a_ptr_5.add(1);
                            
                            // Increment packed_b ptr by 16 (width of strip)
                            b_ptr = b_ptr.add(TILE_N); 
                        }

                        // Store results
                        $crate::simd_primitive!(avx2, $elem, storeu, c.as_mut_ptr().add((m + 0) * n_size + n), c_0_0);
                        $crate::simd_primitive!(avx2, $elem, storeu, c.as_mut_ptr().add((m + 0) * n_size + n + LANES), c_0_1);

                        $crate::simd_primitive!(avx2, $elem, storeu, c.as_mut_ptr().add((m + 1) * n_size + n), c_1_0);
                        $crate::simd_primitive!(avx2, $elem, storeu, c.as_mut_ptr().add((m + 1) * n_size + n + LANES), c_1_1);

                        $crate::simd_primitive!(avx2, $elem, storeu, c.as_mut_ptr().add((m + 2) * n_size + n), c_2_0);
                        $crate::simd_primitive!(avx2, $elem, storeu, c.as_mut_ptr().add((m + 2) * n_size + n + LANES), c_2_1);
                        
                        $crate::simd_primitive!(avx2, $elem, storeu, c.as_mut_ptr().add((m + 3) * n_size + n), c_3_0);
                        $crate::simd_primitive!(avx2, $elem, storeu, c.as_mut_ptr().add((m + 3) * n_size + n + LANES), c_3_1);
                        
                        $crate::simd_primitive!(avx2, $elem, storeu, c.as_mut_ptr().add((m + 4) * n_size + n), c_4_0);
                        $crate::simd_primitive!(avx2, $elem, storeu, c.as_mut_ptr().add((m + 4) * n_size + n + LANES), c_4_1);
                        
                        $crate::simd_primitive!(avx2, $elem, storeu, c.as_mut_ptr().add((m + 5) * n_size + n), c_5_0);
                        $crate::simd_primitive!(avx2, $elem, storeu, c.as_mut_ptr().add((m + 5) * n_size + n + LANES), c_5_1);
                    }
                    n += TILE_N;
                    strip_idx += 1;
                }
                
                // Remainder N handling (Scalar fallback for this strip)
                 while n < n_size {
                    for i in 0..TILE_M {
                        let mut sum: $elem = <$elem as Element>::ZERO;
                        for k in 0..k_size {
                            let val_a = a[(m + i) * k_size + k];
                            let val_b = b[k * n_size + n];
                            sum = val_a.mul_add(val_b, sum);
                        }
                        c[(m + i) * n_size + n] = sum;
                    }
                    n += 1;
                }

                m += TILE_M;
            }

            // Remainder M handling (Scalar fallback)
            while m < m_size {
                for n in 0..n_size {
                     let mut sum: $elem = <$elem as Element>::ZERO;
                    for k in 0..k_size {
                        let val_a = a[m * k_size + k];
                        let val_b = b[k * n_size + n];
                        sum = val_a.mul_add(val_b, sum);
                    }
                    c[m * n_size + n] = sum;
                }
                m += 1;
            }
        }

        // Trampoline to call unsafe implementation
        #[inline(always)]
        pub fn matmul(a: &[$elem], b: &[$elem], c: &mut [$elem], m_size: usize, n_size: usize, k_size: usize) {
            unsafe {
                matmul_avx2(a, b, c, m_size, n_size, k_size);
            }
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

            // Pack B into column strips of width TILE_N
            let n_strips = (n_size + TILE_N - 1) / TILE_N;
            let packed_b_size = n_strips * k_size * TILE_N;
            let mut packed_b: Vec<$elem> = vec![<$elem as Element>::ZERO; packed_b_size];

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

                        for _k in 0..k_size {
                            // Load 3 B vectors from packed buffer
                            let vb_0 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr);
                            let vb_1 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(LANES));
                            let vb_2 = $crate::simd_primitive!(neon, $elem, loadu, b_ptr.add(LANES * 2));

                            // 8 rows × 3 FMA each
                            macro_rules! fma_row {
                                ($a_ptr:ident, $c0:ident, $c1:ident, $c2:ident) => {
                                    let va = $crate::simd_primitive!(neon, $elem, splat, *$a_ptr);
                                    $c0 = $crate::simd_primitive!(neon, $elem, fma, va, vb_0, $c0);
                                    $c1 = $crate::simd_primitive!(neon, $elem, fma, va, vb_1, $c1);
                                    $c2 = $crate::simd_primitive!(neon, $elem, fma, va, vb_2, $c2);
                                };
                            }
                            fma_row!(a_ptr_0, c_0_0, c_0_1, c_0_2);
                            fma_row!(a_ptr_1, c_1_0, c_1_1, c_1_2);
                            fma_row!(a_ptr_2, c_2_0, c_2_1, c_2_2);
                            fma_row!(a_ptr_3, c_3_0, c_3_1, c_3_2);
                            fma_row!(a_ptr_4, c_4_0, c_4_1, c_4_2);
                            fma_row!(a_ptr_5, c_5_0, c_5_1, c_5_2);
                            fma_row!(a_ptr_6, c_6_0, c_6_1, c_6_2);
                            fma_row!(a_ptr_7, c_7_0, c_7_1, c_7_2);

                            b_ptr = b_ptr.add(TILE_N);
                            a_ptr_0 = a_ptr_0.add(1);
                            a_ptr_1 = a_ptr_1.add(1);
                            a_ptr_2 = a_ptr_2.add(1);
                            a_ptr_3 = a_ptr_3.add(1);
                            a_ptr_4 = a_ptr_4.add(1);
                            a_ptr_5 = a_ptr_5.add(1);
                            a_ptr_6 = a_ptr_6.add(1);
                            a_ptr_7 = a_ptr_7.add(1);
                        }

                        // Store 24 results
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
                            sum = a[(m + i) * k_size + k].mul_add(b[k * n_size + n], sum);
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
                        sum = a[m * k_size + k].mul_add(b[k * n_size + n], sum);
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
                        sum = val_a.mul_add(val_b, sum);
                    }
                    c[m * n_size + n] = sum;
                }
            }
        }
    };
}

/// Defines Flash Attention operations (flash_attention, flash_attention_paged)
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

        /// Standard multi-head attention with optional causal masking.
        /// All computation in f32 space; input/output converted from/to $elem.
        #[inline(always)]
        pub fn flash_attention(
            q: &[$elem], k: &[$elem], v: &[$elem], output: &mut [$elem],
            seq_len: usize, num_heads: usize, head_dim: usize,
            scale: f32, causal: bool,
        ) {
            let mut scores = vec![0.0f32; seq_len];
            let mut acc = vec![0.0f32; head_dim];

            let owned_q: Vec<f32>;
            let owned_k: Vec<f32>;
            let owned_v: Vec<f32>;
            let q_f32: &[f32] = if let Some(f) = <$elem as Element>::as_f32_slice(q) { f }
                else { owned_q = q.iter().map(|v| v.to_f32()).collect(); &owned_q };
            let k_f32: &[f32] = if let Some(f) = <$elem as Element>::as_f32_slice(k) { f }
                else { owned_k = k.iter().map(|v| v.to_f32()).collect(); &owned_k };
            let v_f32: &[f32] = if let Some(f) = <$elem as Element>::as_f32_slice(v) { f }
                else { owned_v = v.iter().map(|v| v.to_f32()).collect(); &owned_v };

            for h in 0..num_heads {
                for i in 0..seq_len {
                    let q_off = h * seq_len * head_dim + i * head_dim;
                    let o_off = q_off;
                    let max_j = if causal { i + 1 } else { seq_len };
                    let mut max_val = f32::NEG_INFINITY;

                    let q_ptr = unsafe { q_f32.as_ptr().add(q_off) };
                    for j in 0..max_j {
                        let k_off = h * seq_len * head_dim + j * head_dim;
                        let s = attn_dot_f32(q_ptr, unsafe { k_f32.as_ptr().add(k_off) }, head_dim) * scale;
                        if s > max_val { max_val = s; }
                        scores[j] = s;
                    }

                    let mut sum_exp = 0.0f32;
                    for j in 0..max_j {
                        scores[j] = (scores[j] - max_val).exp();
                        sum_exp += scores[j];
                    }
                    let inv_sum = 1.0 / sum_exp;

                    acc[..head_dim].fill(0.0);
                    for j in 0..max_j {
                        let v_off = h * seq_len * head_dim + j * head_dim;
                        attn_fma_f32(acc.as_mut_ptr(), unsafe { v_f32.as_ptr().add(v_off) }, scores[j] * inv_sum, head_dim);
                    }

                    if let Some(of) = <$elem as Element>::as_f32_slice_mut(output) {
                        of[o_off..o_off + head_dim].copy_from_slice(&acc[..head_dim]);
                    } else {
                        for d in 0..head_dim {
                            output[o_off + d] = <$elem as Element>::from_f32(acc[d]);
                        }
                    }
                }
            }
        }

        /// Paged KV-cache attention with GQA support.
        /// All computation in f32 space; input/output converted from/to $elem.
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

            let mut scores = vec![0.0f32; cache_len];
            let mut acc = vec![0.0f32; head_dim];

            let owned_q: Vec<f32>;
            let owned_k: Vec<f32>;
            let owned_v: Vec<f32>;
            let q_f32: &[f32] = if let Some(f) = <$elem as Element>::as_f32_slice(q) { f }
                else { owned_q = q.iter().map(|v| v.to_f32()).collect(); &owned_q };
            let k_f32: &[f32] = if let Some(f) = <$elem as Element>::as_f32_slice(k_cache) { f }
                else { owned_k = k_cache.iter().map(|v| v.to_f32()).collect(); &owned_k };
            let v_f32: &[f32] = if let Some(f) = <$elem as Element>::as_f32_slice(v_cache) { f }
                else { owned_v = v_cache.iter().map(|v| v.to_f32()).collect(); &owned_v };

            for h in 0..num_heads {
                let kv_h = h / heads_per_kv;
                for i in 0..seq_len {
                    let q_off = h * seq_len * head_dim + i * head_dim;
                    let o_off = q_off;
                    let mut max_val = f32::NEG_INFINITY;

                    let q_ptr = unsafe { q_f32.as_ptr().add(q_off) };
                    for j in 0..cache_len {
                        let page_idx = j / page_size;
                        let page_off = j % page_size;
                        let phys_page = page_table[kv_h * pages_per_kv + page_idx];
                        let k_off = phys_page * page_size * head_dim + page_off * head_dim;
                        let s = attn_dot_f32(q_ptr, unsafe { k_f32.as_ptr().add(k_off) }, head_dim) * scale;
                        if s > max_val { max_val = s; }
                        scores[j] = s;
                    }

                    let mut sum_exp = 0.0f32;
                    for j in 0..cache_len {
                        scores[j] = (scores[j] - max_val).exp();
                        sum_exp += scores[j];
                    }
                    let inv_sum = 1.0 / sum_exp;

                    acc[..head_dim].fill(0.0);
                    for j in 0..cache_len {
                        let page_idx = j / page_size;
                        let page_off = j % page_size;
                        let phys_page = page_table[kv_h * pages_per_kv + page_idx];
                        let v_off = phys_page * page_size * head_dim + page_off * head_dim;
                        attn_fma_f32(acc.as_mut_ptr(), unsafe { v_f32.as_ptr().add(v_off) }, scores[j] * inv_sum, head_dim);
                    }

                    if let Some(of) = <$elem as Element>::as_f32_slice_mut(output) {
                        of[o_off..o_off + head_dim].copy_from_slice(&acc[..head_dim]);
                    } else {
                        for d in 0..head_dim {
                            output[o_off + d] = <$elem as Element>::from_f32(acc[d]);
                        }
                    }
                }
            }
        }
    };
}
