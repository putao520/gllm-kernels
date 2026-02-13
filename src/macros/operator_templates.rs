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
                let val = a[i];
                let one = <$elem as Element>::ONE;
                let neg_val = <$elem as Element>::ZERO - val;
                let sigmoid = one / (one + neg_val.exp());
                out[i] = val * sigmoid;
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
            let mut result = unsafe { $crate::simd_primitive!($isa, $elem, reduce_sum, acc) };
            while i < len { result += a[i]; i += 1; }
            result
        }

        /// max_val: returns maximum element
        #[inline(always)]
        pub fn max_val(a: &[$elem]) -> $elem {
            const LANES: usize = $crate::simd_primitive!($isa, $elem, lanes);
            let len = a.len();
            assert!(len > 0);
            let mut i = 0;
            #[allow(unused_unsafe)]
            let mut acc = unsafe { $crate::simd_primitive!($isa, $elem, splat, <$elem>::NEG_INFINITY) };
            while i + LANES <= len {
                #[allow(unused_unsafe)]
                unsafe {
                    let va = $crate::simd_primitive!($isa, $elem, load, a.as_ptr().add(i));
                    acc = $crate::simd_primitive!($isa, $elem, max, acc, va);
                }
                i += LANES;
            }
            #[allow(unused_unsafe)]
            let mut result = unsafe { $crate::simd_primitive!($isa, $elem, reduce_max, acc) };
            while i < len { result = result.max(a[i]); i += 1; }
            result
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
            let mut result = unsafe { $crate::simd_primitive!($isa, $elem, reduce_sum, acc) };
            while i < len { result += a[i] * a[i]; i += 1; }
            result
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
            while i < len { out[i] = a[i].max(<$elem as Element>::ZERO); i += 1; }
        }

        /// gelu: out[i] = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        #[inline(always)]
        pub fn gelu(a: &[$elem], out: &mut [$elem]) {
            const LANES: usize = $crate::simd_primitive!($isa, $elem, lanes);
            let len = a.len();
            assert!(out.len() == len);

            // Constants for GELU tanh approximation
            let sqrt_2_over_pi: $elem = 0.7978845608;
            let coeff: $elem = 0.044715;

            let mut i = 0;
            while i + LANES <= len {
                #[allow(unused_unsafe)]
                unsafe {
                    let vx = $crate::simd_primitive!($isa, $elem, load, a.as_ptr().add(i));
                    let half = $crate::simd_primitive!($isa, $elem, splat, 0.5 as $elem);
                    let one = $crate::simd_primitive!($isa, $elem, splat, <$elem as Element>::ONE);
                    let vc = $crate::simd_primitive!($isa, $elem, splat, coeff);
                    let vs = $crate::simd_primitive!($isa, $elem, splat, sqrt_2_over_pi);

                    // x^3
                    let x2 = $crate::simd_primitive!($isa, $elem, mul, vx, vx);
                    let x3 = $crate::simd_primitive!($isa, $elem, mul, x2, vx);
                    // x + 0.044715 * x^3
                    let inner = $crate::simd_primitive!($isa, $elem, fma, vc, x3, vx);
                    // sqrt(2/pi) * inner
                    let scaled = $crate::simd_primitive!($isa, $elem, mul, vs, inner);
                    // tanh via (exp(2x)-1)/(exp(2x)+1)
                    let two = $crate::simd_primitive!($isa, $elem, splat, 2.0 as $elem);
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
            // Scalar remainder
            while i < len {
                let x = a[i];
                let inner = sqrt_2_over_pi * (x + coeff * x * x * x);
                let tanh_val = ((2.0 as $elem * inner).exp() - <$elem as Element>::ONE) / ((2.0 as $elem * inner).exp() + <$elem as Element>::ONE);
                out[i] = 0.5 as $elem * x * (<$elem as Element>::ONE + tanh_val);
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
                    let two = $crate::simd_primitive!($isa, $elem, splat, 2.0 as $elem);
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
                let x = a[i];
                let e2x = (2.0 as $elem * x).exp();
                out[i] = (e2x - <$elem as Element>::ONE) / (e2x + <$elem as Element>::ONE);
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
            while i < len { out[i] = a[i].exp(); i += 1; }
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
            let mut max_acc = unsafe { $crate::simd_primitive!($isa, $elem, splat, <$elem>::NEG_INFINITY) };
            while i + LANES <= len {
                #[allow(unused_unsafe)]
                unsafe {
                    let va = $crate::simd_primitive!($isa, $elem, load, a.as_ptr().add(i));
                    max_acc = $crate::simd_primitive!($isa, $elem, max, max_acc, va);
                }
                i += LANES;
            }
            #[allow(unused_unsafe)]
            let mut max_val = unsafe { $crate::simd_primitive!($isa, $elem, reduce_max, max_acc) };
            while i < len { max_val = max_val.max(a[i]); i += 1; }

            // Pass 2: exp(x - max) and accumulate sum
            i = 0;
            #[allow(unused_unsafe)]
            let vmax = unsafe { $crate::simd_primitive!($isa, $elem, splat, max_val) };
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
            let mut sum_val = unsafe { $crate::simd_primitive!($isa, $elem, reduce_sum, sum_acc) };
            while i < len {
                let e = (a[i] - max_val).exp();
                out[i] = e;
                sum_val += e;
                i += 1;
            }

            // Pass 3: divide by sum
            i = 0;
            let inv_sum = <$elem as Element>::ONE / sum_val;
            #[allow(unused_unsafe)]
            let vinv = unsafe { $crate::simd_primitive!($isa, $elem, splat, inv_sum) };
            while i + LANES <= len {
                #[allow(unused_unsafe)]
                unsafe {
                    let va = $crate::simd_primitive!($isa, $elem, load, out.as_ptr().add(i));
                    let res = $crate::simd_primitive!($isa, $elem, mul, va, vinv);
                    $crate::simd_primitive!($isa, $elem, store, out.as_mut_ptr().add(i), res);
                }
                i += LANES;
            }
            while i < len { out[i] *= inv_sum; i += 1; }
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
            let mut ss = unsafe { $crate::simd_primitive!($isa, $elem, reduce_sum, ss_acc) };
            while i < len { ss += a[i] * a[i]; i += 1; }

            // rms = 1/sqrt(mean + eps)
            let inv_rms = <$elem as Element>::ONE / (ss / (len as $elem) + eps).sqrt();

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
                out[i] = a[i] * inv_rms * weight[i];
                i += 1;
            }
        }

        /// layer_norm: out[i] = ((a[i] - mean) / sqrt(var + eps)) * weight[i] + bias[i]
        #[inline(always)]
        pub fn layer_norm(a: &[$elem], weight: &[$elem], bias: &[$elem], out: &mut [$elem], eps: $elem) {
            const LANES: usize = $crate::simd_primitive!($isa, $elem, lanes);
            let len = a.len();
            assert!(weight.len() == len && bias.len() == len && out.len() == len);
            let n = len as $elem;

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
            let mut sum_val = unsafe { $crate::simd_primitive!($isa, $elem, reduce_sum, sum_acc) };
            while i < len { sum_val += a[i]; i += 1; }
            let mean = sum_val / n;

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
            let mut var_val = unsafe { $crate::simd_primitive!($isa, $elem, reduce_sum, var_acc) };
            while i < len { let d = a[i] - mean; var_val += d * d; i += 1; }
            let inv_std = <$elem as Element>::ONE / (var_val / n + eps).sqrt();

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
                out[i] = (a[i] - mean) * inv_std * weight[i] + bias[i];
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
                let mut dot = unsafe { $crate::simd_primitive!($isa, $elem, reduce_sum, acc) };
                while i < k { dot += row_ptr[i] * x[i]; i += 1; }
                y[row] += dot;
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
                        use std::arch::x86_64::*;
                        // 28 accumulators: 14 rows × 2 vectors
                        let mut c_0_0 = _mm512_setzero_ps(); let mut c_0_1 = _mm512_setzero_ps();
                        let mut c_1_0 = _mm512_setzero_ps(); let mut c_1_1 = _mm512_setzero_ps();
                        let mut c_2_0 = _mm512_setzero_ps(); let mut c_2_1 = _mm512_setzero_ps();
                        let mut c_3_0 = _mm512_setzero_ps(); let mut c_3_1 = _mm512_setzero_ps();
                        let mut c_4_0 = _mm512_setzero_ps(); let mut c_4_1 = _mm512_setzero_ps();
                        let mut c_5_0 = _mm512_setzero_ps(); let mut c_5_1 = _mm512_setzero_ps();
                        let mut c_6_0 = _mm512_setzero_ps(); let mut c_6_1 = _mm512_setzero_ps();
                        let mut c_7_0 = _mm512_setzero_ps(); let mut c_7_1 = _mm512_setzero_ps();
                        let mut c_8_0 = _mm512_setzero_ps(); let mut c_8_1 = _mm512_setzero_ps();
                        let mut c_9_0 = _mm512_setzero_ps(); let mut c_9_1 = _mm512_setzero_ps();
                        let mut c_10_0 = _mm512_setzero_ps(); let mut c_10_1 = _mm512_setzero_ps();
                        let mut c_11_0 = _mm512_setzero_ps(); let mut c_11_1 = _mm512_setzero_ps();
                        let mut c_12_0 = _mm512_setzero_ps(); let mut c_12_1 = _mm512_setzero_ps();
                        let mut c_13_0 = _mm512_setzero_ps(); let mut c_13_1 = _mm512_setzero_ps();

                        let mut b_ptr = packed_b.as_ptr().add(strip_idx * k_size * TILE_N);

                        for _k in 0..k_size {
                            // Load 2 vectors from packed B
                            let vb_0 = _mm512_loadu_ps(b_ptr);
                            let vb_1 = _mm512_loadu_ps(b_ptr.add(LANES));

                            // Software prefetch 512B ahead
                            _mm_prefetch(b_ptr.add(TILE_N * 8) as *const i8, _MM_HINT_T0);

                            // Broadcast A values and FMA for each of 14 rows
                            macro_rules! fma_row {
                                ($row:expr, $c0:ident, $c1:ident) => {
                                    let va = _mm512_set1_ps(*a.as_ptr().add((m + $row) * k_size + _k));
                                    $c0 = _mm512_fmadd_ps(va, vb_0, $c0);
                                    $c1 = _mm512_fmadd_ps(va, vb_1, $c1);
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
                                _mm512_storeu_ps(c.as_mut_ptr().add((m + $row) * n_size + n), $c0);
                                _mm512_storeu_ps(c.as_mut_ptr().add((m + $row) * n_size + n + LANES), $c1);
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
