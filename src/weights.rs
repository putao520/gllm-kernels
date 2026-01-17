//! Zero-cost weight containers for neural network parameters.
//!
//! These types are designed to be thin wrappers around `Vec<f32>` with shape metadata,
//! providing zero-cost abstractions through `#[inline(always)]` accessors.
//!
//! # Design Philosophy
//!
//! Unlike Burn's `Tensor<B, N>` which uses generic backends and runtime dispatch,
//! these containers use direct slice access for zero overhead:
//!
//! ```ignore
//! // Zero-cost: compiles to direct pointer access
//! let slice = weight.as_slice();
//! linear_forward(input, slice, ...);
//! ```

/// A 2D weight matrix (e.g., Linear layer weights).
///
/// Stored in row-major order: `[out_features, in_features]`
#[derive(Clone, Debug)]
pub struct WeightMatrix {
    /// Raw weight data in row-major order
    pub data: Vec<f32>,
    /// Number of output features (rows)
    pub rows: usize,
    /// Number of input features (columns)
    pub cols: usize,
}

impl WeightMatrix {
    /// Create a new weight matrix with given dimensions.
    #[inline(always)]
    pub fn new(data: Vec<f32>, rows: usize, cols: usize) -> Self {
        debug_assert_eq!(data.len(), rows * cols, "Data length must match rows * cols");
        Self { data, rows, cols }
    }

    /// Create a zero-initialized weight matrix.
    #[inline(always)]
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    /// Get immutable slice access (zero-cost).
    #[inline(always)]
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Get mutable slice access (zero-cost).
    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Get a specific row as a slice.
    #[inline(always)]
    pub fn row(&self, idx: usize) -> &[f32] {
        let start = idx * self.cols;
        &self.data[start..start + self.cols]
    }

    /// Get the total number of elements.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the matrix is empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Transpose the matrix (creates a new matrix).
    pub fn transpose(&self) -> Self {
        let mut transposed = vec![0.0; self.data.len()];
        for i in 0..self.rows {
            for j in 0..self.cols {
                transposed[j * self.rows + i] = self.data[i * self.cols + j];
            }
        }
        Self {
            data: transposed,
            rows: self.cols,
            cols: self.rows,
        }
    }
}

/// A 3D weight tensor (e.g., multi-head attention QKV weights).
///
/// Stored in row-major order: `[dim0, dim1, dim2]`
#[derive(Clone, Debug)]
pub struct Weight3D {
    /// Raw weight data
    pub data: Vec<f32>,
    /// First dimension (e.g., num_heads)
    pub dim0: usize,
    /// Second dimension (e.g., head_dim)
    pub dim1: usize,
    /// Third dimension (e.g., hidden_size)
    pub dim2: usize,
}

impl Weight3D {
    /// Create a new 3D weight tensor.
    #[inline(always)]
    pub fn new(data: Vec<f32>, dim0: usize, dim1: usize, dim2: usize) -> Self {
        debug_assert_eq!(
            data.len(),
            dim0 * dim1 * dim2,
            "Data length must match dim0 * dim1 * dim2"
        );
        Self { data, dim0, dim1, dim2 }
    }

    /// Create a zero-initialized 3D tensor.
    #[inline(always)]
    pub fn zeros(dim0: usize, dim1: usize, dim2: usize) -> Self {
        Self {
            data: vec![0.0; dim0 * dim1 * dim2],
            dim0,
            dim1,
            dim2,
        }
    }

    /// Get immutable slice access (zero-cost).
    #[inline(always)]
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Get mutable slice access (zero-cost).
    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Get a 2D slice at index `i` along dim0.
    #[inline(always)]
    pub fn slice_2d(&self, idx: usize) -> &[f32] {
        let size = self.dim1 * self.dim2;
        let start = idx * size;
        &self.data[start..start + size]
    }

    /// Get the total number of elements.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the tensor is empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// A 1D weight vector (e.g., LayerNorm gamma/beta, bias).
#[derive(Clone, Debug)]
pub struct WeightVector {
    /// Raw weight data
    pub data: Vec<f32>,
}

impl WeightVector {
    /// Create a new weight vector.
    #[inline(always)]
    pub fn new(data: Vec<f32>) -> Self {
        Self { data }
    }

    /// Create a zero-initialized weight vector.
    #[inline(always)]
    pub fn zeros(len: usize) -> Self {
        Self {
            data: vec![0.0; len],
        }
    }

    /// Create a ones-initialized weight vector (useful for scale parameters).
    #[inline(always)]
    pub fn ones(len: usize) -> Self {
        Self {
            data: vec![1.0; len],
        }
    }

    /// Get immutable slice access (zero-cost).
    #[inline(always)]
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Get mutable slice access (zero-cost).
    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Get the length of the vector.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the vector is empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// A 4D weight tensor (e.g., convolution kernels, batched attention weights).
///
/// Stored in row-major order: `[dim0, dim1, dim2, dim3]`
#[derive(Clone, Debug)]
pub struct Weight4D {
    /// Raw weight data
    pub data: Vec<f32>,
    /// Batch or first dimension
    pub dim0: usize,
    /// Second dimension
    pub dim1: usize,
    /// Third dimension
    pub dim2: usize,
    /// Fourth dimension
    pub dim3: usize,
}

impl Weight4D {
    /// Create a new 4D weight tensor.
    #[inline(always)]
    pub fn new(data: Vec<f32>, dim0: usize, dim1: usize, dim2: usize, dim3: usize) -> Self {
        debug_assert_eq!(
            data.len(),
            dim0 * dim1 * dim2 * dim3,
            "Data length must match dim0 * dim1 * dim2 * dim3"
        );
        Self { data, dim0, dim1, dim2, dim3 }
    }

    /// Create a zero-initialized 4D tensor.
    #[inline(always)]
    pub fn zeros(dim0: usize, dim1: usize, dim2: usize, dim3: usize) -> Self {
        Self {
            data: vec![0.0; dim0 * dim1 * dim2 * dim3],
            dim0,
            dim1,
            dim2,
            dim3,
        }
    }

    /// Get immutable slice access (zero-cost).
    #[inline(always)]
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Get mutable slice access (zero-cost).
    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Get a 3D slice at index `i` along dim0.
    #[inline(always)]
    pub fn slice_3d(&self, idx: usize) -> &[f32] {
        let size = self.dim1 * self.dim2 * self.dim3;
        let start = idx * size;
        &self.data[start..start + size]
    }

    /// Get the total number of elements.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the tensor is empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_matrix_basic() {
        let w = WeightMatrix::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        assert_eq!(w.rows, 2);
        assert_eq!(w.cols, 3);
        assert_eq!(w.len(), 6);
        assert_eq!(w.row(0), &[1.0, 2.0, 3.0]);
        assert_eq!(w.row(1), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_weight_matrix_transpose() {
        let w = WeightMatrix::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let t = w.transpose();
        assert_eq!(t.rows, 3);
        assert_eq!(t.cols, 2);
        assert_eq!(t.as_slice(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_weight_vector() {
        let v = WeightVector::ones(4);
        assert_eq!(v.len(), 4);
        assert_eq!(v.as_slice(), &[1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_weight_3d() {
        let w = Weight3D::zeros(2, 3, 4);
        assert_eq!(w.len(), 24);
        assert_eq!(w.slice_2d(0).len(), 12);
    }

    #[test]
    fn test_weight_4d() {
        let w = Weight4D::zeros(2, 3, 4, 5);
        assert_eq!(w.len(), 120);
        assert_eq!(w.slice_3d(0).len(), 60);
    }
}
