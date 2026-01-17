use crate::kernel_dispatcher::KernelFloat;

/// Cached 3D activation buffer.
#[derive(Debug, Clone)]
pub struct Activation3<T: KernelFloat> {
    data: Vec<T>,
    shape: [usize; 3],
}

impl<T: KernelFloat> Activation3<T> {
    #[inline(always)]
    pub fn new(data: Vec<T>, shape: [usize; 3]) -> Self {
        Self { data, shape }
    }

    #[inline(always)]
    pub fn data(&self) -> &[T] {
        &self.data
    }

    #[inline(always)]
    pub fn shape(&self) -> [usize; 3] {
        self.shape
    }
}

/// Cached 2D activation buffer.
#[derive(Debug, Clone)]
pub struct Activation2<T: KernelFloat> {
    data: Vec<T>,
    shape: [usize; 2],
}

impl<T: KernelFloat> Activation2<T> {
    #[inline(always)]
    pub fn new(data: Vec<T>, shape: [usize; 2]) -> Self {
        Self { data, shape }
    }

    #[inline(always)]
    pub fn data(&self) -> &[T] {
        &self.data
    }

    #[inline(always)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

/// Shared activations cache for draft-verify optimization.
#[derive(Debug, Clone)]
pub struct SharedActivations<T: KernelFloat> {
    layer_hidden: Vec<Option<Activation3<T>>>,
    layer_logits: Vec<Option<Activation3<T>>>,
    layer_confidence: Vec<Option<Activation2<T>>>,
    num_layers: usize,
}

impl<T: KernelFloat> SharedActivations<T> {
    /// Create a new shared activations cache.
    #[inline(always)]
    pub fn new(num_layers: usize) -> Result<Self, &'static str> {
        if num_layers == 0 {
            return Err("num_layers must be > 0");
        }
        Ok(Self {
            layer_hidden: vec![None; num_layers],
            layer_logits: vec![None; num_layers],
            layer_confidence: vec![None; num_layers],
            num_layers,
        })
    }

    /// Store hidden states for a layer.
    #[inline(always)]
    pub fn store_hidden(
        &mut self,
        layer_idx: usize,
        hidden: &[T],
        batch: usize,
        seq_len: usize,
    ) -> Result<(), &'static str> {
        if layer_idx >= self.num_layers {
            return Err("layer_idx out of range");
        }
        if batch == 0 || seq_len == 0 {
            return Err("batch and seq_len must be > 0");
        }
        if hidden.len() % (batch * seq_len) != 0 {
            return Err("hidden length must be multiple of batch * seq_len");
        }

        let hidden_dim = hidden.len() / (batch * seq_len);
        self.layer_hidden[layer_idx] = Some(Activation3::new(
            hidden.to_vec(),
            [batch, seq_len, hidden_dim],
        ));
        Ok(())
    }

    /// Store logits and confidence for a layer.
    #[inline(always)]
    pub fn store_exit_output(
        &mut self,
        layer_idx: usize,
        logits: &[T],
        confidence: &[T],
        batch: usize,
        seq_len: usize,
    ) -> Result<(), &'static str> {
        if layer_idx >= self.num_layers {
            return Err("layer_idx out of range");
        }
        if batch == 0 || seq_len == 0 {
            return Err("batch and seq_len must be > 0");
        }
        let positions = batch * seq_len;
        if positions == 0 {
            return Err("positions must be > 0");
        }
        if confidence.len() != positions {
            return Err("confidence length mismatch");
        }
        if logits.len() % positions != 0 {
            return Err("logits length must be multiple of batch * seq_len");
        }

        let vocab_size = logits.len() / positions;
        self.layer_logits[layer_idx] = Some(Activation3::new(
            logits.to_vec(),
            [batch, seq_len, vocab_size],
        ));
        self.layer_confidence[layer_idx] = Some(Activation2::new(
            confidence.to_vec(),
            [batch, seq_len],
        ));
        Ok(())
    }

    /// Get cached hidden states.
    #[inline(always)]
    pub fn get_hidden(&self, layer_idx: usize) -> Option<&Activation3<T>> {
        self.layer_hidden.get(layer_idx)?.as_ref()
    }

    /// Get cached logits.
    #[inline(always)]
    pub fn get_logits(&self, layer_idx: usize) -> Option<&Activation3<T>> {
        self.layer_logits.get(layer_idx)?.as_ref()
    }

    /// Get cached confidence.
    #[inline(always)]
    pub fn get_confidence(&self, layer_idx: usize) -> Option<&Activation2<T>> {
        self.layer_confidence.get(layer_idx)?.as_ref()
    }

    /// Clear all cached activations.
    #[inline(always)]
    pub fn clear(&mut self) {
        for i in 0..self.num_layers {
            self.layer_hidden[i] = None;
            self.layer_logits[i] = None;
            self.layer_confidence[i] = None;
        }
    }

    /// Clear activations from a specific layer onwards (for partial recompute).
    #[inline(always)]
    pub fn clear_from(&mut self, start_layer: usize) {
        for i in start_layer..self.num_layers {
            self.layer_hidden[i] = None;
            self.layer_logits[i] = None;
            self.layer_confidence[i] = None;
        }
    }
}
