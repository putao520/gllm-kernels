//! Shared memory communication for single-node distributed inference.
//!
//! Zero-cost abstraction: raw bytes + shape, no wrapper types.

use std::thread;

use crossbeam_channel::{unbounded, Receiver, Sender};

use super::traits::{CommError, CommResult, Communicator};

/// Raw tensor message for channel communication.
#[derive(Clone)]
struct RawMessage {
    data: Vec<u8>,
    shape: Vec<usize>,
    dtype: u8,
}

/// A group of shared memory communicators for ring communication.
pub struct SharedMemoryGroup {
    comms: Vec<SharedMemoryComm>,
}

impl SharedMemoryGroup {
    /// Create a new group of communicators with the specified world size.
    pub fn new(world_size: usize) -> CommResult<Self> {
        if world_size == 0 {
            return Err(CommError::InvalidConfig("world_size must be > 0".into()));
        }

        let (send_txs, recv_rxs) = build_ring_channels(world_size);
        let (barrier_txs, barrier_rxs) = build_barrier_channels(world_size);

        let mut comms = Vec::with_capacity(world_size);
        for rank in 0..world_size {
            let recv_idx = (rank + world_size - 1) % world_size;
            comms.push(SharedMemoryComm {
                rank,
                world_size,
                send_tx: send_txs[rank].clone(),
                recv_rx: recv_rxs[recv_idx].clone(),
                barrier_txs: barrier_txs.clone(),
                barrier_rx: barrier_rxs[rank].clone(),
            });
        }

        Ok(Self { comms })
    }

    /// Consume the group and return communicators for each rank.
    pub fn into_comms(self) -> Vec<SharedMemoryComm> {
        self.comms
    }
}

/// Shared memory communicator for a single rank.
pub struct SharedMemoryComm {
    rank: usize,
    world_size: usize,
    send_tx: Sender<RawMessage>,
    recv_rx: Receiver<RawMessage>,
    barrier_txs: Vec<Sender<()>>,
    barrier_rx: Receiver<()>,
}

impl Communicator for SharedMemoryComm {
    fn rank(&self) -> usize {
        self.rank
    }

    fn world_size(&self) -> usize {
        self.world_size
    }

    fn send_raw(&self, data: &[u8], shape: &[usize], dtype: u8) -> CommResult<()> {
        let msg = RawMessage {
            data: data.to_vec(),
            shape: shape.to_vec(),
            dtype,
        };
        self.send_tx.send(msg).map_err(|_| CommError::Disconnected)
    }

    fn recv_raw(&self) -> CommResult<(Vec<u8>, Vec<usize>, u8)> {
        let msg = self.recv_rx.recv().map_err(|_| CommError::Disconnected)?;
        Ok((msg.data, msg.shape, msg.dtype))
    }

    fn barrier(&self) -> CommResult<()> {
        // Signal all other ranks
        for (i, tx) in self.barrier_txs.iter().enumerate() {
            if i != self.rank {
                tx.send(()).map_err(|_| CommError::Disconnected)?;
            }
        }

        // Wait for all other ranks
        for _ in 0..(self.world_size - 1) {
            self.barrier_rx.recv().map_err(|_| CommError::Disconnected)?;
        }

        Ok(())
    }
}

// Allow send_recv to be parallelized
impl SharedMemoryComm {
    /// Optimized send_recv using parallel threads.
    pub fn send_recv_parallel(&self, data: &[u8], shape: &[usize], dtype: u8) -> CommResult<(Vec<u8>, Vec<usize>, u8)> {
        let send_tx = self.send_tx.clone();
        let msg = RawMessage {
            data: data.to_vec(),
            shape: shape.to_vec(),
            dtype,
        };

        let handle = thread::spawn(move || {
            send_tx.send(msg).map_err(|_| CommError::Disconnected)
        });

        let recv_result = self.recv_raw();

        match handle.join() {
            Ok(send_result) => send_result?,
            Err(_) => return Err(CommError::SendFailed("send thread panicked".into())),
        }

        recv_result
    }
}

fn build_ring_channels(world_size: usize) -> (Vec<Sender<RawMessage>>, Vec<Receiver<RawMessage>>) {
    let mut txs = Vec::with_capacity(world_size);
    let mut rxs = Vec::with_capacity(world_size);

    for _ in 0..world_size {
        let (tx, rx) = unbounded();
        txs.push(tx);
        rxs.push(rx);
    }

    (txs, rxs)
}

fn build_barrier_channels(world_size: usize) -> (Vec<Sender<()>>, Vec<Receiver<()>>) {
    let mut txs = Vec::with_capacity(world_size);
    let mut rxs = Vec::with_capacity(world_size);

    for _ in 0..world_size {
        let (tx, rx) = unbounded();
        txs.push(tx);
        rxs.push(rx);
    }

    (txs, rxs)
}
