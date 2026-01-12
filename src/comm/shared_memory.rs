//! Shared memory communication for single-node ring attention.

use std::thread;

use burn::tensor::TensorData;
use crossbeam_channel::{unbounded, Receiver, Sender};

use super::traits::{CommError, CommResult, Communicator};

/// A group of shared memory communicators for ring communication.
pub struct SharedMemoryGroup {
    comms: Vec<SharedMemoryComm>,
}

impl SharedMemoryGroup {
    /// Create a new group of communicators with the specified world size.
    pub fn new(world_size: usize) -> CommResult<Self> {
        if world_size == 0 {
            return Err(CommError::InvalidConfig(
                "world_size must be > 0".to_string(),
            ));
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
    send_tx: Sender<TensorData>,
    recv_rx: Receiver<TensorData>,
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

    fn send(&self, data: &TensorData) -> CommResult<()> {
        self.send_tx
            .send(data.clone())
            .map_err(|_| CommError::Disconnected)
    }

    fn recv(&self) -> CommResult<TensorData> {
        self.recv_rx.recv().map_err(|_| CommError::Disconnected)
    }

    fn send_recv(&self, send_data: &TensorData) -> CommResult<TensorData> {
        let send_tx = self.send_tx.clone();
        let data = send_data.clone();

        let handle = thread::spawn(move || {
            send_tx
                .send(data)
                .map_err(|_| CommError::Disconnected)
        });

        let recv_result = self.recv();
        match handle.join() {
            Ok(send_result) => send_result?,
            Err(_) => {
                return Err(CommError::SendFailed(
                    "send thread panicked".to_string(),
                ))
            }
        }

        recv_result
    }

    fn barrier(&self) -> CommResult<()> {
        if self.world_size <= 1 {
            return Ok(());
        }

        for (idx, tx) in self.barrier_txs.iter().enumerate() {
            if idx != self.rank {
                tx.send(()).map_err(|_| CommError::Disconnected)?;
            }
        }

        for _ in 0..(self.world_size - 1) {
            self.barrier_rx.recv().map_err(|_| CommError::Disconnected)?;
        }

        Ok(())
    }
}

fn build_ring_channels(
    world_size: usize,
) -> (Vec<Sender<TensorData>>, Vec<Receiver<TensorData>>) {
    let mut send_txs = Vec::with_capacity(world_size);
    let mut recv_rxs = Vec::with_capacity(world_size);

    for _ in 0..world_size {
        let (tx, rx) = unbounded();
        send_txs.push(tx);
        recv_rxs.push(rx);
    }

    (send_txs, recv_rxs)
}

fn build_barrier_channels(world_size: usize) -> (Vec<Sender<()>>, Vec<Receiver<()>>) {
    let mut send_txs = Vec::with_capacity(world_size);
    let mut recv_rxs = Vec::with_capacity(world_size);

    for _ in 0..world_size {
        let (tx, rx) = unbounded();
        send_txs.push(tx);
        recv_rxs.push(rx);
    }

    (send_txs, recv_rxs)
}
