//! Shared memory communication for single-node multi-GPU ring attention.

use std::sync::Arc;
use std::thread;

use crossbeam_channel::{bounded, Receiver, Sender};

use super::traits::{CommError, CommResult, Communicator, TensorMessage};

/// A group of shared memory communicators for ring communication.
pub struct SharedMemoryGroup {
    /// Communicators for each rank.
    communicators: Vec<SharedMemoryComm>,
}

impl SharedMemoryGroup {
    /// Create a new group of communicators with the specified world size.
    pub fn new(world_size: usize) -> CommResult<Self> {
        if world_size == 0 {
            return Err(CommError::InvalidConfig(
                "world_size must be > 0".to_string(),
            ));
        }

        // Create channels for ring communication
        // Each rank sends to next and receives from previous
        let mut send_channels: Vec<Sender<TensorMessage>> = Vec::with_capacity(world_size);
        let mut recv_channels: Vec<Receiver<TensorMessage>> = Vec::with_capacity(world_size);

        for _ in 0..world_size {
            let (tx, rx) = bounded(2); // Small buffer for double-buffering
            send_channels.push(tx);
            recv_channels.push(rx);
        }

        // Create barrier channels
        let mut barrier_send: Vec<Sender<()>> = Vec::with_capacity(world_size);
        let mut barrier_recv: Vec<Receiver<()>> = Vec::with_capacity(world_size);

        for _ in 0..world_size {
            let (tx, rx) = bounded(world_size);
            barrier_send.push(tx);
            barrier_recv.push(rx);
        }

        // Build communicators
        // Rank i sends to channel i (received by rank (i+1) % world_size)
        // Rank i receives from channel (i + world_size - 1) % world_size
        let mut communicators = Vec::with_capacity(world_size);

        for rank in 0..world_size {
            let send_idx = rank;
            let recv_idx = (rank + world_size - 1) % world_size;

            let comm = SharedMemoryComm {
                rank,
                world_size,
                send_tx: send_channels[send_idx].clone(),
                recv_rx: recv_channels[recv_idx].clone(),
                barrier_txs: barrier_send.clone(),
                barrier_rx: barrier_recv[rank].clone(),
            };
            communicators.push(comm);
        }

        Ok(Self { communicators })
    }

    /// Get the communicator for a specific rank.
    pub fn get(&self, rank: usize) -> Option<&SharedMemoryComm> {
        self.communicators.get(rank)
    }

    /// Take ownership of the communicator for a specific rank.
    /// Each communicator can only be taken once.
    pub fn take(&mut self, rank: usize) -> Option<SharedMemoryComm> {
        if rank < self.communicators.len() {
            // Replace with a dummy that will fail on use
            let dummy = SharedMemoryComm {
                rank: usize::MAX,
                world_size: 0,
                send_tx: self.communicators[rank].send_tx.clone(),
                recv_rx: self.communicators[rank].recv_rx.clone(),
                barrier_txs: Vec::new(),
                barrier_rx: self.communicators[rank].barrier_rx.clone(),
            };
            Some(std::mem::replace(&mut self.communicators[rank], dummy))
        } else {
            None
        }
    }

    /// Get the world size.
    pub fn world_size(&self) -> usize {
        self.communicators.len()
    }
}

/// Shared memory communicator for a single rank.
pub struct SharedMemoryComm {
    rank: usize,
    world_size: usize,
    send_tx: Sender<TensorMessage>,
    recv_rx: Receiver<TensorMessage>,
    barrier_txs: Vec<Sender<()>>,
    barrier_rx: Receiver<()>,
}

impl Clone for SharedMemoryComm {
    fn clone(&self) -> Self {
        Self {
            rank: self.rank,
            world_size: self.world_size,
            send_tx: self.send_tx.clone(),
            recv_rx: self.recv_rx.clone(),
            barrier_txs: self.barrier_txs.clone(),
            barrier_rx: self.barrier_rx.clone(),
        }
    }
}

impl Communicator for SharedMemoryComm {
    type Data = TensorMessage;

    fn rank(&self) -> usize {
        self.rank
    }

    fn world_size(&self) -> usize {
        self.world_size
    }

    fn send(&self, data: &Self::Data) -> CommResult<()> {
        self.send_tx
            .send(data.clone())
            .map_err(|_| CommError::Disconnected)
    }

    fn recv(&self) -> CommResult<Self::Data> {
        self.recv_rx.recv().map_err(|_| CommError::Disconnected)
    }

    fn send_recv(&self, send_data: &Self::Data) -> CommResult<Self::Data> {
        // For shared memory, we can do true simultaneous send/recv
        // by spawning a thread for send while we recv
        let send_tx = self.send_tx.clone();
        let data_clone = send_data.clone();

        let send_handle = thread::spawn(move || {
            send_tx
                .send(data_clone)
                .map_err(|_| CommError::Disconnected)
        });

        let recv_result = self.recv();

        // Wait for send to complete
        send_handle
            .join()
            .map_err(|_| CommError::SendFailed("Send thread panicked".to_string()))??;

        recv_result
    }

    fn barrier(&self) -> CommResult<()> {
        // Signal all other ranks that we've reached the barrier
        for (i, tx) in self.barrier_txs.iter().enumerate() {
            if i != self.rank {
                tx.send(()).map_err(|_| CommError::Disconnected)?;
            }
        }

        // Wait for signals from all other ranks
        for _ in 0..(self.world_size - 1) {
            self.barrier_rx.recv().map_err(|_| CommError::Disconnected)?;
        }

        Ok(())
    }
}

/// Helper to run ring communication across threads.
pub fn run_ring<F, R>(world_size: usize, f: F) -> CommResult<Vec<R>>
where
    F: Fn(SharedMemoryComm) -> R + Send + Sync + Clone + 'static,
    R: Send + 'static,
{
    let mut group = SharedMemoryGroup::new(world_size)?;
    let f = Arc::new(f);

    let handles: Vec<_> = (0..world_size)
        .map(|rank| {
            let comm = group.take(rank).unwrap();
            let f = Arc::clone(&f);
            thread::spawn(move || f(comm))
        })
        .collect();

    let results: Vec<R> = handles
        .into_iter()
        .map(|h| h.join().expect("Thread panicked"))
        .collect();

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_communication() {
        let world_size = 4;

        let results = run_ring(world_size, |comm| {
            let rank = comm.rank();
            let msg = TensorMessage::new(vec![rank as f32; 4], vec![4]);

            // Each rank sends its own data and receives from prev rank
            let received = comm.send_recv(&msg).unwrap();

            // We should receive from (rank - 1 + world_size) % world_size
            let expected_sender = comm.prev_rank();
            assert_eq!(received.data[0] as usize, expected_sender);

            received.data[0] as usize
        })
        .unwrap();

        // Each rank should have received from its predecessor
        for (rank, received_from) in results.iter().enumerate() {
            let expected = (rank + world_size - 1) % world_size;
            assert_eq!(*received_from, expected);
        }
    }

    #[test]
    fn test_barrier() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let world_size = 4;
        let counter = Arc::new(AtomicUsize::new(0));

        let _ = run_ring(world_size, {
            let counter = Arc::clone(&counter);
            move |comm| {
                // Increment counter
                counter.fetch_add(1, Ordering::SeqCst);

                // Wait at barrier
                comm.barrier().unwrap();

                // After barrier, all ranks should have incremented
                assert_eq!(counter.load(Ordering::SeqCst), world_size);
            }
        })
        .unwrap();
    }
}
