//! Integration tests for shared memory communication.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use gllm_kernels::comm::{Communicator, SharedMemoryGroup};

#[test]
fn test_shared_memory_group_properties() {
    let group = SharedMemoryGroup::new(4).expect("group creation");
    let comms = group.into_comms();

    assert_eq!(comms.len(), 4);
    for (rank, comm) in comms.iter().enumerate() {
        assert_eq!(comm.rank(), rank);
        assert_eq!(comm.world_size(), 4);
    }
}

#[test]
fn test_shared_memory_send_recv_ring() {
    let world_size = 4;
    let group = SharedMemoryGroup::new(world_size).expect("group creation");
    let comms = group.into_comms();

    let handles: Vec<_> = comms
        .into_iter()
        .map(|comm| {
            thread::spawn(move || {
                let rank = comm.rank();
                let data = vec![rank as f32];
                let shape = vec![1usize];
                let (recv_data, _recv_shape) = comm.send_recv(&data, &shape).expect("send_recv");
                (rank, recv_data[0] as usize)
            })
        })
        .collect();

    let mut received_from = vec![0usize; world_size];
    for handle in handles {
        let (rank, recv_rank) = handle.join().expect("thread join");
        received_from[rank] = recv_rank;
    }

    for rank in 0..world_size {
        let expected = (rank + world_size - 1) % world_size;
        assert_eq!(received_from[rank], expected);
    }
}

#[test]
fn test_shared_memory_barrier() {
    let world_size = 4;
    let group = SharedMemoryGroup::new(world_size).expect("group creation");
    let comms = group.into_comms();
    let counter = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = comms
        .into_iter()
        .map(|comm| {
            let counter = Arc::clone(&counter);
            thread::spawn(move || {
                let rank = comm.rank();
                thread::sleep(Duration::from_millis(rank as u64 * 5));
                counter.fetch_add(1, Ordering::SeqCst);
                comm.barrier().expect("barrier");
                counter.load(Ordering::SeqCst)
            })
        })
        .collect();

    for handle in handles {
        let count = handle.join().expect("thread join");
        assert_eq!(count, world_size);
    }
}

#[test]
fn test_shared_memory_invalid_world_size() {
    let result = SharedMemoryGroup::new(0);
    assert!(result.is_err());
}

#[cfg(feature = "nccl")]
mod nccl_tests {
    use gllm_kernels::comm::NcclComm;

    #[test]
    fn test_nccl_id_creation() {
        let id = NcclComm::create_id();
        assert!(id.is_ok());
    }
}
