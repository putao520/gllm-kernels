//! Integration tests for communication backends.

use std::sync::Arc;

use gllm_kernels::comm::{
    run_ring, CommResult, Communicator, SharedMemoryComm, SharedMemoryGroup, TensorMessage,
};

#[test]
fn test_shared_memory_group_creation() {
    let group = SharedMemoryGroup::new(4).unwrap();
    assert_eq!(group.world_size(), 4);

    // Each rank should be accessible
    for rank in 0..4 {
        let comm = group.get(rank).unwrap();
        assert_eq!(comm.rank(), rank);
        assert_eq!(comm.world_size(), 4);
    }
}

#[test]
fn test_shared_memory_ring_simple() {
    let world_size = 4;

    let results: Vec<usize> = run_ring(world_size, |comm| {
        let rank = comm.rank();
        let msg = TensorMessage::new(vec![rank as f32], vec![1]);

        // Send our rank, receive from previous
        let received = comm.send_recv(&msg).unwrap();
        received.data[0] as usize
    })
    .unwrap();

    // Each rank should receive from its predecessor
    for (rank, &received) in results.iter().enumerate() {
        let expected = (rank + world_size - 1) % world_size;
        assert_eq!(received, expected, "Rank {} should receive from {}", rank, expected);
    }
}

#[test]
fn test_shared_memory_full_ring_rotation() {
    let world_size = 4;

    // Each rank will rotate its data through the entire ring
    let results: Vec<Vec<usize>> = run_ring(world_size, |comm| {
        let rank = comm.rank();
        let mut received_from = Vec::new();

        let mut current = TensorMessage::new(vec![rank as f32], vec![1]);

        for _ in 0..world_size {
            received_from.push(current.data[0] as usize);
            current = comm.send_recv(&current).unwrap();
        }

        received_from
    })
    .unwrap();

    // Verify each rank saw all data rotate through
    for (rank, received) in results.iter().enumerate() {
        assert_eq!(received.len(), world_size);
        // First value should be our own rank (we haven't sent yet)
        assert_eq!(received[0], rank);
    }
}

#[test]
fn test_shared_memory_large_tensor() {
    let world_size = 2;
    let tensor_size = 1024 * 1024; // 1M floats

    let results: Vec<bool> = run_ring(world_size, move |comm| {
        let rank = comm.rank();
        let data: Vec<f32> = (0..tensor_size).map(|i| (rank * tensor_size + i) as f32).collect();
        let msg = TensorMessage::new(data, vec![1024, 1024]);

        let received = comm.send_recv(&msg).unwrap();

        // Verify we received from the previous rank
        let prev_rank = (rank + world_size - 1) % world_size;
        let expected_first = (prev_rank * tensor_size) as f32;
        received.data[0] == expected_first && received.shape == vec![1024, 1024]
    })
    .unwrap();

    assert!(results.iter().all(|&ok| ok));
}

#[test]
fn test_barrier_synchronization() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;
    use std::time::Duration;

    let world_size = 4;
    let counter = Arc::new(AtomicUsize::new(0));

    let _ = run_ring(world_size, {
        let counter = Arc::clone(&counter);
        move |comm| {
            let rank = comm.rank();

            // Stagger the increments
            thread::sleep(Duration::from_millis(rank as u64 * 10));
            counter.fetch_add(1, Ordering::SeqCst);

            // After barrier, all should have incremented
            comm.barrier().unwrap();
            let count = counter.load(Ordering::SeqCst);
            assert_eq!(count, world_size, "After barrier, count should be {}", world_size);
        }
    })
    .unwrap();
}

#[test]
fn test_tensor_message_serialization() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = vec![2, 3];
    let msg = TensorMessage::new(data.clone(), shape.clone());

    assert_eq!(msg.numel(), 6);
    assert_eq!(msg.data, data);
    assert_eq!(msg.shape, shape);
}

#[test]
fn test_communicator_next_prev_rank() {
    let group = SharedMemoryGroup::new(4).unwrap();

    let comm0 = group.get(0).unwrap();
    assert_eq!(comm0.next_rank(), 1);
    assert_eq!(comm0.prev_rank(), 3);

    let comm3 = group.get(3).unwrap();
    assert_eq!(comm3.next_rank(), 0);
    assert_eq!(comm3.prev_rank(), 2);
}

#[test]
fn test_world_size_one() {
    // Single node should still work
    let group = SharedMemoryGroup::new(1).unwrap();
    let comm = group.get(0).unwrap();
    assert_eq!(comm.rank(), 0);
    assert_eq!(comm.world_size(), 1);
    assert_eq!(comm.next_rank(), 0);
    assert_eq!(comm.prev_rank(), 0);
}

#[test]
fn test_invalid_world_size() {
    let result = SharedMemoryGroup::new(0);
    assert!(result.is_err());
}

// TCP tests require network coordination, so we only test serialization
mod tcp_tests {
    use gllm_kernels::comm::TensorMessage;

    #[test]
    fn test_tcp_message_roundtrip() {
        use std::io::{Read, Write};

        let msg = TensorMessage::new(vec![1.5, 2.5, 3.5], vec![3]);

        // Simple serialization roundtrip using the same logic as TcpComm
        let mut buf = Vec::new();

        // Serialize
        let shape_len = msg.shape.len() as u32;
        buf.extend_from_slice(&shape_len.to_le_bytes());
        for &dim in &msg.shape {
            buf.extend_from_slice(&(dim as u64).to_le_bytes());
        }
        let data_len = msg.data.len() as u64;
        buf.extend_from_slice(&data_len.to_le_bytes());
        for &val in &msg.data {
            buf.extend_from_slice(&val.to_le_bytes());
        }

        // Deserialize
        let mut pos = 0;
        let decoded_shape_len = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;

        let mut decoded_shape = Vec::new();
        for _ in 0..decoded_shape_len {
            let dim = u64::from_le_bytes(buf[pos..pos + 8].try_into().unwrap()) as usize;
            decoded_shape.push(dim);
            pos += 8;
        }

        let decoded_data_len = u64::from_le_bytes(buf[pos..pos + 8].try_into().unwrap()) as usize;
        pos += 8;

        let mut decoded_data = Vec::new();
        for _ in 0..decoded_data_len {
            let val = f32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap());
            decoded_data.push(val);
            pos += 4;
        }

        assert_eq!(decoded_shape, msg.shape);
        assert_eq!(decoded_data, msg.data);
    }
}
