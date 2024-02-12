// tests/matmul_tests.rs
use llama2_rs::{matmul, QuantizedTensor};


#[cfg(test)]
mod matmul_tests {
    use super::*;

    #[test]
    fn test_basic_multiplication() {
        let x = QuantizedTensor {
            q: vec![1, 2, 3],
            s: vec![1.0],
        };
        let weights = vec![QuantizedTensor {
            q: vec![4, 5, 6, 7, 8, 9],
            s: vec![1.0],
        }];
        let mut xout = vec![0.0; 2];
        let n = 3;
        let d = 2;
        let layer_index = 0;
        let gs = 64;

        matmul(&mut xout, &x, &weights, n, d, layer_index, gs);

        assert_eq!(xout, vec![32.0, 50.0], "The basic multiplication did not match expected output.");
    }
    #[test]
    fn test_different_scales() {
        let x = QuantizedTensor {
            q: vec![2, 4, 6],
            s: vec![0.5],
        };
        let weights = vec![QuantizedTensor {
            q: vec![3, 6, 9, 12, 15, 18],
            s: vec![2.0],
        }];
        let mut xout = vec![0.0; 2];
        let n = 3;
        let d = 2;
        let layer_index = 0;
        let gs = 64;

        matmul(&mut xout, &x, &weights, n, d, layer_index, gs);

        assert_eq!(xout, vec![84.0, 192.0], "Different scales did not produce the expected output.");
    }

    #[test]
    fn test_larger_input() {
        let x = QuantizedTensor {
            q: vec![1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23],
            s: vec![1.0],
        };
        let weights = vec![QuantizedTensor {
            q: vec![2; 24], // 24 elements, all 2
            s: vec![0.5],
        }];
        let mut xout = vec![0.0; 2];
        let n = 12;
        let d = 2;
        let layer_index = 0;
        let gs = 64;

        matmul(&mut xout, &x, &weights, n, d, layer_index, gs);

        // Expected output calculation: Each element of x multiplied by 2 (weight value), 
        // then by 0.5 (weight scale), summed up and scaled by 1.0 (input scale).
        // For simplicity, assuming each row of weights contributes equally.
        assert_eq!(xout, vec![144.0, 144.0], "Larger input did not produce the expected output.");
    }

    #[test]
    fn test_zero_weights() {
        let x = QuantizedTensor {
            q: vec![10, 20, 30],
            s: vec![1.0],
        };
        let weights = vec![QuantizedTensor {
            q: vec![0; 6], // Zero weights
            s: vec![1.0],
        }];
        let mut xout = vec![0.0; 2];
        let n = 3;
        let d = 2;
        let layer_index = 0;
        let gs = 64;

        matmul(&mut xout, &x, &weights, n, d, layer_index, gs);

        assert_eq!(xout, vec![0.0, 0.0], "Zero weights did not produce the expected output.");
    }
    fn test_large_and_different_scales() {
        let x = QuantizedTensor {
            q: (1..=100).map(|x| (x % 127) as i8).collect(),
            s: vec![0.75],
        };
        let weights = vec![QuantizedTensor {
            q: (1..=600).map(|x| (x % 127) as i8).collect(),
            s: vec![1.25],
        }];
        let mut xout = vec![0.0; 6];
        let n = 100;
        let d = 6;
        let layer_index = 0;
        let gs = 64;

        matmul(&mut xout, &x, &weights, n, d, layer_index, gs);

        let expected_xout = vec![317203.125, 231165.9375, 231925.3125, 319481.25, 407156.25, 282661.875];
        assert_eq!(xout, expected_xout, "Large and different scales test with i8 range did not produce the expected output.");
    }
}
