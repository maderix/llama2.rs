use llama2_rs::softmax;
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_basic() {
        let mut x = vec![2.0, 1.0, 0.1];
        softmax(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((1.0 - sum).abs() < 1e-6); // The sum should be close to 1
        for &val in &x {
            assert!(val >= 0.0 && val <= 1.0); // Each value should be in the range [0,1]
        }
    }

    #[test]
    fn test_softmax_large_values() {
        let mut x = vec![1000.0, 1000.0, 1000.0];
        softmax(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((1.0 - sum).abs() < 1e-6);
        for &val in &x {
            assert!((val - (1.0 / x.len() as f32)).abs() < 1e-6); // All values should be close to 1/3
        }
    }

    #[test]
    fn test_softmax_negative_values() {
        let mut x = vec![-1.0, -2.0, -3.0];
        softmax(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((1.0 - sum).abs() < 1e-6);
        for &val in &x {
            assert!(val >= 0.0 && val <= 1.0);
        }
    }

    #[test]
    fn test_softmax_zero_sum() {
        let mut x = vec![0.0, 0.0, 0.0];
        softmax(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((1.0 - sum).abs() < 1e-6);
        for &val in &x {
            assert!((val - (1.0 / x.len() as f32)).abs() < 1e-6); // All values should be close to 1/3
        }
    }
}
