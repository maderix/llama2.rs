#![feature(stdsimd)]
use rayon::prelude::*;
use std::{arch::x86_64::*, cmp::min, time};

use crate::QuantizedTensor;

pub fn matmul(
    xout: &mut Vec<f32>,
    x: &QuantizedTensor,
    ws: &[QuantizedTensor],
    n: usize,
    d: usize,
    layer_index: usize,
    gs: usize,
) {
    assert!(gs % 8 == 0, "Group size must be a multiple of 8 for AVX2");
    let w = &ws[layer_index];

    // Ensure we have a scale per group for both x and w
    assert!(x.s.len() * gs >= x.q.len() && w.s.len() * gs >= w.q.len(), "Scale vector size mismatch");

    xout.par_iter_mut().enumerate().for_each(|(i, xout_elem)| {
        unsafe {
            let mut sum = _mm256_setzero_ps();

            for j in (0..n).step_by(gs) {
                for k in (0..gs).step_by(8) {
                    let idx_x = j + k;
                    let idx_w = i * n + j + k;

                    if idx_x + 7 < x.q.len() && idx_w + 7 < w.q.len() {
                        // Load quantized values and convert to 32-bit floats
                        let x_vals_i32 = _mm256_cvtepi8_epi32(_mm_loadl_epi64(x.q.as_ptr().add(idx_x) as *const __m128i));
                        let w_vals_i32 = _mm256_cvtepi8_epi32(_mm_loadl_epi64(w.q.as_ptr().add(idx_w) as *const __m128i));
                        let x_vals_f32 = _mm256_cvtepi32_ps(x_vals_i32);
                        let w_vals_f32 = _mm256_cvtepi32_ps(w_vals_i32);

                        // Apply scales
                        let scale_x = _mm256_set1_ps(x.s[j / gs]);
                        let scale_w = _mm256_set1_ps(w.s[i * n / gs + j / gs]);
                        let x_scaled = _mm256_mul_ps(x_vals_f32, scale_x);
                        let w_scaled = _mm256_mul_ps(w_vals_f32, scale_w);

                        // Multiply and accumulate
                        sum = _mm256_fmadd_ps(x_scaled, w_scaled, sum);
                    }
                }
            }

            // Horizontal sum to accumulate results
            *xout_elem = hsum_avx2(sum);
        }
    });
}

unsafe fn hsum_avx2(v: __m256) -> f32 {
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps(v, 1);
    let sum = _mm_add_ps(lo, hi);
    let sum_high = _mm_movehl_ps(sum, sum);
    let sum_low = _mm_add_ps(sum, sum_high);
    let sum_scalar = _mm_add_ss(sum_low, _mm_shuffle_ps(sum_low, sum_low, 0x01));
    _mm_cvtss_f32(sum_scalar)
}


fn hsum_ps(vec: __m128) -> f32 {
    unsafe {
        // Shuffle to move elements
        let shuffled = _mm_movehdup_ps(vec); // Duplicate high pair
        let sums1 = _mm_add_ps(vec, shuffled); // Add elements pair-wise
        let shuffled2 = _mm_movehl_ps(sums1, sums1); // Move high pair to low
        let sums2 = _mm_add_ps(sums1, shuffled2); // Add remaining pairs
        _mm_cvtss_f32(sums2) // Extract the lowest element
    }
}

/*unsafe fn hsum_avx2(v: __m256) -> f32 {
    // Step 1: Extract the lower and upper 128-bit lanes as __m128
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps(v, 1);

    // Step 2: Add the lower and upper parts together
    let sum128 = _mm_add_ps(lo, hi);

    // Step 3: Horizontal addition of the resulting __m128
    let sum_high = _mm_movehl_ps(sum128, sum128); // Move high 64 bits of sum128 to low 64 bits
    let sum_low = _mm_add_ps(sum128, sum_high); // Add the original and moved values

    // Step 4: Final shuffle and addition to collapse to a single float
    let sum_scalar = _mm_add_ss(sum_low, _mm_shuffle_ps(sum_low, sum_low, 0x01));

    _mm_cvtss_f32(sum_scalar) // Extract the lowest 32 bits as float
}
*/
