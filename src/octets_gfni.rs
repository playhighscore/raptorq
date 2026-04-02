// GFNI + AVX-512 accelerated GF(2^8) operations using GF2P8AFFINEQB.
// Precomputes an 8x8 GF(2) multiplication matrix per scalar via isomorphism
// through the AES field (0x11B), giving one instruction per 64 bytes.
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
use crate::octet::Octet;

// Packed 8x8 GF(2) isomorphism matrix (self-inverse) between 0x11D and 0x11B
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
const ISO_MATRIX: u64 = 0xFFAACC88F0A0C080;

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
const fn gf256_mul_11b(a: u8, b: u8) -> u8 {
    let mut r: u16 = 0;
    let mut i = 0;
    while i < 8 {
        if b & (1 << i) != 0 {
            r ^= (a as u16) << i;
        }
        i += 1;
    }
    i = 15;
    while i >= 8 {
        if r & (1 << i) != 0 {
            r ^= 0x11B << (i - 8);
        }
        if i == 0 {
            break;
        }
        i -= 1;
    }
    r as u8
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
const fn apply_iso(x: u8) -> u8 {
    let mut result: u8 = 0;
    let mut i = 0;
    while i < 8 {
        let row = (ISO_MATRIX >> ((7 - i) * 8)) as u8;
        let mut masked = row & x;
        masked ^= masked >> 4;
        masked ^= masked >> 2;
        masked ^= masked >> 1;
        if masked & 1 != 0 {
            result |= 1 << i;
        }
        i += 1;
    }
    result
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
const fn gf256_mul_11d(a: u8, b: u8) -> u8 {
    if a == 0 || b == 0 {
        return 0;
    }
    let a_11b = apply_iso(a);
    let b_11b = apply_iso(b);
    let r_11b = gf256_mul_11b(a_11b, b_11b);
    apply_iso(r_11b) // self-inverse
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
const fn build_mul_matrix(scalar: u8) -> u64 {
    let mut mat = [0u8; 8];
    let mut j = 0;
    while j < 8 {
        let basis = 1u8 << j;
        let product = gf256_mul_11d(scalar, basis);
        let mut i = 0;
        while i < 8 {
            if product & (1 << i) != 0 {
                mat[i] |= 1 << j;
            }
            i += 1;
        }
        j += 1;
    }
    let mut packed = 0u64;
    let mut i = 0;
    while i < 8 {
        packed |= (mat[i] as u64) << ((7 - i) * 8);
        i += 1;
    }
    packed
}

// Precomputed GF2P8AFFINEQB multiplication matrices for all 256 scalars
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
pub(crate) const AFFINE_MUL_TABLE: [u64; 256] = {
    let mut table = [0u64; 256];
    let mut scalar = 0usize;
    while scalar < 256 {
        table[scalar] = build_mul_matrix(scalar as u8);
        scalar += 1;
    }
    table
};

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
#[target_feature(enable = "gfni")]
#[target_feature(enable = "avx512f")]
#[target_feature(enable = "avx512bw")]
pub(crate) unsafe fn mulassign_scalar_gfni(octets: &mut [u8], scalar: &Octet) {
    unsafe {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::*;

        let mat = AFFINE_MUL_TABLE[scalar.byte() as usize];
        let v_mat = _mm512_set1_epi64(mat as i64);
        let self_ptr = octets.as_mut_ptr();

        for i in 0..(octets.len() / 64) {
            #[allow(clippy::cast_ptr_alignment)]
            let self_vec = _mm512_loadu_si512(self_ptr.add(i * 64) as *const __m512i);
            let result = _mm512_gf2p8affine_epi64_epi8(self_vec, v_mat, 0);
            #[allow(clippy::cast_ptr_alignment)]
            _mm512_storeu_si512(self_ptr.add(i * 64) as *mut __m512i, result);
        }

        let remainder = octets.len() % 64;
        if remainder > 0 {
            let tail = octets.len() - remainder;
            let mask: __mmask64 = (1u64 << remainder) - 1;
            let self_vec = _mm512_maskz_loadu_epi8(mask, self_ptr.add(tail) as *const i8);
            let result = _mm512_gf2p8affine_epi64_epi8(self_vec, v_mat, 0);
            _mm512_mask_storeu_epi8(self_ptr.add(tail) as *mut i8, mask, result);
        }
    }
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
#[target_feature(enable = "gfni")]
#[target_feature(enable = "avx512f")]
#[target_feature(enable = "avx512bw")]
pub(crate) unsafe fn fused_addassign_mul_scalar_gfni(
    octets: &mut [u8],
    other: &[u8],
    scalar: &Octet,
) {
    unsafe {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::*;

        let mat = AFFINE_MUL_TABLE[scalar.byte() as usize];
        let v_mat = _mm512_set1_epi64(mat as i64);
        let self_ptr = octets.as_mut_ptr();
        let other_ptr = other.as_ptr();

        for i in 0..(octets.len() / 64) {
            #[allow(clippy::cast_ptr_alignment)]
            let other_vec = _mm512_loadu_si512(other_ptr.add(i * 64) as *const __m512i);
            let other_vec = _mm512_gf2p8affine_epi64_epi8(other_vec, v_mat, 0);

            #[allow(clippy::cast_ptr_alignment)]
            let self_vec = _mm512_loadu_si512(self_ptr.add(i * 64) as *const __m512i);
            let result = _mm512_xor_si512(self_vec, other_vec);
            #[allow(clippy::cast_ptr_alignment)]
            _mm512_storeu_si512(self_ptr.add(i * 64) as *mut __m512i, result);
        }

        let remainder = octets.len() % 64;
        if remainder > 0 {
            let tail = octets.len() - remainder;
            let mask: __mmask64 = (1u64 << remainder) - 1;
            let other_vec = _mm512_maskz_loadu_epi8(mask, other_ptr.add(tail) as *const i8);
            let other_vec = _mm512_gf2p8affine_epi64_epi8(other_vec, v_mat, 0);
            let self_vec = _mm512_maskz_loadu_epi8(mask, self_ptr.add(tail) as *const i8);
            let result = _mm512_xor_si512(self_vec, other_vec);
            _mm512_mask_storeu_epi8(self_ptr.add(tail) as *mut i8, mask, result);
        }
    }
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::octet::OCTET_MUL;

    #[test]
    fn verify_mul_matrices() {
        // Verify that AFFINE_MUL_TABLE[s] * x == OCTET_MUL[s][x] for all s, x
        for s in 0..256u16 {
            let mat = AFFINE_MUL_TABLE[s as usize];
            for x in 0..256u16 {
                let expected = OCTET_MUL[s as usize][x as usize];
                // Apply the matrix to x (scalar emulation of GF2P8AFFINEQB)
                let mut result: u8 = 0;
                for i in 0..8 {
                    let row = (mat >> ((7 - i) * 8)) as u8;
                    let mut masked = row & (x as u8);
                    masked ^= masked >> 4;
                    masked ^= masked >> 2;
                    masked ^= masked >> 1;
                    if masked & 1 != 0 {
                        result |= 1 << i;
                    }
                }
                assert_eq!(
                    expected, result,
                    "Mismatch for scalar={s}, x={x}: expected {expected}, got {result}"
                );
            }
        }
    }

    #[test]
    fn mulassign_scalar_correctness() {
        if !is_x86_feature_detected!("gfni")
            || !is_x86_feature_detected!("avx512f")
            || !is_x86_feature_detected!("avx512bw")
        {
            return; // skip on unsupported hardware
        }

        let sizes = [1, 7, 15, 16, 31, 32, 33, 63, 64, 65, 100, 256, 1024, 1280];
        for &size in &sizes {
            for scalar_val in [1u8, 2, 3, 7, 42, 128, 255] {
                let scalar = Octet::new(scalar_val);
                let mut data: Vec<u8> = (0..size).map(|i| (i * 37 + 13) as u8).collect();
                let mut expected = data.clone();

                // Reference
                for byte in expected.iter_mut() {
                    *byte = OCTET_MUL[scalar_val as usize][*byte as usize];
                }

                // GFNI
                unsafe {
                    mulassign_scalar_gfni(&mut data, &scalar);
                }

                assert_eq!(
                    expected, data,
                    "mulassign_scalar mismatch for size={size}, scalar={scalar_val}"
                );
            }
        }
    }

    #[test]
    fn fused_addassign_mul_scalar_correctness() {
        if !is_x86_feature_detected!("gfni")
            || !is_x86_feature_detected!("avx512f")
            || !is_x86_feature_detected!("avx512bw")
        {
            return;
        }

        let sizes = [1, 7, 15, 16, 31, 32, 33, 63, 64, 65, 100, 256, 1024, 1280];
        for &size in &sizes {
            for scalar_val in [2u8, 3, 7, 42, 128, 255] {
                let scalar = Octet::new(scalar_val);
                let other: Vec<u8> = (0..size).map(|i| (i * 37 + 13) as u8).collect();
                let mut dst: Vec<u8> = (0..size).map(|i| (i * 53 + 7) as u8).collect();
                let mut expected = dst.clone();

                // Reference
                for (i, byte) in expected.iter_mut().enumerate() {
                    *byte ^= OCTET_MUL[scalar_val as usize][other[i] as usize];
                }

                // GFNI
                unsafe {
                    fused_addassign_mul_scalar_gfni(&mut dst, &other, &scalar);
                }

                assert_eq!(
                    expected, dst,
                    "fused_addassign mismatch for size={size}, scalar={scalar_val}"
                );
            }
        }
    }
}
