// Copyright (c) 2025, Tom Ouellette
// Licensed under the BSD 3-Clause License

use std::ops::Deref;

use num::{FromPrimitive, ToPrimitive};

use crate::im::ThymeViewBuffer;

#[inline]
pub fn moments_raw<T>(pixels: &[T], width: usize) -> [f32; 10]
where
    T: ToPrimitive,
{
    let mut m00 = 0.0;
    let mut m10 = 0.0;
    let mut m01 = 0.0;
    let mut m11 = 0.0;
    let mut m20 = 0.0;
    let mut m02 = 0.0;
    let mut m12 = 0.0;
    let mut m21 = 0.0;
    let mut m30 = 0.0;
    let mut m03 = 0.0;

    for (i, pixel) in pixels.iter().enumerate() {
        let pixel = pixel.to_f32().unwrap();
        if pixel > 0.0 {
            let x = i % width;
            let y = i / width;
            let xa = x as f32;
            let xb = xa * xa;
            let xc = xb * xa;
            let ya = y as f32;
            let yb = ya * ya;
            let yc = yb * ya;

            m00 += pixel;
            m10 += xa * pixel;
            m01 += ya * pixel;
            m11 += xa * ya * pixel;
            m20 += xb * pixel;
            m02 += yb * pixel;
            m21 += xb * ya * pixel;
            m12 += xa * yb * pixel;
            m30 += xc * pixel;
            m03 += yc * pixel;
        }
    }

    [m00, m10, m01, m11, m20, m02, m21, m12, m30, m03]
}

#[inline]
pub fn moments_central<T>(pixels: &[T], width: usize) -> [f32; 10]
where
    T: ToPrimitive,
{
    let raw_moments = moments_raw(pixels, width);
    let m00 = raw_moments[0];
    let m10 = raw_moments[1];
    let m01 = raw_moments[2];
    let m11 = raw_moments[3];
    let m20 = raw_moments[4];
    let m02 = raw_moments[5];
    let m21 = raw_moments[6];
    let m12 = raw_moments[7];
    let m30 = raw_moments[8];
    let m03 = raw_moments[9];

    if m00 == 0.0 {
        return [0.0; 10];
    }

    let x = m10 / m00;
    let y = m01 / m00;

    let u00 = m00;
    let u10 = 0.0;
    let u01 = 0.0;
    let u11 = m11 - x * m01;
    let u20 = m20 - x * m10;
    let u02 = m02 - y * m01;
    let u21 = m21 - 2.0 * x * m11 - y * m20 + 2.0 * x * x * m01;
    let u12 = m12 - 2.0 * y * m11 - x * m02 + 2.0 * y * y * m10;
    let u30 = m30 - 3.0 * x * m20 + 2.0 * x * x * m10;
    let u03 = m03 - 3.0 * y * m02 + 2.0 * y * y * m01;

    [u00, u10, u01, u11, u20, u02, u21, u12, u30, u03]
}

#[inline]
pub fn moments_hu<T>(pixels: &[T], width: usize) -> [f32; 7]
where
    T: ToPrimitive,
{
    let central_moments = moments_central(pixels, width);
    let u00 = central_moments[0];
    let u20 = central_moments[4];
    let u02 = central_moments[5];
    let u11 = central_moments[3];
    let u30 = central_moments[8];
    let u03 = central_moments[9];
    let u21 = central_moments[6];
    let u12 = central_moments[7];

    if u00 == 0.0 {
        return [0.0; 7];
    }

    let s2 = u00 * u00;
    let s3 = u00.powf(2.5);

    let n20 = u20 / s2;
    let n02 = u02 / s2;
    let n11 = u11 / s2;
    let n30 = u30 / s3;
    let n03 = u03 / s3;
    let n21 = u21 / s3;
    let n12 = u12 / s3;

    let p = n20 - n02;
    let q = n30 - 3.0 * n12;
    let r = n30 + n12;
    let z = n21 + n03;
    let y = 3.0 * n21 - n03;

    let i1 = n20 + n02;
    let i2 = p * p + 4.0 * n11 * n11;
    let i3 = q * q + y * y;
    let i4 = r * r + z * z;
    let i5 = q * r * (r * r - 3.0 * z * z) + y * z * (3.0 * r * r - z * z);
    let i6 = p * (r * r - z * z) + 4.0 * n11 * r * z;
    let i7 = y * r * (r * r - 3.0 * z * z) - q * z * (3.0 * r * r - z * z);

    [i1, i2, i3, i4, i5, i6, i7]
}

#[inline]
pub fn descriptors<T>(pixels: &[T], width: usize) -> [f32; 24]
where
    T: ToPrimitive,
{
    let mut m00 = 0.0;
    let mut m10 = 0.0;
    let mut m01 = 0.0;
    let mut m11 = 0.0;
    let mut m20 = 0.0;
    let mut m02 = 0.0;
    let mut m12 = 0.0;
    let mut m21 = 0.0;
    let mut m30 = 0.0;
    let mut m03 = 0.0;

    for (i, pixel) in pixels.iter().enumerate() {
        let pixel = pixel.to_f32().unwrap();
        if pixel > 0.0 {
            let x = i % width;
            let y = i / width;
            let xa = x as f32;
            let xb = xa * xa;
            let xc = xb * xa;
            let ya = y as f32;
            let yb = ya * ya;
            let yc = yb * ya;

            m00 += pixel;
            m10 += xa * pixel;
            m01 += ya * pixel;
            m11 += xa * ya * pixel;
            m20 += xb * pixel;
            m02 += yb * pixel;
            m21 += xb * ya * pixel;
            m12 += xa * yb * pixel;
            m30 += xc * pixel;
            m03 += yc * pixel;
        }
    }

    if m00 == 0.0 {
        return [0.0; 24];
    }

    let x = m10 / m00;
    let y = m01 / m00;

    let u00 = m00;
    let u11 = m11 - x * m01;
    let u20 = m20 - x * m10;
    let u02 = m02 - y * m01;
    let u21 = m21 - 2.0 * x * m11 - y * m20 + 2.0 * x * x * m01;
    let u12 = m12 - 2.0 * y * m11 - x * m02 + 2.0 * y * y * m10;
    let u30 = m30 - 3.0 * x * m20 + 2.0 * x * x * m10;
    let u03 = m03 - 3.0 * y * m02 + 2.0 * y * y * m01;

    let s2 = u00 * u00;
    let s3 = u00.powf(2.5);

    let n20 = u20 / s2;
    let n02 = u02 / s2;
    let n11 = u11 / s2;
    let n30 = u30 / s3;
    let n03 = u03 / s3;
    let n21 = u21 / s3;
    let n12 = u12 / s3;

    let p = n20 - n02;
    let q = n30 - 3.0 * n12;
    let r = n30 + n12;
    let z = n21 + n03;
    let y = 3.0 * n21 - n03;

    let i1 = n20 + n02;
    let i2 = p * p + 4.0 * n11 * n11;
    let i3 = q * q + y * y;
    let i4 = r * r + z * z;
    let i5 = q * r * (r * r - 3.0 * z * z) + y * z * (3.0 * r * r - z * z);
    let i6 = p * (r * r - z * z) + 4.0 * n11 * r * z;
    let i7 = y * r * (r * r - 3.0 * z * z) - q * z * (3.0 * r * r - z * z);

    [
        m00, m10, m01, m11, m20, m02, m21, m12, m30, m03, u11, u20, u02, u21, u12, u30, u03, i1,
        i2, i3, i4, i5, i6, i7,
    ]
}

#[inline]
pub fn objects<T, Container>(object: &ThymeViewBuffer<T, Container>) -> [f32; 24]
where
    T: ToPrimitive + FromPrimitive,
    Container: Deref<Target = [T]>,
{
    let mut m00 = 0.0;
    let mut m10 = 0.0;
    let mut m01 = 0.0;
    let mut m11 = 0.0;
    let mut m20 = 0.0;
    let mut m02 = 0.0;
    let mut m12 = 0.0;
    let mut m21 = 0.0;
    let mut m30 = 0.0;
    let mut m03 = 0.0;

    for (i, pixel) in object.iter().enumerate() {
        let pixel = pixel.to_f32().unwrap();
        if pixel > 0.0 {
            let x = i % object.width();
            let y = i / object.width();
            let xa = x as f32;
            let xb = xa * xa;
            let xc = xb * xa;
            let ya = y as f32;
            let yb = ya * ya;
            let yc = yb * ya;

            m00 += pixel;
            m10 += xa * pixel;
            m01 += ya * pixel;
            m11 += xa * ya * pixel;
            m20 += xb * pixel;
            m02 += yb * pixel;
            m21 += xb * ya * pixel;
            m12 += xa * yb * pixel;
            m30 += xc * pixel;
            m03 += yc * pixel;
        }
    }

    if m00 == 0.0 {
        return [0.0; 24];
    }

    let x = m10 / m00;
    let y = m01 / m00;

    let u00 = m00;
    let u11 = m11 - x * m01;
    let u20 = m20 - x * m10;
    let u02 = m02 - y * m01;
    let u21 = m21 - 2.0 * x * m11 - y * m20 + 2.0 * x * x * m01;
    let u12 = m12 - 2.0 * y * m11 - x * m02 + 2.0 * y * y * m10;
    let u30 = m30 - 3.0 * x * m20 + 2.0 * x * x * m10;
    let u03 = m03 - 3.0 * y * m02 + 2.0 * y * y * m01;

    let s2 = u00 * u00;
    let s3 = u00.powf(2.5);

    let n20 = u20 / s2;
    let n02 = u02 / s2;
    let n11 = u11 / s2;
    let n30 = u30 / s3;
    let n03 = u03 / s3;
    let n21 = u21 / s3;
    let n12 = u12 / s3;

    let p = n20 - n02;
    let q = n30 - 3.0 * n12;
    let r = n30 + n12;
    let z = n21 + n03;
    let y = 3.0 * n21 - n03;

    let i1 = n20 + n02;
    let i2 = p * p + 4.0 * n11 * n11;
    let i3 = q * q + y * y;
    let i4 = r * r + z * z;
    let i5 = q * r * (r * r - 3.0 * z * z) + y * z * (3.0 * r * r - z * z);
    let i6 = p * (r * r - z * z) + 4.0 * n11 * r * z;
    let i7 = y * r * (r * r - 3.0 * z * z) - q * z * (3.0 * r * r - z * z);

    [
        m00, m10, m01, m11, m20, m02, m21, m12, m30, m03, u11, u20, u02, u21, u12, u30, u03, i1,
        i2, i3, i4, i5, i6, i7,
    ]
}

#[cfg(test)]
mod test {

    use crate::im::ThymeBuffer;

    use super::*;

    fn mask_a() -> [u8; 4] {
        [0, 0, 0, 0]
    }

    fn mask_b() -> [u8; 16] {
        let mut mask: [u8; 16] = [0; 16];
        mask[0] = 1;
        mask[1 + 4] = 1;
        mask[2 + 4 * 2] = 1;
        mask[3 + 4 * 3] = 1;
        mask
    }

    fn mask_c() -> [u8; 100] {
        let mut mask: [u8; 100] = [0; 100];
        for (i, m) in mask.iter_mut().enumerate() {
            let w = i % 10;
            let h = i / 10;
            *m = (w + h) as u8;
        }
        mask
    }

    #[test]
    fn test_moments_raw() {
        let moments_1 = moments_raw(&mask_a(), 2);
        assert_eq!(moments_1, [0.0; 10]);

        let moments_2 = moments_raw(&mask_b(), 4);
        assert_eq!(
            moments_2,
            [4.0, 6.0, 6.0, 14.0, 14.0, 14.0, 36.0, 36.0, 36.0, 36.0]
        );

        let moments_3 = moments_raw(&mask_c(), 10);
        assert_eq!(
            moments_3,
            [
                900.0, 4875.0, 4875.0, 25650.0, 33075.0, 33075.0, 172350.0, 172350.0, 244455.0,
                244455.0
            ]
        );
    }

    #[test]
    fn test_moments_central() {
        let moments_1 = moments_central(&mask_a(), 2);
        assert_eq!(moments_1, [0.0; 10]);

        let moments_2 = moments_central(&mask_b(), 4);
        assert_eq!(
            moments_2,
            [4.0, 0.0, 0.0, 5.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0]
        );

        let moments_3 = moments_central(&mask_c(), 10);
        println!("{:?}", moments_3);
        assert_eq!(
            moments_3,
            [
                900.0, 0.0, 0.0, -756.25, 6668.75, 6668.75, 1386.4375, 1386.4375, -6946.0625,
                -6946.0625
            ]
        );
    }

    #[test]
    fn test_moments_hu() {
        let moments = moments_hu(&mask_a(), 2);
        assert_eq!(moments, [0.0; 7]);

        let moments = moments_hu(&mask_b(), 4);
        assert_eq!(moments, [0.625, 0.390625, 0.0, 0.0, 0.0, 0.0, 0.0]);

        let moments = moments_hu(&mask_c(), 10);
        assert!(
            moments
                .iter()
                .zip(vec![
                    0.016_466_05,
                    3.48674e-06,
                    4.17721e-07,
                    1.04689e-07,
                    -2.189255e-14,
                    -1.954844e-10,
                    0.0
                ])
                .all(|(a, b)| (a - b).abs() < 1e-9)
        );
    }

    #[test]
    fn test_moments_descriptors() {
        let moments = descriptors(&mask_a(), 2);

        let raw_moments = moments_raw(&mask_a(), 2);
        let central_moments = moments_central(&mask_a(), 2);
        let hu_moments = moments_hu(&mask_a(), 2);

        let concat_moments = raw_moments
            .iter()
            .chain(central_moments.iter().skip(3))
            .chain(hu_moments.iter())
            .cloned()
            .collect::<Vec<f32>>();

        assert_eq!(moments.as_slice(), concat_moments.as_slice());

        let moments = descriptors(&mask_b(), 4);
        let raw_moments = moments_raw(&mask_b(), 4);
        let central_moments = moments_central(&mask_b(), 4);
        let hu_moments = moments_hu(&mask_b(), 4);

        let concat_moments = raw_moments
            .iter()
            .chain(central_moments.iter().skip(3))
            .chain(hu_moments.iter())
            .cloned()
            .collect::<Vec<f32>>();

        assert_eq!(moments.as_slice(), concat_moments.as_slice());

        let moments = descriptors(&mask_c(), 10);
        let raw_moments = moments_raw(&mask_c(), 10);
        let central_moments = moments_central(&mask_c(), 10);
        let hu_moments = moments_hu(&mask_c(), 10);

        let concat_moments = raw_moments
            .iter()
            .chain(central_moments.iter().skip(3))
            .chain(hu_moments.iter())
            .cloned()
            .collect::<Vec<f32>>();

        assert_eq!(moments.as_slice(), concat_moments.as_slice());
    }

    #[test]
    fn test_moments_objects() {
        let mask_a = mask_a();
        let mask_b = mask_b();
        let mask_c = mask_c();

        let buffer_a = ThymeBuffer::new(2, 2, 1, mask_a.to_vec()).unwrap();
        let buffer_b = ThymeBuffer::new(4, 4, 1, mask_b.to_vec()).unwrap();
        let buffer_c = ThymeBuffer::new(10, 10, 1, mask_c.to_vec()).unwrap();

        let object_a = ThymeViewBuffer::new(0, 0, 2, 2, &buffer_a);
        let object_b = ThymeViewBuffer::new(0, 0, 4, 4, &buffer_b);
        let object_c = ThymeViewBuffer::new(0, 0, 10, 10, &buffer_c);

        let moments_object_a = objects(&object_a);
        let moments_object_b = objects(&object_b);
        let moments_object_c = objects(&object_c);

        let moments_array_a = descriptors(&mask_a, 2);
        let moments_array_b = descriptors(&mask_b, 4);
        let moments_array_c = descriptors(&mask_c, 10);

        assert_eq!(moments_object_a, moments_array_a);
        assert_eq!(moments_object_b, moments_array_b);
        assert_eq!(moments_object_c, moments_array_c);
    }
}
