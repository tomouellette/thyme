// Copyright (c) 2025, Tom Ouellette
// Licensed under the BSD 3-Clause License

use std::ops::Deref;

use num::{FromPrimitive, ToPrimitive, complex::Complex};

use crate::{constant::FACTORIAL, im::ThymeViewBuffer};

#[inline]
fn radial_polynomial(n: usize, m: usize, r: &mut [Complex<f32>]) {
    let nf = n as i32;
    let nsm = (n - m) / 2;
    let nam = (n + m) / 2;

    for ri in r.iter_mut() {
        let mut r_nm_i = Complex::new(0.0, 0.0);
        for si in 0..=nsm {
            let sf = si as f32;
            let exp = nf - 2 * si as i32;

            let v = ((-1.0f32).powf(sf) * FACTORIAL[n - si])
                / (FACTORIAL[si] * FACTORIAL[nam - si] * FACTORIAL[nsm - si]);

            let pow_term = if exp >= 0 {
                ri.powi(exp)
            } else {
                ri.powf(exp as f32)
            };
            r_nm_i += Complex::new(v, 0.0) * pow_term;
        }
        *ri = r_nm_i;
    }
}

#[inline]
fn zernike_polynomial(n: usize, m: usize, r: &mut [Complex<f32>], theta: &[f32]) {
    let m_theta = Complex::new(0.0, m as f32);
    radial_polynomial(n, m, r);
    for (ri, &theta_i) in r.iter_mut().zip(theta.iter()) {
        *ri *= (m_theta * theta_i).exp();
    }
}

#[inline]
pub fn zernike_moments<T>(pixels: &[T], width: usize, height: usize, n: usize, m: usize) -> f32
where
    T: ToPrimitive,
{
    let half_width = width as f32 / 2.0;
    let half_height = height as f32 / 2.0;

    let mut total_mass = 0.0;

    let capacity = pixels.len();

    let mut circle: Vec<f32> = Vec::with_capacity(capacity);
    let mut theta = Vec::with_capacity(capacity);
    let mut r: Vec<Complex<f32>> = Vec::with_capacity(capacity);

    for (i, pixel) in pixels.iter().enumerate() {
        let x_norm = ((i % width) as f32 - half_width) / half_width;
        let y_norm = ((i / width) as f32 - half_height) / half_height;
        let r_i = (x_norm * x_norm + y_norm * y_norm).sqrt();

        if r_i <= 1.0 {
            let pixel = pixel.to_f32().unwrap();
            total_mass += pixel;
            theta.push(y_norm.atan2(x_norm));
            r.push(Complex::new(r_i, 0.0));
            circle.push(pixel);
        }
    }

    if total_mass == 0.0 {
        return 0.0;
    }

    zernike_polynomial(n, m, &mut r, &theta);

    let inv_mass = 1.0 / total_mass;
    let mut a_nm = Complex::new(0.0, 0.0);

    for (i, z_nm_i) in r.iter().enumerate() {
        a_nm += z_nm_i.conj() * Complex::new(circle[i] * inv_mass, 0.0);
    }

    a_nm *= Complex::new((n as f32 + 1.0) / std::f32::consts::PI, 0.0);

    (a_nm.re.powi(2) + a_nm.im.powi(2)).sqrt()
}

#[inline]
pub fn descriptors<T>(pixels: &[T], width: usize, height: usize) -> [f32; 30]
where
    T: ToPrimitive,
{
    let mut descriptors: [f32; 30] = [0.0; 30];
    let mut i = 0;
    for n in 0..=9 {
        for m in 0..=n {
            if (n - m) % 2 == 0 {
                descriptors[i] = zernike_moments(pixels, width, height, n, m);
                i += 1;
            }
        }
    }

    descriptors
}

#[inline]
pub fn zernike_moments_object<T, Container>(
    object: &ThymeViewBuffer<T, Container>,
    n: usize,
    m: usize,
) -> f32
where
    T: ToPrimitive + FromPrimitive,
    Container: Deref<Target = [T]>,
{
    let width = object.width();
    let half_width = width as f32 / 2.0;
    let half_height = object.height() as f32 / 2.0;

    let mut total_mass = 0.0;

    let capacity = object.len();

    let mut circle: Vec<f32> = Vec::with_capacity(capacity);
    let mut theta = Vec::with_capacity(capacity);
    let mut r: Vec<Complex<f32>> = Vec::with_capacity(capacity);

    for (i, pixel) in object.iter().enumerate() {
        let x_norm = ((i % width) as f32 - half_width) / half_width;
        let y_norm = ((i / width) as f32 - half_height) / half_height;
        let r_i = (x_norm * x_norm + y_norm * y_norm).sqrt();

        if r_i <= 1.0 {
            let pixel = pixel.to_f32().unwrap();
            total_mass += pixel;
            theta.push(y_norm.atan2(x_norm));
            r.push(Complex::new(r_i, 0.0));
            circle.push(pixel);
        }
    }

    if total_mass == 0.0 {
        return 0.0;
    }

    zernike_polynomial(n, m, &mut r, &theta);

    let inv_mass = 1.0 / total_mass;
    let mut a_nm = Complex::new(0.0, 0.0);

    for (i, z_nm_i) in r.iter().enumerate() {
        a_nm += z_nm_i.conj() * Complex::new(circle[i] * inv_mass, 0.0);
    }

    a_nm *= Complex::new((n as f32 + 1.0) / std::f32::consts::PI, 0.0);

    (a_nm.re.powi(2) + a_nm.im.powi(2)).sqrt()
}

#[inline]
pub fn objects<T, Container>(object: &ThymeViewBuffer<T, Container>) -> [f32; 30]
where
    T: ToPrimitive + FromPrimitive,
    Container: Deref<Target = [T]>,
{
    let mut descriptors: [f32; 30] = [0.0; 30];
    let mut i = 0;
    for n in 0..=9 {
        for m in 0..=n {
            if (n - m) % 2 == 0 {
                descriptors[i] = zernike_moments_object(object, n, m);
                i += 1;
            }
        }
    }

    descriptors
}

#[cfg(test)]
mod test {

    use super::*;

    #[test]
    fn test_zernike_moment() {
        let zm = descriptors(&[1], 1, 1);
        for i in zm.iter() {
            assert_eq!(*i, 0.0);
        }
    }

    #[test]
    fn test_radial_polynomial() {
        let mut r = vec![
            Complex::new(0.0, 0.0),
            Complex::new(0.5, 0.0),
            Complex::new(1.0, 0.0),
        ];

        radial_polynomial(2, 0, &mut r);

        let expected = [
            Complex::new(-1.0, 0.0),
            Complex::new(-0.5, 0.0),
            Complex::new(1.0, 0.0),
        ];

        for (res, exp) in r.iter().zip(expected.iter()) {
            assert!((res.re - exp.re).abs() < 1e-6);
        }
    }

    #[test]
    fn test_zernike_polynomial() {
        let mut r = vec![
            Complex::new(1.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(1.0, 0.0),
        ];

        let theta = vec![
            0.0,
            std::f32::consts::FRAC_PI_4,
            std::f32::consts::FRAC_PI_2,
        ];

        zernike_polynomial(2, 2, &mut r, &theta);

        let expected = [
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 1.0),
            Complex::new(-1.0, 0.0),
        ];

        for (res, exp) in r.iter().zip(expected.iter()) {
            assert!((res.re - exp.re).abs() < 1e-6);
            assert!((res.im - exp.im).abs() < 1e-6);
        }
    }
}
