use anyhow::{ensure, Context};
use fast_image_resize::{
    images::Image as FirImage, FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer,
};
use image::{DynamicImage, Rgb, RgbImage};
use ndarray::Array4;

const RECOGNITION_MAX_WIDTH: u32 = 320;
const DETECTION_LIMIT_SIDE_LEN: u32 = 960;
const DETECTION_STRIDE: u32 = 32;
const RECOGNITION_MEAN: [f32; 3] = [0.5, 0.5, 0.5];
const RECOGNITION_STD: [f32; 3] = [0.5, 0.5, 0.5];
const DETECTION_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const DETECTION_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Preprocess an image for OCR recognition.
/// Resizes and pads the image to the expected OpenOCR input dimensions.
pub fn preprocess_for_recognition(
    image: &DynamicImage,
    target_height: u32,
) -> anyhow::Result<DynamicImage> {
    validate_image(image)?;
    ensure!(
        target_height > 0,
        "recognition target height must be greater than zero"
    );

    let rgb = image.to_rgb8();
    let resized_width = (((target_height as f64 * rgb.width() as f64) / rgb.height() as f64).ceil()
        as u32)
        .clamp(1, RECOGNITION_MAX_WIDTH);
    let resized = resize_rgb_image(&rgb, resized_width, target_height)?;

    let mut padded = RgbImage::from_pixel(
        RECOGNITION_MAX_WIDTH,
        target_height,
        Rgb([0_u8, 0_u8, 0_u8]),
    );
    for (x, y, pixel) in resized.enumerate_pixels() {
        padded.put_pixel(x, y, *pixel);
    }

    Ok(DynamicImage::ImageRgb8(padded))
}

/// Convert a preprocessed recognition image into a `[1, 3, H, W]` tensor.
pub fn pack_recognition_tensor(image: &DynamicImage) -> anyhow::Result<Array4<f32>> {
    pack_tensor(image, RECOGNITION_MEAN, RECOGNITION_STD)
}

/// Preprocess an image for text detection.
/// Resizes the image toward the detector limit and aligns dimensions to the model stride.
pub fn preprocess_for_detection(image: &DynamicImage) -> anyhow::Result<DynamicImage> {
    validate_image(image)?;

    let rgb = image.to_rgb8();
    let (width, height) = rgb.dimensions();
    let longest_side = width.max(height);
    let resize_ratio = if longest_side > DETECTION_LIMIT_SIDE_LEN {
        DETECTION_LIMIT_SIDE_LEN as f64 / longest_side as f64
    } else {
        1.0
    };

    let resized_width = align_to_stride(
        ((width as f64 * resize_ratio).floor() as u32).max(1),
        DETECTION_STRIDE,
    );
    let resized_height = align_to_stride(
        ((height as f64 * resize_ratio).floor() as u32).max(1),
        DETECTION_STRIDE,
    );
    let resized = resize_rgb_image(&rgb, resized_width, resized_height)?;

    Ok(DynamicImage::ImageRgb8(resized))
}

/// Convert a preprocessed detection image into a `[1, 3, H, W]` tensor.
pub fn pack_detection_tensor(image: &DynamicImage) -> anyhow::Result<Array4<f32>> {
    pack_tensor(image, DETECTION_MEAN, DETECTION_STD)
}

fn validate_image(image: &DynamicImage) -> anyhow::Result<()> {
    ensure!(
        image.width() > 0 && image.height() > 0,
        "image dimensions must be greater than zero"
    );
    Ok(())
}

fn align_to_stride(value: u32, stride: u32) -> u32 {
    let rounded = ((value + stride / 2) / stride).max(1);
    rounded * stride
}

fn resize_rgb_image(image: &RgbImage, width: u32, height: u32) -> anyhow::Result<RgbImage> {
    ensure!(
        width > 0 && height > 0,
        "resize dimensions must be greater than zero"
    );

    if image.width() == width && image.height() == height {
        return Ok(image.clone());
    }

    let mut dst_image = FirImage::new(width, height, PixelType::U8x3);
    let mut resizer = Resizer::new();
    let options = ResizeOptions::new().resize_alg(ResizeAlg::Interpolation(FilterType::Bilinear));

    resizer
        .resize(image, &mut dst_image, &options)
        .context("failed to resize RGB image")?;

    RgbImage::from_raw(width, height, dst_image.into_vec())
        .context("failed to build resized RGB image")
}

fn pack_tensor(image: &DynamicImage, mean: [f32; 3], std: [f32; 3]) -> anyhow::Result<Array4<f32>> {
    validate_image(image)?;

    let rgb = image.to_rgb8();
    let (width, height) = rgb.dimensions();
    let mut tensor = Array4::zeros((1, 3, height as usize, width as usize));

    for (x, y, pixel) in rgb.enumerate_pixels() {
        for channel in 0..3 {
            let normalized = (pixel[channel] as f32 / 255.0 - mean[channel]) / std[channel];
            tensor[[0, channel, y as usize, x as usize]] = normalized;
        }
    }

    Ok(tensor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < 1e-5,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn recognition_resize_preserves_aspect_and_right_pads() {
        let image = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(20, 10, Rgb([255, 0, 0])));

        let processed =
            preprocess_for_recognition(&image, 48).expect("recognition preprocessing succeeds");
        let rgb = processed.to_rgb8();

        assert_eq!(rgb.dimensions(), (320, 48));
        assert_eq!(rgb.get_pixel(0, 0), &Rgb([255, 0, 0]));
        assert_eq!(rgb.get_pixel(95, 47), &Rgb([255, 0, 0]));
        assert_eq!(rgb.get_pixel(96, 0), &Rgb([0, 0, 0]));
        assert_eq!(rgb.get_pixel(319, 47), &Rgb([0, 0, 0]));
    }

    #[test]
    fn recognition_resize_caps_width_at_maximum() {
        let image = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(100, 10, Rgb([0, 255, 0])));

        let processed =
            preprocess_for_recognition(&image, 48).expect("recognition preprocessing succeeds");
        let rgb = processed.to_rgb8();

        assert_eq!(rgb.dimensions(), (320, 48));
        assert_eq!(rgb.get_pixel(0, 0), &Rgb([0, 255, 0]));
        assert_eq!(rgb.get_pixel(319, 47), &Rgb([0, 255, 0]));
    }

    #[test]
    fn recognition_tensor_uses_openocr_normalization() {
        let image = DynamicImage::ImageRgb8(ImageBuffer::from_fn(2, 1, |x, _| match x {
            0 => Rgb([0, 127, 255]),
            _ => Rgb([255, 127, 0]),
        }));

        let tensor = pack_recognition_tensor(&image).expect("recognition tensor packing succeeds");

        assert_eq!(tensor.shape(), &[1, 3, 1, 2]);
        assert_close(tensor[[0, 0, 0, 0]], -1.0);
        assert_close(tensor[[0, 1, 0, 0]], (127.0 / 255.0 - 0.5) / 0.5);
        assert_close(tensor[[0, 2, 0, 0]], 1.0);
        assert_close(tensor[[0, 0, 0, 1]], 1.0);
        assert_close(tensor[[0, 2, 0, 1]], -1.0);
    }

    #[test]
    fn detection_resize_limits_longest_side_and_aligns_stride() {
        let image = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(2000, 1000, Rgb([1, 2, 3])));

        let processed = preprocess_for_detection(&image).expect("detection preprocessing succeeds");
        let rgb = processed.to_rgb8();

        assert_eq!(rgb.dimensions(), (960, 480));
        assert_eq!(rgb.get_pixel(0, 0), &Rgb([1, 2, 3]));
        assert_eq!(rgb.get_pixel(959, 479), &Rgb([1, 2, 3]));
    }

    #[test]
    fn detection_resize_aligns_small_images_to_stride() {
        let image = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(100, 50, Rgb([9, 8, 7])));

        let processed = preprocess_for_detection(&image).expect("detection preprocessing succeeds");
        let rgb = processed.to_rgb8();

        assert_eq!(rgb.dimensions(), (96, 64));
    }

    #[test]
    fn detection_tensor_uses_imagenet_normalization() {
        let image = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(1, 1, Rgb([255, 0, 0])));

        let tensor = pack_detection_tensor(&image).expect("detection tensor packing succeeds");

        assert_eq!(tensor.shape(), &[1, 3, 1, 1]);
        assert_close(tensor[[0, 0, 0, 0]], (1.0 - 0.485) / 0.229);
        assert_close(tensor[[0, 1, 0, 0]], (0.0 - 0.456) / 0.224);
        assert_close(tensor[[0, 2, 0, 0]], (0.0 - 0.406) / 0.225);
    }
}
