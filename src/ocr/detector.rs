use std::{collections::VecDeque, path::Path, sync::Mutex};

use anyhow::{anyhow, bail, ensure, Context};
use image::DynamicImage;
use ndarray::{Array2, ArrayView2, ArrayViewD, Axis, Ix2, Ix3, Ix4};
use ort::{
    execution_providers::CPUExecutionProvider,
    session::{builder::GraphOptimizationLevel, Session},
    value::TensorRef,
};

use crate::ocr::preprocess::{pack_detection_tensor, preprocess_for_detection};
use crate::ocr::sort_text_boxes_in_reading_order;

#[cfg(feature = "cuda-ep")]
use ort::execution_providers::CUDAExecutionProvider;
#[cfg(feature = "tensorrt-ep")]
use ort::execution_providers::TensorRTExecutionProvider;

const DB_BINARY_THRESHOLD: f32 = 0.3;
const MIN_COMPONENT_PIXELS: usize = 4;
const MIN_BOX_SIDE: u32 = 3;
const BOX_PADDING_RATIO: f32 = 0.1;

/// RepViT/DB text detector backed by ONNX Runtime.
pub struct Detector {
    session: Mutex<Session>,
}

impl Detector {
    pub fn new(model_path: &str) -> anyhow::Result<Self> {
        let path = Path::new(model_path);
        ensure!(
            path.is_file(),
            "detector model not found: {}",
            path.display()
        );

        let execution_providers = vec![
            #[cfg(feature = "tensorrt-ep")]
            TensorRTExecutionProvider::default().build().fail_silently(),
            #[cfg(feature = "cuda-ep")]
            CUDAExecutionProvider::default().build().fail_silently(),
            CPUExecutionProvider::default()
                .with_arena_allocator(true)
                .build(),
        ];

        let session = Session::builder()
            .context("failed to create detector session builder")?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .context("failed to configure detector graph optimizations")?
            .with_execution_providers(execution_providers)
            .context("failed to configure detector execution providers")?
            .commit_from_file(path)
            .with_context(|| format!("failed to load detector model from {}", path.display()))?;

        ensure!(
            !session.inputs.is_empty(),
            "detector model at {} has no inputs",
            path.display()
        );
        ensure!(
            !session.outputs.is_empty(),
            "detector model at {} has no outputs",
            path.display()
        );

        Ok(Self {
            session: Mutex::new(session),
        })
    }

    /// Detect text bounding boxes in an image.
    pub fn detect(&self, image: &DynamicImage) -> anyhow::Result<Vec<TextBox>> {
        ensure!(
            image.width() > 0 && image.height() > 0,
            "image dimensions must be greater than zero"
        );

        let processed = preprocess_for_detection(image)
            .context("failed to preprocess image for text detection")?;
        let tensor =
            pack_detection_tensor(&processed).context("failed to pack detector input tensor")?;

        let detector_input =
            TensorRef::from_array_view(&tensor).context("failed to build detector input tensor")?;

        let mut session = self
            .session
            .lock()
            .map_err(|_| anyhow!("detector session mutex poisoned"))?;
        let outputs = session
            .run(ort::inputs![detector_input])
            .context("detector inference failed")?;

        ensure!(outputs.len() > 0, "detector model produced no outputs");

        let probability_map = extract_probability_map(
            outputs[0]
                .try_extract_array::<f32>()
                .context("failed to extract detector output tensor")?,
        )?;

        Ok(probability_map_to_boxes(
            probability_map.view(),
            image.width(),
            image.height(),
        ))
    }
}

/// A detected text bounding box.
#[derive(Debug, Clone, PartialEq)]
pub struct TextBox {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub confidence: f32,
}

fn extract_probability_map(output: ArrayViewD<'_, f32>) -> anyhow::Result<Array2<f32>> {
    let mut output = output.to_owned();
    if output.iter().any(|value| *value < 0.0 || *value > 1.0) {
        output.mapv_inplace(sigmoid);
    }

    let shape = output.shape().to_vec();
    match output.ndim() {
        2 => output
            .into_dimensionality::<Ix2>()
            .context("failed to interpret detector output as [H, W]"),
        3 => {
            let output = output
                .into_dimensionality::<Ix3>()
                .context("failed to interpret detector output as rank-3 tensor")?;
            if shape[0] == 1 {
                Ok(output.index_axis_move(Axis(0), 0))
            } else if shape[2] == 1 {
                Ok(output.index_axis_move(Axis(2), 0))
            } else {
                bail!("unsupported detector output shape: {shape:?}")
            }
        }
        4 => {
            let output = output
                .into_dimensionality::<Ix4>()
                .context("failed to interpret detector output as rank-4 tensor")?;
            if shape[0] != 1 {
                bail!("unsupported detector batch dimension: {shape:?}");
            }

            if shape[1] == 1 {
                Ok(output
                    .index_axis_move(Axis(0), 0)
                    .index_axis_move(Axis(0), 0))
            } else if shape[3] == 1 {
                Ok(output
                    .index_axis_move(Axis(0), 0)
                    .index_axis_move(Axis(2), 0))
            } else {
                bail!("unsupported detector output shape: {shape:?}")
            }
        }
        _ => bail!("unsupported detector output rank: {shape:?}"),
    }
}

fn probability_map_to_boxes(
    probability_map: ArrayView2<'_, f32>,
    original_width: u32,
    original_height: u32,
) -> Vec<TextBox> {
    let map_height = probability_map.nrows();
    let map_width = probability_map.ncols();
    if map_width == 0 || map_height == 0 || original_width == 0 || original_height == 0 {
        return Vec::new();
    }

    let min_component_pixels = MIN_COMPONENT_PIXELS.max((map_width * map_height) / 100_000);
    let mut visited = vec![false; map_width * map_height];
    let mut boxes = Vec::new();

    for y in 0..map_height {
        for x in 0..map_width {
            let index = y * map_width + x;
            if visited[index] || probability_map[(y, x)] < DB_BINARY_THRESHOLD {
                continue;
            }

            let component = collect_component(probability_map, &mut visited, x, y);
            if component.pixel_count < min_component_pixels {
                continue;
            }

            let bbox_width = component.max_x - component.min_x + 1;
            let bbox_height = component.max_y - component.min_y + 1;
            let bbox_area = bbox_width * bbox_height;
            let fill_ratio = component.pixel_count as f32 / bbox_area as f32;
            let mean_probability = component.probability_sum / component.pixel_count as f32;
            let confidence = (mean_probability * fill_ratio.sqrt()).clamp(0.0, 1.0);

            let pad_x = ((bbox_width as f32 * BOX_PADDING_RATIO).round() as usize).max(1);
            let pad_y = ((bbox_height as f32 * BOX_PADDING_RATIO).round() as usize).max(1);

            let min_x = component.min_x.saturating_sub(pad_x);
            let min_y = component.min_y.saturating_sub(pad_y);
            let max_x = (component.max_x + pad_x).min(map_width - 1);
            let max_y = (component.max_y + pad_y).min(map_height - 1);

            let x0 = scale_start(min_x, map_width, original_width);
            let y0 = scale_start(min_y, map_height, original_height);
            let x1 = scale_end(max_x + 1, map_width, original_width);
            let y1 = scale_end(max_y + 1, map_height, original_height);
            let width = x1.saturating_sub(x0);
            let height = y1.saturating_sub(y0);

            if width < MIN_BOX_SIDE || height < MIN_BOX_SIDE {
                continue;
            }

            boxes.push(TextBox {
                x: x0,
                y: y0,
                width,
                height,
                confidence,
            });
        }
    }

    sort_text_boxes_in_reading_order(&mut boxes);
    boxes
}

#[derive(Debug)]
struct ComponentStats {
    min_x: usize,
    min_y: usize,
    max_x: usize,
    max_y: usize,
    pixel_count: usize,
    probability_sum: f32,
}

fn collect_component(
    probability_map: ArrayView2<'_, f32>,
    visited: &mut [bool],
    start_x: usize,
    start_y: usize,
) -> ComponentStats {
    let map_width = probability_map.ncols();
    let map_height = probability_map.nrows();
    let mut queue = VecDeque::from([(start_x, start_y)]);
    visited[start_y * map_width + start_x] = true;

    let mut component = ComponentStats {
        min_x: start_x,
        min_y: start_y,
        max_x: start_x,
        max_y: start_y,
        pixel_count: 0,
        probability_sum: 0.0,
    };

    while let Some((x, y)) = queue.pop_front() {
        let probability = probability_map[(y, x)];
        component.min_x = component.min_x.min(x);
        component.min_y = component.min_y.min(y);
        component.max_x = component.max_x.max(x);
        component.max_y = component.max_y.max(y);
        component.pixel_count += 1;
        component.probability_sum += probability;

        let x_start = x.saturating_sub(1);
        let y_start = y.saturating_sub(1);
        let x_end = (x + 1).min(map_width - 1);
        let y_end = (y + 1).min(map_height - 1);

        for next_y in y_start..=y_end {
            for next_x in x_start..=x_end {
                let index = next_y * map_width + next_x;
                if visited[index] || probability_map[(next_y, next_x)] < DB_BINARY_THRESHOLD {
                    continue;
                }
                visited[index] = true;
                queue.push_back((next_x, next_y));
            }
        }
    }

    component
}

fn scale_start(position: usize, source_extent: usize, target_extent: u32) -> u32 {
    (((position as u64) * u64::from(target_extent)) / source_extent as u64)
        .min(u64::from(target_extent)) as u32
}

fn scale_end(position: usize, source_extent: usize, target_extent: u32) -> u32 {
    (((position as u64) * u64::from(target_extent)).div_ceil(source_extent as u64))
        .min(u64::from(target_extent)) as u32
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use super::*;
    use image::{DynamicImage, ImageBuffer, Rgb};
    use ndarray::{arr2, ArrayD};

    #[test]
    fn constructor_fails_when_model_is_missing() {
        let error = Detector::new("tests/fixtures/missing-detector-model.onnx")
            .err()
            .expect("missing detector model should fail");
        assert!(error.to_string().contains("detector model not found"));
    }

    #[test]
    fn probability_map_extraction_handles_logits() {
        let output = ArrayD::from_shape_vec(vec![1, 1, 2, 2], vec![-4.0_f32, 0.0, 2.0, 4.0])
            .expect("shape is valid");

        let map = extract_probability_map(output.view()).expect("output extraction succeeds");

        assert_eq!(map.dim(), (2, 2));
        assert!(map[(0, 0)] < 0.05);
        assert!((map[(0, 1)] - 0.5).abs() < 1e-6);
        assert!(map[(1, 0)] > 0.85);
        assert!(map[(1, 1)] > 0.98);
    }

    #[test]
    fn postprocess_detects_connected_regions_and_scales_boxes() {
        let map = arr2(&[
            [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
            [0.05, 0.95, 0.95, 0.95, 0.05, 0.05, 0.05, 0.05],
            [0.05, 0.92, 0.90, 0.93, 0.05, 0.05, 0.05, 0.05],
            [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
            [0.05, 0.05, 0.05, 0.05, 0.05, 0.88, 0.88, 0.05],
            [0.05, 0.05, 0.05, 0.05, 0.05, 0.87, 0.90, 0.05],
            [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
            [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
        ]);

        let boxes = probability_map_to_boxes(map.view(), 160, 80);

        assert_eq!(boxes.len(), 2);

        let first = &boxes[0];
        assert_eq!(
            first,
            &TextBox {
                x: 0,
                y: 0,
                width: 100,
                height: 40,
                confidence: first.confidence,
            }
        );
        assert!(first.confidence > 0.85);

        let second = &boxes[1];
        assert_eq!(
            second,
            &TextBox {
                x: 80,
                y: 30,
                width: 80,
                height: 40,
                confidence: second.confidence,
            }
        );
        assert!(second.confidence > 0.8);
    }

    #[test]
    fn postprocess_filters_single_pixel_noise() {
        let map = arr2(&[
            [0.05, 0.05, 0.05, 0.05],
            [0.05, 0.95, 0.05, 0.05],
            [0.05, 0.05, 0.05, 0.05],
            [0.05, 0.05, 0.05, 0.05],
        ]);

        let boxes = probability_map_to_boxes(map.view(), 80, 40);

        assert!(boxes.is_empty());
    }

    #[test]
    fn smoke_test_runs_when_local_model_is_available() -> anyhow::Result<()> {
        let Some(model_path) = local_detector_model_path() else {
            return Ok(());
        };

        let detector = Detector::new(&model_path)?;
        let image =
            DynamicImage::ImageRgb8(ImageBuffer::from_pixel(256, 128, Rgb([255, 255, 255])));
        let boxes = detector.detect(&image)?;

        for detected in boxes {
            assert!(detected.width > 0);
            assert!(detected.height > 0);
            assert!(detected.x + detected.width <= image.width());
            assert!(detected.y + detected.height <= image.height());
            assert!(detected.confidence.is_finite());
            assert!((0.0..=1.0).contains(&detected.confidence));
        }

        Ok(())
    }

    fn local_detector_model_path() -> Option<String> {
        std::env::var("DOC2MSG_DET_MODEL")
            .ok()
            .filter(|value| Path::new(value).is_file())
            .or_else(|| {
                let repo_model = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("models")
                    .join("det_model.onnx");
                repo_model
                    .is_file()
                    .then(|| repo_model.to_string_lossy().into_owned())
            })
    }
}
