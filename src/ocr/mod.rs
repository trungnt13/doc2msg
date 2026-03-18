pub mod decode;
pub mod detector;
pub mod preprocess;
pub mod recognizer;

use std::env;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use anyhow::{bail, ensure, Context};
use image::DynamicImage;
use serde::Serialize;

use crate::ocr::detector::{Detector, TextBox};
use crate::ocr::recognizer::Recognizer;

pub const OCR_MODEL_NAME: &str = "openocr-repsvtr";

const DEFAULT_DETECTOR_MODEL: &str = "models/det_model.onnx";
const DEFAULT_RECOGNIZER_MODEL: &str = "models/rec_model.onnx";
const DEFAULT_DICTIONARY: &str = "models/ppocr_keys_v1.txt";
const DEFAULT_SESSION_POOL_SIZE: usize = 2;
const DEFAULT_MAX_BATCH: usize = 16;
const DEFAULT_INTRA_THREADS: usize = 4;
const DEFAULT_INTER_THREADS: usize = 1;
const DEFAULT_DEVICE_ID: i32 = -1;

/// Recognized text result from OCR.
#[derive(Debug, Clone, Serialize)]
pub struct OcrResult {
    pub text: String,
    pub confidence: f32,
}

/// Trait for OCR engines.
pub trait OcrEngine: Send + Sync {
    /// Run OCR on a preprocessed image and return recognized text.
    fn recognize(&self, image: &image::DynamicImage) -> anyhow::Result<Vec<OcrResult>>;
}

/// A recognized text region in document-reading order.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RecognizedRegion {
    pub bbox: TextBox,
    pub text: String,
    pub confidence: f32,
}

/// Full OCR output for a page-sized image.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DocumentOcrResult {
    pub markdown: String,
    pub regions: Vec<RecognizedRegion>,
}

pub(crate) struct DocumentOcr {
    detector: Detector,
    recognizer: Recognizer,
}

impl DocumentOcr {
    fn from_environment() -> anyhow::Result<Self> {
        let paths = OcrModelPaths::from_environment()?;
        let detector = Detector::new(&paths.detector_model.to_string_lossy())
            .context("failed to initialize full OCR detector")?;
        let recognizer = Recognizer::new(
            &paths.recognizer_model.to_string_lossy(),
            &paths.dictionary.to_string_lossy(),
            env_usize("DOC2AGENT_SESSION_POOL_SIZE", DEFAULT_SESSION_POOL_SIZE),
            env_usize("DOC2AGENT_MAX_BATCH", DEFAULT_MAX_BATCH),
            env_usize("DOC2AGENT_INTRA_THREADS", DEFAULT_INTRA_THREADS),
            env_usize("DOC2AGENT_INTER_THREADS", DEFAULT_INTER_THREADS),
            env_i32("DOC2AGENT_DEVICE_ID", DEFAULT_DEVICE_ID),
        )
        .context("failed to initialize full OCR recognizer")?;

        Ok(Self {
            detector,
            recognizer,
        })
    }

    pub(crate) fn recognize_document(
        &self,
        image: &DynamicImage,
    ) -> anyhow::Result<DocumentOcrResult> {
        let mut boxes = self
            .detector
            .detect(image)
            .context("failed to detect text regions for image OCR")?;
        sort_text_boxes_in_reading_order(&mut boxes);

        let (boxes, crops) = crop_text_regions(image, &boxes);
        if crops.is_empty() {
            return Ok(DocumentOcrResult {
                markdown: String::new(),
                regions: Vec::new(),
            });
        }

        let results = self
            .recognizer
            .recognize_batch(&crops)
            .context("failed to recognize detected text regions")?;
        ensure!(
            results.len() == boxes.len(),
            "OCR recognizer returned {} results for {} crops",
            results.len(),
            boxes.len()
        );

        let mut regions = Vec::with_capacity(results.len());
        for (bbox, result) in boxes.into_iter().zip(results) {
            let text = result.text.trim().to_string();
            if text.is_empty() {
                continue;
            }

            regions.push(RecognizedRegion {
                confidence: (bbox.confidence * result.confidence).clamp(0.0, 1.0),
                bbox,
                text,
            });
        }

        Ok(DocumentOcrResult {
            markdown: merge_regions_into_markdown(&regions),
            regions,
        })
    }
}

pub(crate) fn shared_document_ocr() -> anyhow::Result<Arc<DocumentOcr>> {
    static SHARED: OnceLock<Arc<DocumentOcr>> = OnceLock::new();

    if let Some(shared) = SHARED.get() {
        return Ok(Arc::clone(shared));
    }

    let created =
        Arc::new(DocumentOcr::from_environment().context("full OCR pipeline is unavailable")?);

    match SHARED.set(Arc::clone(&created)) {
        Ok(()) => Ok(created),
        Err(returned) => Ok(SHARED.get().cloned().unwrap_or(returned)),
    }
}

#[derive(Debug, Clone)]
struct OcrModelPaths {
    detector_model: PathBuf,
    recognizer_model: PathBuf,
    dictionary: PathBuf,
}

impl OcrModelPaths {
    fn from_environment() -> anyhow::Result<Self> {
        let detector_model = resolve_model_path(
            &["DOC2AGENT_DET_MODEL", "DOC2AGENT_TEST_DET_MODEL"],
            DEFAULT_DETECTOR_MODEL,
        );
        let recognizer_model = resolve_model_path(
            &[
                "DOC2AGENT_MODEL_PATH",
                "DOC2AGENT_REC_MODEL",
                "DOC2AGENT_TEST_REC_MODEL",
            ],
            DEFAULT_RECOGNIZER_MODEL,
        );
        let dictionary = resolve_model_path(
            &["DOC2AGENT_DICT_PATH", "DOC2AGENT_TEST_REC_DICT"],
            DEFAULT_DICTIONARY,
        );

        let mut missing = Vec::new();
        if detector_model.is_none() {
            missing.push(format!(
                "detector model not found (set DOC2AGENT_DET_MODEL or add {})",
                DEFAULT_DETECTOR_MODEL
            ));
        }
        if recognizer_model.is_none() {
            missing.push(format!(
                "recognizer model not found (set DOC2AGENT_MODEL_PATH or add {})",
                DEFAULT_RECOGNIZER_MODEL
            ));
        }
        if dictionary.is_none() {
            missing.push(format!(
                "recognizer dictionary not found (set DOC2AGENT_DICT_PATH or add {})",
                DEFAULT_DICTIONARY
            ));
        }

        if !missing.is_empty() {
            bail!("{}.", missing.join("; "));
        }

        Ok(Self {
            detector_model: detector_model.expect("checked above"),
            recognizer_model: recognizer_model.expect("checked above"),
            dictionary: dictionary.expect("checked above"),
        })
    }
}

fn resolve_model_path(env_keys: &[&str], repo_relative: &str) -> Option<PathBuf> {
    env_keys
        .iter()
        .filter_map(|key| env::var_os(key).map(PathBuf::from))
        .find(|path| path.is_file())
        .or_else(|| {
            let fallback = Path::new(env!("CARGO_MANIFEST_DIR")).join(repo_relative);
            fallback.is_file().then_some(fallback)
        })
}

fn env_usize(key: &str, default: usize) -> usize {
    env::var(key)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn env_i32(key: &str, default: i32) -> i32 {
    env::var(key)
        .ok()
        .and_then(|value| value.parse::<i32>().ok())
        .unwrap_or(default)
}

#[derive(Debug, Clone)]
struct LineBand {
    top: u32,
    bottom: u32,
    center_y: f32,
    average_height: f32,
}

impl LineBand {
    fn new(bbox: &TextBox) -> Self {
        Self {
            top: bbox.y,
            bottom: bbox_bottom(bbox),
            center_y: bbox_center_y(bbox),
            average_height: bbox.height as f32,
        }
    }

    fn absorb(&mut self, bbox: &TextBox) {
        self.top = self.top.min(bbox.y);
        self.bottom = self.bottom.max(bbox_bottom(bbox));
        self.center_y = (self.center_y + bbox_center_y(bbox)) / 2.0;
        self.average_height = (self.average_height + bbox.height as f32) / 2.0;
    }
}

fn bbox_bottom(bbox: &TextBox) -> u32 {
    bbox.y.saturating_add(bbox.height)
}

fn bbox_center_y(bbox: &TextBox) -> f32 {
    bbox.y as f32 + (bbox.height as f32 / 2.0)
}

fn fits_line_band(bbox: &TextBox, band: &LineBand) -> bool {
    let center_delta = (bbox_center_y(bbox) - band.center_y).abs();
    let overlap_top = bbox.y.max(band.top);
    let overlap_bottom = bbox_bottom(bbox).min(band.bottom);
    let overlap = overlap_bottom.saturating_sub(overlap_top);
    let min_height = bbox.height.min(band.bottom.saturating_sub(band.top));
    let overlap_ratio = if min_height == 0 {
        0.0
    } else {
        overlap as f32 / min_height as f32
    };
    let tolerance = (band.average_height.max(bbox.height as f32) * 0.6).max(12.0);

    center_delta <= tolerance || overlap_ratio >= 0.35
}

pub(crate) fn sort_text_boxes_in_reading_order(boxes: &mut [TextBox]) {
    if boxes.len() < 2 {
        return;
    }

    let mut ordered = boxes.to_vec();
    ordered.sort_by(|left, right| left.y.cmp(&right.y).then(left.x.cmp(&right.x)));

    let mut bands: Vec<Vec<TextBox>> = Vec::new();
    let mut band_meta: Vec<LineBand> = Vec::new();

    for bbox in ordered {
        if let Some((band, meta)) = bands
            .last_mut()
            .zip(band_meta.last_mut())
            .filter(|(_, meta)| fits_line_band(&bbox, meta))
        {
            meta.absorb(&bbox);
            band.push(bbox);
        } else {
            band_meta.push(LineBand::new(&bbox));
            bands.push(vec![bbox]);
        }
    }

    let mut flattened = Vec::with_capacity(boxes.len());
    for band in &mut bands {
        band.sort_by(|left, right| left.x.cmp(&right.x).then(left.y.cmp(&right.y)));
        flattened.extend(band.iter().cloned());
    }

    boxes.clone_from_slice(&flattened);
}

pub(crate) fn crop_text_regions(
    image: &DynamicImage,
    boxes: &[TextBox],
) -> (Vec<TextBox>, Vec<DynamicImage>) {
    let mut cropped_boxes = Vec::with_capacity(boxes.len());
    let mut crops = Vec::with_capacity(boxes.len());

    for bbox in boxes {
        let Some(clamped) = clamp_text_box(bbox, image.width(), image.height()) else {
            continue;
        };

        crops.push(image.crop_imm(clamped.x, clamped.y, clamped.width, clamped.height));
        cropped_boxes.push(clamped);
    }

    (cropped_boxes, crops)
}

fn clamp_text_box(bbox: &TextBox, image_width: u32, image_height: u32) -> Option<TextBox> {
    if image_width == 0 || image_height == 0 {
        return None;
    }

    let x0 = bbox.x.min(image_width);
    let y0 = bbox.y.min(image_height);
    let x1 = bbox.x.saturating_add(bbox.width).min(image_width);
    let y1 = bbox.y.saturating_add(bbox.height).min(image_height);
    let width = x1.saturating_sub(x0);
    let height = y1.saturating_sub(y0);

    (width > 0 && height > 0).then_some(TextBox {
        x: x0,
        y: y0,
        width,
        height,
        confidence: bbox.confidence,
    })
}

pub(crate) fn merge_regions_into_markdown(regions: &[RecognizedRegion]) -> String {
    if regions.is_empty() {
        return String::new();
    }

    let mut markdown = String::new();
    let mut lines: Vec<Vec<&RecognizedRegion>> = Vec::new();
    let mut bands: Vec<LineBand> = Vec::new();

    for region in regions {
        if let Some((line, band)) = lines
            .last_mut()
            .zip(bands.last_mut())
            .filter(|(_, band)| fits_line_band(&region.bbox, band))
        {
            band.absorb(&region.bbox);
            line.push(region);
        } else {
            bands.push(LineBand::new(&region.bbox));
            lines.push(vec![region]);
        }
    }

    let mut previous_bottom = None;
    let mut previous_height = None;

    for (line, band) in lines.into_iter().zip(bands.into_iter()) {
        let fragments = line
            .iter()
            .map(|region| region.text.trim())
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>();
        if fragments.is_empty() {
            continue;
        }

        if let Some(previous_bottom) = previous_bottom {
            let gap = band.top.saturating_sub(previous_bottom);
            let line_height = band.bottom.saturating_sub(band.top);
            let previous_height = previous_height.unwrap_or(line_height);
            if gap as f32 > previous_height.max(line_height) as f32 * 1.2 {
                markdown.push_str("\n\n");
            } else {
                markdown.push('\n');
            }
        }

        markdown.push_str(&fragments.join(" "));
        previous_bottom = Some(band.bottom);
        previous_height = Some(band.bottom.saturating_sub(band.top));
    }

    markdown
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb, Rgba};

    fn text_box(x: u32, y: u32, width: u32, height: u32) -> TextBox {
        TextBox {
            x,
            y,
            width,
            height,
            confidence: 0.9,
        }
    }

    #[test]
    fn reading_order_sorts_top_to_bottom_then_left_to_right() {
        let mut boxes = vec![
            text_box(140, 54, 48, 18),
            text_box(18, 12, 64, 22),
            text_box(122, 14, 50, 20),
            text_box(20, 52, 70, 20),
        ];

        sort_text_boxes_in_reading_order(&mut boxes);

        assert_eq!(
            boxes,
            vec![
                text_box(18, 12, 64, 22),
                text_box(122, 14, 50, 20),
                text_box(20, 52, 70, 20),
                text_box(140, 54, 48, 18),
            ]
        );
    }

    #[test]
    fn crop_text_regions_clamps_boxes_to_image_bounds() {
        let image = DynamicImage::ImageRgb8(ImageBuffer::from_fn(10, 6, |x, _| {
            if x < 5 {
                Rgb([255, 0, 0])
            } else {
                Rgb([0, 0, 255])
            }
        }));

        let boxes = vec![
            text_box(1, 1, 3, 2),
            text_box(7, 1, 5, 4),
            text_box(30, 30, 4, 4),
        ];

        let (clamped, crops) = crop_text_regions(&image, &boxes);

        assert_eq!(clamped.len(), 2);
        assert_eq!(crops.len(), 2);
        assert_eq!(crops[0].dimensions(), (3, 2));
        assert_eq!(crops[1].dimensions(), (3, 4));
        assert_eq!(crops[0].get_pixel(0, 0), Rgba([255, 0, 0, 255]));
        assert_eq!(crops[1].get_pixel(0, 0), Rgba([0, 0, 255, 255]));
    }

    #[test]
    fn markdown_merge_joins_regions_with_line_and_paragraph_breaks() {
        let regions = vec![
            RecognizedRegion {
                bbox: text_box(10, 10, 40, 16),
                text: "Hello".to_string(),
                confidence: 0.9,
            },
            RecognizedRegion {
                bbox: text_box(60, 12, 40, 16),
                text: "world".to_string(),
                confidence: 0.9,
            },
            RecognizedRegion {
                bbox: text_box(12, 54, 80, 18),
                text: "Second paragraph".to_string(),
                confidence: 0.9,
            },
        ];

        let markdown = merge_regions_into_markdown(&regions);

        assert_eq!(markdown, "Hello world\n\nSecond paragraph");
    }
}
