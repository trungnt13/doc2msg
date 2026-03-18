use std::ops::Range;
use std::path::Path;
use std::slice;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard};

use anyhow::{bail, ensure, Context};
use image::DynamicImage;
use ndarray::{Array2, Array4, ArrayView2, ArrayView3, ArrayViewD, Axis, Ix2, Ix3};
use ort::execution_providers::{CPUExecutionProvider, ExecutionProviderDispatch};
use ort::session::Session;
use ort::value::Tensor;

use crate::ocr::decode::CtcDecoder;
use crate::ocr::preprocess::{pack_recognition_tensor, preprocess_for_recognition};
use crate::ocr::{OcrEngine, OcrResult};

const DEFAULT_RECOGNITION_HEIGHT: u32 = 48;
const PROBABILITY_EPSILON: f32 = 1e-2;

/// OpenOCR RepSVTR-based text recognizer backed by ONNX Runtime.
pub struct Recognizer {
    decoder: CtcDecoder,
    sessions: Vec<Mutex<Session>>,
    next_session: AtomicUsize,
    max_batch: usize,
    input_height: u32,
}

impl Recognizer {
    /// Create a new recognizer with a session pool.
    pub fn new(
        model_path: &str,
        dict_path: &str,
        pool_size: usize,
        max_batch: usize,
        intra_threads: usize,
        inter_threads: usize,
        device_id: i32,
    ) -> anyhow::Result<Self> {
        ensure!(
            pool_size > 0,
            "recognizer session pool size must be greater than zero"
        );
        ensure!(
            max_batch > 0,
            "recognizer max batch must be greater than zero"
        );

        let decoder = CtcDecoder::new(dict_path)
            .with_context(|| format!("failed to load recognizer dictionary from {dict_path}"))?;
        let execution_providers = execution_providers(device_id);

        let first_session = create_session(
            model_path,
            &execution_providers,
            intra_threads,
            inter_threads,
        )
        .with_context(|| format!("failed to create OCR recognizer session from {model_path}"))?;
        let input_height = extract_input_height(&first_session)?;
        validate_model_outputs(&first_session)?;

        let mut sessions = Vec::with_capacity(pool_size);
        sessions.push(Mutex::new(first_session));

        for _ in 1..pool_size {
            let session = create_session(
                model_path,
                &execution_providers,
                intra_threads,
                inter_threads,
            )
            .with_context(|| {
                format!("failed to create OCR recognizer session from {model_path}")
            })?;
            sessions.push(Mutex::new(session));
        }

        Ok(Self {
            decoder,
            sessions,
            next_session: AtomicUsize::new(0),
            max_batch,
            input_height,
        })
    }

    /// Recognize text from a batch of image crops.
    pub fn recognize_batch(&self, images: &[DynamicImage]) -> anyhow::Result<Vec<OcrResult>> {
        if images.is_empty() {
            return Ok(Vec::new());
        }

        let mut results = Vec::with_capacity(images.len());
        for range in batch_ranges(images.len(), self.max_batch) {
            let batch_tensor = build_batch_tensor(&images[range.clone()], self.input_height)?;
            let input = Tensor::from_array(batch_tensor)
                .context("failed to convert OCR batch tensor into ORT input")?;

            let mut session = self.checkout_session()?;
            let outputs = session
                .run(ort::inputs![input])
                .context("failed to run OCR recognizer inference")?;
            let (_, output) = outputs
                .into_iter()
                .next()
                .context("OCR recognizer model returned no outputs")?;
            let output = output
                .try_extract_array::<f32>()
                .context("failed to extract OCR recognizer output tensor as f32 array")?;

            let mut batch_results = self.decode_batch_output(output, range.len())?;
            results.append(&mut batch_results);
        }

        Ok(results)
    }

    fn checkout_session(&self) -> anyhow::Result<MutexGuard<'_, Session>> {
        let index = self.next_session.fetch_add(1, Ordering::Relaxed) % self.sessions.len();
        self.sessions[index]
            .lock()
            .map_err(|_| anyhow::anyhow!("OCR recognizer session mutex was poisoned"))
    }

    fn decode_batch_output(
        &self,
        output: ArrayViewD<'_, f32>,
        batch_size: usize,
    ) -> anyhow::Result<Vec<OcrResult>> {
        match output.ndim() {
            2 => {
                ensure!(
                    batch_size == 1,
                    "OCR recognizer returned rank-2 output for batch size {batch_size}: {:?}",
                    output.shape()
                );
                let sample = output
                    .into_dimensionality::<Ix2>()
                    .context("failed to interpret OCR recognizer output as rank-2 tensor")?;
                Ok(vec![self.decode_sample(sample)?])
            }
            3 => {
                let output = output
                    .into_dimensionality::<Ix3>()
                    .context("failed to interpret OCR recognizer output as rank-3 tensor")?;
                self.decode_rank3_output(output, batch_size)
            }
            rank => bail!(
                "OCR recognizer output tensor must have rank 2 or 3, got rank {rank} with shape {:?}",
                output.shape()
            ),
        }
    }

    fn decode_rank3_output(
        &self,
        output: ArrayView3<'_, f32>,
        batch_size: usize,
    ) -> anyhow::Result<Vec<OcrResult>> {
        let shape = output.shape();
        let batch_axis = if shape[0] == batch_size {
            0
        } else if shape[1] == batch_size {
            1
        } else if shape[2] == batch_size {
            2
        } else {
            bail!(
                "OCR recognizer output shape {:?} does not contain the expected batch size {batch_size}",
                shape
            );
        };

        let mut results = Vec::with_capacity(batch_size);
        for index in 0..batch_size {
            let sample = output.index_axis(Axis(batch_axis), index);
            results.push(self.decode_sample(sample)?);
        }

        Ok(results)
    }

    fn decode_sample(&self, raw_scores: ArrayView2<'_, f32>) -> anyhow::Result<OcrResult> {
        let probabilities = orient_and_normalize_scores(raw_scores, self.decoder.class_count())?;
        let mut decoded = self.decoder.decode(&probabilities);
        let (text, confidence) = decoded
            .pop()
            .context("CTC decoder returned no OCR result for recognizer output")?;

        Ok(OcrResult { text, confidence })
    }
}

impl OcrEngine for Recognizer {
    fn recognize(&self, image: &DynamicImage) -> anyhow::Result<Vec<OcrResult>> {
        self.recognize_batch(slice::from_ref(image))
    }
}

fn create_session(
    model_path: &str,
    execution_providers: &[ExecutionProviderDispatch],
    intra_threads: usize,
    inter_threads: usize,
) -> anyhow::Result<Session> {
    let builder = Session::builder().context("failed to create ORT session builder")?;
    let builder = builder
        .with_independent_thread_pool()
        .context("failed to enable independent ORT thread pool")?;
    let builder = builder
        .with_no_environment_execution_providers()
        .context("failed to disable environment execution providers for recognizer session")?;
    let builder = builder
        .with_intra_threads(intra_threads)
        .with_context(|| format!("failed to set ORT intra-op threads to {intra_threads}"))?;
    let builder = builder
        .with_inter_threads(inter_threads)
        .with_context(|| format!("failed to set ORT inter-op threads to {inter_threads}"))?;
    let builder = builder
        .with_parallel_execution(inter_threads > 1)
        .context("failed to configure ORT parallel execution mode")?;
    let builder = builder
        .with_execution_providers(execution_providers)
        .context("failed to configure ORT execution providers for recognizer session")?;

    builder.commit_from_file(model_path).with_context(|| {
        format!(
            "failed to load OCR recognizer model from {}",
            Path::new(model_path).display()
        )
    })
}

fn execution_providers(_device_id: i32) -> Vec<ExecutionProviderDispatch> {
    vec![
        #[cfg(feature = "tensorrt-ep")]
        ort::execution_providers::TensorRTExecutionProvider::default()
            .with_device_id(_device_id)
            .build()
            .fail_silently(),
        #[cfg(feature = "cuda-ep")]
        ort::execution_providers::CUDAExecutionProvider::default()
            .with_device_id(_device_id)
            .build()
            .fail_silently(),
        CPUExecutionProvider::default()
            .with_arena_allocator(true)
            .build()
            .error_on_failure(),
    ]
}

fn extract_input_height(session: &Session) -> anyhow::Result<u32> {
    let input = session
        .inputs
        .first()
        .context("OCR recognizer model exposes no inputs")?;
    let shape = input
        .input_type
        .tensor_shape()
        .context("OCR recognizer model first input is not a tensor")?;

    if shape.len() < 3 {
        bail!(
            "OCR recognizer input tensor must have at least 3 dimensions, got {:?}",
            shape
        );
    }

    let height = shape[2];
    if height > 0 {
        return u32::try_from(height)
            .with_context(|| format!("OCR recognizer input height {height} is out of range"));
    }

    Ok(DEFAULT_RECOGNITION_HEIGHT)
}

fn validate_model_outputs(session: &Session) -> anyhow::Result<()> {
    let output = session
        .outputs
        .first()
        .context("OCR recognizer model exposes no outputs")?;
    ensure!(
        output.output_type.tensor_shape().is_some(),
        "OCR recognizer model first output is not a tensor"
    );
    Ok(())
}

fn build_batch_tensor(images: &[DynamicImage], input_height: u32) -> anyhow::Result<Array4<f32>> {
    ensure!(
        !images.is_empty(),
        "cannot build OCR recognizer input tensor from an empty image batch"
    );

    let mut tensors = Vec::with_capacity(images.len());
    for image in images {
        let processed = preprocess_for_recognition(image, input_height).with_context(|| {
            format!(
                "failed to preprocess OCR crop with dimensions {}x{}",
                image.width(),
                image.height()
            )
        })?;
        let tensor = pack_recognition_tensor(&processed)
            .context("failed to pack OCR crop into recognition tensor")?;
        tensors.push(tensor);
    }

    let sample_shape = tensors[0].shape().to_vec();
    ensure!(
        sample_shape.len() == 4,
        "OCR recognition tensor must be rank 4, got shape {:?}",
        sample_shape
    );

    let mut batch = Array4::<f32>::zeros((
        images.len(),
        sample_shape[1],
        sample_shape[2],
        sample_shape[3],
    ));

    for (index, tensor) in tensors.iter().enumerate() {
        ensure!(
            tensor.shape() == sample_shape.as_slice(),
            "OCR recognition tensors must share a shape, expected {:?}, got {:?}",
            sample_shape,
            tensor.shape()
        );
        batch
            .index_axis_mut(Axis(0), index)
            .assign(&tensor.index_axis(Axis(0), 0));
    }

    Ok(batch)
}

fn orient_and_normalize_scores(
    raw_scores: ArrayView2<'_, f32>,
    class_count: usize,
) -> anyhow::Result<Array2<f32>> {
    let scores = if raw_scores.shape()[0] == class_count && raw_scores.shape()[1] != class_count {
        raw_scores.t().to_owned()
    } else {
        raw_scores.to_owned()
    };

    normalize_ctc_scores(scores)
}

fn normalize_ctc_scores(scores: Array2<f32>) -> anyhow::Result<Array2<f32>> {
    if looks_like_probabilities(scores.view()) {
        return Ok(scores);
    }

    let mut probabilities = scores;
    for mut row in probabilities.axis_iter_mut(Axis(0)) {
        let mut max_value = f32::NEG_INFINITY;
        for value in row.iter().copied() {
            ensure!(
                value.is_finite(),
                "OCR recognizer output contains non-finite values"
            );
            max_value = max_value.max(value);
        }

        let mut sum = 0.0_f32;
        for value in &mut row {
            *value = (*value - max_value).exp();
            sum += *value;
        }
        ensure!(
            sum.is_finite() && sum > 0.0,
            "failed to normalize OCR recognizer logits into probabilities"
        );

        for value in &mut row {
            *value /= sum;
        }
    }

    Ok(probabilities)
}

fn looks_like_probabilities(scores: ArrayView2<'_, f32>) -> bool {
    scores.axis_iter(Axis(0)).all(|row| {
        let mut sum = 0.0_f32;
        for value in row.iter().copied() {
            if !value.is_finite()
                || !(-PROBABILITY_EPSILON..=1.0 + PROBABILITY_EPSILON).contains(&value)
            {
                return false;
            }
            sum += value;
        }

        sum == 0.0 || (sum - 1.0).abs() <= PROBABILITY_EPSILON
    })
}

fn batch_ranges(total: usize, max_batch: usize) -> Vec<Range<usize>> {
    (0..total)
        .step_by(max_batch)
        .map(|start| start..(start + max_batch).min(total))
        .collect()
}

#[cfg(test)]
mod tests {
    use std::env;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    use anyhow::Result;
    use ndarray::array;

    use super::*;

    struct TempDictionary {
        path: PathBuf,
    }

    impl TempDictionary {
        fn new(tokens: &[&str]) -> Result<Self> {
            let unique_id = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
            let path = env::temp_dir().join(format!(
                "doc2agent-recognizer-dict-{}-{unique_id}.txt",
                std::process::id()
            ));
            fs::write(&path, tokens.join("\n"))?;
            Ok(Self { path })
        }

        fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TempDictionary {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.path);
        }
    }

    #[test]
    fn constructor_requires_positive_pool_size_and_batch_size() {
        let error = Recognizer::new("missing.onnx", "missing.txt", 0, 1, 1, 1, -1)
            .err()
            .expect("invalid pool size should fail");
        assert!(error.to_string().contains("pool size"));

        let error = Recognizer::new("missing.onnx", "missing.txt", 1, 0, 1, 1, -1)
            .err()
            .expect("invalid max batch should fail");
        assert!(error.to_string().contains("max batch"));
    }

    #[test]
    fn constructor_surfaces_missing_dictionary_before_model_load() {
        let error = Recognizer::new("missing.onnx", "missing.txt", 1, 1, 1, 1, -1)
            .err()
            .expect("missing dictionary should fail");
        assert!(error.to_string().contains("dictionary"));
    }

    #[test]
    fn constructor_fails_gracefully_when_model_is_missing() -> Result<()> {
        let dictionary = TempDictionary::new(&["a", "b"])?;
        let dict_path = dictionary.path().to_string_lossy().into_owned();

        let error = Recognizer::new("missing.onnx", &dict_path, 1, 1, 1, 1, -1)
            .err()
            .expect("missing model should fail");

        assert!(error
            .to_string()
            .contains("failed to create OCR recognizer session from missing.onnx"));
        Ok(())
    }

    #[test]
    fn batch_ranges_split_inputs_by_max_batch() {
        assert_eq!(batch_ranges(0, 3), Vec::<Range<usize>>::new());
        assert_eq!(batch_ranges(1, 3), vec![0..1]);
        assert_eq!(batch_ranges(5, 2), vec![0..2, 2..4, 4..5]);
    }

    #[test]
    fn logits_are_softmax_normalized_before_decoding() -> Result<()> {
        let probabilities = normalize_ctc_scores(array![[3.0_f32, 1.0, -1.0], [0.0, 0.0, 0.0]])?;

        for row in probabilities.axis_iter(Axis(0)) {
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() <= PROBABILITY_EPSILON);
        }

        Ok(())
    }

    #[test]
    fn decoder_orients_class_first_output() -> Result<()> {
        let dictionary = TempDictionary::new(&["a", "b"])?;
        let decoder = CtcDecoder::new(&dictionary.path().to_string_lossy())?;
        let raw_scores = array![[0.05_f32, 0.05], [0.90_f32, 0.10], [0.05_f32, 0.85],];

        let probabilities = orient_and_normalize_scores(raw_scores.view(), decoder.class_count())?;
        let decoded = decoder.decode(&probabilities);

        assert_eq!(decoded, vec![("ab".to_string(), 0.875)]);
        Ok(())
    }
}
