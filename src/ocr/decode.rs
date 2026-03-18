use std::fs;

use anyhow::Context;
use ndarray::{Array2, ArrayView1, Axis};

/// CTC (Connectionist Temporal Classification) decoder.
pub struct CtcDecoder {
    /// Character dictionary mapping indices to characters.
    dictionary: Vec<String>,
}

impl CtcDecoder {
    /// Load a CTC dictionary from a file path.
    pub fn new(dict_path: &str) -> anyhow::Result<Self> {
        let dictionary_contents = fs::read_to_string(dict_path)
            .with_context(|| format!("failed to read CTC dictionary from {dict_path}"))?;

        let mut dictionary = Vec::with_capacity(dictionary_contents.lines().count() + 1);
        dictionary.push(String::new());
        dictionary.extend(dictionary_contents.lines().map(ToOwned::to_owned));

        Ok(Self { dictionary })
    }

    /// Decode CTC output probabilities to text.
    pub fn decode(&self, probs: &Array2<f32>) -> Vec<(String, f32)> {
        vec![self.decode_sequence(probs)]
    }

    /// Number of output classes expected by the decoder, including the blank token.
    pub fn class_count(&self) -> usize {
        self.dictionary.len()
    }

    fn decode_sequence(&self, probs: &Array2<f32>) -> (String, f32) {
        if probs.nrows() == 0 || probs.ncols() == 0 {
            return (String::new(), 0.0);
        }

        let mut text = String::new();
        let mut confidence_sum = 0.0_f32;
        let mut retained_tokens = 0_usize;
        let mut previous_index: Option<usize> = None;

        for timestep in probs.axis_iter(Axis(0)) {
            let Some((index, confidence)) = greedy_argmax(timestep) else {
                continue;
            };

            if previous_index == Some(index) {
                continue;
            }
            previous_index = Some(index);

            if index == 0 {
                continue;
            }

            let Some(token) = self.dictionary.get(index) else {
                continue;
            };

            text.push_str(token);
            confidence_sum += confidence;
            retained_tokens += 1;
        }

        let confidence = if retained_tokens == 0 {
            0.0
        } else {
            confidence_sum / retained_tokens as f32
        };

        (text, confidence)
    }
}

fn greedy_argmax(timestep: ArrayView1<'_, f32>) -> Option<(usize, f32)> {
    timestep
        .iter()
        .copied()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
}

#[cfg(test)]
mod tests {
    use std::{
        env,
        path::{Path, PathBuf},
        time::{SystemTime, UNIX_EPOCH},
    };

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
                "doc2msg-ctc-dict-{}-{unique_id}.txt",
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

    fn build_decoder(tokens: &[&str]) -> Result<CtcDecoder> {
        let dictionary = TempDictionary::new(tokens)?;
        let path = dictionary.path().to_string_lossy().into_owned();
        CtcDecoder::new(&path)
    }

    #[test]
    fn test_ctc_decoder_collapses_repeated_tokens() -> Result<()> {
        let decoder = build_decoder(&["a", "b"])?;
        let probs = array![
            [0.1, 0.8, 0.1],
            [0.1, 0.7, 0.2],
            [0.05, 0.05, 0.9],
            [0.1, 0.1, 0.8],
        ];

        let decoded = decoder.decode(&probs);

        assert_eq!(decoded, vec![("ab".to_string(), 0.85)]);
        Ok(())
    }

    #[test]
    fn test_ctc_decoder_removes_blanks_and_resets_repeats() -> Result<()> {
        let decoder = build_decoder(&["a"])?;
        let probs = array![[0.9, 0.1], [0.2, 0.8], [0.1, 0.9], [0.95, 0.05], [0.1, 0.9],];

        let decoded = decoder.decode(&probs);

        assert_eq!(decoded, vec![("aa".to_string(), 0.85)]);
        Ok(())
    }

    #[test]
    fn test_ctc_decoder_aggregates_token_confidence() -> Result<()> {
        let decoder = build_decoder(&["a", "b", "c"])?;
        let probs = array![
            [0.05, 0.9, 0.03, 0.02],
            [0.1, 0.2, 0.6, 0.1],
            [0.2, 0.2, 0.3, 0.3],
        ];

        let decoded = decoder.decode(&probs);

        assert_eq!(decoded[0].0, "abc");
        assert!((decoded[0].1 - 0.6).abs() < f32::EPSILON);
        Ok(())
    }

    #[test]
    fn test_ctc_decoder_skips_out_of_range_indices() -> Result<()> {
        let decoder = build_decoder(&["a"])?;
        let probs = array![
            [0.01, 0.02, 0.03, 0.94],
            [0.1, 0.8, 0.05, 0.05],
            [0.02, 0.03, 0.05, 0.9],
        ];

        let decoded = decoder.decode(&probs);

        assert_eq!(decoded, vec![("a".to_string(), 0.8)]);
        Ok(())
    }

    #[test]
    fn test_ctc_decoder_handles_empty_input() -> Result<()> {
        let decoder = build_decoder(&["a"])?;
        let probs = Array2::<f32>::zeros((0, 2));

        let decoded = decoder.decode(&probs);

        assert_eq!(decoded, vec![(String::new(), 0.0)]);
        Ok(())
    }
}
