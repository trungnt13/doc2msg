use serde::Serialize;

pub const DEFAULT_MIN_CHUNK_CHARS: usize = 1200;
pub const DEFAULT_MAX_CHUNK_CHARS: usize = 2200;
pub const DEFAULT_OVERLAP_CHARS: usize = 150;

/// A single chunk of extracted document text.
#[derive(Debug, Clone, Serialize)]
pub struct Chunk {
    pub id: String,
    pub text: String,
    pub section: Option<String>,
    pub page_start: Option<u32>,
    pub page_end: Option<u32>,
    pub char_count: usize,
    pub token_estimate: usize,
}

/// Full output from a document extraction pipeline.
#[derive(Debug, Clone, Serialize)]
pub struct DocumentOutput {
    pub title: Option<String>,
    pub canonical_url: Option<String>,
    pub markdown: String,
    pub chunks: Vec<Chunk>,
    pub metadata: DocumentMetadata,
    pub diagnostics: PipelineDiagnostics,
    pub image_manifest: Option<Vec<ImageRef>>,
}

/// Metadata about the source document.
#[derive(Debug, Clone, Serialize)]
pub struct DocumentMetadata {
    pub page_count: Option<u32>,
    pub word_count: usize,
    pub char_count: usize,
}

/// Diagnostics from the extraction pipeline.
#[derive(Debug, Clone, Serialize)]
pub struct PipelineDiagnostics {
    pub pipeline_used: String,
    pub ocr_used: bool,
    pub render_used: bool,
    pub fallback_used: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fallback_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text_quality_score: Option<f32>,
    pub latency_ms: u128,
}

/// Reference to an image found in the document.
#[derive(Debug, Clone, Serialize)]
pub struct ImageRef {
    pub page: u32,
    pub url: Option<String>,
    pub alt: Option<String>,
}

/// Approximate token count: byte length / 4.
pub fn estimate_tokens(text: &str) -> usize {
    text.len() / 4
}

/// Extract the first level-1 heading (`# ...`) from markdown.
pub fn extract_title(markdown: &str) -> Option<String> {
    for line in markdown.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("# ") {
            let title = rest.trim();
            if !title.is_empty() {
                return Some(title.to_string());
            }
        }
        // Ensure we only match exactly `# `, not `##` etc.
        // strip_prefix("# ") already handles this correctly.
    }
    None
}

/// Compute document metadata from the full markdown text.
pub fn compute_metadata(markdown: &str, page_count: Option<u32>) -> DocumentMetadata {
    DocumentMetadata {
        page_count,
        word_count: markdown.split_whitespace().count(),
        char_count: markdown.len(),
    }
}

// -- Internal helpers --

/// A section of text associated with an optional heading and page range.
struct Section {
    heading: Option<String>,
    body: String,
    page_start: Option<u32>,
    page_end: Option<u32>,
}

/// Determine whether the document contains page markers (form-feed or `<!-- page N -->`).
fn has_page_markers(markdown: &str) -> bool {
    markdown.contains('\x0c') || markdown.contains("<!-- page ")
}

/// Compute the page number at a given byte offset by counting form-feed chars
/// and `<!-- page N -->` markers preceding it.
fn page_at_offset(markdown: &str, offset: usize) -> Option<u32> {
    if !has_page_markers(markdown) {
        return None;
    }
    let prefix = &markdown[..offset.min(markdown.len())];
    // Start on page 1; every page marker advances to the next page.
    let ff_count = prefix.matches('\x0c').count();
    let marker_count = prefix.matches("<!-- page ").count();
    let total = ff_count + marker_count;
    Some(total as u32 + 1)
}

/// Parse markdown into sections split at heading boundaries, tracking pages.
fn split_into_sections(markdown: &str) -> Vec<Section> {
    let has_pages = has_page_markers(markdown);
    let mut sections: Vec<Section> = Vec::new();
    let mut current_heading: Option<String> = None;
    let mut current_body = String::new();
    let mut section_start_offset: usize = 0;

    for line in markdown.lines() {
        let trimmed = line.trim();
        // Detect headings: lines starting with one or more `#` followed by a space.
        if trimmed.starts_with('#') {
            if let Some(space_pos) = trimmed.find(' ') {
                let hashes = &trimmed[..space_pos];
                if !hashes.is_empty() && hashes.chars().all(|c| c == '#') {
                    // Flush the current section.
                    let body_text = current_body.trim().to_string();
                    if !body_text.is_empty() || current_heading.is_some() {
                        let page_start = if has_pages {
                            page_at_offset(markdown, section_start_offset)
                        } else {
                            None
                        };
                        // end offset is roughly where the body ends
                        let body_end = section_start_offset + current_body.len();
                        let page_end = if has_pages {
                            page_at_offset(markdown, body_end)
                        } else {
                            None
                        };
                        sections.push(Section {
                            heading: current_heading.take(),
                            body: body_text,
                            page_start,
                            page_end,
                        });
                    }
                    current_heading = Some(trimmed[space_pos + 1..].trim().to_string());
                    current_body.clear();
                    // Record the start offset of this new section within the original markdown.
                    // We approximate by searching forward from the previous position.
                    if let Some(pos) = markdown[section_start_offset..].find(trimmed) {
                        section_start_offset += pos;
                    }
                    continue;
                }
            }
        }
        current_body.push_str(line);
        current_body.push('\n');
    }

    // Flush last section.
    let body_text = current_body.trim().to_string();
    if !body_text.is_empty() || current_heading.is_some() {
        let page_start = if has_pages {
            page_at_offset(markdown, section_start_offset)
        } else {
            None
        };
        let page_end = if has_pages {
            page_at_offset(markdown, markdown.len())
        } else {
            None
        };
        sections.push(Section {
            heading: current_heading,
            body: body_text,
            page_start,
            page_end,
        });
    }

    sections
}

/// Split a piece of text that exceeds `max_chars` at paragraph (`\n\n`), sentence,
/// or hard character boundaries.
fn split_oversized(text: &str, max_chars: usize) -> Vec<String> {
    if text.len() <= max_chars {
        return vec![text.to_string()];
    }

    let mut result: Vec<String> = Vec::new();

    // Try splitting on paragraph boundaries first.
    let paragraphs: Vec<&str> = text.split("\n\n").collect();
    if paragraphs.len() > 1 {
        let mut buf = String::new();
        for para in &paragraphs {
            let addition = if buf.is_empty() {
                para.to_string()
            } else {
                format!("\n\n{para}")
            };
            if buf.len() + addition.len() > max_chars && !buf.is_empty() {
                result.push(buf.clone());
                buf.clear();
                buf.push_str(para);
            } else {
                buf.push_str(&addition);
            }
        }
        if !buf.is_empty() {
            result.push(buf);
        }
        // If every fragment is within limit, we're done.
        if result.iter().all(|r| r.len() <= max_chars) {
            return result;
        }
        // Otherwise, recursively split any fragments that are still too large.
        let mut final_result: Vec<String> = Vec::new();
        for fragment in result {
            if fragment.len() <= max_chars {
                final_result.push(fragment);
            } else {
                final_result.extend(split_at_sentences(&fragment, max_chars));
            }
        }
        return final_result;
    }

    // Single paragraph — split at sentences.
    split_at_sentences(text, max_chars)
}

/// Split text at sentence boundaries, falling back to hard-split.
fn split_at_sentences(text: &str, max_chars: usize) -> Vec<String> {
    if text.len() <= max_chars {
        return vec![text.to_string()];
    }

    let mut result: Vec<String> = Vec::new();
    let mut buf = String::new();

    // Split after sentence-ending punctuation followed by a space.
    let mut remaining = text;
    while !remaining.is_empty() {
        // Find the next sentence boundary within reach.
        let search_end = remaining.len().min(max_chars + 200); // look slightly ahead
        let search_slice = &remaining[..search_end.min(remaining.len())];

        if let Some(split_pos) = find_sentence_break(search_slice, max_chars) {
            let sentence = &remaining[..split_pos];
            if buf.len() + sentence.len() > max_chars && !buf.is_empty() {
                result.push(buf.clone());
                buf.clear();
            }
            buf.push_str(sentence);
            remaining = &remaining[split_pos..];
        } else {
            // No sentence boundary found — hard-split.
            if buf.len() + remaining.len() <= max_chars {
                buf.push_str(remaining);
                remaining = "";
            } else if !buf.is_empty() {
                result.push(buf.clone());
                buf.clear();
                // Don't advance remaining — retry with empty buffer.
            } else {
                // Hard-split at max_chars, respecting char boundary.
                let split = find_char_boundary(remaining, max_chars);
                result.push(remaining[..split].to_string());
                remaining = &remaining[split..];
            }
        }
    }
    if !buf.is_empty() {
        result.push(buf);
    }
    result
}

/// Find a sentence-ending position (`. `, `? `, `! `) within `max_chars` of the text.
/// Returns the byte index *after* the sentence-ending punctuation + space.
fn find_sentence_break(text: &str, max_chars: usize) -> Option<usize> {
    let limit = text.len().min(max_chars);
    let search = &text[..limit];
    // Find the last sentence boundary within the limit.
    let mut best: Option<usize> = None;
    for (i, _) in search.char_indices() {
        if i + 2 <= search.len() {
            let two = &search[i..i + 2];
            if two == ". " || two == "? " || two == "! " {
                best = Some(i + 2);
            }
        }
    }
    best
}

/// Find a valid UTF-8 char boundary at or before `pos`.
fn find_char_boundary(s: &str, pos: usize) -> usize {
    let mut p = pos.min(s.len());
    while p > 0 && !s.is_char_boundary(p) {
        p -= 1;
    }
    if p == 0 && !s.is_empty() {
        // Ensure we make progress — advance to at least one char.
        s.char_indices().nth(1).map_or(s.len(), |(i, _)| i)
    } else {
        p
    }
}

/// Build the overlap prefix from the previous chunk's text.
fn overlap_prefix(prev_text: &str) -> String {
    if prev_text.len() <= DEFAULT_OVERLAP_CHARS {
        return prev_text.to_string();
    }
    let start = find_char_boundary(prev_text, prev_text.len() - DEFAULT_OVERLAP_CHARS);
    prev_text[start..].to_string()
}

/// Split markdown text into chunks by headings and size limits.
pub fn chunk_markdown(markdown: &str, max_chars: usize) -> Vec<Chunk> {
    let trimmed = markdown.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }

    let sections = split_into_sections(trimmed);
    let has_pages = has_page_markers(trimmed);

    let mut chunks: Vec<Chunk> = Vec::new();
    let mut chunk_index: usize = 0;
    let mut prev_text = String::new();

    for section in &sections {
        let body = &section.body;
        if body.is_empty() {
            continue;
        }

        let fragments = split_oversized(body, max_chars);

        for (frag_idx, fragment) in fragments.iter().enumerate() {
            let frag_text = fragment.trim();
            if frag_text.is_empty() {
                continue;
            }

            // Apply overlap from previous chunk.
            let full_text = if !prev_text.is_empty() && chunk_index > 0 {
                let prefix = overlap_prefix(&prev_text);
                format!("{prefix}\n\n{frag_text}")
            } else {
                frag_text.to_string()
            };

            // char_count is the length of the fragment's own text (excluding overlap).
            let char_count = frag_text.len();

            let id = if has_pages {
                let page = section.page_start.unwrap_or(1);
                format!("p{page:02}-c{:02}", chunk_index + 1)
            } else {
                format!("c{:02}", chunk_index + 1)
            };

            // Section label: use heading for the first fragment, carry it for subsequent splits.
            let section_label = if frag_idx == 0 {
                section.heading.clone()
            } else {
                section.heading.as_ref().map(|h| format!("{h} (cont.)"))
            };

            chunks.push(Chunk {
                id,
                text: full_text.clone(),
                section: section_label,
                page_start: section.page_start,
                page_end: section.page_end,
                char_count,
                token_estimate: estimate_tokens(&full_text),
            });

            prev_text = frag_text.to_string();
            chunk_index += 1;
        }
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_text_single_chunk() {
        let text = "Hello world, this is a short document.";
        let chunks = chunk_markdown(text, DEFAULT_MAX_CHUNK_CHARS);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].section, None);
        assert_eq!(chunks[0].id, "c01");
        assert!(chunks[0].token_estimate > 0);
    }

    #[test]
    fn test_heading_split() {
        let text = "## Introduction\n\nSome intro text.\n\n## Methods\n\nSome methods text.";
        let chunks = chunk_markdown(text, DEFAULT_MAX_CHUNK_CHARS);
        assert!(chunks.len() >= 2);
        assert_eq!(chunks[0].section, Some("Introduction".to_string()));
        assert_eq!(chunks[1].section, Some("Methods".to_string()));
    }

    #[test]
    fn test_long_section_split() {
        let paragraph = "This is a test paragraph with enough words to be meaningful. ";
        let repeated = paragraph.repeat(100);
        let text = format!("## Section\n\n{repeated}");
        let chunks = chunk_markdown(&text, 2200);
        assert!(
            chunks.len() > 1,
            "Expected multiple chunks, got {}",
            chunks.len()
        );
        for chunk in &chunks {
            // char_count is the fragment's own text length; allow some tolerance.
            assert!(
                chunk.char_count <= 2200 + 200,
                "Chunk too large: {} chars",
                chunk.char_count
            );
        }
    }

    #[test]
    fn test_page_tracking_formfeed() {
        let text = "Page 1 content\x0cPage 2 content";
        let chunks = chunk_markdown(text, DEFAULT_MAX_CHUNK_CHARS);
        assert!(!chunks.is_empty());
        // First chunk should be on page 1.
        assert_eq!(chunks[0].page_start, Some(1));
        assert!(chunks[0].id.starts_with('p'));
    }

    #[test]
    fn test_page_tracking_html_comment() {
        let text = "Page 1 content\n<!-- page 2 -->\nPage 2 content";
        let chunks = chunk_markdown(text, DEFAULT_MAX_CHUNK_CHARS);
        assert!(!chunks.is_empty());
        assert_eq!(chunks[0].page_start, Some(1));
    }

    #[test]
    fn test_chunk_ids_no_pages() {
        let text = "## A\n\nText A\n\n## B\n\nText B\n\n## C\n\nText C";
        let chunks = chunk_markdown(text, DEFAULT_MAX_CHUNK_CHARS);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].id, "c01");
        assert_eq!(chunks[1].id, "c02");
        assert_eq!(chunks[2].id, "c03");
    }

    #[test]
    fn test_chunk_ids_with_pages() {
        let text = "## A\n\nText A\x0c## B\n\nText B";
        let chunks = chunk_markdown(text, DEFAULT_MAX_CHUNK_CHARS);
        assert!(!chunks.is_empty());
        // All IDs should include page prefix.
        for chunk in &chunks {
            assert!(
                chunk.id.starts_with('p'),
                "Expected page prefix in id: {}",
                chunk.id
            );
        }
    }

    #[test]
    fn test_token_estimate() {
        let chunks = chunk_markdown("Hello world this is a test", DEFAULT_MAX_CHUNK_CHARS);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].token_estimate > 0);
        // "Hello world this is a test" is 26 bytes → 26/4 = 6
        assert_eq!(estimate_tokens("Hello world this is a test"), 6);
    }

    #[test]
    fn test_extract_title() {
        assert_eq!(
            extract_title("# My Title\n\nContent"),
            Some("My Title".to_string())
        );
        assert_eq!(extract_title("## Not a title\n\nContent"), None);
        assert_eq!(extract_title("No heading at all"), None);
    }

    #[test]
    fn test_extract_title_with_leading_content() {
        assert_eq!(
            extract_title("Some preamble\n# The Real Title\nMore content"),
            Some("The Real Title".to_string())
        );
    }

    #[test]
    fn test_empty_input() {
        let chunks = chunk_markdown("", DEFAULT_MAX_CHUNK_CHARS);
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_whitespace_only_input() {
        let chunks = chunk_markdown("   \n\n  \n", DEFAULT_MAX_CHUNK_CHARS);
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_compute_metadata() {
        let md = "Hello world\n\nAnother paragraph here.";
        let meta = compute_metadata(md, Some(3));
        assert_eq!(meta.page_count, Some(3));
        assert_eq!(meta.word_count, 5);
        assert_eq!(meta.char_count, md.len());
    }

    #[test]
    fn test_overlap_between_chunks() {
        // Build text long enough to produce multiple chunks.
        let sentence = "This is a moderately long sentence that helps fill up the chunk nicely. ";
        let body = sentence.repeat(80);
        let text = format!("## Long\n\n{body}");
        let chunks = chunk_markdown(&text, 2200);
        assert!(
            chunks.len() >= 2,
            "Expected >=2 chunks, got {}",
            chunks.len()
        );
        // The second chunk's full text should start with the tail of the first chunk's content.
        let first_tail =
            &chunks[0].text[chunks[0].text.len().saturating_sub(DEFAULT_OVERLAP_CHARS)..];
        // The overlap is a prefix of chunk 2's text.
        assert!(
            chunks[1].text.contains(first_tail.trim()),
            "Expected overlap from first chunk in second chunk"
        );
    }

    #[test]
    fn test_hard_split_very_long_word() {
        // A single "word" longer than max_chars must be hard-split.
        let long = "a".repeat(5000);
        let chunks = chunk_markdown(&long, 2200);
        assert!(chunks.len() > 1, "Expected hard-split into multiple chunks");
    }

    #[test]
    fn test_multiple_heading_levels() {
        let text = "# Title\n\nIntro\n\n## Section\n\nBody\n\n### Subsection\n\nDetail";
        let chunks = chunk_markdown(text, DEFAULT_MAX_CHUNK_CHARS);
        // The `# Title` line itself is parsed as a section heading.
        assert!(chunks.len() >= 3);
    }
}
