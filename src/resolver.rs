use bytes::Bytes;
use serde::Serialize;

/// Errors that can occur during URL resolution or content classification.
#[derive(thiserror::Error, Debug)]
pub enum ResolverError {
    #[error("fetch failed: {0}")]
    Fetch(#[from] reqwest::Error),
    #[error("empty response body")]
    EmptyBody,
    #[error("unsupported content type: {0}")]
    UnsupportedType(String),
}

/// The kind of source document detected.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum SourceKind {
    Web,
    Pdf,
    Image,
    Markdown,
    PlainText,
}

/// Descriptor carrying the resolved source data and metadata.
#[derive(Debug, Clone)]
pub struct SourceDescriptor {
    pub canonical_url: Option<String>,
    pub source_kind: SourceKind,
    pub mime: String,
    pub filename: Option<String>,
    pub raw_bytes: Bytes,
}

/// Resolve a URL to a SourceDescriptor by fetching and sniffing content.
pub async fn resolve_url(
    client: &reqwest::Client,
    url: &str,
) -> Result<SourceDescriptor, ResolverError> {
    let effective_url = upgrade_url(url);

    let response = client
        .get(&effective_url)
        .send()
        .await?
        .error_for_status()?;

    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    let filename = extract_filename_from_headers(response.headers())
        .or_else(|| extract_filename_from_url(&effective_url));

    let canonical_url = Some(effective_url);

    let raw_bytes = response.bytes().await?;
    if raw_bytes.is_empty() {
        return Err(ResolverError::EmptyBody);
    }

    let mime = determine_mime(content_type.as_deref(), filename.as_deref(), &raw_bytes);
    let source_kind = classify_mime(&mime);

    Ok(SourceDescriptor {
        canonical_url,
        source_kind,
        mime,
        filename,
        raw_bytes,
    })
}

/// Resolve raw bytes to a SourceDescriptor by sniffing content.
pub fn resolve_bytes(
    data: Bytes,
    filename: Option<&str>,
    mime: Option<&str>,
) -> Result<SourceDescriptor, ResolverError> {
    if data.is_empty() {
        return Err(ResolverError::EmptyBody);
    }

    let detected_mime = determine_mime(mime, filename, &data);
    let source_kind = classify_mime(&detected_mime);

    Ok(SourceDescriptor {
        canonical_url: None,
        source_kind,
        mime: detected_mime,
        filename: filename.map(|s| s.to_string()),
        raw_bytes: data,
    })
}

/// Map a MIME type string to a SourceKind.
pub fn classify_mime(mime: &str) -> SourceKind {
    // Normalize: take only the essence (e.g. "text/html; charset=utf-8" → "text/html")
    let essence = mime.split(';').next().unwrap_or(mime).trim().to_lowercase();

    match essence.as_str() {
        "text/html" | "application/xhtml+xml" => SourceKind::Web,
        "application/pdf" => SourceKind::Pdf,
        "text/markdown" | "text/x-markdown" => SourceKind::Markdown,
        "text/plain" => SourceKind::PlainText,
        _ if essence.starts_with("image/") => SourceKind::Image,
        _ => SourceKind::PlainText,
    }
}

/// Sniff the first bytes of data to guess a MIME type via magic bytes.
pub fn sniff_magic_bytes(data: &[u8]) -> Option<String> {
    if data.len() < 4 {
        return None;
    }

    // PDF: starts with %PDF
    if data.starts_with(b"%PDF") {
        return Some("application/pdf".to_string());
    }

    // PNG: starts with 0x89 P N G
    if data.starts_with(&[0x89, b'P', b'N', b'G']) {
        return Some("image/png".to_string());
    }

    // JPEG: starts with 0xFF 0xD8 0xFF
    if data.len() >= 3 && data.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return Some("image/jpeg".to_string());
    }

    // WEBP: RIFF....WEBP
    if data.len() >= 12 && data.starts_with(b"RIFF") && &data[8..12] == b"WEBP" {
        return Some("image/webp".to_string());
    }

    // HTML: look for common HTML markers (case-insensitive)
    let prefix = &data[..data.len().min(256)];
    let prefix_lower = String::from_utf8_lossy(prefix).to_lowercase();
    let trimmed = prefix_lower.trim_start();
    if trimmed.starts_with("<html") || trimmed.starts_with("<!doctype") {
        return Some("text/html".to_string());
    }

    // Markdown: starts with heading marker
    if data.starts_with(b"# ") || data.starts_with(b"## ") || data.starts_with(b"### ") {
        return Some("text/markdown".to_string());
    }

    None
}

/// Handle special URL transformations (e.g. arxiv abstract → PDF).
pub fn upgrade_url(url: &str) -> String {
    // arxiv.org/abs/<id> → arxiv.org/pdf/<id>.pdf
    if let Some(rest) = url
        .strip_prefix("https://arxiv.org/abs/")
        .or_else(|| url.strip_prefix("http://arxiv.org/abs/"))
    {
        // Strip any trailing query string or fragment from the id
        let id = rest
            .split(['?', '#'])
            .next()
            .unwrap_or(rest)
            .trim_end_matches('/');
        if !id.is_empty() {
            let scheme = if url.starts_with("https") {
                "https"
            } else {
                "http"
            };
            return format!("{scheme}://arxiv.org/pdf/{id}.pdf");
        }
    }

    url.to_string()
}

/// Determine MIME type from multiple sources in priority order:
/// header Content-Type > filename extension > magic bytes.
fn determine_mime(content_type: Option<&str>, filename: Option<&str>, data: &[u8]) -> String {
    // 1. From Content-Type header (if specific enough)
    if let Some(ct) = content_type {
        let essence = ct.split(';').next().unwrap_or(ct).trim();
        if !essence.is_empty() && essence != "application/octet-stream" {
            return essence.to_string();
        }
    }

    // 2. From filename extension via mime_guess
    if let Some(name) = filename {
        let guess = mime_guess::from_path(name).first_raw();
        if let Some(m) = guess {
            return m.to_string();
        }
    }

    // 3. From magic bytes
    if let Some(m) = sniff_magic_bytes(data) {
        return m;
    }

    // Fallback
    "application/octet-stream".to_string()
}

/// Extract filename from Content-Disposition header.
fn extract_filename_from_headers(headers: &reqwest::header::HeaderMap) -> Option<String> {
    let disposition = headers.get("content-disposition")?.to_str().ok()?;

    // Look for filename*= (RFC 5987 extended) first, then filename=
    for param in disposition.split(';').skip(1) {
        let param = param.trim();
        if let Some(value) = param.strip_prefix("filename*=") {
            // Format: UTF-8''encoded_name or utf-8''encoded_name
            if let Some(name) = value.split("''").nth(1) {
                let decoded = percent_decode(name);
                if !decoded.is_empty() {
                    return Some(decoded);
                }
            }
        }
    }
    for param in disposition.split(';').skip(1) {
        let param = param.trim();
        if let Some(value) = param.strip_prefix("filename=") {
            let name = value.trim_matches('"').trim();
            if !name.is_empty() {
                return Some(name.to_string());
            }
        }
    }

    None
}

/// Extract filename from the URL path segment.
fn extract_filename_from_url(url: &str) -> Option<String> {
    let path = url.split('?').next().unwrap_or(url);
    let segment = path.rsplit('/').next()?;
    let decoded = percent_decode(segment);
    if decoded.is_empty() || !decoded.contains('.') {
        return None;
    }
    Some(decoded)
}

/// Simple percent-decoding for filenames.
fn percent_decode(input: &str) -> String {
    let mut result = Vec::new();
    let bytes = input.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(byte) = u8::from_str_radix(&String::from_utf8_lossy(&bytes[i + 1..i + 3]), 16)
            {
                result.push(byte);
                i += 3;
                continue;
            }
        }
        result.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&result).to_string()
}
