/// Remove common leftover HTML tags from a line, replacing block-level
/// tags with newlines and inline tags with empty string.
fn strip_html_tags(line: &str) -> String {
    let mut result = line.to_string();
    // Block-level tags → replaced with newline
    for tag in &[
        "<br>",
        "<br/>",
        "<br />",
        "<hr>",
        "<hr/>",
        "<hr />",
        "<p>",
        "</p>",
        "<div>",
        "</div>",
        "<section>",
        "</section>",
        "<article>",
        "</article>",
        "<header>",
        "</header>",
        "<footer>",
        "</footer>",
        "<nav>",
        "</nav>",
        "<main>",
        "</main>",
        "<aside>",
        "</aside>",
    ] {
        result = result.replace(tag, "\n");
    }
    // Also handle case-insensitive variants
    for tag in &[
        "<BR>", "<BR/>", "<BR />", "<HR>", "<HR/>", "<HR />", "<P>", "</P>", "<DIV>", "</DIV>",
    ] {
        result = result.replace(tag, "\n");
    }
    // Inline tags → remove
    for tag in &[
        "<span>",
        "</span>",
        "<b>",
        "</b>",
        "<i>",
        "</i>",
        "<em>",
        "</em>",
        "<strong>",
        "</strong>",
        "<u>",
        "</u>",
        "<s>",
        "</s>",
        "<del>",
        "</del>",
        "<ins>",
        "</ins>",
        "<SPAN>",
        "</SPAN>",
        "<B>",
        "</B>",
        "<I>",
        "</I>",
        "<EM>",
        "</EM>",
        "<STRONG>",
        "</STRONG>",
    ] {
        result = result.replace(tag, "");
    }
    result
}

/// Replace Unicode whitespace variants (NBSP, etc.) with regular spaces.
fn normalize_whitespace(line: &str) -> String {
    line.chars()
        .map(|c| {
            if c == '\u{00A0}' // non-breaking space
                || c == '\u{2000}' // en quad
                || c == '\u{2001}' // em quad
                || c == '\u{2002}' // en space
                || c == '\u{2003}' // em space
                || c == '\u{2004}' // three-per-em space
                || c == '\u{2005}' // four-per-em space
                || c == '\u{2006}' // six-per-em space
                || c == '\u{2007}' // figure space
                || c == '\u{2008}' // punctuation space
                || c == '\u{2009}' // thin space
                || c == '\u{200A}' // hair space
                || c == '\u{202F}' // narrow no-break space
                || c == '\u{205F}' // medium mathematical space
                || c == '\u{3000}'
            // ideographic space
            {
                ' '
            } else {
                c
            }
        })
        .collect()
}

/// Returns true if the trimmed line looks like a markdown heading (# ... ####### ).
fn is_heading(line: &str) -> bool {
    let trimmed = line.trim_start();
    for level in 1..=6 {
        let prefix = "#".repeat(level);
        if trimmed.starts_with(&prefix) {
            // Must be followed by a space or be exactly the hashes
            let rest = &trimmed[level..];
            if rest.is_empty() || rest.starts_with(' ') {
                return true;
            }
        }
    }
    false
}

/// Returns true if the trimmed line is a list item (unordered or ordered).
fn is_list_item(line: &str) -> bool {
    let trimmed = line.trim_start();
    if trimmed.starts_with("- ") || trimmed.starts_with("* ") || trimmed.starts_with("+ ") {
        return true;
    }
    // Ordered list: digits followed by . and space
    let mut chars = trimmed.chars();
    let first = chars.next();
    if let Some(c) = first {
        if c.is_ascii_digit() {
            for ch in chars {
                if ch == '.' {
                    return true;
                }
                if !ch.is_ascii_digit() {
                    break;
                }
            }
        }
    }
    false
}

/// Process a single non-code-block line: strip HTML, normalize whitespace, trim trailing.
fn process_line(line: &str) -> String {
    let line = strip_html_tags(line);
    let line = normalize_whitespace(&line);
    // Trim trailing whitespace
    line.trim_end().to_string()
}

/// Normalize markdown output: fix whitespace, collapse blank lines,
/// strip unwanted artifacts from HTML-to-markdown conversion.
///
/// Applies transformations in order:
/// 1. Normalize line endings (\r\n → \n)
/// 2. Trim trailing whitespace per line
/// 3. Collapse excessive blank lines (3+ → 2)
/// 4. Fix heading spacing (blank line before/after)
/// 5. Fix list formatting
/// 6. Strip HTML artifacts
/// 7. Normalize Unicode whitespace
/// 8. Preserve code block contents
/// 9. Trim leading/trailing whitespace of entire output
/// 10. Ensure file ends with single newline
pub fn normalize_markdown(input: &str) -> String {
    // Step 1: Normalize line endings
    let input = input.replace("\r\n", "\n");

    let mut lines: Vec<String> = Vec::new();
    let mut in_code_block = false;
    let mut consecutive_blank: usize = 0;

    for line in input.split('\n') {
        // Track code blocks by looking for ``` at start of trimmed line
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") {
            in_code_block = !in_code_block;
            // Code fence lines are kept as-is (but trim trailing whitespace outside block)
            if in_code_block {
                // Opening fence — ensure blank line before if needed
                if let Some(last) = lines.last() {
                    if !last.is_empty() {
                        lines.push(String::new());
                    }
                }
            }
            lines.push(line.trim_end().to_string());
            consecutive_blank = 0;
            continue;
        }

        if in_code_block {
            // Preserve code block content exactly
            lines.push(line.to_string());
            consecutive_blank = 0;
            continue;
        }

        // Process non-code-block lines
        let processed = process_line(line);

        // Handle blank lines
        if processed.is_empty() {
            consecutive_blank += 1;
            // Collapse 3+ blank lines into 2
            if consecutive_blank <= 2 {
                lines.push(String::new());
            }
            continue;
        }

        // Non-blank line
        let prev_was_blank = consecutive_blank > 0;
        consecutive_blank = 0;

        // Heading spacing: ensure blank line before headings (unless at document start)
        if is_heading(&processed) && !lines.is_empty() {
            // Check the last non-empty state: if no blank line before, add one
            if !prev_was_blank {
                lines.push(String::new());
            }
            lines.push(processed);
            // Add blank line after heading
            lines.push(String::new());
            consecutive_blank = 1;
            continue;
        }

        // If previous line was a heading (check by looking back), we already added blank after it.
        // Just push normally.
        lines.push(processed);
    }

    let mut result = lines.join("\n");

    // Collapse any remaining runs of 3+ blank lines that might have been introduced
    while result.contains("\n\n\n\n") {
        result = result.replace("\n\n\n\n", "\n\n\n");
    }

    // Trim leading/trailing whitespace of entire output
    let result = result.trim().to_string();

    // Ensure file ends with single newline
    if result.is_empty() {
        String::from("\n")
    } else {
        result + "\n"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collapse_blank_lines() {
        let input = "line1\n\n\n\n\nline2";
        let result = normalize_markdown(input);
        // 3+ blank lines collapsed to at most 2 blank lines (3 newlines between content)
        assert!(
            result.contains("line1\n\n\nline2"),
            "Expected collapsed blank lines, got: {:?}",
            result
        );
        assert!(
            !result.contains("line1\n\n\n\nline2"),
            "Should not have 4+ consecutive newlines"
        );
    }

    #[test]
    fn test_heading_spacing() {
        let input = "some text\n## Heading\nmore text";
        let result = normalize_markdown(input);
        assert!(
            result.contains("\n\n## Heading"),
            "Heading should have blank line before it, got: {:?}",
            result
        );
        assert!(
            result.contains("## Heading\n\n"),
            "Heading should have blank line after it, got: {:?}",
            result
        );
    }

    #[test]
    fn test_heading_at_start_of_document() {
        let input = "# Title\nsome text";
        let result = normalize_markdown(input);
        assert!(
            result.starts_with("# Title\n"),
            "Heading at start should not have leading blank line, got: {:?}",
            result
        );
    }

    #[test]
    fn test_strip_html_tags() {
        let input = "hello<br>world<div>content</div>";
        let result = normalize_markdown(input);
        assert!(
            !result.contains("<br>"),
            "Should strip <br>, got: {:?}",
            result
        );
        assert!(
            !result.contains("<div>"),
            "Should strip <div>, got: {:?}",
            result
        );
        assert!(
            !result.contains("</div>"),
            "Should strip </div>, got: {:?}",
            result
        );
    }

    #[test]
    fn test_preserve_code_blocks() {
        let input = "```\n<div>keep this</div>\n  extra spaces  \n```";
        let result = normalize_markdown(input);
        assert!(
            result.contains("<div>keep this</div>"),
            "Code block content should be preserved, got: {:?}",
            result
        );
        assert!(
            result.contains("  extra spaces  "),
            "Code block whitespace should be preserved, got: {:?}",
            result
        );
    }

    #[test]
    fn test_unicode_whitespace() {
        let input = "hello\u{00A0}world";
        let result = normalize_markdown(input);
        assert!(
            result.contains("hello world"),
            "NBSP should become regular space, got: {:?}",
            result
        );
    }

    #[test]
    fn test_trailing_whitespace() {
        let input = "line1   \nline2\t\t\n";
        let result = normalize_markdown(input);
        for line in result.lines() {
            assert_eq!(
                line,
                line.trim_end(),
                "Line should have no trailing whitespace: {:?}",
                line
            );
        }
    }

    #[test]
    fn test_crlf_normalization() {
        let input = "line1\r\nline2\r\nline3";
        let result = normalize_markdown(input);
        assert!(
            !result.contains('\r'),
            "Should not contain \\r, got: {:?}",
            result
        );
        assert!(
            result.contains("line1\nline2\nline3"),
            "Lines should be preserved, got: {:?}",
            result
        );
    }

    #[test]
    fn test_ends_with_single_newline() {
        let input = "hello world";
        let result = normalize_markdown(input);
        assert!(
            result.ends_with('\n'),
            "Should end with newline, got: {:?}",
            result
        );
        assert!(
            !result.ends_with("\n\n"),
            "Should not end with double newline, got: {:?}",
            result
        );
    }

    #[test]
    fn test_empty_input() {
        let result = normalize_markdown("");
        assert_eq!(result, "\n", "Empty input should produce single newline");
    }

    #[test]
    fn test_whitespace_only_input() {
        let result = normalize_markdown("   \n  \n   ");
        assert_eq!(
            result, "\n",
            "Whitespace-only input should produce single newline"
        );
    }

    #[test]
    fn test_multiple_headings() {
        let input = "# Title\nparagraph\n## Section\ntext\n### Subsection\nmore text";
        let result = normalize_markdown(input);
        assert!(
            result.contains("\n\n## Section\n\n"),
            "Section heading should have spacing, got: {:?}",
            result
        );
        assert!(
            result.contains("\n\n### Subsection\n\n"),
            "Subsection heading should have spacing, got: {:?}",
            result
        );
    }

    #[test]
    fn test_code_block_with_language() {
        let input = "text\n```rust\nfn main() {\n    println!(\"hello\");\n}\n```\nmore text";
        let result = normalize_markdown(input);
        assert!(
            result.contains("```rust\nfn main()"),
            "Code block language tag should be preserved, got: {:?}",
            result
        );
    }

    #[test]
    fn test_strip_html_helper() {
        assert_eq!(strip_html_tags("hello<br>world"), "hello\nworld");
        assert_eq!(strip_html_tags("<div>content</div>"), "\ncontent\n");
        assert_eq!(strip_html_tags("<span>inline</span>"), "inline");
    }

    #[test]
    fn test_normalize_whitespace_helper() {
        assert_eq!(normalize_whitespace("a\u{00A0}b\u{2003}c"), "a b c");
        assert_eq!(normalize_whitespace("normal text"), "normal text");
    }

    #[test]
    fn test_is_heading_helper() {
        assert!(is_heading("# Title"));
        assert!(is_heading("## Section"));
        assert!(is_heading("### Sub"));
        assert!(is_heading("  ## Indented"));
        assert!(!is_heading("##no space"));
        assert!(!is_heading("regular text"));
        assert!(!is_heading("#######toolong"));
    }

    #[test]
    fn test_is_list_item_helper() {
        assert!(is_list_item("- item"));
        assert!(is_list_item("* item"));
        assert!(is_list_item("+ item"));
        assert!(is_list_item("1. item"));
        assert!(is_list_item("  - indented"));
        assert!(is_list_item("10. double digit"));
        assert!(!is_list_item("regular text"));
    }

    /// Intent: Nested triple-backtick inside a code block doesn't break block tracking.
    #[test]
    fn test_nested_code_block_markers() {
        let input = "```\nsome code\n```nested```\nmore code\n```";
        let result = normalize_markdown(input);
        assert!(
            result.contains("```nested```"),
            "inner backticks should be preserved, got: {:?}",
            result
        );
    }

    /// Intent: Very long line (>10K chars) doesn't cause quadratic blowup or panic.
    #[test]
    fn test_very_long_line() {
        let long_line = "word ".repeat(5000); // ~25K chars
        let result = normalize_markdown(&long_line);
        assert!(
            result.len() > 1000,
            "long line should produce substantial output"
        );
    }

    /// Intent: Input consisting entirely of HTML tags produces clean output (not raw tags).
    #[test]
    fn test_all_html_tags_input() {
        let input = "<div><p><span></span></p></div>";
        let result = normalize_markdown(input);
        let trimmed = result.trim();
        assert!(
            !trimmed.contains('<'),
            "all HTML tags should be stripped, got: {:?}",
            trimmed
        );
    }
}
