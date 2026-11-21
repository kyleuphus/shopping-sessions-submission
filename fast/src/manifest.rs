//! Reads `manifest.json`, which defines the dataset's time window.
//!
//! The file's shape is fixed by `rules.md`:
//!   {"window_start": "2026-10-01T00:00:00Z", "window_end": "2026-10-15T00:00:00Z"}
//! so a 20-line extractor beats pulling in a JSON crate. It is read exactly once
//! per run, so its speed is irrelevant — only its correctness matters.

/// Extract the string value of `"<key>": "<value>"` from JSON bytes.
/// Returns None if the key is absent or the value is not a quoted string.
fn string_field(json: &[u8], key: &str) -> Option<String> {
    let needle = format!("\"{key}\"");
    let start = find(json, needle.as_bytes())? + needle.len();
    let rest = &json[start..];
    // Skip whitespace, then the colon, then whitespace, then expect an opening quote.
    let mut i = 0;
    while i < rest.len() && (rest[i] as char).is_whitespace() {
        i += 1;
    }
    if i >= rest.len() || rest[i] != b':' {
        return None;
    }
    i += 1;
    while i < rest.len() && (rest[i] as char).is_whitespace() {
        i += 1;
    }
    if i >= rest.len() || rest[i] != b'"' {
        return None;
    }
    i += 1;
    let value_start = i;
    while i < rest.len() && rest[i] != b'"' {
        i += 1;
    }
    if i >= rest.len() {
        return None;
    }
    String::from_utf8(rest[value_start..i].to_vec()).ok()
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

/// The dataset's window, as the raw strings from the manifest.
/// Conversion to epoch milliseconds is the caller's job (via `crate::ts::parse_ts`),
/// so there is exactly one timestamp parser in the program.
#[derive(Debug, Clone)]
pub struct Manifest {
    pub window_start: String,
    pub window_end: String,
}

pub fn read(data_dir: &std::path::Path) -> Result<Manifest, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(data_dir.join("manifest.json"))?;
    let window_start =
        string_field(&bytes, "window_start").ok_or("manifest.json: missing window_start")?;
    let window_end =
        string_field(&bytes, "window_end").ok_or("manifest.json: missing window_end")?;
    Ok(Manifest {
        window_start,
        window_end,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_documented_shape() {
        let json = br#"{"window_start": "2026-10-01T00:00:00Z", "window_end": "2026-10-15T00:00:00Z"}"#;
        assert_eq!(
            string_field(json, "window_start").unwrap(),
            "2026-10-01T00:00:00Z"
        );
        assert_eq!(
            string_field(json, "window_end").unwrap(),
            "2026-10-15T00:00:00Z"
        );
    }

    #[test]
    fn parses_pretty_printed_and_reordered_json() {
        // The fixtures ship a pretty-printed manifest, so whitespace must not matter.
        let json = b"{\n  \"window_end\"  :  \"2026-10-15T00:00:00Z\",\n  \"window_start\":\"2026-10-01T00:00:00Z\"\n}";
        assert_eq!(
            string_field(json, "window_start").unwrap(),
            "2026-10-01T00:00:00Z"
        );
        assert_eq!(
            string_field(json, "window_end").unwrap(),
            "2026-10-15T00:00:00Z"
        );
    }

    #[test]
    fn missing_key_is_none() {
        assert!(string_field(br#"{"a": "b"}"#, "window_start").is_none());
    }
}
