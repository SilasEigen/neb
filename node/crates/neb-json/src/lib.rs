//! JSON reader and writer for the JSON-RPC interface (spec/node.md section
//! 7.2) and the wallet file, with no dependency beyond the standard library.
//!
//! Numbers are `i64` integers; a fraction or exponent is a parse error, so
//! amounts and heights never pass through floating point. Duplicate object
//! keys are rejected, so one text has one meaning. Objects are ordered maps:
//! output has sorted keys and no whitespace. The parser is recursive and has
//! no nesting limit; callers must bound the input size.

use std::collections::BTreeMap;
use std::fmt;
use std::fmt::Write as _;

#[derive(Clone, Debug, PartialEq, Eq)]
/// A parsed JSON value.
pub enum Value {
    Null,
    Bool(bool),
    Number(i64),
    String(String),
    Array(Vec<Value>),
    Object(BTreeMap<String, Value>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// A parse error.
pub struct JsonError {
    /// Byte offset in the input where parsing failed.
    pub at: usize,
    pub message: String,
}

impl fmt::Display for JsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "json at byte {}: {}", self.at, self.message)
    }
}

impl std::error::Error for JsonError {}

fn err(at: usize, message: impl Into<String>) -> JsonError {
    JsonError { at, message: message.into() }
}

impl Value {
    /// The member `key` of an object; `None` for other kinds of value.
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Object(map) => map.get(key),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::Number(n) => Some(*n),
            _ => None,
        }
    }

    /// `None` for negative numbers.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::Number(n) if *n >= 0 => Some(*n as u64),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }

    pub fn as_object(&self) -> Option<&BTreeMap<String, Value>> {
        match self {
            Value::Object(o) => Some(o),
            _ => None,
        }
    }

    pub fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Null => f.write_str("null"),
            Value::Bool(true) => f.write_str("true"),
            Value::Bool(false) => f.write_str("false"),
            Value::Number(n) => write!(f, "{n}"),
            Value::String(s) => write_escaped(f, s),
            Value::Array(items) => {
                f.write_str("[")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        f.write_str(",")?;
                    }
                    write!(f, "{item}")?;
                }
                f.write_str("]")
            }
            Value::Object(map) => {
                f.write_str("{")?;
                for (i, (key, value)) in map.iter().enumerate() {
                    if i > 0 {
                        f.write_str(",")?;
                    }
                    write_escaped(f, key)?;
                    f.write_str(":")?;
                    write!(f, "{value}")?;
                }
                f.write_str("}")
            }
        }
    }
}

fn write_escaped(f: &mut fmt::Formatter<'_>, s: &str) -> fmt::Result {
    f.write_str("\"")?;
    for c in s.chars() {
        match c {
            '"' => f.write_str("\\\"")?,
            '\\' => f.write_str("\\\\")?,
            '\n' => f.write_str("\\n")?,
            '\r' => f.write_str("\\r")?,
            '\t' => f.write_str("\\t")?,
            '\u{08}' => f.write_str("\\b")?,
            '\u{0c}' => f.write_str("\\f")?,
            c if (c as u32) < 0x20 => write!(f, "\\u{:04x}", c as u32)?,
            c => f.write_char(c)?,
        }
    }
    f.write_str("\"")
}

/// Parses one complete value; only whitespace may follow it.
pub fn parse(text: &str) -> Result<Value, JsonError> {
    let mut parser = Parser { bytes: text.as_bytes(), pos: 0, depth: 0 };
    let value = parser.value()?;
    parser.skip_ws();
    if parser.pos != parser.bytes.len() {
        return Err(err(parser.pos, "trailing bytes"));
    }
    Ok(value)
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
    /// Number of arrays and objects open at the current position.
    depth: usize,
}

/// Most arrays and objects that may be open at once; deeper input is an error.
pub const MAX_DEPTH: usize = 64;

impl<'a> Parser<'a> {
    fn skip_ws(&mut self) {
        while self.pos < self.bytes.len() && matches!(self.bytes[self.pos], b' ' | b'\t' | b'\n' | b'\r') {
            self.pos += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn expect(&mut self, byte: u8) -> Result<(), JsonError> {
        if self.peek() == Some(byte) {
            self.pos += 1;
            Ok(())
        } else {
            Err(err(self.pos, format!("expected '{}'", byte as char)))
        }
    }

    fn value(&mut self) -> Result<Value, JsonError> {
        self.skip_ws();
        match self.peek() {
            None => Err(err(self.pos, "unexpected end of input")),
            Some(b'n') => {
                self.literal("null")?;
                Ok(Value::Null)
            }
            Some(b't') => {
                self.literal("true")?;
                Ok(Value::Bool(true))
            }
            Some(b'f') => {
                self.literal("false")?;
                Ok(Value::Bool(false))
            }
            Some(b'"') => Ok(Value::String(self.string()?)),
            Some(b'[') | Some(b'{') => {
                if self.depth == MAX_DEPTH {
                    return Err(err(self.pos, "nesting too deep"));
                }
                self.depth += 1;
                let value = if self.peek() == Some(b'[') { self.array() } else { self.object() };
                self.depth -= 1;
                value
            }
            Some(b'-') | Some(b'0'..=b'9') => self.number(),
            Some(other) => Err(err(self.pos, format!("unexpected byte 0x{other:02x}"))),
        }
    }

    fn literal(&mut self, text: &str) -> Result<(), JsonError> {
        if self.bytes[self.pos..].starts_with(text.as_bytes()) {
            self.pos += text.len();
            Ok(())
        } else {
            Err(err(self.pos, format!("expected {text}")))
        }
    }

    fn string(&mut self) -> Result<String, JsonError> {
        self.expect(b'"')?;
        let mut out = String::new();
        loop {
            let Some(byte) = self.peek() else {
                return Err(err(self.pos, "unterminated string"));
            };
            self.pos += 1;
            match byte {
                b'"' => return Ok(out),
                b'\\' => {
                    let Some(escape) = self.peek() else {
                        return Err(err(self.pos, "unterminated escape"));
                    };
                    self.pos += 1;
                    match escape {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{08}'),
                        b'f' => out.push('\u{0c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => out.push(self.unicode_escape()?),
                        other => return Err(err(self.pos - 1, format!("bad escape \\{}", other as char))),
                    }
                }
                byte if byte < 0x20 => return Err(err(self.pos - 1, "control character in string")),
                byte => {
                    // The input is a &str, so it is valid UTF-8; the checks guard
                    // the slice bounds. Copies one whole character.
                    let start = self.pos - 1;
                    let width = utf8_width(byte);
                    let end = start + width;
                    if end > self.bytes.len() {
                        return Err(err(start, "truncated utf8"));
                    }
                    let chunk = std::str::from_utf8(&self.bytes[start..end]).map_err(|_| err(start, "invalid utf8"))?;
                    out.push_str(chunk);
                    self.pos = end;
                }
            }
        }
    }

    /// After `\u`: a surrogate pair (high 0xD800..0xDC00 then low
    /// 0xDC00..0xE000) combines to a code point above 0xFFFF; a lone
    /// surrogate is an error.
    fn unicode_escape(&mut self) -> Result<char, JsonError> {
        let hi = self.hex4()?;
        let code = if (0xD800..0xDC00).contains(&hi) {
            if self.peek() != Some(b'\\') {
                return Err(err(self.pos, "missing low surrogate"));
            }
            self.pos += 1;
            self.expect(b'u')?;
            let lo = self.hex4()?;
            if !(0xDC00..0xE000).contains(&lo) {
                return Err(err(self.pos, "bad low surrogate"));
            }
            0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
        } else if (0xDC00..0xE000).contains(&hi) {
            return Err(err(self.pos, "lone low surrogate"));
        } else {
            hi
        };
        char::from_u32(code).ok_or_else(|| err(self.pos, "invalid code point"))
    }

    fn hex4(&mut self) -> Result<u32, JsonError> {
        if self.pos + 4 > self.bytes.len() {
            return Err(err(self.pos, "truncated \\u escape"));
        }
        let mut value = 0u32;
        for i in 0..4 {
            let b = self.bytes[self.pos + i];
            let digit = match b {
                b'0'..=b'9' => (b - b'0') as u32,
                b'a'..=b'f' => (b - b'a' + 10) as u32,
                b'A'..=b'F' => (b - b'A' + 10) as u32,
                _ => return Err(err(self.pos + i, "bad hex digit")),
            };
            value = value * 16 + digit;
        }
        self.pos += 4;
        Ok(value)
    }

    fn number(&mut self) -> Result<Value, JsonError> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        let digits_start = self.pos;
        // JSON grammar: a leading zero must stand alone.
        if self.peek() == Some(b'0') {
            self.pos += 1;
            if matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(err(self.pos, "leading zero"));
            }
        } else {
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }
        if self.pos == digits_start {
            return Err(err(start, "number has no digits"));
        }
        if matches!(self.peek(), Some(b'.') | Some(b'e') | Some(b'E')) {
            return Err(err(self.pos, "only integer numbers are supported"));
        }
        // Digits and '-' only, hence valid UTF-8.
        let text = std::str::from_utf8(&self.bytes[start..self.pos]).unwrap();
        let n: i64 = text.parse().map_err(|_| err(start, "number out of range"))?;
        Ok(Value::Number(n))
    }

    fn array(&mut self) -> Result<Value, JsonError> {
        self.expect(b'[')?;
        let mut out = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(Value::Array(out));
        }
        loop {
            out.push(self.value()?);
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b']') => {
                    self.pos += 1;
                    return Ok(Value::Array(out));
                }
                _ => return Err(err(self.pos, "expected ',' or ']'")),
            }
        }
    }

    fn object(&mut self) -> Result<Value, JsonError> {
        self.expect(b'{')?;
        let mut out = BTreeMap::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(Value::Object(out));
        }
        loop {
            self.skip_ws();
            let key = self.string()?;
            self.skip_ws();
            self.expect(b':')?;
            let value = self.value()?;
            if out.insert(key, value).is_some() {
                return Err(err(self.pos, "duplicate object key"));
            }
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(Value::Object(out));
                }
                _ => return Err(err(self.pos, "expected ',' or '}'")),
            }
        }
    }
}

/// Length of the UTF-8 sequence that starts with `first`; invalid lead bytes
/// give 1 and fail the later validity check.
fn utf8_width(first: u8) -> usize {
    match first {
        0x00..=0x7F => 1,
        0xC2..=0xDF => 2,
        0xE0..=0xEF => 3,
        0xF0..=0xF4 => 4,
        _ => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 64 nested arrays parse; 65 are refused, and a long run of '[' does not overflow the stack.
    #[test]
    fn nesting_is_limited() {
        let nested = |n: usize| format!("{}{}", "[".repeat(n), "]".repeat(n));
        assert!(parse(&nested(MAX_DEPTH)).is_ok());
        assert!(parse(&nested(MAX_DEPTH + 1)).is_err());
        assert!(parse(&"[".repeat(1_000_000)).is_err());
        assert!(parse(&format!("{}1{}", "{\"a\":".repeat(MAX_DEPTH + 1), "}".repeat(MAX_DEPTH + 1))).is_err());
    }

    // Serialising and parsing again returns the same value.
    #[test]
    fn roundtrip() {
        let text = r#"{"a":[1,-2,true,false,null,"x"],"b":{"c":"d"}}"#;
        let value = parse(text).unwrap();
        assert_eq!(value.get("a").unwrap().as_array().unwrap().len(), 6);
        assert_eq!(value.get("a").unwrap().as_array().unwrap()[1].as_i64(), Some(-2));
        assert_eq!(parse(&value.to_string()).unwrap(), value);
    }

    // Escapes and a surrogate pair decode to the right characters and survive a round trip.
    #[test]
    fn escapes_and_unicode() {
        let value = parse(r#""a\"b\\c\n\u0041\uD83D\uDE00""#).unwrap();
        assert_eq!(value.as_str().unwrap(), "a\"b\\c\nA\u{1F600}");
        assert_eq!(parse(&value.to_string()).unwrap(), value);
    }

    // Malformed documents, fractions, leading zeros, bad escapes and trailing text are errors.
    #[test]
    fn rejects_bad_input() {
        for text in ["", "{", "[1,]", "{\"a\":}", "1.5", "01", "\"\\q\"", "{} x"] {
            assert!(parse(text).is_err(), "{text}");
        }
    }

    // Numbers beyond i64 are errors, not wrapped.
    #[test]
    fn integer_bounds() {
        assert_eq!(parse("9223372036854775807").unwrap().as_i64(), Some(i64::MAX));
        assert!(parse("9223372036854775808").is_err());
    }
}
