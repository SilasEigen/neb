//! What the wallet prints for a person: the package mark and the colours of the node (ANSI 256: 120 green for the
//! amounts, greys for the labels, white for the values), only when the output is a terminal. When it is not a
//! terminal, or `NO_COLOR` is set, the text is the plain text, so scripts that read it see no escape sequence.

use std::io::IsTerminal;

const RESET: &str = "\x1b[0m";
const GREEN: &str = "\x1b[1;38;5;120m";
const WHITE: &str = "\x1b[1;38;5;231m";
const DIM: &str = "\x1b[38;5;244m";
/// The package emoji, two columns wide.
const MARK: &str = "\u{1f4e6}";

pub struct Look {
    color: bool,
}

/// One address of a wallet with its mature and immature amounts, already formatted.
pub struct Row {
    pub address: String,
    pub confirmed: String,
    pub immature: Option<String>,
}

#[cfg(windows)]
fn enable_ansi() {
    use std::ffi::c_void;
    unsafe extern "system" {
        fn GetStdHandle(id: u32) -> *mut c_void;
        fn GetConsoleMode(handle: *mut c_void, mode: *mut u32) -> i32;
        fn SetConsoleMode(handle: *mut c_void, mode: u32) -> i32;
    }
    const STD_OUTPUT_HANDLE: u32 = -11i32 as u32;
    const ENABLE_VIRTUAL_TERMINAL_PROCESSING: u32 = 4;
    // SAFETY: Win32 console calls on the standard output handle; `mode` is a live u32 that the call writes.
    unsafe {
        let handle = GetStdHandle(STD_OUTPUT_HANDLE);
        let mut mode = 0;
        if GetConsoleMode(handle, &mut mode) != 0 {
            SetConsoleMode(handle, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
        }
    }
}

#[cfg(not(windows))]
fn enable_ansi() {}

impl Look {
    pub fn detect() -> Look {
        let color = std::io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none();
        if color {
            enable_ansi();
        }
        Look { color }
    }

    pub fn with_color(color: bool) -> Look {
        Look { color }
    }

    fn paint(&self, color: &str, text: &str) -> String {
        if self.color { format!("{color}{text}{RESET}") } else { text.to_string() }
    }

    /// The start of a line: the mark on the first line, the same width of blanks on the others, nothing without colour.
    fn lead(&self, first: bool) -> String {
        match (self.color, first) {
            (false, _) => String::new(),
            (true, true) => format!("{MARK} "),
            (true, false) => "   ".to_string(),
        }
    }

    /// The output of `balance`.
    pub fn balance(&self, height: u32, rows: &[Row], total: &str) -> String {
        let mut out =
            format!("{}{}   {}\n", self.lead(true), self.paint(DIM, "height"), self.paint(WHITE, &height.to_string()));
        for row in rows {
            out += &format!(
                "{}{}  {}\n",
                self.lead(false),
                self.paint(WHITE, &row.address),
                self.paint(GREEN, &format!("{} NEB", row.confirmed))
            );
            if let Some(immature) = &row.immature {
                out += &format!(
                    "{}{}\n",
                    if self.color { "     " } else { "  " },
                    self.paint(DIM, &format!("immature {immature}"))
                );
            }
        }
        out += &format!(
            "{}{}    {}\n",
            self.lead(false),
            self.paint(DIM, "total"),
            self.paint(GREEN, &format!("{total} NEB"))
        );
        out
    }

    /// The output of `new`.
    pub fn created(&self, path: &str, addresses: &[String]) -> String {
        let mut out = format!("{}{}  {}\n", self.lead(true), self.paint(DIM, "wallet"), path);
        for address in addresses {
            out += &format!("{}{} {}\n", self.lead(false), self.paint(DIM, "address"), self.paint(GREEN, address));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows() -> Vec<Row> {
        vec![Row { address: "neb1abc".into(), confirmed: "266".into(), immature: Some("1881".into()) }]
    }

    #[test]
    fn without_colour_the_text_is_the_plain_text() {
        let look = Look::with_color(false);
        assert_eq!(
            look.balance(113, &rows(), "266"),
            "height   113\nneb1abc  266 NEB\n  immature 1881\ntotal    266 NEB\n"
        );
        assert_eq!(look.created("wallet.txt", &["neb1abc".into()]), "wallet  wallet.txt\naddress neb1abc\n");
    }

    #[test]
    fn with_colour_the_mark_leads_and_the_amounts_are_green() {
        let text = Look::with_color(true).balance(113, &rows(), "266");
        assert!(text.starts_with(&format!("{MARK} ")));
        assert!(text.contains(&format!("{GREEN}266 NEB{RESET}")) && text.contains("immature 1881"));
        assert_eq!(text.matches(MARK).count(), 1);
        let stripped: String = text
            .split('\x1b')
            .enumerate()
            .map(
                |(i, p)| {
                    if i == 0 { p.to_string() } else { p.split_once('m').map_or(String::new(), |x| x.1.to_string()) }
                },
            )
            .collect();
        assert!(stripped.contains("height   113") && stripped.contains("total    266 NEB"));
    }
}
