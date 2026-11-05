//! Console access without dependencies: ANSI output, key presses and the width of the terminal.

/// A key press that the dashboard understands. Only Windows consoles deliver keys.
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(not(windows), allow(dead_code))]
pub enum Key {
    Quit,
    Preview,
    /// Positive: toward older blocks.
    Scroll(i64),
}

/// Escape sequence that leaves the alternate screen and shows the cursor again.
pub const LEAVE: &str = "\x1b[?25h\x1b[?1049l";

#[cfg(windows)]
mod imp {
    use super::{Key, LEAVE};
    use std::ffi::c_void;
    use std::io::Write;

    type Handle = *mut c_void;

    #[repr(C)]
    struct ScreenInfo {
        size: [i16; 2],
        cursor: [i16; 2],
        attributes: u16,
        window: [i16; 4],
        maximum: [i16; 2],
    }

    unsafe extern "system" {
        fn GetStdHandle(id: u32) -> Handle;
        fn GetConsoleMode(handle: Handle, mode: *mut u32) -> i32;
        fn SetConsoleMode(handle: Handle, mode: u32) -> i32;
        fn SetConsoleOutputCP(code_page: u32) -> i32;
        fn GetConsoleScreenBufferInfo(handle: Handle, info: *mut ScreenInfo) -> i32;
        fn SetConsoleCtrlHandler(handler: Option<unsafe extern "system" fn(u32) -> i32>, add: i32) -> i32;
    }

    unsafe extern "C" {
        fn _kbhit() -> i32;
        fn _getch() -> i32;
    }

    const STD_OUTPUT_HANDLE: u32 = -11i32 as u32;
    const ENABLE_VIRTUAL_TERMINAL_PROCESSING: u32 = 4;

    pub fn is_tty() -> bool {
        let mut mode = 0;
        // SAFETY: Win32 calls on the standard output handle; `mode` is a live u32 that the call writes.
        unsafe { GetConsoleMode(GetStdHandle(STD_OUTPUT_HANDLE), &mut mode) != 0 }
    }

    /// Ctrl-C ends the process; the console must not be left on the alternate screen.
    unsafe extern "system" fn on_ctrl_c(_event: u32) -> i32 {
        let mut out = std::io::stdout();
        let _ = out.write_all(LEAVE.as_bytes());
        let _ = out.flush();
        0
    }

    pub fn setup() {
        // SAFETY: Win32 console calls on the standard output handle; `mode` is a live u32, and `on_ctrl_c` has the
        // signature of a console control handler and lives for the whole process.
        unsafe {
            let handle = GetStdHandle(STD_OUTPUT_HANDLE);
            let mut mode = 0;
            if GetConsoleMode(handle, &mut mode) != 0 {
                SetConsoleMode(handle, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
            }
            SetConsoleOutputCP(65001);
            SetConsoleCtrlHandler(Some(on_ctrl_c), 1);
        }
    }

    /// Columns and rows of the visible window.
    pub fn size() -> Option<(usize, usize)> {
        let mut info = ScreenInfo { size: [0; 2], cursor: [0; 2], attributes: 0, window: [0; 4], maximum: [0; 2] };
        // SAFETY: `ScreenInfo` is `repr(C)` with the layout of CONSOLE_SCREEN_BUFFER_INFO, and the call writes it.
        let ok = unsafe { GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &mut info) != 0 };
        let width = info.window[2] as i32 - info.window[0] as i32 + 1;
        let height = info.window[3] as i32 - info.window[1] as i32 + 1;
        (ok && width > 0 && height > 0).then_some((width as usize, height as usize))
    }

    #[repr(C)]
    struct SystemTime([u16; 8]);

    #[repr(C)]
    struct TimeZone {
        bias: i32,
        standard_name: [u16; 32],
        standard_date: SystemTime,
        standard_bias: i32,
        daylight_name: [u16; 32],
        daylight_date: SystemTime,
        daylight_bias: i32,
    }

    unsafe extern "system" {
        fn GetTimeZoneInformation(info: *mut TimeZone) -> u32;
    }

    /// Seconds to add to UTC to get the local time now.
    pub fn utc_offset_secs() -> i64 {
        // SAFETY: `TimeZone` holds integers only, so all-zero bytes are a valid value.
        let mut zone: TimeZone = unsafe { std::mem::zeroed() };
        // SAFETY: `TimeZone` is `repr(C)` with the layout of TIME_ZONE_INFORMATION, and the call writes it.
        let id = unsafe { GetTimeZoneInformation(&mut zone) };
        let bias = zone.bias + if id == 2 { zone.daylight_bias } else { zone.standard_bias };
        -(bias as i64) * 60
    }

    pub fn keys() -> Vec<Key> {
        const PAGE: i64 = 5;
        const FAR: i64 = 1 << 20;
        let mut out = Vec::new();
        // SAFETY: `_kbhit` and `_getch` of the C runtime take no argument and only read the console.
        while unsafe { _kbhit() } != 0 {
            // SAFETY: see `_kbhit` above.
            let c = unsafe { _getch() };
            match c {
                0x71 | 0x51 | 3 => out.push(Key::Quit),
                0x62 | 0x42 => out.push(Key::Preview),
                0x6A | 0x4A => out.push(Key::Scroll(1)),
                0x6B | 0x4B => out.push(Key::Scroll(-1)),
                // SAFETY: see `_kbhit` above; a second `_getch` reads the code of an extended key.
                0 | 0xE0 => match unsafe { _getch() } {
                    80 => out.push(Key::Scroll(1)),
                    72 => out.push(Key::Scroll(-1)),
                    81 => out.push(Key::Scroll(PAGE)),
                    73 => out.push(Key::Scroll(-PAGE)),
                    79 => out.push(Key::Scroll(FAR)),
                    71 => out.push(Key::Scroll(-FAR)),
                    _ => {}
                },
                _ => {}
            }
        }
        out
    }
}

#[cfg(unix)]
mod imp {
    use super::{Key, LEAVE};
    use std::ffi::c_void;

    unsafe extern "C" {
        fn isatty(fd: i32) -> i32;
        fn signal(signal: i32, handler: usize) -> usize;
        fn write(fd: i32, buf: *const c_void, len: usize) -> isize;
        fn _exit(code: i32) -> !;
    }

    const SIGINT: i32 = 2;

    pub fn is_tty() -> bool {
        // SAFETY: `isatty` only inspects file descriptor 1.
        unsafe { isatty(1) == 1 }
    }

    extern "C" fn on_interrupt(_signal: i32) {
        // SAFETY: `write` and `_exit` are async-signal-safe; `LEAVE` is a static string of `LEAVE.len()` bytes.
        unsafe {
            write(1, LEAVE.as_ptr().cast(), LEAVE.len());
            _exit(130)
        }
    }

    pub fn setup() {
        // SAFETY: `on_interrupt` is an `extern "C" fn(i32)` that lives for the whole process.
        unsafe { signal(SIGINT, on_interrupt as extern "C" fn(i32) as usize) };
    }

    pub fn size() -> Option<(usize, usize)> {
        let get = |name: &str| std::env::var(name).ok()?.parse().ok();
        Some((get("COLUMNS")?, get("LINES")?))
    }

    /// The terminal is left in its own mode, so keys need Enter and are not read: Ctrl-C ends the miner.
    pub fn keys() -> Vec<Key> {
        Vec::new()
    }

    #[repr(C)]
    struct Tm {
        fields: [i32; 9],
        gmtoff: i64,
        zone: *const u8,
    }

    unsafe extern "C" {
        fn localtime_r(time: *const i64, out: *mut Tm) -> *mut Tm;
    }

    /// Seconds to add to UTC to get the local time now.
    pub fn utc_offset_secs() -> i64 {
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs() as i64);
        let mut tm = Tm { fields: [0; 9], gmtoff: 0, zone: std::ptr::null() };
        // SAFETY: `Tm` is `repr(C)` with the layout of `struct tm` on 64-bit Linux and macOS, and the call writes it.
        unsafe { localtime_r(&now, &mut tm) };
        tm.gmtoff
    }
}

pub use imp::{is_tty, keys, setup, size, utc_offset_secs};

/// True when the keys of the dashboard can be read.
pub const HAS_KEYS: bool = cfg!(windows);
