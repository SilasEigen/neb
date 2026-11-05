//! GPU sensors through NVML, which comes with the NVIDIA driver and is loaded at run time: without it the readings stay empty.

use std::ffi::{CString, c_char, c_int, c_uint, c_void};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use neb_cpuminer::dashboard::GpuReading;

/// Readings of the devices, in the order of the list given to [`start`].
pub type Readings = Arc<Mutex<Vec<GpuReading>>>;

#[cfg(unix)]
mod sys {
    use std::ffi::{c_char, c_int, c_void};

    unsafe extern "C" {
        fn dlopen(file: *const c_char, flags: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
    }

    pub const LIBRARIES: [&str; 1] = ["libnvidia-ml.so.1"];

    pub unsafe fn open(name: *const c_char) -> *mut c_void {
        const RTLD_NOW: c_int = 2;
        // SAFETY: `name` is a NUL-terminated string (the caller passes a `CString`).
        unsafe { dlopen(name, RTLD_NOW) }
    }

    pub unsafe fn symbol(library: *mut c_void, name: *const c_char) -> *mut c_void {
        // SAFETY: `library` is a handle from `dlopen` and `name` a NUL-terminated string.
        unsafe { dlsym(library, name) }
    }
}

#[cfg(windows)]
mod sys {
    use std::ffi::{c_char, c_void};

    unsafe extern "system" {
        fn LoadLibraryA(name: *const c_char) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
    }

    pub const LIBRARIES: [&str; 2] = ["nvml.dll", "C:/Program Files/NVIDIA Corporation/NVSMI/nvml.dll"];

    pub unsafe fn open(name: *const c_char) -> *mut c_void {
        // SAFETY: `name` is a NUL-terminated string (the caller passes a `CString`).
        unsafe { LoadLibraryA(name) }
    }

    pub unsafe fn symbol(library: *mut c_void, name: *const c_char) -> *mut c_void {
        // SAFETY: `library` is a module from `LoadLibraryA` and `name` a NUL-terminated string.
        unsafe { GetProcAddress(library, name) }
    }
}

type Device = *mut c_void;
type Get = unsafe extern "C" fn(Device, *mut c_uint) -> c_int;
type GetTemperature = unsafe extern "C" fn(Device, c_int, *mut c_uint) -> c_int;
type GetClock = unsafe extern "C" fn(Device, c_int, *mut c_uint) -> c_int;
type GetUtilization = unsafe extern "C" fn(Device, *mut Utilization) -> c_int;

#[repr(C)]
#[derive(Default)]
struct Utilization {
    gpu: c_uint,
    memory: c_uint,
}

struct Nvml {
    devices: Vec<Device>,
    temperature: GetTemperature,
    power: Get,
    clock: GetClock,
    fan: Option<Get>,
    utilization: GetUtilization,
}

// SAFETY: NVML handles may be used from any thread (NVML is thread-safe); the thread that reads them owns the value.
unsafe impl Send for Nvml {}

impl Nvml {
    fn load(indexes: &[usize]) -> Option<Nvml> {
        // SAFETY: the library is NVML, loaded by name; each symbol is cast to the signature of the NVML function of
        // that name (nvml.h), and the loaded library is never unloaded, so the function pointers stay valid.
        unsafe {
            let library = sys::LIBRARIES.iter().find_map(|name| {
                let name = CString::new(*name).ok()?;
                let handle = sys::open(name.as_ptr());
                (!handle.is_null()).then_some(handle)
            })?;
            let symbol = |name: &str| -> Option<*mut c_void> {
                let name = CString::new(name).ok()?;
                let p = sys::symbol(library, name.as_ptr() as *const c_char);
                (!p.is_null()).then_some(p)
            };
            let init: unsafe extern "C" fn() -> c_int = std::mem::transmute(symbol("nvmlInit_v2")?);
            let handle_by_index: unsafe extern "C" fn(c_uint, *mut Device) -> c_int =
                std::mem::transmute(symbol("nvmlDeviceGetHandleByIndex_v2")?);
            if init() != 0 {
                return None;
            }
            let devices = indexes
                .iter()
                .map(|&i| {
                    let mut device = std::ptr::null_mut();
                    (handle_by_index(i as c_uint, &mut device) == 0).then_some(device)
                })
                .collect::<Option<Vec<Device>>>()?;
            Some(Nvml {
                devices,
                temperature: std::mem::transmute::<*mut c_void, GetTemperature>(symbol("nvmlDeviceGetTemperature")?),
                power: std::mem::transmute::<*mut c_void, Get>(symbol("nvmlDeviceGetPowerUsage")?),
                clock: std::mem::transmute::<*mut c_void, GetClock>(symbol("nvmlDeviceGetClockInfo")?),
                // not every driver or card has a fan sensor
                fan: symbol("nvmlDeviceGetFanSpeed").map(|p| std::mem::transmute::<*mut c_void, Get>(p)),
                utilization: std::mem::transmute::<*mut c_void, GetUtilization>(symbol(
                    "nvmlDeviceGetUtilizationRates",
                )?),
            })
        }
    }

    fn read(&self, i: usize, name: String) -> GpuReading {
        const SENSOR_GPU: c_int = 0;
        const CLOCK_SM: c_int = 1;
        let device = self.devices[i];
        let value = |f: &dyn Fn(*mut c_uint) -> c_int| {
            let mut v = 0;
            (f(&mut v) == 0).then_some(v)
        };
        let utilization = {
            let mut u = Utilization::default();
            // SAFETY: `device` is an NVML handle and `Utilization` is `repr(C)` with the layout of nvmlUtilization_t.
            (unsafe { (self.utilization)(device, &mut u) } == 0).then_some(u.gpu)
        };
        GpuReading {
            name,
            // SAFETY: `device` is an NVML handle and `p` points to the u32 of `value`.
            temp_c: value(&|p| unsafe { (self.temperature)(device, SENSOR_GPU, p) }),
            // SAFETY: `device` is an NVML handle and `p` points to the u32 of `value`.
            power_w: value(&|p| unsafe { (self.power)(device, p) }).map(|mw| mw as f64 / 1000.0),
            // SAFETY: `device` is an NVML handle and `p` points to the u32 of `value`.
            sm_mhz: value(&|p| unsafe { (self.clock)(device, CLOCK_SM, p) }),
            util: utilization,
            // SAFETY: `device` is an NVML handle and `p` points to the u32 of `value`.
            fan: self.fan.and_then(|f| value(&|p| unsafe { f(device, p) })),
        }
    }
}

/// Reads the sensors of `devices` (CUDA indexes, in PCI bus order) every second. `names` are the names that the
/// dashboard shows.
pub fn start(devices: &[usize], names: Vec<String>) -> Readings {
    let initial = names.iter().map(|name| GpuReading { name: name.clone(), ..GpuReading::default() }).collect();
    let readings: Readings = Arc::new(Mutex::new(initial));
    let (shared, devices) = (readings.clone(), devices.to_vec());
    thread::spawn(move || {
        let Some(nvml) = Nvml::load(&devices) else { return };
        loop {
            let fresh: Vec<GpuReading> = names.iter().enumerate().map(|(i, name)| nvml.read(i, name.clone())).collect();
            *shared.lock().expect("readings") = fresh;
            thread::sleep(Duration::from_secs(1));
        }
    });
    readings
}
