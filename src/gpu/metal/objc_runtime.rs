//! Objective-C runtime FFI bindings for Metal.framework.
//!
//! Minimal set of Objective-C runtime functions needed to call Metal APIs
//! without any external crate dependency. All interaction with Metal goes
//! through `objc_msgSend` and friends.
//!
//! # Safety
//!
//! All functions in this module are inherently unsafe — they perform raw FFI
//! calls into the Objective-C runtime and Metal.framework.

use std::ffi::{c_void, CStr, CString};

// ── Objective-C type aliases ────────────────────────────────────────────────

/// Objective-C object pointer (`id`).
pub type Id = *mut c_void;
/// Objective-C selector (`SEL`).
pub type Sel = *const c_void;
/// Objective-C class pointer (`Class`).
pub type Class = *mut c_void;
/// NSUInteger (platform-sized unsigned integer).
pub type NSUInteger = usize;
/// NSInteger (platform-sized signed integer).
#[allow(dead_code)]
pub type NSInteger = isize;

/// Null object pointer.
pub const NIL: Id = std::ptr::null_mut();

// ── Objective-C runtime extern functions ────────────────────────────────────

extern "C" {
    /// Look up a class by name.
    pub fn objc_getClass(name: *const i8) -> Class;
    /// Register (or look up) a selector by name.
    pub fn sel_registerName(name: *const i8) -> Sel;
    /// Send a message to an Objective-C object. Variadic — actual signature
    /// depends on the selector being called.
    pub fn objc_msgSend(receiver: Id, sel: Sel, ...) -> Id;
}

// ── Metal framework loader ──────────────────────────────────────────────────

/// Handle to a dynamically loaded Metal.framework.
pub struct MetalFramework {
    _handle: *mut c_void,
    /// Pointer to `MTLCreateSystemDefaultDevice` function.
    pub create_default_device: unsafe extern "C" fn() -> Id,
}

// SAFETY: The framework handle and function pointer are process-global
// and immutable after load.
unsafe impl Send for MetalFramework {}
unsafe impl Sync for MetalFramework {}

impl MetalFramework {
    /// Load Metal.framework via `dlopen` and resolve `MTLCreateSystemDefaultDevice`.
    ///
    /// Returns `None` if Metal.framework is not available (e.g. on Linux).
    pub fn load() -> Option<Self> {
        let path = CString::new("/System/Library/Frameworks/Metal.framework/Metal").ok()?;

        // SAFETY: dlopen with RTLD_LAZY is safe; we check the returned handle.
        let handle = unsafe { libc::dlopen(path.as_ptr(), libc::RTLD_LAZY) };
        if handle.is_null() {
            return None;
        }

        let sym_name = CString::new("MTLCreateSystemDefaultDevice").ok()?;
        let sym = unsafe { libc::dlsym(handle, sym_name.as_ptr()) };
        if sym.is_null() {
            unsafe { libc::dlclose(handle) };
            return None;
        }

        let create_default_device: unsafe extern "C" fn() -> Id =
            unsafe { std::mem::transmute(sym) };

        Some(Self {
            _handle: handle,
            create_default_device,
        })
    }
}

// ── Helper functions ────────────────────────────────────────────────────────

/// Create a selector from a Rust string.
///
/// # Safety
/// The returned `Sel` is valid for the lifetime of the process.
pub unsafe fn sel(name: &str) -> Sel {
    let cstr = CString::new(name).expect("selector name must not contain NUL");
    sel_registerName(cstr.as_ptr())
}

/// Get an Objective-C class by name.
///
/// # Safety
/// Returns a null pointer if the class is not found.
pub unsafe fn class(name: &str) -> Class {
    let cstr = CString::new(name).expect("class name must not contain NUL");
    objc_getClass(cstr.as_ptr())
}

/// Read an NSString's UTF-8 contents into a Rust `String`.
///
/// # Safety
/// `nsstring` must be a valid `NSString *` or nil.
pub unsafe fn nsstring_to_string(nsstring: Id) -> String {
    if nsstring.is_null() {
        return String::new();
    }
    let utf8_sel = sel("UTF8String");
    let cstr_ptr: *const i8 = std::mem::transmute(objc_msgSend(nsstring, utf8_sel));
    if cstr_ptr.is_null() {
        return String::new();
    }
    CStr::from_ptr(cstr_ptr).to_string_lossy().into_owned()
}
