use thiserror::Error;

/// Stable integer status codes shared with the C ABI (`gaussogram-ffi`).
pub const STATUS_OK: i32 = 0;
pub const STATUS_POWER_OF_TWO: i32 = 1;
pub const STATUS_TOO_SMALL: i32 = 2;
pub const STATUS_UNKNOWN_WINDOW: i32 = 3;
pub const STATUS_NOT_INVERTIBLE: i32 = 4;
pub const STATUS_WRONG_INPUT_KIND: i32 = 5;
pub const STATUS_OUTPUT_LEN_MISMATCH: i32 = 6;
pub const STATUS_INPUT_LEN_MISMATCH: i32 = 7;
pub const STATUS_INTERNAL: i32 = 99;

#[derive(Error, Debug, PartialEq, Eq)]
pub enum GaussogramError {
    #[error("N must be a power of two (got {0})")]
    PowerOfTwo(usize),

    #[error("scheme '{scheme}' requires N >= {min} (got {got})")]
    TooSmall {
        scheme: &'static str,
        min: usize,
        got: usize,
    },

    #[error("unknown window_type '{0}' (expected 'gaussian' or 'box')")]
    UnknownWindow(String),

    #[error("scheme '{0}' is not invertible")]
    NotInvertible(&'static str),

    #[error("wrong input kind: this entry point expects {}", if *.expected_complex { "complex input" } else { "real input" })]
    WrongInputKind { expected_complex: bool },

    #[error("output buffer length {got} does not match scheme output_len {expected}")]
    OutputLenMismatch { expected: usize, got: usize },

    #[error("input length {got} does not match scheme N {expected}")]
    InputLenMismatch { expected: usize, got: usize },
}

impl GaussogramError {
    pub fn status_code(&self) -> i32 {
        match self {
            GaussogramError::PowerOfTwo(_) => STATUS_POWER_OF_TWO,
            GaussogramError::TooSmall { .. } => STATUS_TOO_SMALL,
            GaussogramError::UnknownWindow(_) => STATUS_UNKNOWN_WINDOW,
            GaussogramError::NotInvertible(_) => STATUS_NOT_INVERTIBLE,
            GaussogramError::WrongInputKind { .. } => STATUS_WRONG_INPUT_KIND,
            GaussogramError::OutputLenMismatch { .. } => STATUS_OUTPUT_LEN_MISMATCH,
            GaussogramError::InputLenMismatch { .. } => STATUS_INPUT_LEN_MISMATCH,
        }
    }
}
