//! The work service of the NEB node (spec/node.md section 7.1): frames,
//! the mempool, block templates and jobs, and submission rebuilding. This
//! crate is the protocol and the logic; the sockets are in `neb-node`.

pub mod frames;
pub mod mempool;
pub mod template;

#[cfg(test)]
mod tests;

pub use frames::{
    Hello, Job, MSG_JOB, MSG_MODEL, MSG_RESULT, MSG_SET_SHARE_TARGET, MSG_SHARE, MSG_SUBMIT, ResultMsg, ShareMsg,
    Submit, decode_model, frame, model_frame, parse_frame,
};
pub use mempool::{Mempool, MempoolError};
pub use template::Work;
