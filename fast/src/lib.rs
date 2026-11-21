//! Shopping Sessions: the performance implementation.
//!
//! See `BRIEF-fast.md` for the goal and constraints, and `rules.md` for the spec.
//! `rules.md` is authoritative over anything written here.

pub mod hash; // fast non-cryptographic hash + HashMap alias
pub mod csvline; // the rules.md CSV dialect: lines, fields, quotes, malformed
pub mod intern; // byte-string IDs -> dense u32 handles
pub mod manifest;
pub mod out;
pub mod pipeline; // the stages, in rules.md's order
pub mod ts; // timestamp parsing (F1/F2/F3) -> epoch ms
pub mod unionfind; // linked user IDs -> people
