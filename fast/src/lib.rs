//! Shopping Sessions: the performance implementation.
//!
//! See `BRIEF-fast.md` for the goal and constraints, and `rules.md` for the spec.
//! `rules.md` is authoritative over anything written here.

pub mod hash; // fast non-cryptographic hash + HashMap alias
pub mod csvline; // the rules.md CSV dialect: lines, fields, quotes, malformed
pub mod hexid; // lock-free exact table for 33-byte [a-z][0-9a-f]{32} IDs
pub mod intern; // byte-string IDs -> dense u32 handles
pub mod key; // short byte-string IDs -> order-preserving u64
pub mod links2; // v2 parallel links -> integer person-name keys
pub mod manifest;
pub mod out;
pub mod pipeline; // the stages, in rules.md's order
pub mod pipeline2; // integer-key v2, opt-in with FAST_V2=1
pub mod pipeline3; // v3 core (PLAN-v3.md), the default
pub mod scan3; // v3 chunk scanner: one SIMD pass -> fixed field arrays
pub mod sysmem; // madvise helpers: huge pages, parallel release before exit (Linux; no-op elsewhere)
pub mod ts; // timestamp parsing (F1/F2/F3) -> epoch ms
pub mod unionfind; // linked user IDs -> people
