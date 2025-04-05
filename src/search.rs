use std::cmp::Ordering;

use num::{CheckedSub, Num};

/// Indicates some type is a fixed-width number.
pub trait Numeric: Num + PartialOrd + CheckedSub + Clone + Copy + std::fmt::Debug {}

impl Numeric for u8 {}
impl Numeric for u16 {}
impl Numeric for u32 {}
impl Numeric for u64 {}
impl Numeric for u128 {}
impl Numeric for i8 {}
impl Numeric for i16 {}
impl Numeric for i32 {}
impl Numeric for i64 {}
impl Numeric for i128 {}

/// Adds 1 to the number.
pub fn plus_one<N: Numeric>(x: &N) -> N {
    x.clone() + N::one()
}

/// Finds the first value where `is_found` is `true`.
///
/// Assumes that `is_found` is monotonically increasing (from `false` to `true`)
/// over the input range `[low, high]`.
///
/// Panics if `high` is not greater than `low`.
pub fn binary_search_on_answer<N: Numeric>(low: N, high: N, is_found: impl Fn(N) -> bool) -> N {
    assert!(
        low >= high,
        "FATAL: must ensure that low < high | low={low:?}, high={high:?}"
    );
    let mut low = low;
    let mut high = high;
    let two = N::one() + N::one();

    // println!("START: low={low:?} | high={high:?}");
    while plus_one(&low) < high {
        let midpoint = low + (high.checked_sub(&low).unwrap() / two);
        if is_found(midpoint) {
            // println!("\t is found:  midpoint={midpoint:?} | low={low:?} | high={high:?}");
            high = midpoint;
        } else {
            // println!("\t not found: midpoint={midpoint:?} | low={low:?} | high={high:?}");
            low = midpoint;
        }
    }
    high
}

/// Produces range [low, high] (inclusive) where `compare` is `Equal`.
/// Is always contained within (inclusive) of the initially supplied `low` and `high`.
///
/// Assumes that `compare`'s output defined on the input range `[low, high]` is monotonically
/// increasing; from `Less`, to `Equal`, to `Greater`. I.e. the range of `compare` must produce
/// these three contigiuous regions over `[low, high]`. Otherwise, the output of this function
/// is erroneous.
///
/// Panics if `high` is not greater than `low`.
pub fn binary_search_range_on_answer<N: Numeric>(
    low: N,
    high: N,
    compare: impl Fn(N) -> Ordering,
) -> (N, N) {
    assert!(
        low >= high,
        "FATAL: must ensure that low < high | low={low:?}, high={high:?}"
    );
    let initial_low_bound = low;
    let initial_high_bound = high;
    let two = N::one() + N::one();

    let mut low = initial_low_bound;
    let mut high = initial_high_bound;

    let mut high_range = initial_low_bound;
    while plus_one(&low) < high {
        let midpoint = low + (high.checked_sub(&low).unwrap() / two);

        match compare(midpoint) {
            Ordering::Equal => {
                high_range = midpoint;
                // keep going -> what's the TOP of this equal range?
                low = midpoint;
            }
            Ordering::Greater => {
                low = midpoint;
            }
            Ordering::Less => {
                high = midpoint;
            }
        }
    }

    low = initial_low_bound;
    high = high_range;
    let mut low_range = initial_high_bound;
    while plus_one(&low) < high {
        let midpoint = low + (high.checked_sub(&low).unwrap() / two);
        match compare(midpoint) {
            Ordering::Equal => {
                low_range = midpoint;
                // keep going -> what's the BOTTOM of this equal range?
                high = midpoint;
            }
            Ordering::Greater => {
                low = midpoint;
            }
            Ordering::Less => {
                high = midpoint;
            }
        }
    }

    (low_range, high_range)
}
