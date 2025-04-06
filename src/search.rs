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
        low < high,
        "FATAL: must ensure that low < high | low={low:?}, high={high:?}"
    );
    let mut low = low;
    let mut high = high;
    let two = N::one() + N::one();

    // println!("START: low={low:?} | high={high:?}");
    while plus_one(&low) < high {
        let midpoint = low + (high.checked_sub(&low).unwrap() / two);
        if is_found(midpoint) {
            println!("\t is found:  midpoint={midpoint:?} | low={low:?} | high={high:?}");
            high = midpoint;
        } else {
            println!("\t not found: midpoint={midpoint:?} | low={low:?} | high={high:?}");
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
/// If the value is never found in [low, high], then the returned range is [high,low].
///
/// Panics if `high` is not greater than `low`.
pub fn binary_search_range_on_answer<N: Numeric>(
    low: N,
    high: N,
    // is_found: impl Fn(N) -> bool,
    compare: impl Fn(N) -> Ordering,
) -> (N, N) {
    assert!(
        low < high,
        "FATAL: must ensure that low < high | low={low:?}, high={high:?}"
    );

    match (compare(low), compare(high)) {
        (Ordering::Equal, Ordering::Equal) => return (low, high),
        _ => (),
    };

    let low_range = {
        let low_range_found = binary_search_on_answer(low, high, |val: N| match compare(val) {
            Ordering::Greater | Ordering::Equal => true,
            _ => false,
        });
        match compare(low_range_found) {
            Ordering::Equal => low_range_found,
            _ => low_range_found - N::one(),
        }
    };

    let high_range = {
        let high_range_found = binary_search_on_answer(low, high, |val: N| {
            !(match compare(val) {
                Ordering::Less | Ordering::Equal => true,
                _ => false,
            })
        });
        match compare(high_range_found) {
            Ordering::Equal => high_range_found,
            _ => high_range_found - N::one(),
        }
    };

    match (compare(low_range), compare(high_range)) {
        // valid! thing we're looking for is in [low_range, high_range]
        (Ordering::Equal, Ordering::Equal) => (low_range, high_range),
        // invalid! thing we're looking for is outside of the original [low, high] input range
        _ => (high, low),
    }
}

// pub fn binary_search_range_on_answer_2<N: Numeric>(
//     low: N,
//     high: N,
//     compare: impl Fn(N) -> Ordering,
// ) -> (N, N) {
//     assert!(
//         low < high,
//         "FATAL: must ensure that low < high | low={low:?}, high={high:?}"
//     );
//     let initial_low_bound = low;
//     let initial_high_bound = high;
//     let two = N::one() + N::one();

//     let mut low = initial_low_bound;
//     let mut high = initial_high_bound;

//     let mut high_found_once = false;
//     let mut high_range = initial_high_bound;
//     while plus_one(&low) < high {
//         let midpoint = low + (high.checked_sub(&low).unwrap() / two);

//         match compare(midpoint) {
//             Ordering::Equal => {
//                 high_found_once = true;
//                 high_range = midpoint;
//                 // keep going -> what's the TOP of this equal range?
//                 low = midpoint;
//                 println!("\t[find high] =: [{low:?}, {high:?}]");
//             }
//             Ordering::Greater => {
//                 low = midpoint;
//                 println!("\t[find high] >: [{low:?}, {high:?}]");
//             }
//             Ordering::Less => {
//                 high = midpoint;
//                 println!("\t[find high] <: [{low:?}, {high:?}]");
//             }
//         }
//     }
//     // if !high_found_once {
//     //     high_range = high;
//     // }

//     // let high_range = high;
//     // high_range = high;
//     println!("\t[find high] STOP: [{low:?}, {high:?}] high_range={high_range:?}");

//     low = initial_low_bound;
//     let mut low_range = low;
//     // high = high_range;

//     let mut low_found_once = false;
//     // if plus_one(&low) >= high {
//     //     low_range = high_range;
//     //     println!("\t\tSET: {low_range:?} - {high_range:?}");
//     // } else {
//     while plus_one(&low) < high {
//         let midpoint = low + (high.checked_sub(&low).unwrap() / two);
//         match compare(midpoint) {
//             Ordering::Equal => {
//                 low_found_once = true;
//                 low_range = midpoint;
//                 // keep going -> what's the BOTTOM of this equal range?
//                 high = midpoint;
//                 println!("\t[find low] =: [{low:?}, {high:?}]");
//             }
//             Ordering::Greater => {
//                 low = midpoint;
//                 println!("\t[find low] >: [{low:?}, {high:?}]");
//             }
//             Ordering::Less => {
//                 high = midpoint;
//                 println!("\t[find low] <: [{low:?}, {high:?}]");
//             }
//         }
//     }

//     // if !low_found_once {
//     //     low_range = high;
//     // }
//     // }
//     // let low_range = low;
//     // low_range = low;
//     println!("\t[find low] STOP: [{low:?}, {high:?}] low_range={low_range:?}");

//     if plus_one(&low_range) >= high_range {
//         if compare(low_range) == Ordering::Equal {
//             let ans = low_range;
//             println!("\t[STOP-CHK] answer: [{ans:?},{ans:?}]");
//             (ans, ans)
//         } else if compare(high_range) == Ordering::Equal {
//             let ans = high_range;
//             println!("\t[STOP-CHK] answer: [{ans:?},{ans:?}]");
//             (ans, ans)
//         } else {
//             println!("\t[STOP-ERR] answer: [{initial_low_bound:?}, {initial_high_bound:?}]");
//             (initial_low_bound, initial_high_bound)
//         }
//     } else {
//         println!("\t[STOP-FND] answer: [{low_range:?}, {high_range:?}]");
//         (low_range, high_range)
//     }

//     // if !high_found_once
//     //     && !low_found_once
//     //     && compare(low_range) != Ordering::Equal
//     //     && compare(high_range) != Ordering::Equal
//     // {
//     //     // failed to find even a single instance!
//     //     println!("\t[STOP-ERR] answer: [{initial_low_bound:?}, {initial_high_bound:?}]");
//     //     (initial_low_bound, initial_high_bound)
//     // } else {
//     //     println!("\t[STOP-FND] answer: [{low_range:?}, {high_range:?}]");
//     //     (low_range, high_range)
//     // }
// }

#[cfg(test)]
mod test {
    use std::collections::HashSet;

    use super::*;

    type N = u16;

    fn compare(checking: N) -> impl Fn(N) -> Ordering {
        move |v: N| -> Ordering {
            // let x = checking.cmp(&v);
            if v < checking {
                Ordering::Less
            } else if v > checking {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
            // let x = v.cmp(&checking);
            // println!("checking={checking:?} value={v:?} => {x:?}");
            // x
        }
    }

    fn compare_range(min: N, max: N) -> impl Fn(N) -> Ordering {
        move |v: N| -> Ordering {
            if v < min {
                Ordering::Less
            } else if v > max {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        }
    }

    #[test]
    fn range_nothing() {
        for compare in [
            |v: N| -> Ordering { v.cmp(&11) },
            |v: N| -> Ordering { 11.cmp(&v) },
        ] {
            let (low, high) = binary_search_range_on_answer(0, 10, compare);
            // let (low, high) = binary_search_range_on_answer_2(0, 10, compare);
            println!("low={low} | high={high}");
            assert_eq!((high, low), (0, 10));
        }
    }

    #[test]
    fn range_first() {
        // single element, first
        let (low, high) = binary_search_range_on_answer(0, 10, compare(0));
        // let (low, high) = binary_search_range_on_answer_2(0, 10, compare(0));
        println!("low={low} | high={high}");
        assert_eq!((low, high), (0, 0));
    }

    #[test]
    fn range_last() {
        // single element, last
        let (low, high) = binary_search_range_on_answer(0, 10, compare(10));
        // let (low, high) = binary_search_range_on_answer_2(0, 10, compare(10));
        println!("low={low} | high={high}");
        assert_eq!((low, high), (10, 10));
    }

    #[test]
    fn range_single() {
        // single element, middle
        let (low, high) = binary_search_range_on_answer(0, 10, compare(5));
        // let (low, high) = binary_search_range_on_answer_2(0, 10, compare(5));
        println!("low={low} | high={high}");
        assert_eq!((low, high), (5, 5));
    }

    #[test]
    fn range_middle() {
        let (low, high) = binary_search_range_on_answer(0, 10, compare_range(3, 7));
        // let (low, high) = binary_search_range_on_answer_2(0, 10, compare_range(3, 7));
        println!("low={low} | high={high}");
        assert_eq!((low, high), (3, 7));
    }

    #[test]
    fn range_all() {
        let (low, high) = binary_search_range_on_answer(0, 10, compare_range(0, 10));
        // let (low, high) = binary_search_range_on_answer_2(0, 10, compare_range(0, 10));
        println!("low={low} | high={high}");
        assert_eq!((low, high), (0, 10));
    }
}
