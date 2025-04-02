use num::{CheckedSub, Num};

pub fn binary_search_on_answer<N: Num + PartialOrd + CheckedSub + Clone + std::fmt::Debug>(
    low: N,
    high: N,
    is_found: impl Fn(N) -> bool,
) -> N {
    let mut low = low.clone();
    let mut high = high.clone();
    let two = N::one() + N::one();
    let plus_one = |low: &N| -> N { low.clone() + N::one() };
    println!("START: low={low:?} | high={high:?}");
    while plus_one(&low) < high {
        // loop {
        //     if high <= plus_one(&low) {
        //         println!("FOUND! low={low:?} | high={high:?}");
        //         break;
        //     }
        let midpoint = low.clone() + (high.checked_sub(&low).unwrap() / two.clone());
        if is_found(midpoint.clone()) {
            println!("\t is found:  midpoint={midpoint:?} | low={low:?} | high={high:?}");
            high = midpoint;
        } else {
            println!("\t not found: midpoint={midpoint:?} | low={low:?} | high={high:?}");
            low = midpoint;
        }
    }
    high
}
