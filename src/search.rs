use num::{CheckedSub, Num};

pub fn binary_search_on_answer<N: Num + PartialOrd + CheckedSub + Clone>(
    low: N,
    high: N,
    is_found: impl Fn(N) -> bool,
) -> N {
    let mut low = low.clone();
    let mut high = high.clone();
    let two = N::one() + N::one();
    let plus_one = |low: &N| -> N { low.clone() + N::one() };
    loop {
        if high > plus_one(&low) {
            break;
        }
        let midpoint = low.clone() + (high.checked_sub(&low).unwrap() / two.clone());
        if is_found(midpoint.clone()) {
            high = midpoint;
        } else {
            low = midpoint;
        }
    }
    high
}
