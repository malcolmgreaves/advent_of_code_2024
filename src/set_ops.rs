use std::collections::HashSet;
use std::hash::Hash;

pub enum CheckSameElements<T> {
    Equal,
    ActualMissing(Vec<T>),
    ActualExtra(Vec<T>),
    ActualProblems { missing: Vec<T>, extra: Vec<T> },
}

pub fn check_same_elements<T: Hash + Clone + Eq>(
    actual: &[T],
    expected: &[T],
) -> CheckSameElements<T> {
    let h_actual = slice_to_set(actual);
    let h_expected = slice_to_set(expected);
    // let correct_actual = h_expected.intersection(&h_actual);
    let missing_from_actual = h_expected
        .difference(&h_actual)
        .into_iter()
        .map(|x| (*x).clone())
        .collect::<Vec<T>>();
    let extra_in_actual = h_actual
        .difference(&h_expected)
        .into_iter()
        .map(|x| (*x).clone())
        .collect::<Vec<T>>();
    match (missing_from_actual.len() > 0, extra_in_actual.len() > 0) {
        (true, true) => CheckSameElements::ActualProblems {
            missing: missing_from_actual,
            extra: extra_in_actual,
        },
        (true, false) => CheckSameElements::ActualMissing(missing_from_actual),
        (false, true) => CheckSameElements::ActualExtra(extra_in_actual),
        (false, false) => CheckSameElements::Equal,
    }
}

pub fn slice_to_set<T: Hash + Clone + Eq>(xs: &[T]) -> HashSet<T> {
    xs.iter().map(|t| (*t).clone()).collect()
}

// pub fn intersection<T:Hash+Clone+Eq>(h1: HashSet<T>, h2: HashSet<T>) -> HashSet<T> {
//     // elements in both sets
//     let (smaller, larger) = if h1.len() < h2.len() { (h1, h2) } else { (h2, h1) };
//     smaller.iter().filter(|x| larger.contains(*x)).map(|x| (*x).clone()).collect()
// }

// pub fn difference<T:Hash+Clone+Eq>(h1: HashSet<T>, h2: HashSet<T>) -> HashSet<T> {
//     // DIRECTIONAL: the elements of h1 that are **NOT IN** h2.
//     h1.iter().filter(|x| !h2.contains(*x)).map(|x| (*x).clone()).collect()
// }

// pub fn union<T:Hash+Clone+Eq>(h1: HashSet<T>, h2: HashSet<T>) -> HashSet<T> {
//     // the elements in both sets
//     let (smaller, larger) = if h1.len() < h2.len() { (h1, h2) } else { (h2, h1) };
//     let mut u = larger.clone();
//     smaller.difference(other)
//     u
// }
