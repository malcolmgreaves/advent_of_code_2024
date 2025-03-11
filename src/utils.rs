use std::{
    cmp::Ordering,
    collections::HashMap,
    error::Error,
    hash::Hash,
    ops::{AddAssign, SubAssign},
};

pub const LOG_ENABLED: bool = false;

#[macro_export]
macro_rules! log {
    () => {
        if crate::utils::LOG_ENABLED {
            println!();
        }
    };
    ($($arg:tt)*) => {
        if crate::utils::LOG_ENABLED {
            println!($($arg)*);
        }
    };
}

#[allow(dead_code)]
pub fn argmin<F, T, C: Ord>(values: impl Iterator<Item = T>, key: F) -> usize
where
    F: Fn(&T) -> C,
{
    if values.len() == 0 {
        panic!("must supply non-empty values!");
    }
    if values.len() == 1 {
        return 0;
    }
    let mut min_index = 0;
    let mut min_val = key(&values[min_index]);
    values.iter().enumerate().for_each(|(i, x)| {
        let m = key(&x);
        match min_val.cmp(&m) {
            Ordering::Less | Ordering::Equal => (),
            Ordering::Greater => {
                min_index = i;
                min_val = m;
            }
        }
    });
    min_index
}

#[allow(dead_code)]
pub fn collect_results<T, E>(results: impl Iterator<Item = Result<T, E>>) -> (Vec<T>, Vec<E>) {
    results.fold((Vec::new(), Vec::new()), |(mut oks, mut errs), x| {
        match x {
            Result::Ok(t) => oks.push(t),
            Result::Err(e) => errs.push(e),
        };
        (oks, errs)
    })
}

#[allow(dead_code)]
pub fn group_by<K, V>(key: fn(&V) -> K, values: Vec<V>) -> HashMap<K, Vec<V>>
where
    K: Hash + Eq,
{
    values.into_iter().fold(HashMap::new(), |mut m, v| {
        let v_key = key(&v);
        match m.get_mut(&v_key) {
            Some(existing) => _ = existing.push(v),
            None => _ = m.insert(v_key, vec![v]),
        };
        m
    })
}

#[allow(unused)]
pub fn sorted_keys<K, _V>(m: &HashMap<K, _V>) -> Vec<K>
where
    K: Hash + Clone + Ord,
{
    let mut ks = m.keys().map(|x| x.clone()).collect::<Vec<_>>();
    ks.sort();
    ks
}

#[allow(unused)]
pub fn pairs<T>(elements: &[T]) -> impl Iterator<Item = (&T, &T)> {
    assert!(
        elements.len() > 1,
        "elements.len() > 1 VIOLATED because length is: {}",
        elements.len()
    );
    (0..(elements.len() - 1)).map(|i| (&elements[i], &elements[i + 1]))
}

#[allow(unused)]
pub fn increment<K, V>(m: &mut HashMap<K, V>, key: K, val: V)
where
    K: Hash + Eq,
    V: AddAssign,
{
    match m.get_mut(&key) {
        Some(existing) => *existing += val,
        None => _ = m.insert(key, val),
    }
}

#[allow(unused)]
pub fn decrement<K, V>(m: &mut HashMap<K, V>, key: K, val: V)
where
    K: Hash + Eq,
    V: SubAssign,
{
    match m.get_mut(&key) {
        Some(existing) => *existing -= val,
        None => _ = m.insert(key, val),
    }
}

pub fn reverse_string(x: String) -> String {
    x.chars().rev().collect::<String>()
}

pub type Res<T> = Result<T, Box<dyn Error>>;

pub fn proc_elements_result<A, B>(process: fn(&A) -> Res<B>, elements: &[A]) -> Res<Vec<B>> {
    let mut collected: Vec<B> = Vec::new();
    for x in elements.iter() {
        match process(x) {
            Ok(result) => collected.push(result),
            Err(error) => {
                return Err(error);
            }
        }
    }
    Ok(collected)
}

// #[cfg(test)]
// mod test {
//     use super::*;
//     #[test]
//     fn ...() {}
// }
