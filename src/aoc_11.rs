use std::{
    collections::HashMap,
    sync::{Arc, Mutex, mpsc::channel},
    thread::available_parallelism,
};

use threadpool::ThreadPool;

use crate::io_help;

///////////////////////////////////////////////////////////////////////////////////////////////////
/// solutions

pub fn solution_pt1() -> Result<u64, String> {
    let lines = io_help::read_lines("./inputs/11").collect::<Vec<String>>();
    let stone_line = construct(lines[0].as_str());
    let final_stone_line = iterate_stones(&stone_line, 25);
    Ok(final_stone_line.len() as u64)
}

pub fn solution_pt2() -> Result<u64, String> {
    let lines = io_help::read_lines("./inputs/11").collect::<Vec<String>>();
    if lines.len() != 1 {
        return Err(format!("Can only handle one input stone line."));
    }
    let stone_line = construct(lines[0].as_str());
    // iterate_stones_parallel_count(&stone_line, 75, Some(100))
    // iterate_stones_depth_first(&stone_line, 75, Some(25))
    Ok(iterate_stones_count(&stone_line, 75))
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// common / part1

fn construct(line: &str) -> StoneLine {
    line.trim()
        .split(" ")
        .map(|v| {
            let stone_value = v.parse::<u64>().unwrap();
            Stone::new(stone_value)
        })
        .collect()
}

type StoneLine = Vec<Stone>;

#[derive(Clone, Debug, PartialEq, PartialOrd, Eq, Ord, Hash)]
struct Stone {
    /// The number is stored in reverse-order. The least significant digit is at position 0.
    ///
    /// Increasing index increases significance, according to:
    ///     10^i * d
    /// for digit d at position i.
    num_rev: Vec<u8>,
}

impl Stone {
    fn value(&self) -> u64 {
        self.num_rev.iter().enumerate().fold(0, |s, (i, digit)| {
            let base: u64 = 10_u64.pow(i.try_into().unwrap());
            let val = (*digit as u64) * base;
            s + val
        })
    }

    fn split_digits(&self) -> (Self, Self) {
        let n = self.num_rev.len();
        assert!(n > 0 && n % 2 == 0, "{n} must be even to split!");
        let midpoint = n / 2;
        let mut left = self.num_rev[midpoint..n].to_vec();
        let mut right = self.num_rev[0..midpoint].to_vec();
        Self::clear_leading_zeros(&mut left);
        Self::clear_leading_zeros(&mut right); // shouldn't be necessary by construction
        (Self { num_rev: left }, Self { num_rev: right })
    }

    fn clear_leading_zeros(num_rev: &mut Vec<u8>) {
        // remember: reverse-order digits
        //      num_rev[0] is the least significant
        //      num_rev[-1] is the most significant
        // => remove 0s from **END** until hit first non-zero
        loop {
            match num_rev.last() {
                Some(v) => {
                    if *v == 0 {
                        num_rev.pop();
                    } else {
                        return;
                    }
                }
                None => {
                    num_rev.push(0);
                    return;
                }
            }
        }
    }

    fn new(value: u64) -> Self {
        if value == 0 {
            return Self { num_rev: vec![0] };
        }
        let mut num_rev = Vec::new();
        let mut m = value.clone();
        while m > 0 {
            let digit = m % 10;
            m = m / 10;
            num_rev.push(digit.try_into().unwrap());
        }
        Self { num_rev }
    }
}

enum State {
    ZeroToOne(Stone),
    EvenSplit { left: Stone, right: Stone },
    Mul2024(Stone),
}

// RULES:
//
// If the stone is engraved with the number 0, it is replaced by a stone engraved with the number 1.
//
// If the stone is engraved with a number that has an even number of digits, it is replaced by two stones.
// The left half of the digits are engraved on the new left stone, and the right half of the digits are engraved on the new right stone.
// (The new numbers don't keep extra leading zeroes: 1000 would become stones 10 and 0.)
//
// If none of the other rules apply, the stone is replaced by a new stone; the old stone's number multiplied by 2024 is engraved on the new stone.
fn action(s: &Stone) -> State {
    if s.num_rev.len() == 0 {
        panic!("unitialized Stone -- no digits!");
    }
    if s.num_rev.len() == 1 && s.num_rev[0] == 0 {
        return State::ZeroToOne(Stone { num_rev: vec![1] });
    }
    if s.num_rev.len() % 2 == 0 {
        let (left, right) = s.split_digits();
        return State::EvenSplit { left, right };
    }
    let new_value = s.value() * 2024;
    State::Mul2024(Stone::new(new_value))
}

fn iterate_stones_count(stone_line: &StoneLine, times: usize) -> u64 {
    assert_ne!(stone_line.len(), 0);

    let mut builder = HashMap::<Stone, u64>::with_capacity(1_000_000);
    for s in stone_line {
        builder.insert(s.clone(), 1);
    }

    for _ in 0..times {
        iterate_count(&mut builder);
    }

    builder.iter().fold(0, |s, (_, occurences)| s + *occurences)
}

fn iterate_count(builder: &mut HashMap<Stone, u64>) {
    let stone_occurences = builder
        .iter()
        .map(|(s, o)| (s.clone(), o.clone()))
        .collect::<Vec<_>>();
    for (stone, occurences) in stone_occurences {
        match action(&stone) {
            State::EvenSplit { left, right } => {
                increase_count(builder, left, occurences);
                increase_count(builder, right, occurences);
            }
            State::ZeroToOne(new) | State::Mul2024(new) => increase_count(builder, new, occurences),
        }
        decrease_count(builder, stone, occurences);
    }
}

fn increase_count(b: &mut HashMap<Stone, u64>, s: Stone, o: u64) {
    match b.get_mut(&s) {
        Some(v) => *v += o,
        None => {
            b.insert(s, o);
        }
    }
}

fn decrease_count(b: &mut HashMap<Stone, u64>, s: Stone, o: u64) {
    match b.get(&s).map(|x| x.clone()) {
        Some(v) => {
            if o >= v {
                b.remove(&s);
            } else {
                *(b.get_mut(&s).unwrap()) -= o;
            }
        }
        None => (),
    }
}

fn iterate_stones(stone_line: &StoneLine, times: usize) -> StoneLine {
    // we need to copy & we want it as a vec
    let mut final_stone_line: Vec<Stone> = stone_line.clone();
    for _ in 0..times {
        final_stone_line = iterate(&final_stone_line);
    }
    final_stone_line
}

fn iterate(stone_line: &StoneLine) -> StoneLine {
    let mut new = Vec::with_capacity(stone_line.len());
    for stone in stone_line {
        // ORDER MATTERS: preserve the original ordering from stone_line when performing action(..)
        match action(&stone) {
            State::EvenSplit { left, right } => {
                new.push(left);
                new.push(right);
            }
            State::ZeroToOne(new_stone) | State::Mul2024(new_stone) => new.push(new_stone),
        }
    }
    new
}

const FAST_ITERATE_LIMIT: usize = 30;

#[allow(dead_code)]
fn iterate_stones_parallel_count(
    stone_line: &StoneLine,
    times: usize,
    n_workers: Option<usize>,
) -> u64 {
    if times <= FAST_ITERATE_LIMIT {
        iterate_stones(stone_line, times).len() as u64
    } else {
        let n_parallel = n_workers.unwrap_or(available_parallelism().unwrap().get());
        let pool = ThreadPool::new(n_parallel);
        let stone_line = iterate_stones(stone_line, FAST_ITERATE_LIMIT);
        let times = times - FAST_ITERATE_LIMIT;

        let (s, r) = channel::<u64>();
        let tx = Arc::new(Mutex::new(s));

        for part_stone_line in partition(stone_line, n_parallel) {
            let tx = tx.clone();
            pool.execute(move || {
                let final_stone_line_part = iterate_stones(&part_stone_line, times);
                let n_in_part = final_stone_line_part.len() as u64;
                tx.lock().unwrap().send(n_in_part).unwrap();
            });
        }
        let final_count = r.iter().take(n_parallel).fold(0, |s, x| s + x);
        pool.join();
        drop(pool);
        final_count
    }
}

#[allow(dead_code)]
fn partition<T>(elements: Vec<T>, n_parts: usize) -> Vec<Vec<T>> {
    if n_parts == 0 {
        panic!("Cannot partition 0 times!")
    } else if n_parts == 1 {
        vec![elements]
    } else {
        let n_per_part = elements.len() / n_parts;
        let mut parts = (0..n_parts).map(|_| vec![]).collect::<Vec<_>>();
        let mut i = 0;
        let mut part_index = 0;
        for e in elements {
            parts[part_index].push(e);
            i += 1;
            if part_index < n_parts - 1 && i % n_per_part == 0 {
                part_index += 1;
            }
        }
        parts
    }
}

#[allow(dead_code)]
fn iterate_stones_depth_first(
    stone_line: &StoneLine,
    times: usize,
    n_workers: Option<usize>,
) -> u64 {
    if times <= FAST_ITERATE_LIMIT {
        iterate_stones(stone_line, times).len() as u64
    } else {
        let n_parallel = n_workers.unwrap_or(available_parallelism().unwrap().get());
        let pool = ThreadPool::new(n_parallel);

        let (s, r) = channel::<u64>();
        let tx = Arc::new(Mutex::new(s));
        // take a stone
        // iterate it
        // next stone

        for stone in stone_line {
            let tx = tx.clone();
            let stone = (*stone).clone();
            pool.execute(move || {
                let final_for_stone = iterate_stones(&vec![stone], times);
                let n_for_stone = final_for_stone.len() as u64;
                tx.lock().unwrap().send(n_for_stone).unwrap();
            });
        }

        let final_count = r.iter().take(stone_line.len()).fold(0, |s, x| s + x);
        pool.join();
        drop(pool);
        final_count
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod test {

    use indoc::indoc;
    use lazy_static::lazy_static;

    use crate::io_help::read_lines_in_memory;

    use super::*;

    const EXAMPLE_INPUT_STR_INITIAL: &str = "125 17";

    const EXAMPLE_LINES_SEQ: &str = indoc! {"
            253000 1 7
            253 0 2024 14168
            512072 1 20 24 28676032
            512 72 2024 2 0 2 4 2867 6032
            1036288 7 2 20 24 4048 1 4048 8096 28 67 60 32
            2097446912 14168 4048 2 0 2 4 40 48 2024 40 48 80 96 2 8 6 7 6 0 3 2
    "};

    lazy_static! {

        // Initial arrangement:
        static ref EXAMPLE_EXPECTED_INITIAL: StoneLine = vec![Stone::new(125), Stone::new(17)];

        static ref EXAMPLE_EXPECTED_SEQ: Vec<StoneLine> = vec![
            // After 1 blink:
            vec![Stone::new(253000), Stone::new(1), Stone::new(7)],
            // After 2 blinks:
            vec![Stone::new(253), Stone::new(0), Stone::new(2024), Stone::new(14168)],
            // After 3 blinks:
            vec![Stone::new(512072), Stone::new(1), Stone::new(20), Stone::new(24), Stone::new(28676032)],
            // After 4 blinks:
            vec![Stone::new(512), Stone::new(72), Stone::new(2024), Stone::new(2), Stone::new(0), Stone::new(2), Stone::new(4), Stone::new(2867), Stone::new(6032)],
            // After 5 blinks:
            vec![Stone::new(1036288), Stone::new(7), Stone::new(2), Stone::new(20), Stone::new(24), Stone::new(4048), Stone::new(1), Stone::new(4048), Stone::new(8096), Stone::new(28), Stone::new(67), Stone::new(60), Stone::new(32)],
            // After 6 blinks:
            vec![Stone::new(2097446912), Stone::new(14168), Stone::new(4048), Stone::new(2), Stone::new(0), Stone::new(2), Stone::new(4), Stone::new(40), Stone::new(48), Stone::new(2024), Stone::new(40), Stone::new(48), Stone::new(80), Stone::new(96), Stone::new(2), Stone::new(8), Stone::new(6), Stone::new(7), Stone::new(6), Stone::new(0), Stone::new(3), Stone::new(2)],
        ];
    }

    #[test]
    fn construction_single() {
        let a = construct(EXAMPLE_INPUT_STR_INITIAL);
        let e: &StoneLine = &EXAMPLE_EXPECTED_INITIAL;
        assert_eq!(a, *e);
    }

    #[test]
    fn iterate_few() {
        let initial_stone_line: &StoneLine = &EXAMPLE_EXPECTED_INITIAL;
        println!("initial_stone_line: {initial_stone_line:?}");
        let first_stone_line = iterate(initial_stone_line);
        println!("first_stone_line:   {first_stone_line:?}");
        let second_stone_line = iterate(&first_stone_line);
        println!("second_stone_line:  {second_stone_line:?}");
        assert_eq!(second_stone_line, *EXAMPLE_EXPECTED_SEQ.get(1).unwrap());
    }

    #[test]
    fn construction_many() {
        let a = construct(EXAMPLE_INPUT_STR_INITIAL);
        let e: &StoneLine = &EXAMPLE_EXPECTED_INITIAL;
        assert_eq!(a, *e);
        let actual = read_lines_in_memory(&EXAMPLE_LINES_SEQ)
            .map(|line| construct(line.as_str()))
            .collect::<Vec<_>>();
        let expected: &Vec<StoneLine> = &EXAMPLE_EXPECTED_SEQ;
        assert_eq!(actual, *expected);
    }

    #[test]
    fn iterate_stones_once() {
        let expected = &(&EXAMPLE_EXPECTED_SEQ)[0];
        let actual = iterate_stones(&EXAMPLE_EXPECTED_INITIAL, 1);
        assert_eq!(actual, *expected);
    }

    #[test]
    fn iterate_stones_six() {
        let n = EXAMPLE_EXPECTED_SEQ.len();
        let expected = &EXAMPLE_EXPECTED_SEQ[n - 1];
        let actual = iterate_stones(&EXAMPLE_EXPECTED_INITIAL, n);
        assert_eq!(actual, *expected);
    }

    #[test]
    fn iterate_repeated() {
        let handle = {
            let mut stone_line: StoneLine = (&EXAMPLE_EXPECTED_INITIAL).to_vec();
            move |(i, e): (usize, &StoneLine)| {
                let actual = iterate(&stone_line);
                assert_eq!(actual, *e, "Iteration #{} did not match.", i + 1);
                stone_line = actual;
            }
        };
        let expected: &Vec<StoneLine> = &EXAMPLE_EXPECTED_SEQ;
        expected.iter().enumerate().for_each(handle);
    }

    #[test]
    fn pt1_soln_example() {
        let actual = iterate_stones(&EXAMPLE_EXPECTED_INITIAL, 6).len();
        assert_eq!(actual, 22);
        let actual = iterate_stones(&EXAMPLE_EXPECTED_INITIAL, 25).len();
        assert_eq!(actual, 55312);
    }

    #[test]
    fn pt2_soln_example() {
        let actual = iterate_stones_count(&EXAMPLE_EXPECTED_INITIAL, 6);
        assert_eq!(actual, 22);
        let actual = iterate_stones_count(&EXAMPLE_EXPECTED_INITIAL, 25);
        assert_eq!(actual, 55312);
    }
}
