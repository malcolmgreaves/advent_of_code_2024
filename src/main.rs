mod aoc_1;
mod aoc_10;
mod aoc_11;
mod aoc_12;
mod aoc_2;
mod aoc_3;
mod aoc_4;
mod aoc_5;
mod aoc_6;
mod aoc_7;
mod aoc_8;
mod aoc_9;
mod geometry;
mod io_help;
mod matrix;
mod nums;
mod set_ops;
mod utils;

use std::cmp::Ordering;
use std::collections::BTreeMap;

use std::fmt::Display;
use std::io::Write;
use std::process::exit;
use std::sync::mpsc::{Receiver, channel};
use std::sync::{Arc, Mutex};
use std::{io, thread};

use clap::Parser;

/// Advent of Code 2024 Solutions
#[derive(Parser, Debug)]
#[command(version, about, long_about=None)]
struct Args {
    /// Which problem to solve. (0 means all problems)
    #[arg(short=None, long, default_value_t = 0)]
    problem: usize,

    /// Which part of the problem to solve. (0 means all parts)
    #[arg(short=None, long, default_value_t = 0)]
    part: usize,
}

// #[derive(Debug, Clone, PartialEq, Eq)]
// struct posint(usize);

// impl posint {
//     fn new(v: usize) -> posint {
//         if v == 0 {
//             panic!("cannot create zero-valued posint!");
//         }
//         posint(v)
//     }
// }

// impl Display for posint {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         write!(f, "{}", self.0)
//     }
// }

type PosInt = usize;

#[derive(Debug, Clone, PartialEq, Eq, Ord)]
struct Aoc {
    problem: PosInt,
    part: PosInt,
    func: fn() -> u64,
}

impl Aoc {
    fn new(problem: PosInt, part: PosInt, func: fn() -> u64) -> Aoc {
        if problem == 0 {
            panic!("problem number must be positive!");
        }
        if part == 0 {
            panic!("part number must be positive!");
        }
        Aoc {
            problem,
            part,
            func,
        }
    }

    fn stringify(&self) -> String {
        format!("AOC #{} pt.{}", self.problem, self.part)
    }
}

impl PartialOrd for Aoc {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match self.problem.partial_cmp(&other.problem) {
            Some(Ordering::Equal) => {}
            ord => return ord,
        }
        match self.part.partial_cmp(&other.part) {
            Some(Ordering::Equal) => {}
            ord => return ord,
        }
        Some(Ordering::Equal)
    }
}

impl Display for Aoc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.stringify())
    }
}

struct FutureResults<T> {
    r: Receiver<(Aoc, T)>,
    n_expected: usize,
}

// impl <T: 'static> FutureResults<T> {
impl<T> FutureResults<T> {
    fn collect(self) -> BTreeMap<Aoc, T> {
        self.r.iter().take(self.n_expected).collect()
    }

    // fn iter(self) -> std::iter::Take<std::sync::mpsc::Iter<'static, (String, T)>> {
    // fn iter(self) -> impl Iterator<Item=(String, T)> {
    //     self.r.iter().take(self.n_expected)
    // }
}

fn concurrently_run(problems: &[Aoc]) -> FutureResults<u64> {
    let (s, r) = channel::<(Aoc, u64)>();
    let tx = Arc::new(Mutex::new(s));

    problems.iter().for_each(|aoc| {
        let tx = tx.clone();
        let f = aoc.func;
        let aoc = aoc.clone();
        thread::spawn(move || {
            let result = f();
            tx.lock().unwrap().send((aoc, result)).unwrap(); //.expect(format!("Could not compute result for: {name}!").as_str());
        });
    });

    FutureResults {
        r,
        n_expected: problems.len(),
    }
}

pub fn main() {
    // let problems: BTreeMap<&str, fn() -> u64> = BTreeMap::from([
    let all_problems = [
        Aoc::new(1, 1, aoc_1::solution_pt1),
        Aoc::new(1, 2, aoc_1::solution_pt2),
        Aoc::new(2, 1, aoc_2::solution_pt1),
        Aoc::new(2, 2, aoc_2::solution_pt2),
        Aoc::new(3, 1, aoc_3::solution_pt1),
        Aoc::new(3, 2, aoc_3::solution_pt2),
        Aoc::new(4, 1, aoc_4::solution_pt1),
        Aoc::new(4, 2, aoc_4::solution_pt2),
        Aoc::new(5, 1, aoc_5::solution_pt1),
        Aoc::new(5, 2, aoc_5::solution_pt2),
        Aoc::new(6, 1, aoc_6::solution_pt1),
        // Aoc::new(6, 2, aoc_6::solution_pt2),
        Aoc::new(7, 1, aoc_7::solution_pt1),
        Aoc::new(7, 2, aoc_7::solution_pt2),
        Aoc::new(8, 1, aoc_8::solution_pt1),
        Aoc::new(8, 2, aoc_8::solution_pt2),
        Aoc::new(9, 1, aoc_9::solution_pt1),
        Aoc::new(9, 2, aoc_9::solution_pt2),
        Aoc::new(10, 1, aoc_10::solution_pt1),
        Aoc::new(10, 2, aoc_10::solution_pt2),
        Aoc::new(11, 1, aoc_11::solution_pt1),
        Aoc::new(11, 2, aoc_11::solution_pt2),
        Aoc::new(12, 1, aoc_12::solution_pt1),
        Aoc::new(12, 2, aoc_12::solution_pt2),
    ];

    let args = Args::parse();

    let problems: &[Aoc] = {
        if args.problem == 0 {
            &all_problems
        } else {
            if args.part == 0 {
                &all_problems
                    .into_iter()
                    .filter(
                        |Aoc {
                             problem,
                             part: _,
                             func: _,
                         }| *problem == args.problem,
                    )
                    .collect::<Vec<_>>()
            } else {
                &all_problems
                    .into_iter()
                    .filter(
                        |Aoc {
                             problem,
                             part,
                             func: _,
                         }| *problem == args.problem && *part == args.part,
                    )
                    .collect::<Vec<_>>()
            }
        }
    };

    match problems.len() {
        0 => {
            eprintln!(
                "must select at least one problem to run! Invalid problem: {} and invalid part: {}",
                args.problem, args.part
            );
            exit(1);
        }
        1 => {
            let name = problems[0].stringify();
            let func = problems[0].func;
            print!("{name}: ");
            io::stdout().flush().expect("Could not flush STDOUT.");
            let r = func();
            println!("{r}");
        }
        _ => {
            println!(
                "Computing solutions for {} Advent of Code 2024 problems.",
                problems.len()
            );
            let solutions = concurrently_run(&problems);
            // for (name, r) in solutions.r.iter().take(solutions.n_expected) {
            for (name, r) in solutions.collect() {
                println!("{name}: {r}");
            }
        }
    }
}
