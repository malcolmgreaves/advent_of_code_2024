// mod aoc_1;
// mod aoc_2;
// mod aoc_3;
// mod aoc_4;
// mod aoc_5;
// mod aoc_6;
// mod aoc_7;
// mod aoc_8;
// mod geometry;
// mod io_help;
// mod nums;
// mod set_ops;
// mod utils;

// use std::collections::BTreeMap;

// fn foobar(x: fn() -> u64) {
//     println!("before");
//     println!("after: {}", x());
// }

// struct FnUn {
//     f: fn() -> u64
// }

// pub fn main() {

//     foobar(aoc_1::solution_pt1);

//     let solutions: BTreeMap<&str, u64> = BTreeMap::from([
//         ("AOC #1 pt.1", aoc_1::solution_pt1()),
//         ("AOC #1 pt.2", aoc_1::solution_pt2()),
//         ("AOC #2 pt.1", aoc_2::solution_pt1()),
//         ("AOC #2 pt.2", aoc_2::solution_pt2()),
//         ("AOC #3 pt.1", aoc_3::solution_pt1()),
//         ("AOC #3 pt.2", aoc_3::solution_pt2()),
//         ("AOC #4 pt.1", aoc_4::solution_pt1()),
//         ("AOC #4 pt.2", aoc_4::solution_pt2()),
//         ("AOC #5 pt.1", aoc_5::solution_pt1()),
//         ("AOC #5 pt.2", aoc_5::solution_pt2()),
//         ("AOC #6 pt.1", aoc_6::solution_pt1()),
//         // ("AOC #6 pt.2", aoc_6::solution_pt2()),
//         ("AOC #7 pt.1", aoc_7::solution_pt1()),
//         ("AOC #7 pt.2", aoc_7::solution_pt2()),
//         ("AOC #8 pt.1", aoc_8::solution_pt1()),
//         ("AOC #8 pt.2", aoc_8::solution_pt2()),
//     ]);
//     for (name, result) in solutions.iter() {
//         println!("{name} Solution: {result}");
//     }
// }

use crossbeam_channel::{after, select, tick};
use std::time::{Duration, Instant};

pub fn main() {
    let start = Instant::now();
    let ticker = tick(Duration::from_millis(50));
    let timeout = after(Duration::from_secs(1));

    loop {
        select! {
            recv(ticker) -> _ => println!("elapsed: {:?}", start.elapsed()),
            recv(timeout) -> _ => break,
        }
    }
}
