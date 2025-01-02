mod aoc_1;
mod aoc_2;
mod aoc_3;
mod io_help;

use std::collections::HashMap;

pub fn main() {
    let solutions = HashMap::from([
        ("AOC #1", aoc_1::solution()),
        ("AOC #2", aoc_2::solution()),
        ("AOC #3 pt.1", aoc_3::solution_pt1()),
        ("AOC #3 pt.2", aoc_3::solution_pt2()),
    ]);
    for (name, result) in solutions.iter() {
        println!("{name} Solution: {result}");
    }
}
