mod aoc_1;
mod aoc_2;
mod aoc_3;
mod io_help;

use std::collections::BTreeMap;

pub fn main() {
    let solutions = BTreeMap::from([
        ("AOC #1 pt.1", aoc_1::solution_pt1()),
        ("AOC #1 pt.2", aoc_1::solution_pt2()),
        ("AOC #2 pt.1", aoc_2::solution_pt1()),
        ("AOC #2 pt.2", aoc_2::solution_pt2()),
        ("AOC #3 pt.1", aoc_3::solution_pt1()),
        ("AOC #3 pt.2", aoc_3::solution_pt2()),
    ]);
    for (name, result) in solutions.iter() {
        println!("{name} Solution: {result}");
    }
}
