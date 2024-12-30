mod aoc_1;
mod aoc_2;

// use crate::aoc_1::solution;

pub fn main() {
    let solutions = [aoc_1::solution(), aoc_2::solution()];
    for i in 0..solutions.len() {
        let ip1 = i + 1;
        let soln = solutions[i];
        println!("AOC {ip1} Solution: {soln}");
    }
}
