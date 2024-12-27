mod aoc_1;

use crate::aoc_1::solve_aoc_1;


pub fn main() {
    let soln = solve_aoc_1();
    println!("AOC 1 solution: {soln}");
    assert!(soln == 11)
}
