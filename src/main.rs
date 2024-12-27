mod aoc_1;
mod aoc_2;

// use crate::aoc_1::solution;


pub fn main() {
    let solutions_and_expected = [
        (aoc_1::solution(), 11),
        (aoc_2::solution(), 2),
    ];

    // (0..(solutions_and_expected.len())).map(|i| )
    for i in 0..solutions_and_expected.len() {
        let (soln, expected) = solutions_and_expected[i];
        let ip1 = i+1;
        println!("AOC {ip1} Solution: {soln}");
        assert!(soln == expected);

    }
}
