mod aoc_1;
mod aoc_2;

// use crate::aoc_1::solution;


pub fn main() {
    let solutions_and_expected = [
        (aoc_1::solution(), 11),
        (aoc_2::solution(), 2),
    ];

    let mut failures: Vec<usize> = [].to_vec();
    // (0..(solutions_and_expected.len())).map(|i| )
    for i in 0..solutions_and_expected.len() {
        let (soln, expected) = solutions_and_expected[i];
        let ip1 = i+1;
        println!("AOC {ip1} Solution: {soln}");
        if soln != expected { 
            failures.push(i);
            println!("\tFAILED! Expecting {expected} but obtained {soln}")
        }
    }
    if failures.len() > 0 {
        let n = failures.len();
        let s = failures.iter().map(
            |f| {
                let fp1 = f+1;
                format!("AOC #{fp1}")
            }
        ).fold(
            "".to_string(), 
            |acc, f| 
                if acc.len() == 0 { 
                    f
                } else { 
                    format!("{acc}, {f}") 
                }
        );
        println!("Found {n} failure(s): '{s}'")
    } else {
        println!("Success!")
    }
}