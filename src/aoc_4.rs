use core::iter::Iterator;
use std::collections::HashSet;

use crate::io_help;

// https://adventofcode.com/2024/day/4

pub fn solution_pt1() -> i32 {
    // &io_help::read_lines("./inputs/4").collect::<Vec<String>>()
    panic!("UNIMPLEMENTED");
}

fn count_terms<const L: usize, const N: usize>(term: &str, lines: &[[char; L]; N]) -> i32 {
    // let mut lines = format_matrix(term, lines);

    panic!("");
}

// fn format_matrix<const L:usize, const N:usize>(term: &str, lines: &[[char; L]; N]) -> [[char; L]; N] {
//     let term_set = term.chars().collect::<HashSet<char>>();
//     let mut lines = lines.clone();
//     for i in 0..N {
//         for j in 0..L {
//             if !(term_set.contains(&lines[i][j])) {
//                 lines[i][j] = '.';
//             }
//         }
//     }
//     lines
// }

pub fn solution_pt2() -> i32 {
    panic!("UNIMPLEMENTED");
    // io_help::read_lines("./inputs/4").collect::<String>()
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn example_solution_pt1() {
        let example_input = [
            ['M', 'M', 'M', 'S', 'X', 'X', 'M', 'A', 'S', 'M'],
            ['M', 'S', 'A', 'M', 'X', 'M', 'S', 'M', 'S', 'A'],
            ['A', 'M', 'X', 'S', 'X', 'M', 'A', 'A', 'M', 'M'],
            ['M', 'S', 'A', 'M', 'A', 'S', 'M', 'S', 'M', 'X'],
            ['X', 'M', 'A', 'S', 'A', 'M', 'X', 'A', 'M', 'M'],
            ['X', 'X', 'A', 'M', 'M', 'X', 'X', 'A', 'M', 'A'],
            ['S', 'M', 'S', 'M', 'S', 'A', 'S', 'X', 'S', 'S'],
            ['S', 'A', 'X', 'A', 'M', 'A', 'S', 'A', 'A', 'A'],
            ['M', 'A', 'M', 'M', 'M', 'X', 'M', 'M', 'M', 'M'],
            ['M', 'X', 'M', 'X', 'A', 'X', 'M', 'A', 'S', 'X'],
        ];
        let result = count_terms("XMAS", &example_input);
        assert!(result == 18);
    }

    #[test]
    fn example_solution_pt2() {
        println!("UNIMPLEMENTED");
        // let result = solve_inputs_conditionals(example_input.to_string());
        // assert!(result == 48);
    }
}
