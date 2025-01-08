use core::iter::Iterator;

use std::cmp::{max, min};

use crate::io_help;

// https://adventofcode.com/2024/day/4

pub fn solution_pt1() -> i32 {
    // &io_help::read_lines("./inputs/4").collect::<Vec<String>>()
    panic!("UNIMPLEMENTED");
}

fn count_terms<const L: usize, const N: usize>(term: &str, lines: &[[char; L]; N]) -> i32 {
    // take the matrix and get out each full-length
    //  - horizontal
    //  - vertical
    //  - diagonal
    // and also take the reverse of each
    // take each of these (6 kinds in total) and break up into windows of term.len()
    // check if each window equals term, if so, increment++
    // return increment total

    let expanded_all_lines: Vec<&str> = Vec::new();

    panic!("");
}

fn horizontals<const L: usize, const N: usize>(lines: &[[char; L]; N]) -> Vec<String> {
    lines.map(|line| line.iter().collect::<String>()).to_vec()
}

fn verticals<const L: usize, const N: usize>(lines: &[[char; L]; N]) -> Vec<String> {
    (0..N)
        .map(|row| String::from_iter((0..L).map(|col| lines[row][col])))
        .collect::<Vec<_>>()
}

fn diagonals<const L: usize, const N: usize>(lines: &[[char; L]; N]) -> Vec<String> {
    // take the (NxL) matrix and convert into lists of index pairs
    // each list corresponds to a full diagonal
    // then, take each list and reindex into `lines` to get the full String
    panic!("");
}

fn diagonals_r2l<const L: usize, const N: usize>(lines: &[[char; L]; N]) -> Vec<String> {
    let n = TryInto::<i32>::try_into(N).unwrap();
    let m = TryInto::<i32>::try_into(L).unwrap();

    (0..(n + m - 1))
        .map(|d| {
            (max(0, d - m + 1)..min(n, d + 1))
                .map(|x| {
                    // (x, d-x)
                    lines[TryInto::<usize>::try_into(x).unwrap()]
                        [TryInto::<usize>::try_into(d - x).unwrap()]
                })
                .collect::<String>()
        })
        .collect::<Vec<_>>()
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

    #[test]
    fn testing_diagonals() {
        let example_input = [
            ['a', 'b', 'c', 'd'],
            ['e', 'f', 'g', 'h'],
            ['i', 'j', 'k', 'l'],
            ['m', 'n', 'o', 'p'],
            ['q', 'r', 's', 't'],
            ['u', 'v', 'w', 'x'],
        ];
        let result = diagonals_r2l(&example_input);
        let expected = ["a", "be", "cfi", "dgjm", "hknq", "loru", "psv", "tw", "x"];
        for e in expected {
            assert!(result.contains(&e.to_string()));
        }
        assert_eq!(result.len(), expected.len());
    }
}
