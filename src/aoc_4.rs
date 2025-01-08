use core::iter::Iterator;

use std::{
    borrow::Borrow,
    cmp::{max, min},
    collections::HashSet,
};

use crate::{io_help, utils};

// https://adventofcode.com/2024/day/4

pub fn solution_pt1() -> i32 {
    let lines = io_help::read_lines("./inputs/4").collect::<Vec<String>>();
    assert_ne!(lines.len(), 0);

    let ROWS = lines.len();
    let COLS = lines[0].len();

    count_terms("XMAS", utils::convert_to_char_matrix::<ROWS, COLS>(&lines))
}

fn count_terms<const L: usize, const N: usize>(term: &str, lines: &[[char; L]; N]) -> i32 {
    // (a) take the matrix and get out each full-length
    //  - horizontal
    //  - vertical
    //  - diagonal
    // (b) and also take the reverse of each
    // (c) take each of these (6 kinds in total) and break up into windows of term.len()
    // (d) check if each window equals term, if so, increment++
    // (e) return increment total

    let increment = |expanded: Vec<String>| -> i32 {
        let c1 = count(term, expanded.iter());
        let rev_expanded: Vec<String> = expanded
            .into_iter()
            .map(utils::reverse_string)
            .collect::<Vec<_>>();
        let c2 = count(term, rev_expanded.iter());
        c1 + c2
    };

    let mut found = 0;
    found += increment(horizontals(lines));
    found += increment(verticals(lines));
    found += increment(diagonals(lines));
    found
}

fn count<'a>(term: &str, expanded: impl Iterator<Item = &'a String>) -> i32 {
    let mut found = 0;
    // (c, d, e) window, check, increment
    expanded.for_each(|line| {
        line.chars()
            .collect::<Vec<_>>()
            .windows(term.len())
            .map(|x| x.into_iter().collect::<String>())
            .for_each(|candidate| {
                if candidate == term {
                    found += 1;
                }
            })
    });
    found
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
    let mut result = HashSet::new();

    let d1 = diagonals_r2l(lines);
    result.extend(d1);

    let transposed = utils::transpose(lines);
    let d2 = diagonals_r2l(&transposed);
    result.extend(d2);

    // destructive: move each element **OUT OF** the hashset so we can construct a Vec<String<>
    // note that .iter() will borrow each element, meaning we'd have Vec<&String>
    // we don't need the hashset after this function, so we `move`
    result.into_iter().collect::<Vec<_>>()
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
        let expected = ["a", "be", "cfi", "dgjm", "hknq", "loru", "psv", "tw", "x"];

        let result = diagonals_r2l(&example_input);
        for e in expected {
            assert!(result.contains(&e.to_string()));
        }
        assert_eq!(result.len(), expected.len());

        let transposed_example = utils::transpose(&example_input);
        let result_transposed = diagonals_r2l(&transposed_example);
        for e in expected {
            assert!(result_transposed.contains(&utils::reverse_string(e.to_string())));
        }
        assert_eq!(result_transposed.len(), expected.len());
    }
}
