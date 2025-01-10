use core::iter::Iterator;

use std::{
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

    let matrix = utils::convert_to_char_matrix(ROWS, COLS, &lines);
    // let matrix: [[char; COLS]; ROWS] = utils::convert_to_char_matrix::<ROWS, COLS>(&lines);

    count_terms(COLS, ROWS, "XMAS", matrix)
    // count_terms("XMAS", &matrix)
}

type CharMatrix = utils::Matrix<char>;

fn count_terms(L: usize, N: usize, term: &str, lines: CharMatrix) -> i32 {
    // (a) take the matrix and get out each full-length
    //  - horizontal
    //  - vertical
    //  - diagonal
    // (b) and also take the reverse of each
    // (c) take each of these (6 kinds in total) and break up into windows of term.len()
    // (d) check if each window equals term, if so, increment++
    // (e) return increment total

    let increment = |expanded: Vec<String>| -> i32 {
        let _rows = expanded.len();
        let _cols = expanded[0].len();

        let c1 = count(term, expanded.iter());

        let rev_expanded: Vec<String> = expanded
            .into_iter()
            .map(utils::reverse_string)
            .collect::<Vec<_>>();
        let c2 = count(term, rev_expanded.iter());

        // println!("{} x {} -> c1: {} , c2: {}", _rows, _cols, c1, c2);
        c1 + c2
    };

    let increment_diagonals = || -> i32 {
        let d1 = diagonals_r2l(L, N, &lines);
        let c1 = count(term, d1.iter());

        let _d2 = utils::char_matrix_to_lines(lines.clone());
        let reversed_char_matrix = utils::convert_to_char_matrix(
            N,
            L,
            &_d2.into_iter()
                .map(|x| x.chars().rev().collect::<String>())
                .collect::<Vec<_>>(),
        );
        let d2 = diagonals_r2l(L, N, &reversed_char_matrix);
        let c2 = count(term, d2.iter());

        c1 + c2
    };

    let mut found = 0;
    println!("[start]       found: {found}");

    let h = increment(horizontals(L, N, &lines));
    found += h;
    println!("[horizontals] found: {found} (+{h})");

    let v = increment(verticals(L, N, &lines));
    found += v;
    println!("[verticals]   found: {found} (+{v})");

    let d = increment_diagonals();
    found += d;
    println!("[diagonals]   found: {found} (+{d})");

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

fn horizontals(L: usize, N: usize, lines: &CharMatrix) -> Vec<String> {
    assert_eq!(lines.len(), N, "H: char matrix rows != expected");
    lines
        .iter()
        .map(|line| {
            assert_eq!(line.len(), L, "H: char matrix cols != expected");
            line.iter().collect::<String>()
        })
        .collect::<Vec<_>>()
}

fn verticals(L: usize, N: usize, lines: &CharMatrix) -> Vec<String> {
    assert_eq!(lines.len(), N, "V: char matrix rows != expected");
    (0..L)
        .map(move |col| {
            (0..N)
                .map(move |row: usize| lines[row][col])
                .collect::<String>()
        })
        .collect::<Vec<String>>()
}

fn diagonals(L: usize, N: usize, lines: &CharMatrix) -> Vec<String> {
    // take the (NxL) matrix and convert into lists of index pairs
    // each list corresponds to a full diagonal
    // then, take each list and reindex into `lines` to get the full String
    let mut result = HashSet::new();

    let d1 = diagonals_r2l(L, N, lines);
    result.extend(d1);

    let transposed = utils::transpose(L, N, &lines);
    let d2 = diagonals_r2l(L, N, &transposed);
    result.extend(d2);

    // destructive: move each element **OUT OF** the hashset so we can construct a Vec<String<>
    // note that .iter() will borrow each element, meaning we'd have Vec<&String>
    // we don't need the hashset after this function, so we `move`
    result.into_iter().collect::<Vec<_>>()
}

fn diagonals_r2l(L: usize, N: usize, lines: &CharMatrix) -> Vec<String> {
    assert_eq!(lines.len(), N, "D: char matrix rows != expected");
    assert_eq!(lines[0].len(), L, "D: char matrix cols != expected");

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
    println!("UNIMPLEMENTED");
    // io_help::read_lines("./inputs/4").collect::<String>()
    -1
}

#[cfg(test)]
mod test {

    use super::*;

    fn convert<const R: usize, const C: usize>(example_input: [[char; C]; R]) -> CharMatrix {
        example_input
            .into_iter()
            .map(|x| x.to_vec())
            .collect::<Vec<_>>()
    }

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
        let result = count_terms(10, 10, "XMAS", convert(example_input));
        // let result = count_terms("XMAS", &example_input);
        assert!(result == 18);
    }

    #[test]
    fn example_solution_pt2() {
        println!("UNIMPLEMENTED");
        // let result = solve_inputs_conditionals(example_input.to_string());
        // assert!(result == 48);
    }

    fn square_xmas_once() -> [[char; 4]; 4] {
        [
            ['X', 'M', 'A', 'S'],
            ['X', 'X', 'X', 'X'],
            ['X', 'X', 'X', 'X'],
            ['X', 'X', 'X', 'X'],
        ]
    }

    #[test]
    fn example_solution_small_repeats() {
        let example_input = square_xmas_once();
        let result = count_terms(
            example_input[0].len(),
            example_input.len(),
            "XMAS",
            convert(example_input),
        );
        assert!(result == 1, "result: {result} != 1");
    }

    #[test]
    fn check_shapes_simple_example() {
        let example_input = convert(square_xmas_once());
        // println!("example:");
        // for i in 0..example_input.len() {
        //     println!("{:?}", example_input[i]);
        // }
        // println!("------------------------------");

        // println!("HORIZONTAL:");
        // for (index, h) in horizontals(4, 4, &example_input).iter().enumerate() {
        //     println!("[{index}]: {h}", );
        // }
        // println!("--------------");
        assert_eq!(
            horizontals(4, 4, &example_input),
            vec!["XMAS", "XXXX", "XXXX", "XXXX"],
            "horizontals failed"
        );

        // println!("VERTICAL:");
        // for (index, v) in verticals(4, 4, &example_input).iter().enumerate() {
        //     println!("[{index}]: {v}", );
        // }
        // println!("--------------");
        assert_eq!(
            verticals(4, 4, &example_input),
            vec!["XXXX", "MXXX", "AXXX", "SXXX"],
            "verticals failed"
        );

        // println!("DIAGONTAL r2l:");
        // for (index, d) in diagonals_r2l(4, 4, &example_input).iter().enumerate() {
        //     println!("[{index}]: {d}", );
        // }
        // println!("--------------");
        assert_eq!(
            diagonals_r2l(4, 4, &example_input),
            vec!["X", "MX", "AXX", "SXXX", "XXX", "XX", "X",],
            "diagonals failed"
        );
    }

    #[test]
    fn testing_diagonals() {
        const ROWS: usize = 6;
        const COLS: usize = 4;

        let example_input: [[char; COLS]; ROWS] = [
            ['a', 'b', 'c', 'd'],
            ['e', 'f', 'g', 'h'],
            ['i', 'j', 'k', 'l'],
            ['m', 'n', 'o', 'p'],
            ['q', 'r', 's', 't'],
            ['u', 'v', 'w', 'x'],
        ];
        let expected = ["a", "be", "cfi", "dgjm", "hknq", "loru", "psv", "tw", "x"];

        let result = diagonals_r2l(COLS, ROWS, &convert(example_input));
        // let result = diagonals_r2l(&example_input);
        for e in expected {
            assert!(result.contains(&e.to_string()));
        }
        assert_eq!(result.len(), expected.len());

        let transposed_example = utils::transpose(ROWS, COLS, &convert(example_input));
        // let transposed_example = utils::transpose(&example_input);

        // println!("transposed example:\nrows: {} | cols: {}", transposed_example.len(), transposed_example[0].len());

        let result_transposed = diagonals_r2l(ROWS, COLS, &transposed_example);
        // let result_transposed = diagonals_r2l(&transposed_example);
        for e in expected {
            assert!(result_transposed.contains(&utils::reverse_string(e.to_string())));
        }
        assert_eq!(result_transposed.len(), expected.len());
    }
}
