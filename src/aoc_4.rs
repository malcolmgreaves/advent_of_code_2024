use core::iter::Iterator;

use std::cmp::{max, min};

use crate::{io_help, utils};

// https://adventofcode.com/2024/day/4

pub fn solution_pt1() -> u64 {
    let lines = io_help::read_lines("./inputs/4").collect::<Vec<String>>();
    assert_ne!(lines.len(), 0);

    let max_rows = lines.len();
    let max_cols = lines[0].len();

    let matrix = utils::convert_to_char_matrix(max_rows, max_cols, &lines);
    // let matrix: [[char; COLS]; ROWS] = utils::convert_to_char_matrix::<ROWS, COLS>(&lines);

    count_terms(max_cols, max_rows, "XMAS", matrix)
    // count_terms("XMAS", &matrix)
}

type CharMatrix = utils::Matrix<char>;

fn count_terms(max_cols: usize, max_rows: usize, term: &str, lines: CharMatrix) -> u64 {
    // (a) take the matrix and get out each full-length
    //  - horizontal
    //  - vertical
    //  - diagonal
    // (b) and also take the reverse of each
    // (c) take each of these (6 kinds in total) and break up into windows of term.len()
    // (d) check if each window equals term, if so, increment++
    // (e) return increment total

    let increment = |expanded: Vec<String>| -> u64 {
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

    // let increment_diagonals = || -> u64 {
    //     let d1 = diagonals_r2l(L, N, &lines);
    //     for x in d1.iter() {
    //         println!("diagonal (first): {x}")
    //     }
    //     let c1 = count(term, d1.iter());

    //     let _d2 = utils::char_matrix_to_lines(lines.clone());
    //     let reversed_char_matrix = utils::convert_to_char_matrix(
    //         N,
    //         L,
    //         &_d2.into_iter()
    //             .map(|x| x.chars().rev().collect::<String>())
    //             .collect::<Vec<_>>(),
    //     );
    //     let d2 = diagonals_r2l(L, N, &reversed_char_matrix);
    //     for x in d2.iter() {
    //         println!("diagonal (reversed): {x}")
    //     }
    //     let c2 = count(term, d2.iter());

    //     c1 + c2
    // };

    let mut found = 0;
    // println!("[start]       found: {found}");

    let h = increment(horizontals(max_cols, max_rows, &lines));
    found += h;
    // println!("[horizontals] found: {found} (+{h})");

    let v = increment(verticals(max_cols, max_rows, &lines));
    found += v;
    // println!("[verticals]   found: {found} (+{v})");

    // let d = increment_diagonals();
    let d = increment(diagonals(max_cols, max_rows, &lines));
    found += d;
    // println!("[diagonals]   found: {found} (+{d})");

    found
}

fn count<'a>(term: &str, expanded: impl Iterator<Item = &'a String>) -> u64 {
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

fn horizontals(max_cols: usize, max_rows: usize, lines: &CharMatrix) -> Vec<String> {
    assert_eq!(lines.len(), max_rows, "H: char matrix rows != expected");
    lines
        .iter()
        .map(|line| {
            assert_eq!(line.len(), max_cols, "H: char matrix cols != expected");
            line.iter().collect::<String>()
        })
        .collect::<Vec<_>>()
}

fn verticals(max_cols: usize, max_rows: usize, lines: &CharMatrix) -> Vec<String> {
    assert_eq!(lines.len(), max_rows, "V: char matrix rows != expected");
    (0..max_cols)
        .map(move |col| {
            (0..max_rows)
                .map(move |row: usize| lines[row][col])
                .collect::<String>()
        })
        .collect::<Vec<String>>()
}

fn diagonals(max_cols: usize, max_rows: usize, lines: &CharMatrix) -> Vec<String> {
    assert_eq!(lines.len(), max_rows, "D: char matrix rows != expected");
    assert_eq!(lines[0].len(), max_rows, "D: char matrix rows != expected");
    let access_diagonals_of = |m: &utils::Matrix<char>| -> Vec<String> {
        utils::diagonal_coordinates(max_rows as i32, max_cols as i32)
            .iter()
            .map(|diag| diag.iter().map(|(i, j)| m[*i][*j]).collect::<String>())
            .collect::<Vec<_>>()
    };
    let anti_diagonals = access_diagonals_of(lines);
    let mut true_diagonals = access_diagonals_of(&utils::fliplr(lines));
    true_diagonals.extend(anti_diagonals);
    true_diagonals
}

#[allow(dead_code)]
fn diagonals_r2l(max_cols: usize, max_rows: usize, lines: &CharMatrix) -> Vec<String> {
    assert_eq!(lines.len(), max_rows, "D: char matrix rows != expected");
    assert_eq!(lines[0].len(), max_cols, "D: char matrix cols != expected");

    let n = TryInto::<i32>::try_into(max_rows).unwrap();
    let m = TryInto::<i32>::try_into(max_cols).unwrap();

    (0..(n + m - 1))
        .map(|d| {
            // let right_of_diagonal = max(0, d - m + 1);
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

pub fn solution_pt2() -> u64 {
    let lines = io_help::read_lines("./inputs/4").collect::<Vec<String>>();
    assert_ne!(lines.len(), 0);
    let max_rows = lines.len();
    let max_cols = lines[0].len();
    let chars = utils::convert_to_char_matrix(max_rows, max_cols, &lines);

    count_mas_x(&chars)
}

pub fn count_mas_x(chars: &CharMatrix) -> u64 {
    let windows = window_2d((3, 3), &chars);

    let term = "MAS";

    let mut count = 0;
    for w in windows {
        if check_window(term, &w) {
            count += 1;
        }
    }
    count
}

pub fn window_2d<T: Default + Copy>(
    shape: (usize, usize),
    m: &utils::Matrix<T>,
) -> Vec<utils::Matrix<T>> {
    let (w_rows, w_cols) = shape;

    let make_window = |top_left: (usize, usize)| -> Option<utils::Matrix<T>> {
        let mut window = vec![vec![Default::default(); w_cols]; w_rows];
        for i in 0..w_rows {
            if top_left.0 + i >= m.len() {
                return None;
            }
            for j in 0..w_cols {
                if top_left.1 + j >= m[0].len() {
                    return None;
                }
                window[i][j] = m[top_left.0 + i][top_left.1 + j];
            }
        }
        Some(window)
    };

    let mut windows = Vec::new();
    for i in 0..m.len() {
        for j in 0..m[0].len() {
            match make_window((i, j)) {
                Some(window) => windows.push(window),
                None => (),
            }
        }
    }
    windows
}

pub fn check_window(term: &str, window: &utils::Matrix<char>) -> bool {
    assert_eq!(
        term.len(),
        window.len(),
        "[height] window must equal term length!"
    );
    assert_eq!(
        term.len(),
        window[0].len(),
        "[width] window must equal term length!"
    );

    let check_diag = |d: String| -> bool { term == d || term == utils::reverse_string(d) };

    // written this way so that we short-circut and don't compute anti_diagonal
    // keep style the same for regularity
    check_diag({
        let diagonal = (0..term.len()).map(|i| window[i][i]).collect::<String>();
        diagonal
    }) && check_diag({
        let anti_diagonal = {
            let end = term.len() - 1;
            (0..term.len())
                .map(|i| window[i][end - i])
                .collect::<String>()
        };
        anti_diagonal
    })
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_windows_single() {
        let example: utils::Matrix<char> = vec![
            vec!['M', '.', 'S'],
            vec!['.', 'A', '.'],
            vec!['M', '.', 'S'],
        ];
        let windows = window_2d((3, 3), &example);
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0], example);
    }

    #[test]
    fn part_2() {
        let example: utils::Matrix<char> = vec![
            vec!['.', 'M', '.', 'S', '.', '.', '.', '.', '.', '.'],
            vec!['.', '.', 'A', '.', '.', 'M', 'S', 'M', 'S', '.'],
            vec!['.', 'M', '.', 'S', '.', 'M', 'A', 'A', '.', '.'],
            vec!['.', '.', 'A', '.', 'A', 'S', 'M', 'S', 'M', '.'],
            vec!['.', 'M', '.', 'S', '.', 'M', '.', '.', '.', '.'],
            vec!['.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],
            vec!['S', '.', 'S', '.', 'S', '.', 'S', '.', 'S', '.'],
            vec!['.', 'A', '.', 'A', '.', 'A', '.', 'A', '.', '.'],
            vec!['M', '.', 'M', '.', 'M', '.', 'M', '.', 'M', '.'],
            vec!['.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],
        ];

        let windows = window_2d((3, 3), &example);

        let term = "MAS";

        let mut count = 0;
        for w in windows {
            if check_window(term, &w) {
                println!("OK window!:\n{:?}", w);
                count += 1;
            }
        }

        assert_eq!(count, 9, "actual != expected");
    }

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
        assert_eq!(result, 18, "actual (left) | expected (right)");
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
            // ['X', 'X', 'X', 'X'],
        ]
    }

    /*

    let x = [
        [(0,0), (0,1), (0,2), (0,3)],
        [(1,0), (1,1), (1,2), (1,3)],
        [(2,0), (2,1), (2,2), (2,3)],
        [(3,0), (3,1), (3,2), (3,3)],
        [(4,0), (4,1), (4,2), (4,3)],
    ]

    R=5, C=4

    d=0
    x=d
    while x < C;

        (0,0), (1,1), (2, 2), (3,3)

    d=1



    (3,0)
    (2,0), (3,1)


    */

    #[test]
    fn sqaure_diagonal() {
        let example_input = convert(square_xmas_once());
        let d = diagonals_r2l(4, 4, &example_input);
        for x in d {
            println!("{x}");
        }
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
