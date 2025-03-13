use std::collections::HashMap;
use std::fmt::Debug;
use std::num::ParseIntError;
use std::str::FromStr;

use regex::Regex;

use crate::io_help;
use crate::matrix::{Coordinate, GridMovement, Matrix};
use crate::utils::{collect_results, sum_bools};

///////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone)]
struct Velocity {
    x: i32,
    y: i32,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone)]
struct Robot {
    p: Coordinate,
    v: Velocity,
}

fn construct(lines: impl Iterator<Item = String>) -> Result<Vec<Robot>, String> {
    let (robots, errors) = collect_results(lines.map(|s| parse_robot(s.as_str())));
    if errors.len() > 0 {
        return Result::Err(format!(
            "failed to parse {} lines (successfully parsed {} lines):\n\t{}",
            errors.len(),
            robots.len(),
            errors.join("\n\t"),
        ));
    }
    Result::Ok(robots)
}

fn parse_robot(s: &str) -> Result<Robot, String> {
    // p=0,4 v=3,-3
    let bits = s.trim().split(" ").collect::<Vec<_>>();
    if bits.len() != 2 {
        return Result::Err(format!(
            "expected to have Coordinate & velocity substrings, but found {}: '{bits:?}' - input: '{s}'",
            bits.len()
        ));
    }

    let p = parse_position(bits[0])?;
    let v = parse_velocity(bits[1])?;
    Result::Ok(Robot { p, v })
}

fn parse_position(s: &str) -> Result<Coordinate, String> {
    // p=0,4
    let re = Regex::new(r"p=(\d+),(\d+)").unwrap();
    let (x, y) = _parse_number_pair::<usize, ParseIntError>(re, s)?;
    // x is the column -> it's how many places FROM the left
    // y is the row    -> it's how many places FROM the top
    Result::Ok(Coordinate { row: y, col: x })
}

fn parse_velocity(s: &str) -> Result<Velocity, String> {
    // v=3,-3
    let re = Regex::new(r"v=(-?\d+),(-?\d+)").unwrap();
    let (x, y) = _parse_number_pair::<i32, ParseIntError>(re, s)?;
    Result::Ok(Velocity { x, y })
}

fn _parse_number_pair<Int, E>(re: Regex, s: &str) -> Result<(Int, Int), String>
where
    E: Debug,
    Int: FromStr<Err = E> + Debug,
{
    let parse = |m: &str| -> Result<Int, String> {
        m.parse::<Int>()
            .map_err(|e| format!("{e:?} - input: '{s}'"))
    };

    let (first, second) = match re.captures(s) {
        Some(caps) => match (caps.get(1), caps.get(2)) {
            (Some(first), Some(second)) => (first.as_str(), second.as_str()),
            _ => {
                return Result::Err(format!(
                    "failed to capture two items from regex: first={:?}, second={:?}",
                    caps.get(1),
                    caps.get(2)
                ));
            }
        },
        None => {
            return Result::Err(format!(
                "regex matched nothing! re={}, input='{s}'",
                re.as_str()
            ));
        }
    };

    let a = parse(first)?;
    let b = parse(second)?;

    Result::Ok((a, b))
}

///////////////////////////////////////////////////////////////////////////////////////////////////

pub fn solution_pt1() -> u64 {
    let lines = io_help::read_lines("./inputs/14");
    let robots = construct(lines).unwrap();
    let board = create_board_state(robots);
    assert_eq!(board.g.max_row + 1, 103);
    assert_eq!(board.g.max_col + 1, 101);
    let final_positions = run_robot_paths(&board, 100);
    safety_factor(&final_positions)
}

#[derive(Debug, PartialEq, Eq, Clone)]
struct BoardState {
    positions: HashMap<Robot, Coordinate>,
    g: GridMovement,
}

fn create_board_state(
    robots: Vec<Robot>,
    // constraint_max_row: Option<usize>,
    // constraint_max_y: Option<usize>,
) -> BoardState {
    let mut max_row = 0;
    let mut max_col = 0;

    let positions = robots
        .into_iter()
        .map(|r| {
            let p = r.p.clone();
            max_row = std::cmp::max(p.row, max_row);
            max_col = std::cmp::max(p.col, max_col);
            (r, p)
        })
        .collect();

    // let check_bounds = |max: usize, constraint: Option<usize>, msg: &str| -> usize {
    //     match constraint {
    //         Some(v) => {
    //             if max > v {
    //                 panic!(
    //                     "constraining {msg} to be {v} but there's at least one robot who's {msg} value is greater: {max}"
    //                 );
    //             } else {
    //                 v
    //             }
    //         }
    //         None => max,
    //     }
    // };
    // let max_x = check_bounds(max_x, constraint_max_row, "X");
    // let max_y = check_bounds(max_y, constraint_max_y, "Y");

    BoardState {
        positions,
        g: GridMovement { max_row, max_col },
    }
}

fn run_robot_paths(board: &BoardState, iterations: u64) -> BoardState {
    if iterations == 0 {
        return board.clone();
    }
    let iterated = board
        .positions
        .iter()
        .map(|(r, p)| {
            let updated_positions = run_robot_path(&board.g, &p, &r.v, iterations);
            (r.clone(), updated_positions)
        })
        .collect();
    BoardState {
        positions: iterated,
        g: board.g.clone(),
    }
}

fn run_robot_path(
    // #[allow(dead_code)]
    // fn run_robot_path_closed_form(
    g: &GridMovement,
    initial: &Coordinate,
    v: &Velocity,
    iterations: u64,
) -> Coordinate {
    // position after X seconds: pos = (pos + time*velocity) mod (max width/height)
    let row_signed =
        (initial.row as i64 + (v.y as i64 * iterations as i64)) % (g.max_row + 1) as i64;
    let col_signed =
        (initial.col as i64 + (v.x as i64 * iterations as i64)) % (g.max_col + 1) as i64;

    let into_coordinate = |signed: i64, max: usize| -> usize {
        if signed < 0 {
            max - TryInto::<usize>::try_into(signed.abs()).unwrap()
        } else {
            TryInto::<usize>::try_into(signed).unwrap()
        }
    };

    let row = into_coordinate(row_signed, g.max_row + 1);
    let col = into_coordinate(col_signed, g.max_col + 1);

    Coordinate { row, col }
}

// fn run_robot_path(
#[allow(dead_code)]
fn run_robot_path_iterate(
    g: &GridMovement,
    initial: &Coordinate,
    v: &Velocity,
    iterations: u64,
) -> Coordinate {
    let mut row = initial.row;
    let mut col = initial.col;
    for _ in 0..iterations {
        row = g.wrap_increment_row(row, v.y);
        col = g.wrap_increment_col(col, v.x);
    }

    Coordinate { row, col }
}

fn safety_factor(board: &BoardState) -> u64 {
    let quads = quadrant_split(board);
    quads.top_left.len() as u64
        * quads.top_right.len() as u64
        * quads.bottom_left.len() as u64
        * quads.bottom_right.len() as u64
}

struct Quadrants<'a> {
    top_left: Vec<&'a Robot>,
    top_right: Vec<&'a Robot>,
    bottom_left: Vec<&'a Robot>,
    bottom_right: Vec<&'a Robot>,
}

fn quadrant_split<'a>(board: &'a BoardState) -> Quadrants<'a> {
    // Robots that are exactly in the middle (horizontally or vertically) don't count as being in any quadrant.

    // len is odd => midpoint is column => discount
    // len is even => midpoint is fractional => no middle so no discount

    let (end_row, start_row) = _split_no_midpoint(board.g.max_row + 1);

    let (end_col, start_col) = _split_no_midpoint(board.g.max_col + 1);

    let top_left = filter_robots_by_coordinate(
        &board.positions,
        0,       //
        0,       //
        end_row, //
        end_col, //
    );

    let top_right =
        filter_robots_by_coordinate(&board.positions, 0, start_col, end_row, board.g.max_col);

    let bottom_left =
        filter_robots_by_coordinate(&board.positions, start_row, 0, board.g.max_row, end_col);

    let bottom_right = filter_robots_by_coordinate(
        &board.positions,
        start_row,
        start_col,
        board.g.max_row,
        board.g.max_col,
    );

    Quadrants {
        top_left,
        top_right,
        bottom_left,
        bottom_right,
    }
}

// Takes in maximum number and produces indices.
fn _split_no_midpoint(max_number_not_index: usize) -> (usize, usize) {
    let x_mid = max_number_not_index / 2;
    if max_number_not_index % 2 == 0 {
        // even => no middle index!
        (x_mid, x_mid + 1)
    } else {
        // odd => x_mid is middle index thus NOT x_mid !!
        (x_mid - 1, x_mid + 1)
    }
}

fn filter_robots_by_coordinate<'a>(
    positions: &'a HashMap<Robot, Coordinate>,
    min_row: usize,
    min_col: usize,
    max_row: usize,
    max_col: usize,
) -> Vec<&'a Robot> {
    positions
        .iter()
        .filter(|(_, position)| {
            position.row >= min_row
                && position.row <= max_row
                && position.col >= min_col
                && position.col <= max_col
        })
        .map(|(r, _)| r)
        .collect::<Vec<_>>()
}

///////////////////////////////////////////////////////////////////////////////////////////////////

pub fn solution_pt2() -> u64 {
    let lines = io_help::read_lines("./inputs/14");
    let robots = construct(lines).unwrap();
    let board = create_board_state(robots);
    assert_eq!(board.g.max_row + 1, 103);
    assert_eq!(board.g.max_col + 1, 101);
    find_christmas_tree_iteration(&board)
}

fn find_christmas_tree_iteration(board: &BoardState) -> u64 {
    let mut active = board.clone();
    let mut i = 0;
    loop {
        if check_christmas_tree(&active, 35) {
            return i;
        }
        i += 1;
        active = run_robot_paths(&active, 1);
    }
}

fn check_christmas_tree(board: &BoardState, heuristic_limit: usize) -> bool {
    let present: Matrix<bool> = {
        let mut present = vec![vec![false; board.g.max_col + 1]; board.g.max_row + 1];
        board
            .positions
            .values()
            .for_each(|Coordinate { row, col }| present[*row][*col] = true);
        present
    };

    let sum_by_row = present
        .iter()
        .map(|row| sum_bools(row.iter()))
        .collect::<Vec<_>>();

    let maybe_row = sum_by_row.iter().any(|s| *s > heuristic_limit);
    // println!(
    //     "\tmax sum by rows is: {:?} & limit={heuristic_limit}",
    //     sum_by_row.iter().max()
    // );
    if maybe_row {
        return true;
    }

    let sum_by_col = (0..board.g.max_col + 1)
        .map(|col_index| {
            let col_values = present.iter().map(|row| row[col_index]);
            sum_bools(col_values)
        })
        .collect::<Vec<_>>();
    let maybe_col = sum_by_col.iter().any(|s| *s > heuristic_limit);
    // println!(
    //     "\tmax sum by column is: {:?} and limit={heuristic_limit}",
    //     sum_by_col.iter().max()
    // );
    maybe_col
    // maybe_row || maybe_col
}

///////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod test {

    use indoc::indoc;
    use lazy_static::lazy_static;

    use crate::{io_help::read_lines_in_memory, matrix::Coords};

    use super::*;

    ///////////////////////////////////////////////

    const EXAMPLE_INPUT_STR: &str = indoc! {"
        p=0,4 v=3,-3
        p=6,3 v=-1,-3
        p=10,3 v=-1,2
        p=2,0 v=2,-1
        p=0,0 v=1,3
        p=3,0 v=-2,-2
        p=7,6 v=-1,-3
        p=3,0 v=-1,-2
        p=9,3 v=2,3
        p=7,3 v=-1,2
        p=2,4 v=2,-3
        p=9,5 v=-3,-3
    "};

    lazy_static! {
        static ref EXAMPLE_EXPECTED: Vec<Robot> = vec![
            Robot {
                p: Coordinate { col: 0, row: 4 },
                v: Velocity { x: 3, y: -3 }
            },
            Robot {
                p: Coordinate { col: 6, row: 3 },
                v: Velocity { x: -1, y: -3 }
            },
            Robot {
                p: Coordinate { col: 10, row: 3 },
                v: Velocity { x: -1, y: 2 }
            },
            Robot {
                p: Coordinate { col: 2, row: 0 },
                v: Velocity { x: 2, y: -1 }
            },
            Robot {
                p: Coordinate { col: 0, row: 0 },
                v: Velocity { x: 1, y: 3 }
            },
            Robot {
                p: Coordinate { col: 3, row: 0 },
                v: Velocity { x: -2, y: -2 }
            },
            Robot {
                p: Coordinate { col: 7, row: 6 },
                v: Velocity { x: -1, y: -3 }
            },
            Robot {
                p: Coordinate { col: 3, row: 0 },
                v: Velocity { x: -1, y: -2 }
            },
            Robot {
                p: Coordinate { col: 9, row: 3 },
                v: Velocity { x: 2, y: 3 }
            },
            Robot {
                p: Coordinate { col: 7, row: 3 },
                v: Velocity { x: -1, y: 2 }
            },
            Robot {
                p: Coordinate { col: 2, row: 4 },
                v: Velocity { x: 2, y: -3 }
            },
            Robot {
                p: Coordinate { col: 9, row: 5 },
                v: Velocity { x: -3, y: -3 }
            },
        ];
    }

    const INITIAL_ROBOT: Robot = Robot {
        p: Coordinate { row: 4, col: 2 },
        v: Velocity { x: 2, y: -3 },
    };

    ///////////////////////////////////////////////

    #[test]
    fn construction() {
        let expected: &[Robot] = &EXAMPLE_EXPECTED;
        match construct(read_lines_in_memory(EXAMPLE_INPUT_STR)) {
            Result::Ok(actual) => assert_eq!(actual, expected),
            Result::Err(error) => assert!(false, "{error}"),
        };
    }

    #[test]
    fn board() {
        let board = create_board_state(EXAMPLE_EXPECTED.clone());
        assert_eq!(
            board.g,
            GridMovement {
                max_row: 6,
                max_col: 10
            }
        );
        let expected = EXAMPLE_EXPECTED
            .iter()
            .map(|r| (r.clone(), r.p.clone()))
            .collect();
        assert_eq!(board.positions, expected);
    }

    #[test]
    fn iterate_1() {
        let board = run_robot_paths(&create_board_state(EXAMPLE_EXPECTED.clone()), 1);
        let actual = board.positions.get(&INITIAL_ROBOT).unwrap();
        assert_eq!(actual, &Coordinate { row: 1, col: 4 })
    }

    #[test]
    fn iterate_2() {
        let board = run_robot_paths(&create_board_state(EXAMPLE_EXPECTED.clone()), 2);
        let actual = board.positions.get(&INITIAL_ROBOT).unwrap();
        assert_eq!(actual, &Coordinate { row: 5, col: 6 })
    }

    #[test]
    fn iterate_3() {
        let board = run_robot_paths(&create_board_state(EXAMPLE_EXPECTED.clone()), 3);
        let actual = board.positions.get(&INITIAL_ROBOT).unwrap();
        assert_eq!(actual, &Coordinate { row: 2, col: 8 })
    }

    #[test]
    fn iterate_4() {
        let board = run_robot_paths(&create_board_state(EXAMPLE_EXPECTED.clone()), 4);
        let actual = board.positions.get(&INITIAL_ROBOT).unwrap();
        assert_eq!(actual, &Coordinate { row: 6, col: 10 })
    }

    #[test]
    fn iterate_5() {
        let board = run_robot_paths(&create_board_state(EXAMPLE_EXPECTED.clone()), 5);
        let actual = board.positions.get(&INITIAL_ROBOT).unwrap();
        assert_eq!(actual, &Coordinate { row: 3, col: 1 })
    }

    ///////////////////////////////////////////////

    #[test]
    fn splitting_quadrants() {
        let board = create_board_state(EXAMPLE_EXPECTED.clone());
        /*
        1.12.......
        ...........
        ...........
        ......11.11
        1.1........
        .........1.
        .......1...
         */

        let quads = quadrant_split(&board);
        /*
        ==> AS QUADRANTS <==
        --------------------
        1.12. . .....
        ..... . .....
        ..... . .....

        ..... . 11.11

        1.1.. . .....
        ..... . ...1.
        ..... . .1...
        --------------------


        top-left
        --------
        1.12.
        .....
        .....
        --------
        => 4

        top-right
        ---------
        .....
        .....
        .....
        ---------
        => 0

        bottom-left
        -----------
        1.1..
        .....
        .....
        -----------
        => 2


        bottom-right
        ------------
        .....
        ...1.
        .1...
        ------------
        => 2
         */

        assert_eq!(
            quads.top_left.len(),
            4,
            "top left: {}",
            Coords(
                &quads
                    .top_left
                    .iter()
                    .map(|r| r.p.clone())
                    .collect::<Vec<_>>()
            )
        );
        assert_eq!(
            quads.top_right.len(),
            0,
            "top right: {}",
            Coords(
                &quads
                    .top_right
                    .iter()
                    .map(|r| r.p.clone())
                    .collect::<Vec<_>>()
            )
        );
        assert_eq!(
            quads.bottom_left.len(),
            2,
            "bottom left: {}",
            Coords(
                &quads
                    .bottom_left
                    .iter()
                    .map(|r| r.p.clone())
                    .collect::<Vec<_>>()
            )
        );
        assert_eq!(
            quads.bottom_right.len(),
            2,
            "bottom right: {}",
            Coords(
                &quads
                    .bottom_right
                    .iter()
                    .map(|r| r.p.clone())
                    .collect::<Vec<_>>()
            )
        );
    }

    #[test]
    fn iterate_100() {
        let expected = {
            let mut e = vec![
                Coordinate { row: 0, col: 6 },
                Coordinate { row: 0, col: 6 },
                Coordinate { row: 0, col: 9 },
                Coordinate { row: 2, col: 0 },
                Coordinate { row: 3, col: 1 },
                Coordinate { row: 3, col: 2 },
                Coordinate { row: 4, col: 5 },
                Coordinate { row: 5, col: 3 },
                Coordinate { row: 5, col: 4 },
                Coordinate { row: 5, col: 4 },
                Coordinate { row: 6, col: 1 },
                Coordinate { row: 6, col: 6 },
            ];
            e.sort();
            e.iter().map(|c| (c.row, c.col)).collect::<Vec<_>>()
        };
        /*
        ......2..1.
        ...........
        1..........
        .11........
        .....1.....
        ...12......
        .1....1....
         */

        let final_board = run_robot_paths(&create_board_state(EXAMPLE_EXPECTED.clone()), 100);
        let actual = {
            let mut a = final_board
                .positions
                .values()
                .map(|p| p.clone())
                .collect::<Vec<_>>();
            a.sort();
            a.into_iter().map(|c| (c.row, c.col)).collect::<Vec<_>>()
        };
        assert_eq!(actual, expected);
    }

    #[test]
    fn safety_factor_initial() {
        let actual = safety_factor(&create_board_state(EXAMPLE_EXPECTED.clone()));
        assert_eq!(actual, 0);
    }

    #[test]
    fn safety_factor_100() {
        let actual = safety_factor(&run_robot_paths(
            &create_board_state(EXAMPLE_EXPECTED.clone()),
            100,
        ));
        assert_eq!(actual, 12);
    }

    #[test]
    fn pt1_soln_example() {
        assert_eq!(solution_pt1(), 228457125);
    }

    ///////////////////////////////////////////////

    #[test]
    fn pt2_soln_example() {
        assert_eq!(solution_pt2(), 6493);
    }
}
