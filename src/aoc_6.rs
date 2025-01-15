use std::collections::HashSet;

use crate::{
    io_help,
    utils::{self, Coordinate},
};

type PatrolMap = utils::Matrix<State>;

#[derive(Clone, Debug, PartialEq)]
enum State {
    Guard(Direction),
    Visited,
    Unvisited,
    Obstruction,
}

#[derive(Clone, Debug, PartialEq)]
enum Direction {
    Up,
    Down,
    Left,
    Right,
}

fn rotate_clockwise(d: &Direction) -> Direction {
    match d {
        Direction::Up => Direction::Right,
        Direction::Right => Direction::Down,
        Direction::Down => Direction::Left,
        Direction::Left => Direction::Up,
    }
}

fn create_patrol_map(lines: &[String]) -> PatrolMap {
    lines.iter().map(|line| {
        line.chars().map(|position| {
            match position {
                '.' => State::Unvisited,
                '#' => State::Obstruction,
                'X' => State::Visited, // should *never* be in the inputs/6 ! ONLY an intermediate state!
                maybe_guard_direction => State::Guard(
                    match maybe_guard_direction {
                        '^' => Direction::Up,
                        '>' => Direction::Right,
                        '<' => Direction::Left,
                        'v' => Direction::Down,
                        unknown => panic!("Unexpected character at position: '{unknown}' --> cannot build Patrol Map !! line: {line}"),
                    }
                )
            }
        }).collect::<Vec<_>>()
    }).collect::<Vec<_>>()
}

// assert: the patrol map is non-empty and rectangular
fn assert_patrol_map_correctness(patrol_map: &PatrolMap) {
    let rows = patrol_map.len();
    assert_ne!(
        rows, 0,
        "An empty patrol map is not allowed! Found no rows!"
    );
    let cols = patrol_map[0].len();
    assert_ne!(
        cols, 0,
        "An empty patrol map is not allowed! Found no columns!"
    );
    match utils::is_rectangular(rows, cols, patrol_map) {
        Some(invalid_parts) => assert!(false, "Expected rectangular patrol map ({rows} x {cols}). Dimension mismatch found {} times:\n{:?}", invalid_parts.len(), invalid_parts),
        None => (), // expected outcome: no invalid parts!
    }
}

pub fn solution_pt1() -> i32 {
    let lines = io_help::read_lines("./inputs/6").collect::<Vec<String>>();
    let patrol_map = create_patrol_map(&lines);
    trace_guards_and_count_visisted(&patrol_map) as i32
}

fn trace_guards_and_count_visisted(patrol_map: &PatrolMap) -> usize {
    assert_patrol_map_correctness(&patrol_map);
    let final_patrol_map = trace_guard_route(&patrol_map);
    count_visted(&final_patrol_map)
}

fn trace_guard_route(patrol_map: &PatrolMap) -> PatrolMap {
    let mut final_patrol = patrol_map.clone();
    // while exists_at_least_one_guard(&final_patrol) {
    //     step_guard_positions(&mut final_patrol);
    // }
    let mut can_take_a_step = true;
    while can_take_a_step {
        can_take_a_step = step_guard_positions(&mut final_patrol);
    }
    final_patrol
}

fn exists_at_least_one_guard(patrol_map: &PatrolMap) -> bool {
    for row in patrol_map {
        for position in row {
            match position {
                State::Guard(_) => {
                    return true;
                }
                _ => (),
            }
        }
    }
    false
}

fn guard_coordinates(patrol_map: &PatrolMap) -> Vec<Coordinate> {
    patrol_map
        .iter()
        .enumerate()
        .flat_map(|(idx_row, row)| {
            row.iter()
                .enumerate()
                .flat_map(move |(idx_col, position)| match position {
                    State::Guard(_) => Some(Coordinate {
                        row: idx_row,
                        col: idx_col,
                    }),
                    _ => None,
                })
        })
        .collect()
}

// Iterate guard locations by one step. Return satus indicates whether or not steps have finished.
//
// Returns true if there is at least one guard on the resulting board. False means all guards fell off the
// board as a result of the performed step.
fn step_guard_positions(patrol_map: &mut PatrolMap) -> bool {
    let guards = guard_coordinates(&patrol_map);
    if guards.len() == 0 {
        return false;
    }

    let mut new_guard_positions: HashSet<Coordinate> = HashSet::new();

    guards.iter().for_each(|guard_coord|  {
        let g = &patrol_map[guard_coord.row][guard_coord.col];
        match g {
            State::Guard(direction) => {
                // NOTE: Does not consider case where a guard starts being obstructed in all 4 directions.
                //       We'd technically have to keep the guard around and note a cycle in the above direction-
                //       seeking logic as a new part of the terminal case (where all guards "fall off" the patrol map).
                //
                //       Instead, we just assume that this never happens :)
                //       If it does, then `guard_step(..)`` code will panic!().
                // match guard_step(patrol_map, &new_guard_positions, &guard_coord, direction) {
                match guard_step(patrol_map, &guard_coord, direction) {
                    Some((new_guard_position, new_direction)) => {
                        patrol_map[new_guard_position.row][new_guard_position.col] = State::Guard(new_direction);
                        // NOTE: We don't handle the situation where 2 or more guards attempt to move to the same position.
                        //       They all cannot occupy the same space!
                        if new_guard_positions.contains(&new_guard_position) {
                            panic!("Unexpected! At least two guards moved to the same location! This is unsuported! New location that is a violation: {new_guard_position:?}");
                        }
                        new_guard_positions.insert(new_guard_position);
                    }
                    None => (), // guard fell off the patrol map!
                }

                // Always update the guard's previous position to be visisted!
                patrol_map[guard_coord.row][guard_coord.col] = State::Visited
            }
            _ => panic!("Expecting guard at ({guard_coord:?}) but found {g:?}"),
        }
    });

    // all guards fell off the map!  => new_guard_positions is empty      => cannot take any more steps (false)
    // at least one guard on the map => new_guard_positions is non-empty  => can take another step      (true)
    !new_guard_positions.is_empty()
}

fn guard_step(
    patrol_map: &PatrolMap,
    // updated_guard_positions: &HashSet<Coordinate>,
    guard_coord: &Coordinate,
    direction: &Direction,
) -> Option<(Coordinate, Direction)> {
    let max_n_rows = patrol_map.len();
    let max_n_cols = patrol_map[0].len();

    let mut direction_attempt = direction.clone();
    let mut attempts = 0;

    // the <= is necessary because we treat attempts as a count, not an index
    while attempts <= 4 {
        attempts += 1;
        match new_coordinate(max_n_rows, max_n_cols, guard_coord, &direction_attempt) {
            Some(new_pos) => {
                // if is_obstructed(&patrol_map, updated_guard_positions, &new_pos) {
                if is_obstructed(&patrol_map, &new_pos) {
                    // uh-oh! let's try another direction!
                    direction_attempt = rotate_clockwise(&direction_attempt);
                } else {
                    // ok! we found a direction the guard can take a step in
                    return Some((new_pos, direction_attempt));
                }
            }
            None => {
                // guard fell off the map!
                // this is ok! we now don't have a position for the guard so we return None
                return None;
            }
        }
    }
    panic!("Unexpected! Guard ({direction:?} @ {guard_coord:?}) is stuck: no direction lets it take a single step!")
}

fn new_coordinate(
    max_n_rows: usize,
    max_n_cols: usize,
    xy: &Coordinate,
    d: &Direction,
) -> Option<Coordinate> {
    // origin (0,0) is TOP-LEFT CORNER: this is like AN IMAGE
    //      x+1 means GO DOWN
    //      x-1 means GO UP
    //  y+1 and y-1 are the expected go right and left, respectively
    //
    // ==> Some(x,y): new valid coordinate in one step of direction
    // ==> None:      step is off of board
    assert_ne!(max_n_rows, 0);
    assert_ne!(max_n_cols, 0);
    match d {
        Direction::Up => {
            if xy.row == 0 {
                None
            } else {
                Some(Coordinate {
                    row: xy.row - 1,
                    col: xy.col,
                })
            }
        }
        Direction::Down => {
            if xy.row + 1 >= max_n_rows {
                None
            } else {
                Some(Coordinate {
                    row: xy.row + 1,
                    col: xy.col,
                })
            }
        }
        Direction::Left => {
            if xy.col == 0 {
                None
            } else {
                Some(Coordinate {
                    row: xy.row,
                    col: xy.col - 1,
                })
            }
        }
        Direction::Right => {
            if xy.col + 1 >= max_n_cols {
                None
            } else {
                Some(Coordinate {
                    row: xy.row,
                    col: xy.col + 1,
                })
            }
        }
    }
}

// fn is_obstructed(patrol_map: &PatrolMap, updated_guard_positions: &HashSet<Coordinate>, xy: &Coordinate) -> bool {
fn is_obstructed(patrol_map: &PatrolMap, xy: &Coordinate) -> bool {
    match &patrol_map[xy.row][xy.col] {
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        // // obstructions and guards are treated as an obstruction!
        // State::Obstruction | State::Guard(_) => true,
        // // visisted or unvisisted spots are not obstructions!
        // State::Visited | State::Unvisited => false,
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        State::Obstruction => true,
        // INVALID: under the assumption that any Guard will move,
        // INVALID: we lump this + Visited + Unvisted as all not obstrcuted positions
        //
        // WHY?
        //          This is invalid because we are viewing a *MUTATING* PatrolMap!
        //          Guards can be either new or old! We don't differentiate.
        // we lump this + Visited + Unvisted as all not obstrcuted positions
        _ => false,
        // -- OLD --
    }
}

fn count_visted(patrol_map: &PatrolMap) -> usize {
    patrol_map
        .iter()
        .map(|row| {
            row.iter().fold(0 as usize, |count, state| match state {
                State::Visited => count + 1,
                _ => count,
            })
        })
        .sum()
}

pub fn solution_pt2() -> i32 {
    let lines = io_help::read_lines("./inputs/6").collect::<Vec<String>>();
    todo!()
}

#[cfg(test)]
mod test {
    use indoc::indoc;
    use io_help::read_lines_in_memory;
    use lazy_static::lazy_static;

    use super::*;

    const EXAMPLE_INPUT_STR: &str = indoc! {"
        ....#.....
        .........#
        ..........
        ..#.......
        .......#..
        ..........
        .#..^.....
        ........#.
        #.........
        ......#...
    "};

    lazy_static! {
        static ref EXAMPLE_PATROL_MAP: PatrolMap = {
            vec![
                vec![
                    State::Unvisited,
                    State::Unvisited,
                    State::Unvisited,
                    State::Unvisited,
                    State::Obstruction,
                    State::Unvisited,
                    State::Unvisited,
                    State::Unvisited,
                    State::Unvisited,
                    State::Unvisited,
                ],
                vec![
                    State::Unvisited,
                    State::Unvisited,
                    State::Unvisited,
                    State::Unvisited,
                    State::Unvisited,
                    State::Unvisited,
                    State::Unvisited,
                    State::Unvisited,
                    State::Unvisited,
                    State::Obstruction,
                ],
                vec![
                    State::Unvisited,
                    State::Unvisited,
                    State::Unvisited,
                    State::Unvisited,
                    State::Unvisited,
                    State::Unvisited,
                    State::Unvisited,
                    State::Unvisited,
                    State::Unvisited,
                    State::Unvisited,
                ],
                vec![
                    State::Unvisited,
                    State::Unvisited,
                    State::Obstruction,
                    State::Unvisited,
                    State::Unvisited,
                    State::Unvisited,
                    State::Unvisited,
                    State::Unvisited,
                    State::Unvisited,
                    State::Unvisited,
                ],
                vec![
                    State::Unvisited,
                    State::Unvisited,
                    State::Unvisited,
                    State::Unvisited,
                    State::Unvisited,
                    State::Unvisited,
                    State::Unvisited,
                    State::Obstruction,
                    State::Unvisited,
                    State::Unvisited,
                ],
                vec![
                    State::Unvisited,
                    State::Unvisited,
                    State::Unvisited,
                    State::Unvisited,
                    State::Unvisited,
                    State::Unvisited,
                    State::Unvisited,
                    State::Unvisited,
                    State::Unvisited,
                    State::Unvisited,
                ],
                vec![
                    State::Unvisited,
                    State::Obstruction,
                    State::Unvisited,
                    State::Unvisited,
                    State::Guard(Direction::Up),
                    State::Unvisited,
                    State::Unvisited,
                    State::Unvisited,
                    State::Unvisited,
                    State::Unvisited,
                ],
                vec![
                    State::Unvisited,
                    State::Unvisited,
                    State::Unvisited,
                    State::Unvisited,
                    State::Unvisited,
                    State::Unvisited,
                    State::Unvisited,
                    State::Unvisited,
                    State::Obstruction,
                    State::Unvisited,
                ],
                vec![
                    State::Obstruction,
                    State::Unvisited,
                    State::Unvisited,
                    State::Unvisited,
                    State::Unvisited,
                    State::Unvisited,
                    State::Unvisited,
                    State::Unvisited,
                    State::Unvisited,
                    State::Unvisited,
                ],
                vec![
                    State::Unvisited,
                    State::Unvisited,
                    State::Unvisited,
                    State::Unvisited,
                    State::Unvisited,
                    State::Unvisited,
                    State::Obstruction,
                    State::Unvisited,
                    State::Unvisited,
                    State::Unvisited,
                ],
            ]
        };
    }

    #[test]
    fn construct_patrol_map_from_input() {
        let lines = read_lines_in_memory(EXAMPLE_INPUT_STR).collect::<Vec<_>>();
        let patrol_map = create_patrol_map(&lines);
        assert_patrol_map_correctness(&patrol_map);
        let expected: &PatrolMap = &EXAMPLE_PATROL_MAP;
        assert_eq!(patrol_map, *expected, "actual != expected");
    }

    #[test]
    fn pt1_soln_example() {
        let n_visisted = trace_guards_and_count_visisted(
            &(create_patrol_map(&(read_lines_in_memory(EXAMPLE_INPUT_STR).collect::<Vec<_>>()))),
        );
        assert_eq!(n_visisted, 41, "actual != expected");
    }
}
