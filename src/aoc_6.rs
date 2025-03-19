use std::{cmp::Ordering, collections::HashSet, mem};

use crate::{
    io_help,
    matrix::{self, Coordinate, Matrix},
};

type PatrolMap = Matrix<State>;

#[derive(Clone, Debug, PartialEq)]
enum State {
    Guard(Direction),
    Visited(Option<Direction>),
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

// type ExtendedPatrolMap = utils::Matrix<ExtendedState>;

// #[derive(Clone, Debug, PartialEq)]
// enum ExtendedState {
//     Guard(Direction),
//     Visited(Direction),
//     Unvisited,
//     Obstruction,
// }

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
                'X' => {
                    panic!("WARNING!!!! should *never* be in the inputs! ! ONLY an intermediate state!");
                    // State::Visited(None)
                },
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
    match matrix::is_rectangular(rows, cols, patrol_map) {
        Some(invalid_parts) => assert!(
            false,
            "Expected rectangular patrol map ({rows} x {cols}). Dimension mismatch found {} times:\n{:?}",
            invalid_parts.len(),
            invalid_parts
        ),
        None => (), // expected outcome: no invalid parts!
    }
}

pub fn solution_pt1() -> Result<u64, String> {
    let lines = io_help::read_lines("./inputs/6").collect::<Vec<String>>();
    let patrol_map = create_patrol_map(&lines);
    Ok(trace_guards_and_count_visisted(&patrol_map))
}

fn trace_guards_and_count_visisted(patrol_map: &PatrolMap) -> u64 {
    assert_patrol_map_correctness(&patrol_map);
    let final_patrol_map = trace_guard_route(&patrol_map).unwrap();
    count_visted(&final_patrol_map)
}

fn trace_guard_route(patrol_map: &PatrolMap) -> Option<PatrolMap> {
    let mut final_patrol = patrol_map.clone();
    // while _exists_at_least_one_guard(&final_patrol) {
    //     step_guard_positions(&mut final_patrol);
    // }
    let mut cycle_visisted = vec![vec![0; patrol_map[0].len()]; patrol_map.len()];
    let mut can_take_a_step = StepResult::StepsRemain; // true
    while can_take_a_step == StepResult::StepsRemain {
        can_take_a_step = step_guard_positions(&mut final_patrol, &mut cycle_visisted);
    }
    if can_take_a_step == StepResult::CycleDetected {
        None
    } else {
        Some(final_patrol)
    }
}

fn _exists_at_least_one_guard(patrol_map: &PatrolMap) -> bool {
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

#[derive(Clone, Debug, PartialEq)]
enum StepResult {
    NoMoreSteps,
    StepsRemain,
    CycleDetected,
}

// Iterate guard locations by one step. Return satus indicates whether or not steps have finished.
//
// Returns true if there is at least one guard on the resulting board. False means all guards fell off the
// board as a result of the performed step.
fn step_guard_positions(
    patrol_map: &mut PatrolMap,
    cycle_visisted: &mut Matrix<usize>,
) -> StepResult {
    let guards = guard_coordinates(&patrol_map);
    if guards.len() == 0 {
        return StepResult::NoMoreSteps;
    }

    let mut new_guard_positions: HashSet<Coordinate> = HashSet::new();

    // guards.iter().for_each(|guard_coord|  {
    for guard_coord in guards.iter() {
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

                if cycle_visisted[guard_coord.row][guard_coord.col] > 2 {
                    // println!(
                    //     "ALREADY VISISTED {} times: {},{}",
                    //     cycle_visisted[guard_coord.row][guard_coord.col],
                    //     guard_coord.row,
                    //     guard_coord.col
                    // );
                    return StepResult::CycleDetected;
                }

                let new_direction_for_visisted: Direction = match guard_step(
                    patrol_map,
                    &guard_coord,
                    direction,
                ) {
                    Some((new_guard_position, new_direction)) => {
                        patrol_map[new_guard_position.row][new_guard_position.col] =
                            State::Guard(new_direction.clone());
                        // NOTE: We don't handle the situation where 2 or more guards attempt to move to the same position.
                        //       They all cannot occupy the same space!
                        if new_guard_positions.contains(&new_guard_position) {
                            panic!(
                                "Unexpected! At least two guards moved to the same location! This is unsuported! New location that is a violation: {new_guard_position:?}"
                            );
                        }
                        new_guard_positions.insert(new_guard_position);
                        new_direction
                    }
                    None => {
                        // guard fell off the patrol map!
                        direction.clone()
                    }
                };

                // Always update the guard's previous position to be visisted!
                patrol_map[guard_coord.row][guard_coord.col] =
                    State::Visited(Some(new_direction_for_visisted));
                cycle_visisted[guard_coord.row][guard_coord.col] += 1;
            }
            _ => panic!("Expecting guard at ({guard_coord:?}) but found {g:?}"),
        }
    }
    // });

    // all guards fell off the map!  => new_guard_positions is empty      => cannot take any more steps (false)
    // at least one guard on the map => new_guard_positions is non-empty  => can take another step      (true)
    // !new_guard_positions.is_empty()
    if new_guard_positions.is_empty() {
        StepResult::NoMoreSteps
    } else {
        StepResult::StepsRemain
    }
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
    panic!(
        "Unexpected! Guard ({direction:?} @ {guard_coord:?}) is stuck: no direction lets it take a single step!"
    )
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

fn count_visted(patrol_map: &PatrolMap) -> u64 {
    patrol_map
        .iter()
        .map(|row| {
            row.iter().fold(0 as u64, |count, state| match state {
                State::Visited(_) => count + 1,
                _ => count,
            })
        })
        .sum()
}

// AoC #6 part 2
//
// We are completing the box. We only place one (1) obstruction at a time. Therefore, we always need
// three (3) existing obstructions that are already causing the guard's patrol route to form the
// rectangle's edges.
//
// There are exactly four (4) kinds of guard(direction)-obstruction types. Since the gaurd always
// rotates 90 degrees right, it means that the the trace of all rectangles will follow the exact
// same sequence (modulo any starting/loopback point in the sequence). Without loss of generality,
// starting from the top-left of a box, these types are:
//     - guard UP, obstruction above
//     - guard RIGHT, obstriction right
//     - guard DOWN, obstruction below
//     - guard LEFT, obstriction left
//
// These guard-obstriction types defined by the guard's position and its direction. The obstruction's
// position is always determined as one (1) unit in the direction the guard is facing (this is what
// causes the guard to rotate 90 degrees right).
//
#[allow(dead_code)]
pub fn solution_pt2() -> Result<u64, String> {
    let lines = io_help::read_lines("./inputs/6").collect::<Vec<String>>();
    let initial_patrol_map = create_patrol_map(&lines);
    if false {
        Ok(solve_n_places_to_put_obstruction_to_cause_loop(
            &initial_patrol_map,
        ))
    } else {
        Err(format!("aoc #6 isn't correct yet!"))
    }
}

#[allow(dead_code)]
fn solve_n_places_to_put_obstruction_to_cause_loop(patrol_map: &PatrolMap) -> u64 {
    // let final_patrol_map = trace_guard_route(&patrol_map).unwrap();
    // let new_loop_causing_obstructions = positions_of_obstructions_to_add(&final_patrol_map);
    let new_loop_causing_obstructions = positions_of_obstructions_to_add(&patrol_map);
    new_loop_causing_obstructions.len() as u64
}

#[allow(dead_code)]
// fn positions_of_obstructions_to_add(final_patrol_map: &PatrolMap) -> Vec<Coordinate> {
fn positions_of_obstructions_to_add(patrol_map: &PatrolMap) -> Vec<Coordinate> {
    // for each VISISTED, x
    //     obs_candidate = x's direction + 1 unit
    //     match trace_march_route_from(patrol_map, obs_candidate)
    //         cycle => good!
    //         no_cycle => bad!
    match trace_guard_route(&patrol_map) {
        Some(final_patrol_map) => _all_visisted(&final_patrol_map)
            .iter()
            .flat_map(|x| {
                // let mut border_attempts = _bordering(patrol_map.len(), patrol_map[0].len(), x)
                //     .into_iter()
                //     .flat_map(|b| {
                //         cycle_causing_obstruction(patrol_map, &final_patrol_map, &b, true)
                //     })
                //     .collect::<Vec<Coordinate>>();
                // match cycle_causing_obstruction(patrol_map, &final_patrol_map, x, false) {
                //     Some(c) => border_attempts.push(c),
                //     None => (),
                // };
                // border_attempts
                cycle_causing_obstruction(patrol_map, &final_patrol_map, x, false)
            })
            .collect::<HashSet<Coordinate>>()
            .into_iter()
            .collect::<Vec<Coordinate>>(),
        None => panic!("inital patrol map has cycles!"),
    }
}

fn _bordering(max_rows: usize, max_cols: usize, x: &Coordinate) -> Vec<Coordinate> {
    //    ________________
    //    | lt | tt | rt |
    //    | ll | xx | rr |
    //    | lb | bb | br |
    //    ----------------
    let safe_coordinates = |row_mod: i32, col_mod: i32| -> Option<Coordinate> {
        let new_row: Option<usize> = if row_mod < 0 {
            if x.row == 0 {
                None
            } else {
                Some(((x.row as i32) + row_mod) as usize)
            }
        } else {
            Some(x.row + row_mod as usize)
        };

        let new_col: Option<usize> = if col_mod < 0 {
            if x.col == 0 {
                None
            } else {
                Some(((x.col as i32) + col_mod) as usize)
            }
        } else {
            Some(x.col + col_mod as usize)
        };

        match (new_row, new_col) {
            (Some(r), Some(c)) => {
                if r >= max_rows || c >= max_cols {
                    None
                } else {
                    Some(Coordinate { row: r, col: c })
                }
            }
            _ => None,
        }
    };
    let lt = safe_coordinates(-1, -1);
    let tt = safe_coordinates(-1, 0);
    let rt = safe_coordinates(-1, 1);
    let ll = safe_coordinates(0, -1);
    let rr = safe_coordinates(0, 1);
    let lb = safe_coordinates(1, -1);
    let bb = safe_coordinates(1, 0);
    let br = safe_coordinates(1, 1);
    // only return the ones that are in-bounds
    [lt, tt, rt, ll, rr, lb, bb, br]
        .into_iter()
        .flatten()
        .collect()
}

fn cycle_causing_obstruction(
    patrol_map: &PatrolMap,
    final_patrol_map: &PatrolMap,
    visisted: &Coordinate,
    use_as_obstruction: bool,
) -> Option<Coordinate> {
    let try_coord_as_obstruction = if use_as_obstruction {
        Some(visisted.clone())
    } else {
        match &final_patrol_map[visisted.row][visisted.col] {
            State::Visited(Some(d)) => next_from(patrol_map, visisted, d),
            _ => panic!(
                "expected visisted (with direction) at {visisted:?}, got: {:?}",
                final_patrol_map[visisted.row][visisted.col]
            ),
        }
    };

    match try_coord_as_obstruction {
        Some(coord) => {
            let mut obstructed_patrol_map = patrol_map.clone();
            // let before = obstructed_patrol_map[coord.row][coord.col].clone();
            obstructed_patrol_map[coord.row][coord.col] = State::Obstruction;
            // println!("\ttrying {coord:?} as obstruction (from: {visisted:?}) | before: {before:?}, after: {:?}", obstructed_patrol_map[coord.row][coord.col]);
            // if coord.row == 7 && coord.col == 7 {
            //     _print_patrol_map(patrol_map);
            // }
            match trace_guard_route(&obstructed_patrol_map) {
                Some(_) => None,
                None => {
                    // println!("\tWORKS!!!!");
                    Some(coord)
                }
            }
        }
        None => None,
    }
}

fn _print_patrol_map(patrol_map: &PatrolMap) {
    for row in patrol_map.iter() {
        for col in row.iter() {
            let x = match col {
                State::Visited(_) => 'x',
                State::Unvisited => '.',
                State::Guard(d) => match d {
                    Direction::Up => '^',
                    Direction::Right => '>',
                    Direction::Down => 'v',
                    Direction::Left => '<',
                },
                State::Obstruction => 'O',
            };
            print!("{}", x);
        }
        print!("\n");
    }
}

fn next_from(patrol_map: &PatrolMap, visisted: &Coordinate, d: &Direction) -> Option<Coordinate> {
    match d {
        Direction::Up => {
            if visisted.row == 0 {
                None
            } else {
                if some_obstruction_in(patrol_map, &visisted, Direction::Right) {
                    Some(Coordinate {
                        row: visisted.row - 1,
                        col: visisted.col,
                    })
                } else {
                    None
                }
            }
        }
        Direction::Right => {
            if visisted.col + 1 == patrol_map[0].len() {
                None
            } else {
                if some_obstruction_in(patrol_map, &visisted, Direction::Down) {
                    Some(Coordinate {
                        row: visisted.row,
                        col: visisted.col + 1,
                    })
                } else {
                    None
                }
            }
        }
        Direction::Down => {
            if visisted.row + 1 == patrol_map.len() {
                None
            } else {
                if some_obstruction_in(patrol_map, &visisted, Direction::Left) {
                    Some(Coordinate {
                        row: visisted.row + 1,
                        col: visisted.col,
                    })
                } else {
                    None
                }
            }
        }
        Direction::Left => {
            if visisted.col == 0 {
                None
            } else {
                if some_obstruction_in(patrol_map, &visisted, Direction::Up) {
                    Some(Coordinate {
                        row: visisted.row,
                        col: visisted.col - 1,
                    })
                } else {
                    None
                }
            }
        }
    }
}

fn some_obstruction_in(patrol_map: &PatrolMap, from: &Coordinate, dir: Direction) -> bool {
    match dir {
        Direction::Up => {
            let mut i = from.row;
            while i > 0 {
                i -= 1;
                match patrol_map[i][from.col] {
                    State::Obstruction => return true,
                    _ => (),
                }
            }
            false
        }
        Direction::Right => {
            let mut i = from.col + 1;
            while i < patrol_map[0].len() - 1 {
                i += 1;
                match patrol_map[from.row][i] {
                    State::Obstruction => return true,
                    _ => (),
                }
            }
            false
        }
        Direction::Down => {
            let mut i = from.row + 1;
            while i < patrol_map.len() - 1 {
                i += 1;
                match patrol_map[i][from.col] {
                    State::Obstruction => return true,
                    _ => (),
                }
            }
            false
        }
        Direction::Left => {
            let mut i = from.col - 1;
            while i > 0 {
                i -= 1;
                match patrol_map[from.row][i] {
                    State::Obstruction => return true,
                    _ => (),
                }
            }
            false
        }
    }
}

fn _positions_of_obstructions_to_add_rect(final_patrol_map: &PatrolMap) -> Vec<Coordinate> {
    // each resulting coordinate DOES NOT have an obstruction,
    // but if it did, it would cause a guard path to result in a cycle (rectangle)

    // get all boxes
    // then filter down on critera -- must have:
    //  - 3 obstructions in the right positions:
    //      - top left     => above
    //      - top right    => right
    //      - bottom-right => bottom
    //      - bottom-left  => left
    //  - a NotVisisted spot in the correct position for the remaining no-obstruction

    // make sure that the returned coordinate is where the obstruction SHOULD go
    // check if it would be out-of-bounds too!

    _rectangular_guard_paths(final_patrol_map)
        .iter()
        .flat_map(|rect_path| {
            // println!("rectangle path candidate: {rect_path:?}");
            let x = _check_suitable_obstruction_placement(final_patrol_map, rect_path);
            match x {
                Some(ref _y) => {
                    /*
                    expected:
                        (6,3)
                        (7,6)
                        (7,7)
                        (8,1)
                        (8,3)
                        (9,7)
                     */
                    // println!(
                    //     "<!> rectangle {:?} is completed by putting obstruction at: {y:?}",
                    //     rect_path
                    // );
                    ()
                }
                _ => (),
            };
            x
        })
        .collect()
}

fn _check_suitable_obstruction_placement(
    patrol_map: &PatrolMap,
    rect: &_Rectangle,
) -> Option<Coordinate> {
    // for each, take the other 3 and check:
    //  - does the taken one have UNVISISTED in the spot?
    //  - do they all have OBSTRUCTION in their spots?
    //  - will all of these locations be in-bounds?
    //
    // ==> if so, then taken one is Some(Coordinate)!
    // ==> ensure that there is EXACTLY ONE of these! Otherwise NONE !

    let check_in_bounds_and_has_obstruction_or_unvisisted =
        |maybe_in_bounds_coord: Option<Coordinate>| -> Option<Coordinate> {
            match maybe_in_bounds_coord {
                Some(coord) => match patrol_map[coord.row][coord.col] {
                    State::Unvisited | State::Obstruction | State::Visited(_) => Some(coord),
                    _ => {
                        println!(
                            "\t\tfail: location has: {:?}",
                            patrol_map[coord.row][coord.col]
                        );
                        None
                    }
                },
                None => None,
            }
        };

    if rect._bottom_left() == (Coordinate { row: 8, col: 4 })
        && rect.top_left == (Coordinate { row: 1, col: 4 })
        && rect.bottom_right == (Coordinate { row: 8, col: 6 })
    {
        println!(
            "\n\nrectangle is:  [TL] {:?}   [TR] {:?}   [BL]: {:?}   [BR]: {:?}!\n\n",
            rect.top_left,
            rect._top_right(),
            rect._bottom_left(),
            rect.bottom_right
        );
    }

    // top-left needs one?
    let obs_tl = match check_in_bounds_and_has_obstruction_or_unvisisted(
        rect._obstruction_top_left(),
    ) {
        Some(c) => c,
        None => {
            println!(
                "\tFAIL: top-left can't have obstruction / is out of bounds! top-left: {:?} -> obstruction: {:?}, matrix size: {} x {}",
                rect.top_left,
                rect._obstruction_top_left(),
                patrol_map.len(),
                patrol_map[0].len()
            );
            return None;
        }
    };

    // top-right needs one?
    let obs_tr = match check_in_bounds_and_has_obstruction_or_unvisisted(
        rect._obstruction_top_right(patrol_map),
    ) {
        Some(c) => c,
        None => {
            println!(
                "\tFAIL: top-right can't have obstruction / is out of bounds! top-right: {:?} -> obstruction: {:?}, matrix size: {} x {}",
                rect._top_right(),
                rect._obstruction_top_right(patrol_map),
                patrol_map.len(),
                patrol_map[0].len()
            );
            return None;
        }
    };

    // bottom-right needs one?
    let obs_br = match check_in_bounds_and_has_obstruction_or_unvisisted(
        rect._obstruction_bottom_right(patrol_map),
    ) {
        Some(c) => c,
        None => {
            println!(
                "\tFAIL: bottom-right can't have obstruction / is out of bounds! | bottom-right: {:?} -> obstruction: {:?}, matrix size: {} x {}",
                rect.bottom_right,
                rect._obstruction_bottom_right(patrol_map),
                patrol_map.len(),
                patrol_map[0].len()
            );
            return None;
        }
    };

    // bottom-left needs one?
    let obs_bl = match check_in_bounds_and_has_obstruction_or_unvisisted(
        rect._obstruction_bottom_left(),
    ) {
        Some(c) => c,
        None => {
            println!(
                "\tFAIL: bottom-left can't have obstruction / is out of bounds! bottom-left: {:?} -> obstruction: {:?}, matrix size: {} x {}",
                rect._bottom_left(),
                rect._obstruction_bottom_left(),
                patrol_map.len(),
                patrol_map[0].len()
            );
            return None;
        }
    };

    //
    // ** CRITICAL **
    // ** CRITICAL **
    //
    // DO NOT CHANGE THE ORDER OF possibilities WITHOUT ALSO CORRECTLY UPDATING all_attempts !!!!
    //
    // ** CRITICAL **
    // ** CRITICAL **
    //
    let possibilities = [
        &obs_tl, //
        &obs_tr, //
        &obs_br, //
        &obs_bl, //
    ];
    let all_attempts = [
        // top-left can add obstruction?
        _attempt(patrol_map, &obs_tl, &obs_tr, &obs_br, &obs_bl),
        // top-right can add obstruction?
        _attempt(patrol_map, &obs_tr, &obs_br, &obs_bl, &obs_tl),
        // bottom-right can add obstruction?
        _attempt(patrol_map, &obs_br, &obs_bl, &obs_tl, &obs_tr),
        // bottom-left can add obstruction?
        _attempt(patrol_map, &obs_bl, &obs_tl, &obs_tr, &obs_br),
    ];

    if rect.top_left == (Coordinate { row: 8, col: 4 }) {
        println!(
            "TOP LEFT IS (8,4): OBSTRUCTION ATTEMPTS:\ntop-left: {obs_tl:?}\ntop-right: {obs_tr:?}\nbottom-right: {obs_br:?}\nbottom-left: {obs_bl:?}"
        );
    }

    let coordinate_that_can_have_an_obstruction = {
        let mut index_into_possibilities = 0;
        let mut n_true = 0;
        for i in 0..all_attempts.len() {
            if all_attempts[i] {
                n_true += 1;
                index_into_possibilities = i;
            }
        }
        if n_true == 1 {
            Some(possibilities[index_into_possibilities].clone())
        } else {
            // println!("n_true: {n_true}");
            None
        }
    };
    coordinate_that_can_have_an_obstruction
}

fn _attempt(
    patrol_map: &PatrolMap,
    taken: &Coordinate,
    other_1: &Coordinate,
    other_2: &Coordinate,
    other_3: &Coordinate,
) -> bool {
    match patrol_map[taken.row][taken.col] {
        State::Unvisited | State::Visited(_) => {
            let n_others_ok: i32 = [other_1, other_2, other_3]
                .into_iter()
                .map(|x| match patrol_map[x.row][x.col] {
                    State::Obstruction => {
                        // println!("\t\t{},{} -> obstruction", x.row, x.col);
                        1
                    }
                    _ => {
                        // println!("\t\t{},{} -> {:?}", x.row, x.col, patrol_map[x.row][x.col]);
                        0
                    }
                })
                .sum();
            // if n_others_ok != 3 {
            //     println!("\tonly {n_others_ok}/3 have obstructions in the right places for {taken:?}");
            // } else {
            //     println!("********OK!******** {taken:?} works because {other_1:?}, {other_2:?}, {other_3:?} are all obstructions!\n\n\n\n\n\n")
            // }
            n_others_ok == 3
        }
        _ => {
            // println!("\t{taken:?} is not unvisited, it is {:?}", patrol_map[taken.row][taken.col]);
            false
        }
    }
}

#[derive(Hash, PartialEq, Eq, Debug)]
struct _Rectangle {
    top_left: Coordinate,
    bottom_right: Coordinate,
}

impl _Rectangle {
    fn _top_right(&self) -> Coordinate {
        Coordinate {
            row: self.top_left.row,
            col: self.bottom_right.col,
        }
    }

    fn _bottom_left(&self) -> Coordinate {
        Coordinate {
            row: self.bottom_right.row,
            col: self.top_left.col,
        }
    }

    fn _obstruction_top_left(&self) -> Option<Coordinate> {
        let mut v = self.top_left.clone();
        if v.row > 0 {
            v.row -= 1;
            Some(v)
        } else {
            None
        }
    }

    fn _obstruction_top_right(&self, patrol_map: &PatrolMap) -> Option<Coordinate> {
        let mut v = self._top_right();
        v.col += 1;
        if v.col < patrol_map[0].len() {
            Some(v)
        } else {
            None
        }
    }

    fn _obstruction_bottom_left(&self) -> Option<Coordinate> {
        let mut v: Coordinate = self._bottom_left();
        if v.col > 0 {
            v.col -= 1;
            Some(v)
        } else {
            None
        }
    }

    fn _obstruction_bottom_right(&self, patrol_map: &PatrolMap) -> Option<Coordinate> {
        let mut v = self.bottom_right.clone();
        v.row += 1;
        if v.row < patrol_map.len() {
            Some(v)
        } else {
            None
        }
    }
}

fn _all_visisted(patrol_map: &PatrolMap) -> Vec<Coordinate> {
    let mut visisted = filter_patrol_map_by_state(patrol_map, |st| match st {
        State::Visited(_) => true,
        _ => false,
    });
    visisted.sort_by(|coord_a, coord_b| {
        // origin (0,0) is **TOP-lEFT**
        //
        // we want our visited iteration order to go:
        //  (1) things to the LEFT first
        //  (2) things ABOVE others second
        if coord_a.col == coord_b.col {
            // same column
            if coord_a.row == coord_b.row {
                // same coordinate!
                Ordering::Equal
            } else if coord_a.row < coord_b.row {
                // A is above B
                Ordering::Less
            } else {
                // A is below B
                Ordering::Greater
            }
        } else if coord_a.col < coord_b.col {
            // A is left of B
            Ordering::Less
        } else {
            // A is right of B
            Ordering::Greater
        }
    });
    visisted
}

fn _rectangular_guard_paths(final_patrol_map: &PatrolMap) -> Vec<_Rectangle> {
    let visisted = _all_visisted(final_patrol_map);

    let all_cands = visisted
        .iter()
        .flat_map(|top_left| {
            let top_right_cands = _scan_right(final_patrol_map, top_left);
            let bottom_left_cands = _scan_down(final_patrol_map, top_left);
            let completed_boxes = _boxes_from(
                final_patrol_map,
                top_left,
                &top_right_cands,
                &bottom_left_cands,
            );
            // completed_boxes.iter().for_each(|x| {
            //     println!("CANDIDATE: {x:?}");
            // });
            completed_boxes
        })
        .collect::<HashSet<_Rectangle>>();

    let hashset_len = all_cands.len();
    let final_boxes: Vec<_> = all_cands.into_iter().collect();
    if hashset_len != final_boxes.len() {
        println!(
            "WANRING: generating {} duplicates!",
            hashset_len - final_boxes.len()
        );
    }
    final_boxes
}

fn filter_patrol_map_by_state(
    patrol_map: &PatrolMap,
    matching_state: fn(&State) -> bool,
) -> Vec<Coordinate> {
    (0..patrol_map.len())
        .flat_map(|row| {
            (0..patrol_map[0].len()).flat_map(move |col| {
                if matching_state(&patrol_map[row][col]) {
                    Some(Coordinate { row, col })
                } else {
                    None
                }
            })
        })
        .collect()
}

// fn scan_right(patrol_map: &PatrolMap, v: &Coordinate) -> Vec<Coordinate> {
//     patrol_map[v.row]
//         .iter()
//         .enumerate()
//         .flat_map(|(i, st)| match st {
//             State::Visited => Some(Coordinate { row: v.row, col: i }),
//             _ => None,
//         })
//         .collect()
// }
fn _scan_right(patrol_map: &PatrolMap, v: &Coordinate) -> Vec<Coordinate> {
    let mut idx_col = v.col + 1;
    let mut coords = Vec::new();
    while idx_col < patrol_map[0].len() && _is_visisted(&patrol_map[v.row][idx_col]) {
        coords.push(Coordinate {
            row: v.row,
            col: idx_col,
        });
        idx_col += 1;
    }
    coords
}

fn _is_visisted(x: &State) -> bool {
    mem::discriminant(x) == mem::discriminant(&State::Visited(None))
}

// fn scan_down(patrol_map: &PatrolMap, v: &Coordinate) -> Vec<Coordinate> {
//     (0..patrol_map.len())
//         .flat_map(|row| match patrol_map[row][v.col] {
//             State::Visited => Some(Coordinate {
//                 row: row,
//                 col: v.col,
//             }),
//             _ => None,
//         })
//         .collect()
// }
fn _scan_down(patrol_map: &PatrolMap, v: &Coordinate) -> Vec<Coordinate> {
    let mut idx_row = v.row + 1;
    let mut coords = Vec::new();
    while idx_row < patrol_map.len() && _is_visisted(&patrol_map[idx_row][v.col]) {
        coords.push(Coordinate {
            row: idx_row,
            col: v.col,
        });
        idx_row += 1;
    }
    coords
}

// fn boxes_from(
//     patrol_map: &PatrolMap,
//     top_left: &Coordinate,
//     tr_cands: &[Coordinate],
//     bl_cands: &[Coordinate],
// ) -> Vec<Rectangle> {
//     // complete the box! => find a bottom-right candidate!
//     tr_cands
//         .iter()
//         .flat_map(|top_right| {
//             bl_cands.iter().flat_map(|bottom_left| {
//                 let (row, col) = (bottom_left.row, top_right.col);
//                 match patrol_map[row][col] {
//                     State::Visited => Some(Rectangle {
//                         top_left: top_left.clone(),
//                         bottom_right: Coordinate { row, col },
//                     }),
//                     _ => None,
//                 }
//             })
//         })
//         .collect()
// }
fn _boxes_from(
    patrol_map: &PatrolMap,
    top_left: &Coordinate,
    tr_cands: &[Coordinate],
    bl_cands: &[Coordinate],
) -> Vec<_Rectangle> {
    // complete the box! => find a bottom-right candidate!
    tr_cands
        .iter()
        .flat_map(|top_right| {
            bl_cands.iter().flat_map(|bottom_left| {
                let (row, col) = (bottom_left.row, top_right.col);

                // are ALL of the positions from here UP visisted?
                // are ALL of the positions from here LEFT visisted?
                // yes => true , no => false

                match patrol_map[row][col] {
                    State::Visited(_) => Some(_Rectangle {
                        top_left: top_left.clone(),
                        bottom_right: Coordinate { row, col },
                    }),
                    _ => None,
                }
            })
        })
        .collect()
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

    #[test]
    fn pt2_soln_coords() {
        let lines = read_lines_in_memory(EXAMPLE_INPUT_STR).collect::<Vec<_>>();
        let coords = positions_of_obstructions_to_add(&create_patrol_map(&lines));
        let actual: HashSet<Coordinate> = HashSet::from_iter(coords);
        let expected: HashSet<Coordinate> = HashSet::from_iter(vec![
            Coordinate { row: 6, col: 3 },
            Coordinate { row: 7, col: 6 },
            Coordinate { row: 7, col: 7 },
            Coordinate { row: 8, col: 1 },
            Coordinate { row: 8, col: 3 },
            Coordinate { row: 9, col: 7 },
        ]);
        assert_eq!(actual, expected);
    }

    #[ignore]
    #[test]
    fn pt2_soln() {
        let lines = read_lines_in_memory(EXAMPLE_INPUT_STR).collect::<Vec<_>>();
        let actual = solve_n_places_to_put_obstruction_to_cause_loop(&create_patrol_map(&lines));
        assert_eq!(actual, 6, "actual != expected");
    }

    #[test]
    fn break_test() {
        let lines = read_lines_in_memory(EXAMPLE_INPUT_STR).collect::<Vec<_>>();
        let mut patrol_map = create_patrol_map(&lines);
        patrol_map[7][7] = State::Obstruction;
        let traced = trace_guard_route(&patrol_map);
        assert_eq!(traced, None);
    }
}
