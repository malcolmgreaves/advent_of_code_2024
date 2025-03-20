use crate::{
    // fun_list::{GrowableList, List, ListOps},
    fun_list::Appendable,
    io_help,
    matrix::{Coordinate, GridMovement, Matrix},
    utils::collect_results,
};

///////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Debug, PartialEq, Eq)]
struct Warehouse {
    floor: Floor,
    moves: Moves,
    robot: Coordinate,
}

type Floor = Matrix<Space>;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum Space {
    Wall,  // #
    Empty, // .
    Box,   // O
    Robot, // @
}

type Moves = Vec<Move>;

#[derive(Debug, PartialEq, Eq, Clone)]
enum Move {
    Up,    // ^
    Down,  // v
    Left,  // <
    Right, // >
}

fn construct(lines: impl Iterator<Item = String>) -> Result<Warehouse, String> {
    let (f, m, _) = lines.fold(
        // (List::<String>::Empty, List::<String>::Empty, false),
        (Vec::new(), Vec::new(), false),
        |(floor, moves, switch), line| {
            let l = line.trim();
            if l.len() == 0 {
                (floor, moves, true)
            } else {
                if !switch {
                    (floor.append(line), moves, switch)
                } else {
                    (floor, moves.append(line), switch)
                }
            }
        },
    );

    let floor = parse_floor(f.to_vec().into_iter())?;
    let moves = {
        let move_list = m.to_vec();
        let moves_str = if move_list.len() != 1 {
            &move_list.join("")
        } else {
            &move_list[0]
        };
        parse_moves(moves_str)?
    };
    let mut robot_positions = floor
        .iter()
        .enumerate()
        .flat_map(|(i, row)| {
            row.iter()
                .enumerate()
                .flat_map(move |(j, space)| match space {
                    Space::Robot => Some(Coordinate { row: i, col: j }),
                    _ => None,
                })
        })
        .collect::<Vec<_>>();
    if robot_positions.len() != 1 {
        Result::Err(format!(
            "There must be exactly 1 robot, but found {}.",
            robot_positions.len()
        ))
    } else {
        let robot = robot_positions.swap_remove(0);
        Result::Ok(Warehouse {
            floor,
            moves,
            robot,
        })
    }
}

fn parse_floor(lines: impl Iterator<Item = String>) -> Result<Floor, String> {
    let (floor, errors) = collect_results(lines.map(|l| {
        let (spaces, errors) = collect_results(l.chars().map(|c| match c {
            '#' => Result::Ok(Space::Wall),
            '.' => Result::Ok(Space::Empty),
            '@' => Result::Ok(Space::Robot),
            'O' => Result::Ok(Space::Box),
            _ => Result::Err(format!("unexpected character: '{c}'")),
        }));
        if errors.len() > 0 {
            Result::Err(format!(
                "failed to parse {} spaces: {errors:?}",
                errors.len()
            ))
        } else {
            Result::Ok(spaces)
        }
    }));
    if errors.len() > 0 {
        let e_str = errors.join("\n\t");
        Result::Err(format!(
            "[parse floor] failed on {} lines of input:\n\t{}",
            errors.len(),
            e_str
        ))
    } else {
        Result::Ok(floor)
    }
}

fn parse_moves(line: &str) -> Result<Moves, String> {
    let (moves, errors) = collect_results(line.chars().map(|c| match c {
        '>' => Result::Ok(Move::Right),
        '<' => Result::Ok(Move::Left),
        '^' => Result::Ok(Move::Up),
        'v' => Result::Ok(Move::Down),
        _ => Result::Err(format!("unrecognized move: '{c}'")),
    }));
    if errors.len() > 0 {
        let e_str = errors.join("\n\t");
        Result::Err(format!(
            "[parse moves] failed on {} lines of input:\n\t{}",
            errors.len(),
            e_str
        ))
    } else {
        Result::Ok(moves)
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

pub fn solution_pt1() -> Result<u64, String> {
    let lines = io_help::read_lines("./inputs/15");
    let warehouse = construct(lines)?;
    Ok(solve_1(&warehouse))
}

fn solve_1(warehouse: &Warehouse) -> u64 {
    let final_floor: Floor = move_to_completion(&warehouse);
    let g = gps_sum_boxes(&final_floor);
    g
}

fn move_to_completion(w: &Warehouse) -> Floor {
    let g = GridMovement::new(&w.floor);
    let mut robot_position = w.robot.clone();
    let mut final_floor = w.floor.clone();
    w.moves.iter().for_each(|m| {
        robot_position = step(&g, &mut final_floor, &robot_position, m);
    });
    final_floor
}

fn step(g: &GridMovement, f: &mut Floor, robot: &Coordinate, m: &Move) -> Coordinate {
    assert_eq!(f[robot.row][robot.col], Space::Robot);
    let (new_row, new_col) = match m {
        Move::Up => (g.sub(robot.row), Some(robot.col)),
        Move::Down => (g.add_row(robot.row), Some(robot.col)),
        Move::Left => (Some(robot.row), g.sub(robot.col)),
        Move::Right => (Some(robot.row), g.add_col(robot.col)),
    };
    match (new_row, new_col) {
        (Some(r), Some(c)) => {
            let movement_occurred = match f[r][c] {
                Space::Empty => true,
                Space::Box => try_push_boxes(g, f, robot, m),
                Space::Wall => false,
                Space::Robot => panic!("impossible! only one robot allowed!"),
            };
            if movement_occurred {
                f[robot.row][robot.col] = Space::Empty;
                f[r][c] = Space::Robot;
                Coordinate { row: r, col: c }
            } else {
                robot.clone()
            }
        }
        _ => robot.clone(),
    }
}

fn try_push_boxes(g: &GridMovement, f: &mut Floor, robot: &Coordinate, m: &Move) -> bool {
    // in direction of *m*
    // check along that axis to see if there are any EMPTY positions
    // there must be at least one empty position after the box
    // AND no wall before this empty position
    let spaces_to_check = match m {
        Move::Up => g.positions_up(robot),
        Move::Down => g.positions_down(robot),
        Move::Left => g.positions_left(robot),
        Move::Right => g.positions_right(robot),
    };

    let mut move_until = None;
    for (i, Coordinate { row, col }) in spaces_to_check.iter().enumerate() {
        match f[*row][*col] {
            Space::Box => (),
            Space::Wall => break,
            Space::Empty => {
                move_until = Some(i);
                break;
            }
            Space::Robot => panic!("impossible! only one robot space allowed!"),
        }
    }

    match move_until {
        Some(index_to_move_until) => {
            // work backwards from this empty position *c*:
            // take the
            let rev_spaces = {
                let mut s = spaces_to_check[0..=index_to_move_until].to_vec();
                s.reverse();
                s
            };
            for window in rev_spaces.windows(2) {
                let from = &window[0];
                let to = &window[1];
                f[from.row][from.col] = f[to.row][to.col];
            }
            true
        }
        None => {
            // println!("\tno move until");
            false
        }
    }
}

fn gps_sum_boxes(floor: &Floor) -> u64 {
    floor.iter().enumerate().fold(0, |s, (row, r)| {
        r.iter().enumerate().fold(s, |rsum, (col, space)| {
            rsum + match space {
                Space::Box => gps(&Coordinate { row, col }),
                _ => 0,
            }
        })
    })
}

fn gps(c: &Coordinate) -> u64 {
    100 * c.row as u64 + c.col as u64
}

#[allow(unused)]
fn step_n(warehouse: &Warehouse, n_steps: usize) -> Floor {
    move_to_completion(&Warehouse {
        floor: warehouse.floor.clone(),
        robot: warehouse.robot.clone(),
        moves: warehouse.moves[0..n_steps].to_vec(),
    })
}

#[allow(unused)]
fn locate_robot(floor: &Floor) -> Result<Coordinate, String> {
    let mut robots = floor
        .iter()
        .enumerate()
        .flat_map(|(row, r)| {
            r.iter().enumerate().flat_map(move |(col, s)| match s {
                Space::Robot => Some(Coordinate { row, col }),
                _ => None,
            })
        })
        .collect::<Vec<_>>();
    match robots.len() {
        1 => Ok(robots.swap_remove(0)),
        n => Err(format!(
            "there must be exactly 1 robot in the floor, found: {n}"
        )),
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

pub fn solution_pt2() -> Result<u64, String> {
    let lines = io_help::read_lines("./inputs/15");
    let warehouse = construct(lines)?;
    Err(format!(
        "part2 is incomplete! warehouse's robot={:?}",
        warehouse.robot
    ))
}

///////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod test {

    use indoc::indoc;
    use lazy_static::lazy_static;

    use crate::{io_help::read_lines_in_memory, testing_utilities::check_matrices};

    use super::*;

    ///////////////////////////////////////////////

    const EXAMPLE_INPUT_STR_SM: &str = indoc! {"
        ########
        #..O.O.#
        ##@.O..#
        #...O..#
        #.#.O..#
        #...O..#
        #......#
        ########

        <^^>>>vv<v>>v<<
    "};

    lazy_static! {
        static ref EXAMPLE_EXPECTED_SM: Warehouse = Warehouse {
            floor: vec![
                vec![
                    Space::Wall,
                    Space::Wall,
                    Space::Wall,
                    Space::Wall,
                    Space::Wall,
                    Space::Wall,
                    Space::Wall,
                    Space::Wall
                ],
                vec![
                    Space::Wall,
                    Space::Empty,
                    Space::Empty,
                    Space::Box,
                    Space::Empty,
                    Space::Box,
                    Space::Empty,
                    Space::Wall
                ],
                vec![
                    Space::Wall,
                    Space::Wall,
                    Space::Robot,
                    Space::Empty,
                    Space::Box,
                    Space::Empty,
                    Space::Empty,
                    Space::Wall
                ],
                vec![
                    Space::Wall,
                    Space::Empty,
                    Space::Empty,
                    Space::Empty,
                    Space::Box,
                    Space::Empty,
                    Space::Empty,
                    Space::Wall
                ],
                vec![
                    Space::Wall,
                    Space::Empty,
                    Space::Wall,
                    Space::Empty,
                    Space::Box,
                    Space::Empty,
                    Space::Empty,
                    Space::Wall
                ],
                vec![
                    Space::Wall,
                    Space::Empty,
                    Space::Empty,
                    Space::Empty,
                    Space::Box,
                    Space::Empty,
                    Space::Empty,
                    Space::Wall
                ],
                vec![
                    Space::Wall,
                    Space::Empty,
                    Space::Empty,
                    Space::Empty,
                    Space::Empty,
                    Space::Empty,
                    Space::Empty,
                    Space::Wall
                ],
                vec![
                    Space::Wall,
                    Space::Wall,
                    Space::Wall,
                    Space::Wall,
                    Space::Wall,
                    Space::Wall,
                    Space::Wall,
                    Space::Wall
                ],
            ],
            moves: vec![
                Move::Left,
                Move::Up,
                Move::Up,
                Move::Right,
                Move::Right,
                Move::Right,
                Move::Down,
                Move::Down,
                Move::Left,
                Move::Down,
                Move::Right,
                Move::Right,
                Move::Down,
                Move::Left,
                Move::Left
            ],
            robot: Coordinate { row: 2, col: 2 },
        };
    }

    const EXAMPLE_INPUT_STR_LG: &str = indoc! { "
        ##########
        #..O..O.O#
        #......O.#
        #.OO..O.O#
        #..O@..O.#
        #O#..O...#
        #O..O..O.#
        #.OO.O.OO#
        #....O...#
        ##########

        <vv>^<v^>v>^vv^v>v<>v^v<v<^vv<<<^><<><>>v<vvv<>^v^>^<<<><<v<<<v^vv^v>^vvv<<^>^v^^><<>>><>^<<><^vv^^<>vvv<>><^^v>^>vv<>v<<<<v<^v>^<^^>>>^<v<v><>vv>v^v^<>><>>>><^^>vv>v<^^^>>v^v^<^^>v^^>v^<^v>v<>>v^v^<v>v^^<^^vv<<<v<^>>^^^^>>>v^<>vvv^><v<<<>^^^vv^<vvv>^>v<^^^^v<>^>vvvv><>>v^<<^^^^^^><^><>>><>^^<<^^v>>><^<v>^<vv>>v>>>^v><>^v><<<<v>>v<v<v>vvv>^<><<>^><^>><>^v<><^vvv<^^<><v<<<<<><^v<<<><<<^^<v<^^^><^>>^<v^><<<^>>^v<v^v<v^>^>>^v>vv>^<<^v<>><<><<v<<v><>v<^vv<<<>^^v^>^^>>><<^v>>v^v><^^>>^<>vv^<><^^>^^^<><vvvvv^v<v<<>^v<v>v<<^><<><<><<<^^<<<^<<>><<><^^^>^^<>^>v<>^^>vv<^v^v<vv>^<><v<^v>^^^>>>^^vvv^>vvv<>>>^<^>>>>>^<<^v>^vvv<>^<><<v>v^^>>><<^^<>>^v^<v^vv<>v^<<>^<^v^v><^<<<><<^<v><v<>vv>>v><v^<vv<>v^<<^
    "};

    ///////////////////////////////////////////////

    #[test]
    fn construction() {
        match construct(read_lines_in_memory(EXAMPLE_INPUT_STR_SM)) {
            Result::Ok(actual) => {
                let expected: &Warehouse = &EXAMPLE_EXPECTED_SM;
                assert_eq!(actual, *expected);
                assert_eq!(
                    actual.floor[actual.robot.row][actual.robot.col],
                    Space::Robot
                );
            }
            Result::Err(e) => assert!(false, "{}", e),
        }
    }

    fn check_step_sm(input_floor: &str, steps: usize, robot: Coordinate) {
        let expected: Floor = parse_floor(read_lines_in_memory(input_floor)).unwrap();
        let actual: Floor = step_n(&EXAMPLE_EXPECTED_SM, steps);
        assert_eq!(
            locate_robot(&actual).unwrap(),
            robot,
            "robot is in incorrect position"
        );
        check_matrices(&actual, &expected).unwrap()
    }

    #[test]
    fn step_1() {
        check_step_sm(
            indoc! {
                "
                ########
                #..O.O.#
                ##@.O..#
                #...O..#
                #.#.O..#
                #...O..#
                #......#
                ########
                "
            },
            1,
            Coordinate { row: 2, col: 2 },
        );
    }

    #[test]
    fn step_2() {
        check_step_sm(
            indoc! {
                "
                ########
                #.@O.O.#
                ##..O..#
                #...O..#
                #.#.O..#
                #...O..#
                #......#
                ########
                "
            },
            2,
            Coordinate { row: 1, col: 2 },
        );
    }

    #[test]
    fn step_3() {
        check_step_sm(
            indoc! {
                "
                ########
                #.@O.O.#
                ##..O..#
                #...O..#
                #.#.O..#
                #...O..#
                #......#
                ########
                "
            },
            3,
            Coordinate { row: 1, col: 2 },
        );
    }

    #[test]
    fn step_4() {
        check_step_sm(
            indoc! {
                "
                ########
                #..@OO.#
                ##..O..#
                #...O..#
                #.#.O..#
                #...O..#
                #......#
                ########
                "
            },
            4,
            Coordinate { row: 1, col: 3 },
        );
    }

    #[test]
    fn step_5() {
        check_step_sm(
            indoc! {
                "
                ########
                #...@OO#
                ##..O..#
                #...O..#
                #.#.O..#
                #...O..#
                #......#
                ########
                "
            },
            5,
            Coordinate { row: 1, col: 4 },
        );
    }

    #[test]
    fn gps_calc_0() {
        assert_eq!(gps(&Coordinate { row: 1, col: 4 }), 104);

        let simple = construct(read_lines_in_memory(indoc! {
            "
            #######
            #...O..
            #.....@

            <
            "
        }))
        .unwrap();

        assert_eq!(gps_sum_boxes(&simple.floor), 104);
    }

    #[test]
    fn pt1_small() {
        let small_example: &Warehouse = &EXAMPLE_EXPECTED_SM;
        let g = solve_1(small_example);
        assert_eq!(g, 2028);
    }

    #[test]
    fn pt1_lg() {
        let large_example: Warehouse =
            construct(read_lines_in_memory(&EXAMPLE_INPUT_STR_LG)).unwrap();
        let g = solve_1(&large_example);
        assert_eq!(g, 10092);
    }

    ///////////////////////////////////////////////

    #[test]
    fn pt1_soln_example() {
        assert_eq!(solution_pt1().unwrap(), 1398947);
    }

    ///////////////////////////////////////////////

    #[ignore]
    #[test]
    fn pt2_soln_example() {
        panic!();
    }
}
