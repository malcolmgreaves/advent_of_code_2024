use std::collections::HashSet;

use crate::{
    io_help,
    matrix::{Coordinate, Coords, Direction, GridMovement, Matrix},
    utils::collect_results,
};

use stacker;

///////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Debug, PartialEq, Eq, Clone)]
enum Tile {
    Wall,
    Empty,
    Start,
    End,
}

type Puzzle = Matrix<Tile>;

fn construct(lines: impl Iterator<Item = String>) -> Result<Puzzle, String> {
    let mut n_start = 0_u32;
    let mut n_end = 0_u32;

    let (tile_rows, errors) = collect_results(lines.map(|l| {
        let (ts, es) = collect_results(l.chars().map(parse_tile).map(|x| {
            match x {
                Ok(Tile::Start) => n_start += 1,
                Ok(Tile::End) => n_end += 1,
                _ => (),
            };
            x
        }));
        if es.len() > 0 {
            Err(format!(
                "failed to parse {} characters into tiles:\n\t{}",
                es.len(),
                es.join("\n\t")
            ))
        } else {
            Ok(ts)
        }
    }));

    if errors.len() > 0 {
        Err(format!(
            "failed to parse {} lines (rows):\n\t{}",
            errors.len(),
            errors.join("\n\t")
        ))
    } else if n_start != 1 {
        Err(format!(
            "found {n_start} Start tiles when there must be exactly one!"
        ))
    } else if n_end != 1 {
        Err(format!(
            "found {n_end} End tiles when there must be exactly one!"
        ))
    } else {
        Ok(tile_rows)
    }
}

fn parse_tile(c: char) -> Result<Tile, String> {
    match c {
        '#' => Ok(Tile::Wall),
        '.' => Ok(Tile::Empty),
        'S' => Ok(Tile::Start),
        'E' => Ok(Tile::End),
        _ => Err(format!("unrecognized Tile character: '{c}'")),
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

pub fn solution_pt1() -> Result<u64, String> {
    let lines = io_help::read_lines("./inputs/???");
    let puzzle: Puzzle = construct(lines)?;
    // let (_, cost) = lowest_cost_path_dijkstras(&puzzle);
    let (_, cost) = brute_force_lowest_cost(&puzzle);
    Ok(cost)
}

#[derive(Debug, PartialEq, Eq, Clone)]
enum Move {
    Rotate90Clockwise,
    Rotate90CounterCW,
    Step(Direction),
}

fn cost(path: &[Move]) -> u64 {
    path.iter()
        .map(|m| match m {
            Move::Rotate90Clockwise | Move::Rotate90CounterCW => 1000,
            Move::Step(_) => 1,
        })
        .sum()
}

fn brute_force_lowest_cost(puzzle: &Puzzle) -> (Vec<Move>, u64) {
    let start = locate(puzzle, Tile::Start).unwrap();
    let end = locate(puzzle, Tile::End).unwrap();

    let results = all_paths(puzzle, &start, &end);
    println!("results.len()= {:?}", results.len());

    results
        .into_iter()
        .map(|p| {
            let c = cost(&p);
            (p, c)
        })
        .min_by_key(|(_, c)| *c)
        .unwrap()
}

fn locate(puzzle: &Puzzle, start_or_end: Tile) -> Result<Coordinate, String> {
    match start_or_end {
        Tile::Empty | Tile::Wall => Err(format!("only locate start or end!")),
        s_e => {
            let ptr_se = &s_e;
            let mut locs = puzzle
                .iter()
                .enumerate()
                .flat_map(|(row, r)| {
                    r.iter().enumerate().flat_map(move |(col, t)| {
                        if t == ptr_se {
                            Some(Coordinate { row, col })
                        } else {
                            None
                        }
                    })
                })
                .collect::<Vec<_>>();
            if locs.len() != 1 {
                Err(format!(
                    "expecting to find exactly 1 location for {s_e:?} but found {}: {}",
                    locs.len(),
                    Coords(&locs)
                ))
            } else {
                Ok(locs.swap_remove(0))
            }
        }
    }
}

fn all_paths(puzzle: &Puzzle, start: &Coordinate, end: &Coordinate) -> Vec<Path> {
    assert_eq!(
        puzzle[start.row][start.col],
        Tile::Start,
        "expecting start to be at {start} but found {:?}",
        puzzle[start.row][start.col]
    );
    assert_eq!(
        puzzle[end.row][end.col],
        Tile::End,
        "expecting end to be at {end} but found {:?}",
        puzzle[end.row][end.col]
    );

    let mut path_accumulator = Vec::new();
    // start facing EAST aka right
    stacker::maybe_grow(32 * 1024, 1024 * 1024, || {
        walk(
            &GridMovement::new(puzzle),
            puzzle,
            start.clone(),
            Direction::Right,
            vec![],
            &mut path_accumulator,
            &mut HashSet::new(),
        )
    });
    path_accumulator
}

type Path = Vec<Move>;

fn walk(
    g: &GridMovement,
    puzzle: &Puzzle,
    loc: Coordinate,
    facing: Direction,
    current: Path,
    finished_paths: &mut Vec<Path>,
    visisted: &mut HashSet<Coordinate>,
) {
    if puzzle[loc.row][loc.col] == Tile::End {
        println!("FINISHED!");
        finished_paths.push(current);
        return;
    }

    let try_advance = |finished_paths: &mut Vec<Path>,
                       visisted: &mut HashSet<Coordinate>,
                       current: &Path,
                       d: Direction| {
        // println!("\ttry_advance(.., loc={loc}, facing={d:?})");
        match g.next_advance(&loc, &d) {
            Some(continuing) => {
                if visisted.contains(&continuing) {
                    println!("\t\t\talready visisted {continuing} on this run");
                    return;
                }

                match &puzzle[continuing.row][continuing.col] {
                    Tile::Empty => {
                        println!("\tadvancing from {loc} to {continuing} along {d:?}!");
                        let mut extended_path = current.clone();
                        extended_path.push(Move::Step(d.clone()));
                        let mut new_visisted = visisted.clone();
                        new_visisted.insert(continuing.clone());
                        stacker::maybe_grow(32 * 1024, 1024 * 1024, || {
                            // guaranteed to have at least 32M of stack
                            walk(
                                g,
                                puzzle,
                                continuing,
                                d,
                                extended_path,
                                finished_paths,
                                &mut new_visisted,
                            )
                        });
                    }
                    t => {
                        println!(
                            "\t\tcannot move {d:?} from {loc} into {continuing} because the tile is non-empty: {t:?}",
                        );
                        ();
                    }
                }
            }
            None => {
                ();
                // println!("\t\tcannot go {d:?} from {loc} as it is out of bounds\n");
            }
        }
    };

    // println!("conituning in direction: {facing:?} from {loc}");
    try_advance(finished_paths, visisted, &current, facing.clone());

    // println!(
    //     "trying clockwise rotation: {:?} from {loc}",
    //     facing.clockwise()
    // );
    try_advance(
        finished_paths,
        visisted,
        &{
            let mut p = current.clone();
            p.push(Move::Rotate90Clockwise);
            p
        },
        facing.clockwise(),
    );

    // one more clockwise would be going BACKWARDS
    // so we do 2 clockwise rotations
    // ==> this is equivalent to a counter-clockwise rotation from the original direction
    // println!(
    //     "trying counter-clockwise rotation: {:?} from {loc}",
    //     facing.counter_clockwise()
    // );
    try_advance(
        finished_paths,
        visisted,
        &{
            let mut p = current.clone();
            p.push(Move::Rotate90CounterCW);
            p
        },
        facing.counter_clockwise(),
    );

    // we ONLY ROTATE TO FACE WHERE WE CAME FROM AT THE START
    // otherwise, if we do this, we will infinite loop!
    if current.len() == 0 {
        // println!(
        //     "initial: turning 180 -> facing {:?} from {loc}",
        //     facing.opposite()
        // );
        try_advance(
            finished_paths,
            visisted,
            &{
                let mut p = current.clone();
                p.push(Move::Rotate90Clockwise);
                p.push(Move::Rotate90Clockwise);
                p
            },
            facing.opposite(),
        );
    }
}

fn lowest_cost_path_dijkstras(puzzle: &Puzzle) -> (Vec<Move>, u64) {
    // create graph from Puzzle
    // create priority queue
    // create cost ("distance") map from start -> each vertx (empty space)
    // while queue is not empty
    //      v = take lowest cost from queue
    //      if v is end: return cost(v)
    //
    panic!()
}

///////////////////////////////////////////////////////////////////////////////////////////////////

pub fn solution_pt2() -> Result<u64, String> {
    let lines = io_help::read_lines("./inputs/???");
    let puzzle: Puzzle = construct(lines)?;
    let _ = puzzle;
    Err(format!("part 2 is unimplemented!"))
}

///////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod test {

    use indoc::indoc;
    use lazy_static::lazy_static;

    use crate::{io_help::read_lines_in_memory, testing_utilities::check_matrices};

    use super::*;

    ///////////////////////////////////////////////

    const EXAMPLE_INPUT_STR_1: &str = indoc! {"
        ###############
        #.......#....E#
        #.#.###.#.###.#
        #.....#.#...#.#
        #.###.#####.#.#
        #.#.#.......#.#
        #.#.#####.###.#
        #...........#.#
        ###.#.#####.#.#
        #...#.....#.#.#
        #.#.#.###.#.#.#
        #.....#...#.#.#
        #.###.#.#.#.#.#
        #S..#.....#...#
        ###############
    "};

    lazy_static! {
        static ref EXAMPLE_EXPECTED_1: Puzzle = vec![
            vec![
                Tile::Wall,
                Tile::Wall,
                Tile::Wall,
                Tile::Wall,
                Tile::Wall,
                Tile::Wall,
                Tile::Wall,
                Tile::Wall,
                Tile::Wall,
                Tile::Wall,
                Tile::Wall,
                Tile::Wall,
                Tile::Wall,
                Tile::Wall,
                Tile::Wall,
            ],
            vec![
                Tile::Wall,
                Tile::Empty,
                Tile::Empty,
                Tile::Empty,
                Tile::Empty,
                Tile::Empty,
                Tile::Empty,
                Tile::Empty,
                Tile::Wall,
                Tile::Empty,
                Tile::Empty,
                Tile::Empty,
                Tile::Empty,
                Tile::End,
                Tile::Wall,
            ],
            vec![
                Tile::Wall,
                Tile::Empty,
                Tile::Wall,
                Tile::Empty,
                Tile::Wall,
                Tile::Wall,
                Tile::Wall,
                Tile::Empty,
                Tile::Wall,
                Tile::Empty,
                Tile::Wall,
                Tile::Wall,
                Tile::Wall,
                Tile::Empty,
                Tile::Wall
            ],
            vec![
                Tile::Wall,
                Tile::Empty,
                Tile::Empty,
                Tile::Empty,
                Tile::Empty,
                Tile::Empty,
                Tile::Wall,
                Tile::Empty,
                Tile::Wall,
                Tile::Empty,
                Tile::Empty,
                Tile::Empty,
                Tile::Wall,
                Tile::Empty,
                Tile::Wall
            ],
            vec![
                Tile::Wall,
                Tile::Empty,
                Tile::Wall,
                Tile::Wall,
                Tile::Wall,
                Tile::Empty,
                Tile::Wall,
                Tile::Wall,
                Tile::Wall,
                Tile::Wall,
                Tile::Wall,
                Tile::Empty,
                Tile::Wall,
                Tile::Empty,
                Tile::Wall,
            ],
            vec![
                Tile::Wall,
                Tile::Empty,
                Tile::Wall,
                Tile::Empty,
                Tile::Wall,
                Tile::Empty,
                Tile::Empty,
                Tile::Empty,
                Tile::Empty,
                Tile::Empty,
                Tile::Empty,
                Tile::Empty,
                Tile::Wall,
                Tile::Empty,
                Tile::Wall
            ],
            vec![
                Tile::Wall,
                Tile::Empty,
                Tile::Wall,
                Tile::Empty,
                Tile::Wall,
                Tile::Wall,
                Tile::Wall,
                Tile::Wall,
                Tile::Wall,
                Tile::Empty,
                Tile::Wall,
                Tile::Wall,
                Tile::Wall,
                Tile::Empty,
                Tile::Wall,
            ],
            vec![
                Tile::Wall,
                Tile::Empty,
                Tile::Empty,
                Tile::Empty,
                Tile::Empty,
                Tile::Empty,
                Tile::Empty,
                Tile::Empty,
                Tile::Empty,
                Tile::Empty,
                Tile::Empty,
                Tile::Empty,
                Tile::Wall,
                Tile::Empty,
                Tile::Wall
            ],
            vec![
                Tile::Wall,
                Tile::Wall,
                Tile::Wall,
                Tile::Empty,
                Tile::Wall,
                Tile::Empty,
                Tile::Wall,
                Tile::Wall,
                Tile::Wall,
                Tile::Wall,
                Tile::Wall,
                Tile::Empty,
                Tile::Wall,
                Tile::Empty,
                Tile::Wall
            ],
            vec![
                Tile::Wall,
                Tile::Empty,
                Tile::Empty,
                Tile::Empty,
                Tile::Wall,
                Tile::Empty,
                Tile::Empty,
                Tile::Empty,
                Tile::Empty,
                Tile::Empty,
                Tile::Wall,
                Tile::Empty,
                Tile::Wall,
                Tile::Empty,
                Tile::Wall
            ],
            vec![
                Tile::Wall,
                Tile::Empty,
                Tile::Wall,
                Tile::Empty,
                Tile::Wall,
                Tile::Empty,
                Tile::Wall,
                Tile::Wall,
                Tile::Wall,
                Tile::Empty,
                Tile::Wall,
                Tile::Empty,
                Tile::Wall,
                Tile::Empty,
                Tile::Wall
            ],
            vec![
                Tile::Wall,
                Tile::Empty,
                Tile::Empty,
                Tile::Empty,
                Tile::Empty,
                Tile::Empty,
                Tile::Wall,
                Tile::Empty,
                Tile::Empty,
                Tile::Empty,
                Tile::Wall,
                Tile::Empty,
                Tile::Wall,
                Tile::Empty,
                Tile::Wall,
            ],
            vec![
                Tile::Wall,
                Tile::Empty,
                Tile::Wall,
                Tile::Wall,
                Tile::Wall,
                Tile::Empty,
                Tile::Wall,
                Tile::Empty,
                Tile::Wall,
                Tile::Empty,
                Tile::Wall,
                Tile::Empty,
                Tile::Wall,
                Tile::Empty,
                Tile::Wall,
            ],
            vec![
                Tile::Wall,
                Tile::Start,
                Tile::Empty,
                Tile::Empty,
                Tile::Wall,
                Tile::Empty,
                Tile::Empty,
                Tile::Empty,
                Tile::Empty,
                Tile::Empty,
                Tile::Wall,
                Tile::Empty,
                Tile::Empty,
                Tile::Empty,
                Tile::Wall
            ],
            vec![
                Tile::Wall,
                Tile::Wall,
                Tile::Wall,
                Tile::Wall,
                Tile::Wall,
                Tile::Wall,
                Tile::Wall,
                Tile::Wall,
                Tile::Wall,
                Tile::Wall,
                Tile::Wall,
                Tile::Wall,
                Tile::Wall,
                Tile::Wall,
                Tile::Wall,
            ],
        ];
    }

    const EXAMPLE_INPUT_STR_2: &str = indoc! {"
        #################
        #...#...#...#..E#
        #.#.#.#.#.#.#.#.#
        #.#.#.#...#...#.#
        #.#.#.#.###.#.#.#
        #...#.#.#.....#.#
        #.#.#.#.#.#####.#
        #.#...#.#.#.....#
        #.#.#####.#.###.#
        #.#.#.......#...#
        #.#.###.#####.###
        #.#.#...#.....#.#
        #.#.#.#####.###.#
        #.#.#.........#.#
        #.#.#.#########.#
        #S#.............#
        #################
    "};

    ///////////////////////////////////////////////

    #[test]
    fn construction_1() {
        // 1
        let expected: &Puzzle = &EXAMPLE_EXPECTED_1;
        let acutal: Puzzle = construct(read_lines_in_memory(EXAMPLE_INPUT_STR_1)).unwrap();
        match check_matrices(&acutal, expected) {
            Ok(()) => (),
            Err(e) => assert!(false, "{e}"),
        };
        // 2
        match construct(read_lines_in_memory(EXAMPLE_INPUT_STR_2)) {
            Ok(_) => (),
            Err(e) => assert!(false, "{e}"),
        };
    }

    #[ignore]
    #[test]
    fn construction_2() {
        panic!();
    }

    #[test]
    fn brute_force_all_paths() {
        // let example: &Puzzle = &EXAMPLE_EXPECTED_1;

        /*

        */
        let example: &Puzzle = &construct(read_lines_in_memory(indoc! {
        "
            #######
            #####E#
            #...#.#
            #.#.#.#
            #S#...#
            #######
            "}))
        .unwrap();

        let (path, cost) = brute_force_lowest_cost(example);
        println!("minimum cost is: {cost}");
        println!(
            "path has {} members | {} are moves and {} are rotations",
            path.len(),
            path.iter()
                .filter(|m| match m {
                    Move::Step(_) => true,
                    _ => false,
                })
                .fold(0, |s, _| s + 1),
            path.iter()
                .filter(|m| match m {
                    Move::Step(_) => false,
                    _ => true,
                })
                .fold(0, |s, _| s + 1)
        )
    }

    ///////////////////////////////////////////////

    #[ignore]
    #[test]
    fn pt1_soln_example() {
        panic!();
    }

    ///////////////////////////////////////////////

    #[ignore]
    #[test]
    fn pt2_soln_example() {
        panic!();
    }
}
