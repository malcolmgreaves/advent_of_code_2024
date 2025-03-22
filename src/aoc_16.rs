use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::{
    graph::{self, Graph, GraphBuilder, Node, SparseBuilder, SparseGraph},
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
    let lines = io_help::read_lines("./inputs/16");
    let puzzle: Puzzle = construct(lines)?;
    match lowest_cost_path_dijkstras(&puzzle) {
        Some(cost) => Ok(cost),
        None => Err(format!("no path from start -> finish was found!")),
    }
    // let (_, cost) = brute_force_lowest_cost(&puzzle);
    // Ok(cost)
}

#[derive(Debug, PartialEq, Eq, Clone, Hash)]
enum Move {
    Rotate90Clockwise,
    Rotate90CounterCW,
    Step(Direction),
}

fn path_cost<'a>(path: impl Iterator<Item = &'a Move>) -> u64 {
    path.map(|m| match m {
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
            let c = path_cost(p.iter());
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
        // println!("FINISHED!");
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
                    // println!("\t\t\talready visisted {continuing} on this run");
                    return;
                }

                match &puzzle[continuing.row][continuing.col] {
                    Tile::Empty | Tile::End => {
                        // println!("\tadvancing from {loc} to {continuing} along {d:?}!");
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
                        // println!(
                        //     "\t\tcannot move {d:?} from {loc} into {continuing} because the tile is non-empty: {t:?}",
                        // );
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

///////////////////////////////////////////////////////////////////////////////////////////////////

pub fn solution_pt2() -> Result<u64, String> {
    let lines = io_help::read_lines("./inputs/16");
    let puzzle: Puzzle = construct(lines)?;
    let _ = puzzle;
    Err(format!("part 2 is unimplemented!"))
}

#[derive(Debug, PartialEq, Eq, PartialOrd)]
struct Search {
    loc: Coordinate,
    dir: Direction,
    cost: u64,
}

impl Ord for Search {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.cost.cmp(&other.cost)
    }
}

fn lowest_cost_path_dijkstras(puzzle: &Puzzle) -> Option<u64> {
    let start = locate(puzzle, Tile::Start).unwrap();
    let end = locate(puzzle, Tile::End).unwrap();

    // create graph from Puzzle
    // let graph = create_graph(puzzle);

    let g = GridMovement::new(puzzle);

    // create priority queue
    let mut priority_queue = BinaryHeap::<Search>::new();
    priority_queue.push(Search {
        loc: start.clone(),
        dir: Direction::Right,
        cost: 0,
    });

    // create cost ("distance") map from start -> each vertx (empty space)
    let mut distance: HashMap<Coordinate, u64> = GridMovement::new(puzzle)
        .coordinates()
        .map(|c| {
            let cost_from_start = if c == start { 0 } else { u64::MAX };
            (c, cost_from_start)
        })
        .collect();

    // while queue is not empty
    //      v = take lowest cost from queue
    //      if v is end: return cost(v)

    let mut hit_end_once = false;
    while let Some(Search { loc, dir, cost }) = priority_queue.pop() {
        if loc == end {
            println!("\treached end! cost={cost}, skipping");
            // return Some(cost);
            hit_end_once = true;
            continue;
        }
        if cost > *distance.get(&loc).unwrap() {
            // lower-cose path to loc has already been found
            continue;
        }

        // graph.neighbors(node)
        for (neighbor, new_dir) in g.cardinal_neighbor_directions(&loc) {
            if puzzle[neighbor.row][neighbor.col] == Tile::Wall {
                // we can't go into a wall!
                continue;
            }

            // fold the step + rotation cost into next_cost
            let next_cost = if new_dir == dir {
                // we're always stepping, which is cost 1
                // in this case, we are not rotating
                1
            } else {
                if new_dir == dir.opposite() {
                    if loc == start {
                        // we only consider rotating backwards if we are at START
                        2001
                    } else {
                        continue;
                    }
                } else {
                    // we're rotating, so we need to incorporate this higher cost
                    1001
                }
            };

            let considering_next = Search {
                loc: neighbor,
                cost: cost + next_cost,
                dir: new_dir,
            };

            let previous_min_cost = distance.get_mut(&considering_next.loc).unwrap();
            println!(
                "\t{loc} ->{} next={} vs. min={}",
                considering_next.loc, considering_next.cost, previous_min_cost
            );

            if considering_next.cost < *previous_min_cost {
                // the path we took to get here is lower than the minimum cost of
                // some other path we took to get here!
                let new_min_cost = considering_next.cost.clone();
                println!("\t\tsetting to new min={}", considering_next.cost);
                priority_queue.push(considering_next);
                // update distance (cost) for the location
                *previous_min_cost = new_min_cost;
            }
        }
    }

    distance
        .get(&end)
        .filter(|_| hit_end_once)
        .map(|c| c.clone())
}

/// Represents moving into Coorindate and facing Direction.
type MovedInto = (Coordinate, Direction);

fn create_graph(puzzle: &Puzzle) -> impl Graph<MovedInto> {
    let g = &GridMovement::new(puzzle);
    let mut graph_builder = SparseBuilder::with_capacity(puzzle.len());

    // initial rotation?
    // or does the if puizzle[...][...] != Start in insert(.) work??
    // graph_builder.insert(
    //     Node(),
    //     Node(),
    // );

    let insert = |graph_builder: &mut SparseBuilder<MovedInto>,
                  source: Coordinate,
                  destination: Coordinate| {
        for d1 in [
            Direction::Up,
            Direction::Down,
            Direction::Left,
            Direction::Right,
        ] {
            for d2 in [
                Direction::Up,
                Direction::Down,
                Direction::Left,
                Direction::Right,
            ] {
                if puzzle[source.row][source.col] != Tile::Start && d1.opposite() == d2 {
                    continue;
                }
                graph_builder.insert(
                    Node((source.clone(), d1.clone())),
                    Node((destination.clone(), d2.clone())),
                );
            }
        }
    };

    for (row, r) in puzzle.iter().enumerate() {
        for (col, t) in r.iter().enumerate() {
            if *t != Tile::Wall {
                let c = Coordinate { row, col };
                for neighbor_coordinate in g.cardinal_neighbors(&c) {
                    if puzzle[neighbor_coordinate.row][neighbor_coordinate.col] != Tile::Wall {
                        insert(&mut graph_builder, c.clone(), neighbor_coordinate);
                        // (&mut graph_builder).insert(
                        //     Node((c.clone(), )),
                        //     Node((neighbor_coordinate, )),
                        // );
                    }
                }
            }
        }
    }
    graph_builder.to_graph()
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

    const EXAMPLE_INPUT_STR_ONE_PATH: &str = indoc! {"
        #######
        #####E#
        #...#.#
        #.#.#.#
        #S#...#
        #######
    "};

    lazy_static! {
        static ref EXAMPLE_ONLY_ONE_PATH: Path = vec![
            Move::Rotate90CounterCW,      // rotate
            Move::Step(Direction::Up),    // move
            Move::Step(Direction::Up),    // move
            Move::Rotate90Clockwise,      // rotate
            Move::Step(Direction::Right), // move
            Move::Step(Direction::Right), // move
            Move::Rotate90Clockwise,      // rotate
            Move::Step(Direction::Down),  // move
            Move::Step(Direction::Down),  // move
            Move::Rotate90CounterCW,      // rotate
            Move::Step(Direction::Right), // move
            Move::Step(Direction::Right), // move
            Move::Rotate90CounterCW,      // rotate
            Move::Step(Direction::Up),    // move
            Move::Step(Direction::Up),    // move
            Move::Step(Direction::Up),    // move
        ];
    }

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
    fn brute_force_paths_one_path_example() {
        let example: Puzzle = construct(read_lines_in_memory(EXAMPLE_INPUT_STR_ONE_PATH)).unwrap();
        let expected_moves: &Path = &EXAMPLE_ONLY_ONE_PATH;
        brute_force_all_paths(&example, None, Some(expected_moves));
    }

    #[test]
    fn brute_force_paths_example_1() {
        let example: &Puzzle = &EXAMPLE_EXPECTED_1;
        brute_force_all_paths(example, Some(7036), None);
    }

    fn brute_force_all_paths(
        example: &Puzzle,
        expected_cost: Option<u64>,
        expected_path: Option<&[Move]>,
    ) {
        let (path, min_cost) = brute_force_lowest_cost(example);
        println!("minimum cost is: {min_cost}");
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
        );

        let mut both_none = false;
        match expected_cost {
            Some(c) => assert_eq!(
                c, min_cost,
                "incorrect minimum cost - expecting {min_cost} but got {c}"
            ),
            None => both_none = true,
        }
        match expected_path {
            Some(p) => {
                let c = path_cost(p.iter());
                assert_eq!(
                    c, min_cost,
                    "incorrect minimum cost - expecting {min_cost} from supplied expected path but got {c}"
                );
                assert_eq!(p, path);
            }
            None => assert!(
                !both_none,
                "can't have expected cost and path both be none!"
            ),
        }
    }

    #[test]
    fn dijkstras_one_path_example() {
        let example: Puzzle = construct(read_lines_in_memory(EXAMPLE_INPUT_STR_ONE_PATH)).unwrap();
        let expected_moves: &Path = &EXAMPLE_ONLY_ONE_PATH;
        let expected = path_cost(expected_moves.iter());
        let actual = lowest_cost_path_dijkstras(&example).unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn dijkstras_example_1() {
        let example: Puzzle = construct(read_lines_in_memory(EXAMPLE_INPUT_STR_1)).unwrap();
        let expected = 7036;
        let actual = lowest_cost_path_dijkstras(&example).unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn dijkstras_example_2() {
        let example: Puzzle = construct(read_lines_in_memory(EXAMPLE_INPUT_STR_2)).unwrap();
        let expected = 11048;
        let actual = lowest_cost_path_dijkstras(&example).unwrap();
        assert_eq!(actual, expected);
    }

    ///////////////////////////////////////////////

    #[test]
    fn pt1_soln_example() {
        assert_eq!(solution_pt1().unwrap(), 98416);
    }

    ///////////////////////////////////////////////

    #[ignore]
    #[test]
    fn pt2_soln_example() {
        panic!();
    }
}
