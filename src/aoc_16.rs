use std::{
    collections::{BinaryHeap, HashMap, HashSet, VecDeque},
    hash::{DefaultHasher, Hash, Hasher},
};

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
    // println!("results.len()= {:?}", results.len());

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
                    _ => {
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
    match number_tiles_in_all_lowest_cost_paths(&puzzle) {
        Some(actual) => Ok(actual.n_tiles()),
        None => Err(format!("could not find a path from start to end!")),
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Clone)]
struct Search {
    loc: Coordinate,
    dir: Direction,
    cost: u64,
}

type HashKey = u64;

fn hashkey(loc: &Coordinate, dir: &Direction) -> u64 {
    let mut h = DefaultHasher::new();
    loc.hash(&mut h);
    dir.hash(&mut h);
    h.finish()
}

impl Search {
    fn hashkey(&self) -> u64 {
        hashkey(&self.loc, &self.dir)
    }
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

    let mut prev_for_best_path: HashMap<Coordinate, Option<Coordinate>> = GridMovement::new(puzzle)
        .coordinates()
        .map(|c| (c, None))
        .collect();

    // while queue is not empty
    //      v = take lowest cost from queue
    //      if v is end: return cost(v)

    let mut hit_end_once = false;
    while let Some(Search { loc, dir, cost }) = priority_queue.pop() {
        if loc == end {
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

            if considering_next.cost < *previous_min_cost {
                // the path we took to get here is lower than the minimum cost of
                // some other path we took to get here!
                let new_min_cost = considering_next.cost.clone();
                let neighbor = considering_next.loc.clone();
                priority_queue.push(considering_next);
                // update distance (cost) for the location
                *previous_min_cost = new_min_cost;
                // update: we got to <neighbor> via <loc>
                prev_for_best_path.insert(neighbor, Some(loc.clone()));
            }
        }
    }

    distance
        .get(&end)
        .filter(|_| hit_end_once)
        .map(|c| c.clone())
}

struct BestPaths {
    visited: HashSet<Coordinate>,
}

impl BestPaths {
    fn n_tiles(&self) -> u64 {
        self.visited.len() as u64
    }
}

fn number_tiles_in_all_lowest_cost_paths(puzzle: &Puzzle) -> Option<BestPaths> {
    let start = locate(puzzle, Tile::Start).unwrap();
    let end = locate(puzzle, Tile::End).unwrap();

    // create graph from Puzzle
    let g = GridMovement::new(puzzle);

    let is_wall_in_next_step_in_dir = |considering_next: &Search| -> bool {
        match g.next_advance(&considering_next.loc, &considering_next.dir) {
            Some(Coordinate {
                row: next_row,
                col: next_col,
            }) => match puzzle[next_row][next_col] {
                Tile::Wall => true,
                _ => false,
            },
            None => false,
        }
    };

    // create priority queue
    let mut priority_queue = BinaryHeap::<Search>::new();
    priority_queue.push(Search {
        loc: start.clone(),
        dir: Direction::Right,
        cost: 0,
    });

    // create cost ("distance") map from start -> each vertx (empty space)
    let mut distance: HashMap<HashKey, u64> = GridMovement::new(puzzle)
        .coordinates()
        .flat_map(|c| {
            let cost_from_start = if c == start { 0 } else { u64::MAX };
            Direction::ALL.map(|d| {
                let k = hashkey(&c, &d);
                (k, cost_from_start)
            })
        })
        .collect();

    let is_higher_than_min = |distance: &HashMap<HashKey, u64>, current: &Search| -> bool {
        // check if lower-cose path to loc has already been found
        if current.cost > *distance.get(&current.hashkey()).unwrap() {
            return true;
        }

        // if not, check if there is one going in the OPPOSITE direction
        // => means that we are now backtracking from whence we came and thus
        // => are going to be higher cost!
        if current.cost
            > *distance
                .get(&hashkey(&current.loc, &current.dir.opposite()))
                .unwrap()
        {
            return true;
        }

        false
    };

    let mut prev_for_best_path: HashMap<HashKey, Vec<(u64, Coordinate, Direction)>> =
        GridMovement::new(puzzle)
            .coordinates()
            .flat_map(|c| {
                Direction::ALL.map(|d| {
                    let k = hashkey(&c, &d);
                    (k, vec![])
                })
            })
            .collect::<HashMap<_, _>>();

    let mut hit_end_once = false;
    while let Some(current_search) = priority_queue.pop() {
        if current_search.loc == end {
            hit_end_once = true;
            continue;
        }

        if is_higher_than_min(&distance, &current_search) {
            continue;
        }

        let Search { loc, dir, cost } = current_search;
        // graph.neighbors(node)
        for (neighbor, new_dir) in g.cardinal_neighbor_directions(&loc) {
            if puzzle[neighbor.row][neighbor.col] == Tile::Wall {
                // we can't go into a wall!
                continue;
            }

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

            let in_consideration = if is_wall_in_next_step_in_dir(&considering_next) {
                let mut cs = Vec::with_capacity(2);

                let mut c1 = considering_next.clone();
                c1.cost += 1000;
                c1.dir = c1.dir.clockwise();
                cs.push(c1);

                let mut c2 = considering_next.clone();
                c2.cost += 1000;
                c2.dir = c2.dir.counter_clockwise();
                cs.push(c2);

                cs
            } else {
                vec![considering_next]
            };

            for considering_next in in_consideration {
                let previous_min_cost = distance.get_mut(&considering_next.hashkey()).unwrap();
                if considering_next.cost <= *previous_min_cost {
                    // either a new minimum OR just equal to the existing minimum

                    // update: we got to <neighbor> via <loc>
                    //      this also means we need to ignore any paths
                    //      in the value that are LESS than the min we just found
                    let existing = prev_for_best_path
                        .get_mut(&considering_next.hashkey())
                        .unwrap();

                    if considering_next.cost < *previous_min_cost {
                        // the path we took to get here is lower than the minimum cost of
                        // some other path we took to get here!
                        //      update distance (cost) for the location => de-reference & write
                        *previous_min_cost = considering_next.cost.clone();
                        // ignore all other prior paths that got to the *OLD* min cost
                        // since we are now at a **NEW** min cost
                        let new_existing = existing
                            .iter()
                            .filter(|(c, _, _)| *c <= considering_next.cost)
                            .map(|(c, l, d)| (c.clone(), l.clone(), d.clone()))
                            .collect::<Vec<_>>();
                        *existing = new_existing;
                    } else {
                        // it's equal ==> so we are just keeping track that this is
                        // one of the current min-cost paths
                        // we don't need to do anything since we only clear in the true minimum case
                    }

                    let nc = considering_next.cost.clone();
                    priority_queue.push(considering_next);
                    existing.push((nc, loc.clone(), dir.clone()));
                }
            }
        }
    }

    // BACKTRACK
    Direction::ALL
        .map(|d| Search {
            loc: end.clone(),
            dir: d,
            cost: u64::MAX,
        })
        .into_iter()
        .flat_map(|s| distance.get(&s.hashkey()))
        .min()
        .filter(|_| hit_end_once)
        .map(|_| {
            let mut visited = HashSet::new();

            let mut queue = VecDeque::new();
            queue.push_back((
                end.clone(),
                // find the start of the minimum cost path
                //   => which direction did we take to end?
                {
                    let (min_dir, _) = Direction::ALL
                        .map(|d| {
                            let k = hashkey(&end, &d);
                            (d, distance.get(&k).unwrap())
                        })
                        .into_iter()
                        .min_by_key(|(_, cost)| *cost)
                        .unwrap();
                    min_dir
                },
            ));

            while let Some((location, dir)) = queue.pop_front() {
                // add each next (_"previous" direction_) step
                // in every path that took us to `location`
                let key = hashkey(&location, &dir);

                // take that one's path and push onto queue
                match prev_for_best_path.remove(&key) {
                    Some(paths_to_location) => {
                        for (_, prev_loc, prev_dir) in paths_to_location {
                            queue.push_back((prev_loc, prev_dir));
                        }
                    }
                    None => (),
                };

                // record that we have visited this tile
                visited.insert(location);
            }

            BestPaths { visited }
        })
}

fn print_paths_taken(puzzle: &Puzzle, visited: &HashSet<Coordinate>) {
    for row in 0..puzzle.len() {
        for col in 0..puzzle[0].len() {
            let loc = Coordinate { row, col };
            match puzzle[row][col] {
                Tile::Wall => print!("#"),
                Tile::End => print!("E"),
                Tile::Start => print!("S"),
                _ => {
                    if visited.contains(&loc) {
                        print!("O");
                    } else {
                        print!(".");
                    }
                }
            }
        }
        print!("\n");
    }
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

    #[test]
    fn backtracking_one_path_example() {
        let example: Puzzle = construct(read_lines_in_memory(EXAMPLE_INPUT_STR_ONE_PATH)).unwrap();
        let actual = number_tiles_in_all_lowest_cost_paths(&example).unwrap();
        print_paths_taken(&example, &actual.visited);
        assert_eq!(actual.n_tiles(), 12);
    }

    #[test]
    fn backtracking_example_1() {
        let example: Puzzle = construct(read_lines_in_memory(EXAMPLE_INPUT_STR_1)).unwrap();
        let actual = number_tiles_in_all_lowest_cost_paths(&example).unwrap();
        print_paths_taken(&example, &actual.visited);
        assert_eq!(actual.n_tiles(), 45);
    }

    #[test]
    fn backtracking_example_2() {
        let example: Puzzle = construct(read_lines_in_memory(EXAMPLE_INPUT_STR_2)).unwrap();
        let actual = number_tiles_in_all_lowest_cost_paths(&example).unwrap();
        print_paths_taken(&example, &actual.visited);
        assert_eq!(actual.n_tiles(), 64);
    }

    #[test]
    fn backtracking_example_micro() {
        let example_input = indoc! {"
            ###############
            #############E#
            #...........#.#
            ###.#.#####.#.#
            #...#.....#.#.#
            #.#.#.###.#.#.#
            #.....#...#.#.#
            #.###.#.#.#.#.#
            #S..#.....#...#
            ###############
        "};
        let example: Puzzle = construct(read_lines_in_memory(example_input)).unwrap();
        let actual = number_tiles_in_all_lowest_cost_paths(&example).unwrap();
        print_paths_taken(&example, &actual.visited);
        assert_eq!(actual.n_tiles(), 40);
        /* EXPECTED:
           ###############
           #############O#
           #..OOOOOOOOO#O#
           ###O#O#####O#O#
           #OOO#O....#O#O#
           #O#O#O###.#O#O#
           #OOOOO#...#O#O#
           #O###.#.#.#O#O#
           #O..#.....#OOO#
           ###############
        */
    }

    ///////////////////////////////////////////////

    #[test]
    fn pt1_soln_example() {
        assert_eq!(solution_pt1().unwrap(), 98416);
    }

    ///////////////////////////////////////////////

    #[test]
    fn pt2_soln_example() {
        let _ = solution_pt2();
        assert_eq!(solution_pt2().unwrap(), 471);
    }
}
