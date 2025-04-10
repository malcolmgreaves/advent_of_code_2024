use std::{
    cmp::max,
    collections::{BinaryHeap, HashMap},
};

use crate::{
    io_help,
    matrix::{Coordinate, Coords, GridMovement, Matrix},
    search::binary_search_on_answer,
    utils::collect_results,
};

///////////////////////////////////////////////////////////////////////////////////////////////////

type Memory = Matrix<bool>;

fn construct(
    max_row_index: Option<usize>,
    max_col_index: Option<usize>,
    lines: impl Iterator<Item = String>,
) -> Result<Memory, String> {
    let coords = parse_coordinates(lines)?;
    make(max_row_index, max_col_index, coords.iter())
}

fn make<'a>(
    max_row_index: Option<usize>,
    max_col_index: Option<usize>,
    coords: impl Iterator<Item = &'a Coordinate> + Clone,
) -> Result<Memory, String> {
    let mut mem: Memory = {
        let (max_row_found, max_col_found) =
            coords
                .clone()
                .fold((0, 0), |(max_row, max_col), coordinate| {
                    (max(max_row, coordinate.row), max(max_col, coordinate.col))
                });

        let max_row_using = match max_row_index {
            Some(m) => {
                if m + 1 < max_row_found {
                    return Err(format!(
                        "found a coordinate with row={max_row_found} but requested row limit of {}",
                        m + 1
                    ));
                } else {
                    m
                }
            }
            None => max_row_found,
        };

        let max_col_using = match max_col_index {
            Some(m) => {
                if m + 1 < max_col_found {
                    return Err(format!(
                        "found a coordinate with col={max_col_found} but requested col limit of {}",
                        m + 1
                    ));
                } else {
                    m
                }
            }
            None => max_row_found,
        };

        vec![vec![false; max_col_using + 1]; max_row_using + 1]
    };

    for Coordinate { row, col } in coords {
        mem[*row][*col] = true;
    }
    Ok(mem)
}

fn parse_coordinates(lines: impl Iterator<Item = String>) -> Result<Vec<Coordinate>, String> {
    let (coordinates, errors) = collect_results(lines.map(parse_coordinate));
    if errors.len() > 0 {
        Err(format!(
            "Found {} coordinates but had {} errors:\n\t{}",
            coordinates.len(),
            errors.len(),
            errors.join("\n\t")
        ))
    } else {
        Ok(coordinates)
    }
}

fn parse_coordinate(l: String) -> Result<Coordinate, String> {
    let bits = l.trim().split(",").collect::<Vec<_>>();
    if bits.len() != 2 {
        Err(format!(
            "expecting a pair -- (distance from left: Y), (distance from top: X) but got {} comma-separated parts: '{l}'",
            bits.len()
        ))
    } else {
        Ok(Coordinate {
            row: bits[1].parse::<usize>().map_err(|e| format!("{e:?}"))?,
            col: bits[0].parse::<usize>().map_err(|e| format!("{e:?}"))?,
        })
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

pub fn solution_pt1() -> Result<u64, String> {
    let lines = io_help::read_lines("./inputs/18");
    let memory: Memory = construct(Some(70), Some(70), lines.take(1024))?;
    lowest_step_path_dijkstras(
        &memory,
        &Coordinate { row: 0, col: 0 },
        &Coordinate { row: 70, col: 70 },
    )
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Search {
    loc: Coordinate,
    cost: u64,
}

fn lowest_step_path_dijkstras(
    memory: &Memory,
    start: &Coordinate,
    end: &Coordinate,
) -> Result<u64, String> {
    validate(memory, start, end)?;

    // create graph
    let g = GridMovement::new(memory);

    // create priority queue
    let mut priority_queue: BinaryHeap<_> = BinaryHeap::<Search>::new();
    priority_queue.push(Search {
        loc: start.clone(),
        cost: 0,
    });

    // create cost ("distance") map from start -> each vertex (empty space)
    let mut distance: HashMap<Coordinate, u64> = GridMovement::new(memory)
        .coordinates()
        .map(|c| {
            let cost_from_start = if &c == start { 0 } else { u64::MAX };
            (c, cost_from_start)
        })
        .collect();

    let mut prev_for_best_path: HashMap<Coordinate, Option<Coordinate>> = GridMovement::new(memory)
        .coordinates()
        .map(|c| (c, None))
        .collect();

    // while queue is not empty
    //      v = take lowest cost from queue
    //      if v is end: return cost(v)

    let mut hit_end_once = false;
    while let Some(Search { loc, cost }) = priority_queue.pop() {
        if &loc == end {
            hit_end_once = true;
            continue;
        }
        if cost > *distance.get(&loc).unwrap() {
            // lower-cose path to loc has already been found
            continue;
        }

        // graph.neighbors(node)
        for (neighbor, new_dir) in g.cardinal_neighbor_directions(&loc) {
            if memory[neighbor.row][neighbor.col] == true {
                // we can't go into an occupied memory space!
                continue;
            }

            let next_cost = 1;

            let considering_next = Search {
                loc: neighbor,
                cost: cost + next_cost,
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

    match distance
        .get(&end)
        .filter(|_| hit_end_once)
        .map(|c| c.clone())
    {
        Some(min_steps) => Ok(min_steps),
        None => Err(format!(
            "no valid path found from start={start} to end={end}"
        )),
    }
}

fn validate(memory: &Memory, start: &Coordinate, end: &Coordinate) -> Result<(), String> {
    if memory.len() == 0 || memory[0].len() == 0 {
        return Err(format!("empty memory!"));
    }
    if start.row >= memory.len() {
        return Err(format!(
            "start's row is out of bounds: {} >= {}",
            start.row,
            memory.len()
        ));
    }
    if start.col >= memory[0].len() {
        return Err(format!(
            "start's col is out of bounds: {} >= {}",
            start.col,
            memory[0].len()
        ));
    }

    if end.row >= memory.len() {
        return Err(format!(
            "end's row is out of bounds: {} >= {}",
            end.row,
            memory.len()
        ));
    }
    if end.col >= memory[0].len() {
        return Err(format!(
            "end's col is out of bounds: {} >= {}",
            end.col,
            memory[0].len()
        ));
    }
    Ok(())
}

///////////////////////////////////////////////////////////////////////////////////////////////////

pub fn solution_pt2() -> Result<String, String> {
    let lines = io_help::read_lines("./inputs/18");
    let coordinates = parse_coordinates(lines)?;
    match find_first_coordinate_unreachable_end(70, 70, &coordinates)? {
        Some(c) => Ok(format!("{},{}", c.col, c.row)),
        None => Err(format!(
            "All {} coordinates allow for a path from the start (0,0) to the end (70,70)",
            coordinates.len()
        )),
    }
}

fn find_first_coordinate_unreachable_end(
    max_row_index: usize,
    max_col_index: usize,
    coordinates: &[Coordinate],
) -> Result<Option<Coordinate>, String> {
    let n_coordinates = coordinates.len();

    let iter = || coordinates.iter();

    // ensure we can actually make the Memory
    // this will ensure that the .unwrap() in makes_path_not_found will never panic!()
    _ = make(Some(max_row_index), Some(max_col_index), iter())?;

    // We want to know the first coordinate that prevents us from getting to the end
    // For binary search on answer, we need to arrange the data such that it's `false`s
    // follwed by `true`s.
    // Thus, we invert: if we _can_ find a path, then this function returns false.
    // If we cannot, then we return true.

    let makes_path_not_found = {
        let start = Coordinate { row: 0, col: 0 };
        let end = Coordinate {
            row: max_row_index,
            col: max_col_index,
        };
        move |coordinate_index: usize| -> bool {
            let coords_up_to_i = iter()
                .take(coordinate_index + 1)
                .map(|c| c.clone())
                .collect::<Vec<_>>();
            let memory: Memory = make(
                Some(max_row_index),
                Some(max_col_index),
                coords_up_to_i.iter(),
            )
            .unwrap();
            match lowest_step_path_dijkstras(&memory, &start, &end) {
                Ok(_) => false,
                Err(_) => true,
            }
        }
    };

    let index_coordinate_that_makes_path_not_found =
        binary_search_on_answer(0, n_coordinates - 1, &makes_path_not_found);
    if index_coordinate_that_makes_path_not_found >= coordinates.len() {
        Ok(None)
    } else {
        if makes_path_not_found(index_coordinate_that_makes_path_not_found) {
            let c = &coordinates[index_coordinate_that_makes_path_not_found];
            Ok(Some(c.clone()))
        } else {
            Ok(None)
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod test {

    use indoc::indoc;
    use lazy_static::lazy_static;

    use crate::io_help::read_lines_in_memory;

    use super::*;

    ///////////////////////////////////////////////

    const EXAMPLE_INPUT_STR: &str = indoc! {"
        5,4
        4,2
        4,5
        3,0
        2,1
        6,3
        2,4
        1,5
        0,6
        3,3
        2,6
        5,1
        1,2
        5,5
        2,5
        6,5
        1,4
        0,4
        6,4
        1,1
        6,1
        1,0
        0,5
        1,6
        2,0
    "};

    lazy_static! {
        /*
            ...#...
            ..#..#.
            ....#..
            ...#..#
            ..#..#.
            .#..#..
            #.#....
         */
        static ref EXAMPLE_EXPECTED: Memory = vec![
            vec![false, false, false, true, false, false, false],
            vec![false, false, true, false, false, true, false],
            vec![false, false, false, false, true, false, false],
            vec![false, false, false, true, false, false, true],
            vec![false, false, true, false, false, true, false],
            vec![false, true, false, false, true, false, false],
            vec![true, false, true, false, false, false, false],
        ];
    }

    ///////////////////////////////////////////////

    #[test]
    fn construction() {
        let expected: &Memory = &EXAMPLE_EXPECTED;

        let check = |actual: &Memory| {
            assert_eq!(expected.len(), actual.len(), "row check failed");
            assert_eq!(expected[0].len(), actual[0].len(), "col check failed");
            for row in 0..expected.len() {
                assert_eq!(expected[row], actual[row], "failed on row {row}");
            }
        };

        let lines = || read_lines_in_memory(EXAMPLE_INPUT_STR).take(12);

        check(&construct(Some(6), Some(6), lines()).unwrap());

        check(&construct(None, None, lines()).unwrap());
    }

    ///////////////////////////////////////////////

    #[test]
    fn dijkstras_example() {
        let memory = construct(
            Some(6),
            Some(6),
            read_lines_in_memory(EXAMPLE_INPUT_STR).take(12),
        )
        .unwrap();
        let n_steps = lowest_step_path_dijkstras(
            &memory,
            &Coordinate { row: 0, col: 0 },
            &Coordinate { row: 6, col: 6 },
        )
        .unwrap();
        assert_eq!(n_steps, 22);
    }

    #[test]
    fn pt1_soln_example() {
        assert_eq!(solution_pt1().unwrap(), 278);
    }

    ///////////////////////////////////////////////

    #[test]
    fn coordinate_to_make_no_valid_path() {
        match find_first_coordinate_unreachable_end(
            6,
            6,
            &parse_coordinates(read_lines_in_memory(EXAMPLE_INPUT_STR).take(20)).unwrap(),
        ) {
            Ok(Some(x)) => assert!(false, "not expecting to find solution: {x}"),
            Ok(None) => (),
            Err(e) => assert!(false, "expecting to find solution - error: '{e}'"),
        };
        match find_first_coordinate_unreachable_end(
            6,
            6,
            &parse_coordinates(read_lines_in_memory(EXAMPLE_INPUT_STR).take(21)).unwrap(),
        ) {
            Ok(Some(_)) => (),
            Ok(None) => assert!(false, "expecting to find a solution"),
            Err(e) => assert!(false, "not expecting to find solution - error: '{e}'"),
        };
    }

    #[test]
    fn pt2_soln_example() {
        assert_eq!(solution_pt2().unwrap(), "43,12")
    }
}
