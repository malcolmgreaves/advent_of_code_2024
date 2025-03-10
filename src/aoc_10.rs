use std::collections::{HashMap, HashSet};

use crate::{
    io_help,
    matrix::{Coordinate, Matrix},
};

type Topo = Matrix<u8>;

type Path = Vec<Coordinate>;

fn construct(lines: &[String]) -> Topo {
    lines
        .iter()
        .map(|l| {
            l.trim()
                .chars()
                .map(|c| {
                    c.to_digit(10)
                        .map(|x| {
                            // x has to be >= 0
                            if x < 10 {
                                x as u8
                            } else {
                                panic!(
                                    "Expected number in [0,9] but got a number larger than 9: {x}"
                                )
                            }
                        })
                        .expect(
                            format!("Expected number in [0,9] but got '{c}': line: '{l}'").as_str(),
                        )
                })
                .collect()
        })
        .collect()
}

///////////////////////////////////////////////////////////////////////////////////////////////////

pub fn solution_pt1() -> u64 {
    let lines = io_help::read_lines("./inputs/10").collect::<Vec<String>>();
    let trailhead_paths = construct_and_find_all_trailhead_paths(&lines);
    score_trailhead_paths(&trailhead_paths)
}

pub fn solution_pt2() -> u64 {
    let lines = io_help::read_lines("./inputs/10").collect::<Vec<String>>();
    let trailhead_paths = construct_and_find_all_trailhead_paths(&lines);
    rate_trailhead_paths(&trailhead_paths)
}

///////////////////////////////////////////////////////////////////////////////////////////////////

fn construct_and_find_all_trailhead_paths(lines: &[String]) -> HashMap<Coordinate, Vec<Path>> {
    let topo_map: Topo = construct(&lines);
    let trailhead_paths = paths_from_trailheads(&topo_map);
    trailhead_paths
}

fn paths_from_trailheads(topo_map: &Topo) -> HashMap<Coordinate, Vec<Path>> {
    trailheads(&topo_map)
        .into_iter()
        .map(|trailhead| {
            let paths = find_hiking_paths(&topo_map, &trailhead);
            (trailhead, paths)
        })
        .collect()
}

fn trailheads(topo_map: &Topo) -> Vec<Coordinate> {
    if topo_map.len() == 0 {
        return Vec::new();
    }
    (0..topo_map.len())
        .flat_map(|row| {
            (0..topo_map[0].len()).flat_map(move |col| {
                if topo_map[row][col] == 0 {
                    Some(Coordinate { row, col })
                } else {
                    None
                }
            })
        })
        .collect()
}

fn find_hiking_paths(topo_map: &Topo, trailhead: &Coordinate) -> Vec<Path> {
    assert_eq!(topo_map[trailhead.row][trailhead.col], 0);

    let value = |loc: &Coordinate| -> u8 { topo_map[loc.row][loc.col] };

    // seed candidates
    let mut candidates: Vec<Path> = neighborhood(topo_map, trailhead)
        .into_iter()
        .map(|n| vec![n])
        .collect();

    let mut final_candidates = Vec::new();

    loop {
        if candidates.len() == 0 {
            break;
        }

        let previous_candidates = candidates.clone();
        candidates.clear();

        for working_path in previous_candidates {
            let neighbors = neighborhood(topo_map, working_path.last().unwrap());
            for new_extension in neighbors {
                // println!("\t\tnew_extension: {}", new_extension);
                let is_terminal = value(&new_extension) == 9;
                let new_candidate = {
                    let mut new = working_path.clone();
                    new.push(new_extension);
                    new
                };
                if is_terminal {
                    final_candidates.push(new_candidate)
                } else {
                    candidates.push(new_candidate);
                }
            }
        }
    }

    final_candidates
}

/// within +1 (**NOT -1**) from loc
/// in-bounds
fn neighborhood(topo_map: &Topo, loc: &Coordinate) -> Vec<Coordinate> {
    let sub = |x: usize| -> Option<usize> { if x > 0 { Some(x - 1) } else { None } };

    let add_row = |x: usize| -> Option<usize> {
        if x + 1 < topo_map.len() {
            Some(x + 1)
        } else {
            None
        }
    };

    let add_col = |x: usize| -> Option<usize> {
        if x + 1 < topo_map[0].len() {
            Some(x + 1)
        } else {
            None
        }
    };

    let Coordinate { row: r, col: c } = *loc;
    let val_loc: u8 = topo_map[r][c];
    // [
    //     [           (r-1,c),          ],
    //     [(r,  c-1),          (r,  c+1)],
    //     [           (r+1,c),          ],
    // ]
    // #[rustfmt::skip]
    [
        (sub(r), Some(c)),
        (Some(r), sub(c)),
        (Some(r), add_col(c)),
        (add_row(r), Some(c)),
    ]
    .into_iter()
    .flat_map(|(maybe_row, maybe_col)| match (maybe_row, maybe_col) {
        (Some(row), Some(col)) => Some(Coordinate { row, col }),
        _ => None,
    })
    .filter(|candidate| {
        let val_candidate = topo_map[candidate.row][candidate.col];
        val_candidate > val_loc && val_candidate - val_loc == 1
    })
    .collect()
}

fn score_trailhead_paths(trailhead_paths: &HashMap<Coordinate, Vec<Path>>) -> u64 {
    trailhead_paths.iter().map(|(_, paths)| score(paths)).sum()
}

fn score(paths: &Vec<Path>) -> u64 {
    paths
        .iter()
        .map(|path| path.last().unwrap())
        .collect::<HashSet<_>>()
        .len() as u64
}

fn rate_trailhead_paths(trailhead_paths: &HashMap<Coordinate, Vec<Path>>) -> u64 {
    trailhead_paths.iter().map(|(_, paths)| rate(paths)).sum()
}

fn rate(paths: &Vec<Path>) -> u64 {
    paths.clone().into_iter().collect::<HashSet<_>>().len() as u64
}

///////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod test {

    use indoc::indoc;

    use crate::io_help::read_lines_in_memory;

    use super::*;

    const EXAMPLE_INPUT_STR_SIMPLE: &str = indoc! {"
        0123
        1234
        8765
        9876
    "};

    const EXAMPLE_INPUT_STR_COMPLEX: &str = indoc! {"
        89010123
        78121874
        87430965
        96549874
        45678903
        32019012
        01329801
        10456732
    "};

    #[test]
    fn pt1_soln_example() {
        [
            (EXAMPLE_INPUT_STR_SIMPLE, 1),
            (EXAMPLE_INPUT_STR_COMPLEX, 36),
        ]
        .into_iter()
        .for_each(|(contents, expected)| {
            let lines = read_lines_in_memory(contents).collect::<Vec<_>>();
            let actual = score_trailhead_paths(&construct_and_find_all_trailhead_paths(&lines));
            assert_eq!(actual, expected, "contents: {}", contents);
        })
    }

    #[test]
    fn pt2_soln_example() {
        let contents = EXAMPLE_INPUT_STR_COMPLEX;
        let lines = read_lines_in_memory(contents).collect::<Vec<_>>();
        let actual = rate_trailhead_paths(&construct_and_find_all_trailhead_paths(&lines));
        assert_eq!(actual, 81, "contents: {}", contents);
    }
}
