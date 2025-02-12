use std::collections::{HashMap, HashSet};

use crate::{
    io_help,
    utils::{cardinal_neighbors, exterior_perimiter, trace_perimiter, Coordinate, Matrix},
};

///////////////////////////////////////////////////////////////////////////////////////////////////

pub fn solution_pt1() -> u64 {
    let lines = io_help::read_lines("./inputs/12").collect::<Vec<String>>();
    let garden = construct(&lines);
    let regions = determine_regions(&garden);
    cost(&regions)
}

pub fn solution_pt2() -> u64 {
    let lines = io_help::read_lines("./inputs/12").collect::<Vec<String>>();
    let garden = construct(&lines);
    let regions = determine_regions(&garden);
    cost_sides(&garden, &regions)
}

fn cost_sides(garden: &Garden, regions: &[Region]) -> u64 {
    regions
        .iter()
        .fold(0, |s, region| s + cost_sides_region(garden, &region))
}

fn cost_sides_region(garden: &Garden, region: &Region) -> u64 {
    let n_sides = count_sides(&garden, &region);
    let price = n_sides * region.area;
    price
}

fn count_sides(garden: &Garden, region: &Region) -> u64 {
    // let membership: Matrix<bool> = {
    //     let in_region = region.members.iter().collect::<HashSet<_>>();
    //     garden
    //         .iter()
    //         .enumerate()
    //         .map(|(row, r)| {
    //             r.iter()
    //                 .enumerate()
    //                 .map(|(col, _)| in_region.contains(&Coordinate { row, col }))
    //                 .collect()
    //         })
    //         .collect()
    // };

    /*

    to get sides:

    get all pairs such that they are exactly 1 distance apart in up/down or left/right


     */

    let exterior = {
        let mut e = trace_perimiter(garden, &region.members);
        e.sort();
        e
    };

    let create_2coords = |row_view: bool| -> HashMap<usize, Vec<&Coordinate>> {
        let mut m = HashMap::<usize, Vec<&Coordinate>>::new();
        exterior.iter().for_each(|c| {
            let v = if row_view { c.row } else { c.col };
            match m.get_mut(&v) {
                Some(existing) => existing.push(c),
                None => {
                    m.insert(v, vec![c]);
                }
            }
        });
        m
    };

    let perform_merge_split = |row_view: bool, coords: &[&Coordinate]| -> Vec<Vec<&Coordinate>> {
        if coords.len() == 0 {
            panic!("Cannot handle empty coordinates list!");
        }
        if coords.len() == 1 {
            return vec![vec![*coords.first().unwrap()]];
        }

        let coords = {
            let mut cs = coords.to_vec();
            let cmp = if row_view {
                |a: &&Coordinate, b: &&Coordinate| a.row.cmp(&b.row)
            } else {
                |a: &&Coordinate, b: &&Coordinate| a.col.cmp(&b.col)
            };
            cs.sort_by(cmp);
            cs
        };

        let mut new = Vec::new();
        let mut current = vec![coords[0]];

        for c in &coords[1..coords.len()] {
            match current.last() {
                Some(x) => {
                    let (a, b) = if row_view {
                        (x.row, c.row)
                    } else {
                        (x.col, c.col)
                    };
                    if a.abs_diff(b) == 1 {
                        current.push(c);
                    } else {
                        new.push(current);
                        current = vec![c];
                    }
                }
                None => panic!(),
            }
        }

        new.push(current);
        new
    };

    let create_final = |row_view: bool| -> HashMap<usize, Vec<Vec<&Coordinate>>> {
        create_2coords(row_view)
            .iter()
            .map(|(v, coords)| (*v, perform_merge_split(row_view, coords)))
            .collect::<HashMap<_, _>>()
    };

    let row2finals = create_final(true);
    let col2finals = create_final(false);

    let sum_sides = |finals: &HashMap<_, Vec<Vec<&Coordinate>>>| -> u64 {
        finals.iter().fold(0, |s, (_, groups)| {
            // each element of groups is a side
            s + (groups.len() as u64)
        })
    };

    let horizontal_sides = sum_sides(&row2finals);
    let vertical_sides = sum_sides(&col2finals);
    horizontal_sides + vertical_sides
}

impl HasCharacter for char {
    fn character(&self) -> char {
        *self
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

type Garden = Matrix<char>;

fn construct(lines: &[String]) -> Garden {
    assert_ne!(lines.len(), 0);
    lines.iter().map(|l| l.chars().collect()).collect()
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Region {
    letter: char,
    area: u64,
    perimiter: u64,
    members: Vec<Coordinate>,
}

impl Region {
    fn price(&self) -> u64 {
        self.area * self.perimiter
    }

    fn new<T>(mat: &Matrix<T>, letter: char, members: Vec<Coordinate>) -> Self {
        let area = members.len() as u64;
        let perimiter = exterior_perimiter(mat, &members);
        Self {
            letter,
            area,
            perimiter,
            members,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum State {
    Building(char),
    Finished(char),
}

impl HasCharacter for State {
    fn character(&self) -> char {
        match self {
            Self::Building(c) | Self::Finished(c) => *c,
        }
    }
}

fn determine_regions(garden: &Garden) -> Vec<Region> {
    assert_ne!(garden.len(), 0);

    let mut region_builder: Matrix<State> = garden
        .iter()
        .map(|r| r.iter().map(|c| State::Building(*c)).collect())
        .collect();

    let mut regions = Vec::new();

    let mut unfinished_business = true;
    while unfinished_business {
        unfinished_business = false;
        for row in 0..garden.len() {
            for col in 0..garden[0].len() {
                let val = region_builder[row][col].character();
                match expanded_neighborhood(&region_builder, row, col) {
                    FloodFill::New(available) => {
                        unfinished_business = true;
                        for Coordinate { row: r, col: c } in available.iter() {
                            region_builder[*r][*c] = State::Finished(val);
                        }
                        let region = Region::new(&garden, val, available);
                        regions.push(region);
                    }
                    FloodFill::Solo => {
                        region_builder[row][col] = State::Finished(val);
                        let region = Region::new(&garden, val, vec![Coordinate { row, col }]);
                        regions.push(region)
                    }
                    FloodFill::Prefilled => {
                        // we have already included this position in a previously obtained FloodFill::New(..) result
                        // we kept track of this by marking this visisted position with a State::Finished
                        // when we call expanded_neighborhood(..) on a Finished, we get this Prefilled
                        continue;
                    }
                }
            }
        }
    }

    regions
}

trait HasCharacter {
    fn character(&self) -> char;
}

/// Neighbors: up, below, left, and right of (row, col) while still being in-bounds.
fn immediate_neighbors<T: HasCharacter>(
    region_builder: &Matrix<T>,
    row: usize,
    col: usize,
) -> Vec<Coordinate> {
    let center_character = region_builder[row][col].character();

    cardinal_neighbors(region_builder, row, col)
        .iter()
        .flat_map(|x| match x {
            (Some(new_row), Some(new_col)) => {
                // this character check ensures we're only considering compatible positions
                if region_builder[*new_row][*new_col].character() == center_character {
                    Some(Coordinate::new(*new_row, *new_col))
                } else {
                    None
                }
            }
            _ => None,
        })
        .collect::<Vec<_>>()
}

enum FloodFill {
    Solo,
    Prefilled,
    New(Vec<Coordinate>),
}

/// The entire contigious neighborhood from (row,col) that are all in-bounds and have the same character.
fn expanded_neighborhood(region_builder: &Matrix<State>, row: usize, col: usize) -> FloodFill {
    if let State::Finished(_) = region_builder[row][col] {
        return FloodFill::Prefilled;
    }

    let mut neighborhood = HashSet::<Coordinate>::new();
    neighborhood.extend(immediate_neighbors(region_builder, row, col));
    if neighborhood.len() == 0 {
        return FloodFill::Solo;
    }

    let mut inserted_once = true;
    while inserted_once {
        inserted_once = false;
        let current_neighbors = neighborhood.iter().map(|x| x.clone()).collect::<Vec<_>>();
        current_neighbors.into_iter().for_each(|n| {
            for neigh in immediate_neighbors(region_builder, n.row, n.col) {
                if !neighborhood.contains(&neigh) {
                    inserted_once = true;
                    neighborhood.insert(neigh);
                }
            }
        });
    }

    FloodFill::New(neighborhood.into_iter().collect())
}

// fn neighborhood(region_builder: &Matrix<State>, row: usize, col: usize) -> Option<Vec<Coordinate>> {
//     let character_at = |row: usize, col: usize| -> Option<char> {
//         match region_builder[row][col] {
//             State::Building(c) => Some(c),
//             State::Finished(_) => None,
//         }
//     };

//     let char_at_center = match character_at(row, col) {
//         Some(c) => c,
//         None => return None,
//     };

//     /*
//                   (row-1, col)
//                  ---------------
//     (row, col-1)| (row,  col) |  (row, col+1)
//                  ---------------
//                   (row+1, col)
//       */
//     let neighbor_positions = vec![
//         (sub_row(row), Some(col)),
//         (Some(row), sub_col(col)),
//         (Some(row), add_col(region_builder, col)),
//         (add_row(region_builder, row), Some(col)),
//     ]
//     .iter()
//     .flat_map(|x| match x {
//         (Some(new_row), Some(new_col)) => Some(Coordinate {
//             row: *new_row,
//             col: *new_col,
//         }),
//         _ => None,
//     })
//     .collect::<Vec<_>>();

//     if neighbor_positions.len() == 0 {
//         return None;
//     }

//     let neighbor_positions = neighbor_positions
//         .into_iter()
//         .filter(
//             |coordinate| match character_at(coordinate.row, coordinate.col) {
//                 Some(char_coordinate) => char_coordinate == char_at_center,
//                 None => false,
//             },
//         )
//         .collect::<Vec<_>>();
//     if neighbor_positions.len() == 0 {
//         return None;
//     }
//     Some(neighbor_positions)
// }

fn cost(regions: &[Region]) -> u64 {
    regions.iter().fold(0, |s, r| s + r.price())
}

///////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod test {

    use std::{cmp::Ordering, collections::HashMap};

    use indoc::indoc;
    use lazy_static::lazy_static;

    use crate::io_help::read_lines_in_memory;

    use super::*;

    const EXAMPLE_INPUT_STR_SM: &str = indoc! {"
        AAAA
        BBCD
        BBCC
        EEEC
    "};

    const EXAMPLE_INPUT_STR_2P: &str = indoc! {"
        OOOOO
        OXOXO
        OOOOO
        OXOXO
        OOOOO
    "};

    const EXAMPLE_INPUT_STR_LG: &str = indoc! {"
        RRRRIICCFF
        RRRRIICCCF
        VVRRRCCFFF
        VVRCCCJFFF
        VVVVCJJCFE
        VVIVCCJJEE
        VVIIICJJEE
        MIIIIIJJEE
        MIIISIJEEE
        MMMISSJEEE
    "};

    lazy_static! {
        static ref EXAMPLE_SM: Garden = vec![
            vec!['A', 'A', 'A', 'A'],
            vec!['B', 'B', 'C', 'D'],
            vec!['B', 'B', 'C', 'C'],
            vec!['E', 'E', 'E', 'C'],
        ];
        static ref EXAMPLE_2P: Garden = vec![
            vec!['O', 'O', 'O', 'O', 'O'],
            vec!['O', 'X', 'O', 'X', 'O'],
            vec!['O', 'O', 'O', 'O', 'O'],
            vec!['O', 'X', 'O', 'X', 'O'],
            vec!['O', 'O', 'O', 'O', 'O'],
        ];
        static ref EXAMPLE_LG: Garden = vec![
            vec!['R', 'R', 'R', 'R', 'I', 'I', 'C', 'C', 'F', 'F'],
            vec!['R', 'R', 'R', 'R', 'I', 'I', 'C', 'C', 'C', 'F'],
            vec!['V', 'V', 'R', 'R', 'R', 'C', 'C', 'F', 'F', 'F'],
            vec!['V', 'V', 'R', 'C', 'C', 'C', 'J', 'F', 'F', 'F'],
            vec!['V', 'V', 'V', 'V', 'C', 'J', 'J', 'C', 'F', 'E'],
            vec!['V', 'V', 'I', 'V', 'C', 'C', 'J', 'J', 'E', 'E'],
            vec!['V', 'V', 'I', 'I', 'I', 'C', 'J', 'J', 'E', 'E'],
            vec!['M', 'I', 'I', 'I', 'I', 'I', 'J', 'J', 'E', 'E'],
            vec!['M', 'I', 'I', 'I', 'S', 'I', 'J', 'E', 'E', 'E'],
            vec!['M', 'M', 'M', 'I', 'S', 'S', 'J', 'E', 'E', 'E'],
        ];
    }

    fn into_lines(contents: &str) -> Vec<String> {
        read_lines_in_memory(contents).collect::<Vec<_>>()
    }

    #[test]
    fn construction() {
        let actual = construct(&into_lines(EXAMPLE_INPUT_STR_2P));
        let expected: &Garden = &EXAMPLE_2P;
        assert_eq!(actual, *expected);
        //
        let actual = construct(&into_lines(EXAMPLE_INPUT_STR_SM));
        let expected: &Garden = &EXAMPLE_SM;
        assert_eq!(actual, *expected);
        //
        let actual = construct(&into_lines(EXAMPLE_INPUT_STR_LG));
        let expected: &Garden = &EXAMPLE_LG;
        assert_eq!(actual, *expected);
    }

    #[test]
    fn regions() {
        regions_test(&EXAMPLE_2P, vec![('X', 1, 4), ('O', 21, 36)]);
        regions_test(
            &EXAMPLE_SM,
            vec![
                ('A', 4, 10),
                ('B', 4, 8),
                ('C', 4, 10),
                ('D', 1, 4),
                ('E', 3, 8),
            ],
        );
    }

    #[test]
    fn price() {
        price_test(
            &EXAMPLE_2P,
            vec![('X', 4), ('X', 4), ('X', 4), ('X', 4), ('O', 756)],
        );
        price_test(
            &EXAMPLE_SM,
            vec![('A', 40), ('B', 32), ('C', 40), ('D', 4), ('E', 24)],
        );
        price_test(
            &EXAMPLE_LG,
            vec![
                ('R', 216),
                ('I', 32),
                ('C', 392),
                ('F', 180),
                ('V', 260),
                ('J', 220),
                ('C', 4),
                ('E', 234),
                ('I', 308),
                ('M', 60),
                ('S', 24),
            ],
        );
    }

    fn price_test(garden: &Garden, expected_prices: Vec<(char, u64)>) {
        let compare =
            |(a_char, a_price): &(char, u64), (b_char, b_price): &(char, u64)| -> Ordering {
                match a_char.cmp(&b_char) {
                    Ordering::Equal => a_price.cmp(&b_price),
                    other => other,
                }
            };

        let expected_prices = {
            let mut e = expected_prices.clone();
            e.sort_by(compare);
            e
        };
        let expected_cost = expected_prices.iter().fold(0, |s, (_, p)| s + *p);

        let regions = determine_regions(garden);
        let actual_region_prices = {
            let mut r: Vec<(char, u64)> = determine_regions(garden)
                .iter()
                .map(|r| (r.letter, r.price()))
                .collect();
            r.sort_by(compare);
            r
        };
        let actual_cost = cost(&regions);

        assert_eq!(actual_region_prices, expected_prices);
        assert_eq!(actual_cost, expected_cost);
    }

    fn regions_test(garden: &Garden, expected_region_info: Vec<(char, u64, u64)>) {
        for r in determine_regions(garden) {
            println!(
                "REGION: '{}': area={} perimiter={} # members: {}",
                r.letter,
                r.area,
                r.perimiter,
                r.members.len()
            );
            let cap = (r.letter, r.area, r.perimiter);
            assert!(
                expected_region_info.iter().any(|expected| cap == *expected),
                "expecting {cap:?} to be one of {}: {expected_region_info:?}",
                expected_region_info.len(),
            )
        }
    }

    #[test]
    fn pt1_soln_example() {
        assert_eq!(solution_pt1(), 1473620);
    }

    #[test]
    fn pt2_soln_example() {
        panic!();
    }
}
