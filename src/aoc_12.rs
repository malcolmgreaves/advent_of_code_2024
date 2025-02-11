use std::collections::HashMap;

use crate::{
    io_help,
    utils::{Coordinate, Matrix},
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
    panic!();
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
}

fn determine_regions(garden: &Garden) -> Vec<Region> {
    assert_ne!(garden.len(), 0);

    /*

       enum State {
           Island(char),
           Building(char),
           Finished(char),
       }


       region_builder = copy(garden, |c| State::Island{c} )

       let mut unfinished_business = true;

       while unfinished_buinsess {
           unfinished_business = false;

           for row in 0..garden.len() {
               for col in 0..garden[0].len() {

                   match region_builder[row][col] {

                       State::Island(c) => {
                           let available_neighborhood = neighborhood(&garden, &region_builder, row, col);
                           if available_neighborhood.len() == 0 {
                               region_builder[row][col] = State::Finished(c);
                           } else {
                               unfinished_business = true;
                               region_builder[row][col] = State::Building(c);
                               for (neighbor_row, neighbor_col) in available_neighborhood {
                                   region_builder[neighbor_row][neighbor_col] = State::Building(c);
                               }
                           }
                       },

                       State::Building(c) => {
                           let available_neighborhood = neighborhood(&garden, &region_builder, row, col);
                           if available_neighborhood.len() == 0 {
                               region_builder[row][col] = State::Finished(c);
                           } else {
                               unfinished_business = true;
                               for (neighbor_row, neighbor_col) in available_neighborhood {
                                   region_builder[neighbor_row][neighbor_col] = State::Building(c);
                               }
                           }

                       },

                       _ => (),
                   }
               }
           }
       }


    */

    let mut region_builder = HashMap::<char, Vec<Coordinate>>::new();
    // initialize
    garden.iter().enumerate().for_each(|(row, r)| {
        r.iter().enumerate().for_each(|(col, c)| {
            let coordinate = Coordinate { row, col };
            match region_builder.get_mut(c) {
                Some(members) => members.push(coordinate),
                None => {
                    region_builder.insert(*c, vec![coordinate]);
                }
            }
        })
    });

    panic!()
}

fn cost(regions: &[Region]) -> u64 {
    regions.iter().fold(0, |s, r| s + r.price())
}

///////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod test {

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
        let actual = construct(&into_lines(EXAMPLE_INPUT_STR_SM));
        let expected: &Garden = &EXAMPLE_SM;
        assert_eq!(actual, *expected);
        //
        let actual = construct(&into_lines(EXAMPLE_INPUT_STR_2P));
        let expected: &Garden = &EXAMPLE_2P;
        assert_eq!(actual, *expected);
        //
        let actual = construct(&into_lines(EXAMPLE_INPUT_STR_LG));
        let expected: &Garden = &EXAMPLE_LG;
        assert_eq!(actual, *expected);
    }

    #[test]
    fn pt1_soln_example() {
        panic!();
    }

    #[test]
    fn pt2_soln_example() {
        panic!();
    }
}
