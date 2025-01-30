use std::collections::HashMap;

use crate::{
    geometry::{antinode_location, antinode_location_resonant_harmonics, AntinodeLoc},
    io_help,
    utils::{Coordinate, Matrix},
};

#[derive(Clone, Debug, PartialEq, Eq)]
enum State {
    Nothing,
    Antenna(char),
    Antinote { antenna: char, ant_loc: Coordinate },
}

impl Default for State {
    fn default() -> Self {
        State::Nothing
    }
}

type SignalMat = Matrix<State>;

type FullSignalMat = Matrix<Vec<State>>;

///////////////////////////////////////////////////////////////////////////////////////////////////

pub fn solution_pt1() -> u64 {
    // How many unique locations within the bounds of the map contain an antinode?
    let lines = io_help::read_lines("./inputs/8").collect::<Vec<String>>();
    let signals = construct_signal_matrix(&lines);
    n_unique_antinodes(&signals)
}

fn construct_signal_matrix(lines: &[String]) -> SignalMat {
    lines.iter().map(convert_line_to_row).collect()
}

fn convert_line_to_row(line: &String) -> Vec<State> {
    line.trim()
        .chars()
        .map(|c| match c {
            '.' => State::Nothing,
            'a'..='z' | 'A'..='Z' | '0'..='9' => State::Antenna(c),
            // '#' => {
            //     println!("WARNING: found antinode without associated antenna information. Using fake name ('-') and coordinates (usize::MAX={})", usize::MAX);
            //     State::Antinote { antenna: '-', ant_loc: Coordinate { row: usize::MAX, col: usize::MAX } }
            // },
            _ => panic!("Unrecognized character! Must be a letter or number, not: '{c}'"),
        })
        .collect()
}

fn n_unique_antinodes(signals: &SignalMat) -> u64 {
    // How many unique locations within the bounds of the map contain an antinode?

    // In particular, an antinode occurs at any point that is perfectly in line with two antennas of the same frequency -
    // but only when one of the antennas is twice as far away as the other. This means that for any pair of antennas with
    // the same frequency, there are two antinodes, one on either side of them.
    let full_signals = build_full_signals(signals);
    count_unique_antinode_locations(&full_signals)
    // _count_distinct_antinode(&full_signals)
    //     .iter()
    //     .fold(0, |n_uniq, (_, n_uniq_antenna)| n_uniq + n_uniq_antenna)
}

fn count_unique_antinode_locations(full_signals: &FullSignalMat) -> u64 {
    full_signals.iter().enumerate().fold(0, |c, (i, row)| {
        row.iter().enumerate().fold(c, |count, (j, states)| {
            let at_least_one_antinode_here = states.iter().any(|state| match state {
                State::Antinote { antenna, ant_loc } => {
                    // println!(
                    //     "[{}] ({i},{j}) has antinode for {antenna} @ ({},{})",
                    //     count + 1,
                    //     ant_loc.row,
                    //     ant_loc.col
                    // );
                    true
                }
                _ => false,
            });
            if at_least_one_antinode_here {
                count + 1
            } else {
                count
            }
        })
    })

    // n = 0
    // for x in signals:
    //      for y in x:
    //          if y is antinode:
    //              n += 1
    // return n
}

fn build_full_signals(signals: &SignalMat) -> FullSignalMat {
    let max_rows = signals.len();
    let max_cols = signals[0].len();

    // dictionary: antenna type (char) -> locations ([]Coordinate)
    let ant2loc = antenna_to_locs(signals);

    let mut builder: FullSignalMat = initialize_full_signal_matrix(signals);
    let mut update = |antenna: char, ant_loc: Coordinate, og_loc: &Coordinate| {
        // match builder
        // .get_mut(ant_loc.row)
        // .unwrap()
        // .get_mut(ant_loc.col) {
        //     None => panic!("ERROR: antinode column doesn't exist! coordinate: {},{}", ant_loc.row, ant_loc.col),
        //     x => x,
        // }
        // .unwrap()
        // .push(State::Antinote { antenna, ant_loc });
        builder
            .get_mut(ant_loc.row)
            .unwrap()
            .get_mut(ant_loc.col)
            .unwrap()
            .push(State::Antinote {
                antenna,
                ant_loc: og_loc.clone(),
            });
    };

    // for each key in dictionary:
    //      for each pair of locations -> a,b:
    //          loc = antinode_location(a,b)
    //          full_signals[loc] += antinode(key antenna type)
    ant2loc.iter().for_each(|(antenna, locations)| {
        (0..locations.len()).for_each(|i| {
            let line_source = locations.get(i).unwrap();
            (i + 1..locations.len()).for_each(|j| {
                let line_destination = locations.get(j).unwrap();
                match antinode_location(max_rows, max_cols, line_source, line_destination) {
                    AntinodeLoc::OutOfBounds => (),
                    AntinodeLoc::TwoLocations { src, dst } => {
                        update(*antenna, src, line_source);
                        update(*antenna, dst, line_destination);
                    }
                    AntinodeLoc::OneLocationSrc(antinode) => {
                        update(*antenna, antinode, line_source)
                    }
                    AntinodeLoc::OneLocationDst(antinode) => {
                        update(*antenna, antinode, line_destination)
                    }
                }
            });
        });
    });

    builder
}

fn build_full_signals_resonant_harmonics(signals: &SignalMat) -> FullSignalMat {
    let max_rows = signals.len();
    let max_cols = signals[0].len();

    // dictionary: antenna type (char) -> locations ([]Coordinate)
    let ant2loc = antenna_to_locs(signals);

    let mut builder: FullSignalMat = initialize_full_signal_matrix(signals);
    let mut update = |antenna: char, ant_loc: Coordinate, og_loc: &Coordinate| {
        // match builder
        // .get_mut(ant_loc.row)
        // .unwrap()
        // .get_mut(ant_loc.col) {
        //     None => panic!("ERROR: antinode column doesn't exist! coordinate: {},{}", ant_loc.row, ant_loc.col),
        //     x => x,
        // }
        // .unwrap()
        // .push(State::Antinote { antenna, ant_loc });
        builder
            .get_mut(ant_loc.row)
            .unwrap()
            .get_mut(ant_loc.col)
            .unwrap()
            .push(State::Antinote {
                antenna,
                ant_loc: og_loc.clone(),
            });
    };

    // for each key in dictionary:
    //      for each pair of locations -> a,b:
    //          loc = antinode_location(a,b)
    //          full_signals[loc] += antinode(key antenna type)
    ant2loc.iter().for_each(|(antenna, locations)| {
        (0..locations.len()).for_each(|i| {
            let line_source = locations.get(i).unwrap();
            (i + 1..locations.len()).for_each(|j| {
                let line_destination = locations.get(j).unwrap();
                antinode_location_resonant_harmonics(
                    max_rows,
                    max_cols,
                    line_source,
                    line_destination,
                )
                .into_iter()
                .for_each(|x| update(*antenna, x, line_source));
            });
        });
    });

    builder
}

fn initialize_full_signal_matrix(signals: &SignalMat) -> FullSignalMat {
    signals
        .iter()
        .map(|row| row.iter().map(|state| vec![state.clone()]).collect())
        .collect()
}

fn _count_distinct_antinode(full_signals: &FullSignalMat) -> HashMap<char, u64> {
    // n = 0
    // for x in signals:
    //      for y in x:
    //          if y is antinode:
    //              n += 1
    // return n
    full_signals
        .iter()
        .flat_map(|row| {
            row.iter().flat_map(|states| {
                states.iter().flat_map(|state| match state {
                    State::Antinote {
                        antenna,
                        ant_loc: _,
                    } => Some(antenna),
                    _ => None,
                })
            })
        })
        .fold(HashMap::new(), |mut cd, antinode| {
            match cd.get_mut(antinode) {
                Some(n_unique_for_antinode) => {
                    *n_unique_for_antinode += 1;
                }
                None => {
                    cd.insert(*antinode, 1);
                }
            }
            cd
        })
}

fn antenna_to_locs(signals: &SignalMat) -> HashMap<char, Vec<Coordinate>> {
    signals.iter().enumerate().flat_map(|(row, state_row)| {
        state_row.iter().enumerate().flat_map(move |(col, position)| match position {
            State::Antenna(c) => Some((c, Coordinate{row, col})),
            State::Nothing => None,
            invalid => panic!("No antinodes should be present! Only Antennas & Nothing, but found: {invalid:?}"),
        })
    }).fold(HashMap::<char, Vec<Coordinate>>::new(), |mut a2ls, (antenna, location)| {
        match a2ls.get_mut(antenna) {
            Some(coordinates) => coordinates.push(location),
            None => {
                a2ls.insert(*antenna, vec![location]);
            }
        }
        a2ls
    })
}

///////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////

pub fn solution_pt2() -> u64 {
    let lines = io_help::read_lines("./inputs/8").collect::<Vec<String>>();
    let signals: SignalMat = construct_signal_matrix(&lines);
    n_unique_antinodes_resonant_harmonics(&signals)
}

fn n_unique_antinodes_resonant_harmonics(signals: &SignalMat) -> u64 {
    let antinodes_resonant_harmonics: FullSignalMat =
        build_full_signals_resonant_harmonics(signals);
    count_unique_antinode_locations(&antinodes_resonant_harmonics)
}

///////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod test {

    use std::fmt::Display;

    use indoc::indoc;
    use io_help::read_lines_in_memory;
    use lazy_static::lazy_static;

    use crate::set_ops::{check_same_elements, CheckSameElements};

    use super::*;

    const EXAMPLE_INPUT_STR: &str = indoc! {"
        ............
        ........0...
        .....0......
        .......0....
        ....0.......
        ......A.....
        ............
        ............
        ........A...
        .........A..
        ............
        ............
    "};

    #[test]
    fn expected_example_output() {
        //} -> FullSignalMat {

        let s = indoc! { "
            ......#....#
            ...#....0...
            ....#0....#.
            ..#....0....
            ....0....#..
            .#....A.....
            ...#........
            #......#....
            ........A...
            .........A..
            ..........#.
            ..........#.
        "};

        let handle = |i: usize, row: &String| -> Vec<Coordinate> {
            row.chars()
                .enumerate()
                .flat_map(|(j, c)| {
                    if c == '#' {
                        Some(Coordinate { row: i, col: j })
                    } else {
                        None
                    }
                })
                .collect()
        };

        let lines = read_lines_in_memory(s).collect::<Vec<_>>();

        let antinode_locs = {
            let mut x = lines
                .iter()
                .enumerate()
                .flat_map(|(i, row)| handle(i, row))
                .collect::<Vec<_>>();
            x.push(Coordinate { row: 5, col: 6 });
            x
        };

        for a in antinode_locs.iter() {
            println!("{a:?}");
        }
        assert_eq!(antinode_locs.len(), 14);
    }

    lazy_static! {
        static ref EXAMPLE_INPUT: SignalMat = vec![
            // ............
            vec![
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing
            ],
            // ........0...
            vec![
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Antenna('0'),
                State::Nothing,
                State::Nothing,
                State::Nothing
            ],
            // .....0......
            vec![
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Antenna('0'),
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing
            ],
            vec![
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Antenna('0'),
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
            ],
            // ....0.......
            vec![
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Antenna('0'),
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing
            ],
            // ......A.....
            vec![
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Antenna('A'),
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing
            ],
            // ............
            vec![
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing
            ],
            // ............
            vec![
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing
            ],
            // ........A...
            vec![
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Antenna('A'),
                State::Nothing,
                State::Nothing,
                State::Nothing
            ],
            // .........A..
            vec![
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Antenna('A'),
                State::Nothing,
                State::Nothing
            ],
            // ............
            vec![
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing
            ],
            // ............
            vec![
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing,
                State::Nothing
            ],
        ];
    }

    #[test]
    fn construct() {
        let lines = read_lines_in_memory(EXAMPLE_INPUT_STR).collect::<Vec<_>>();
        let created = construct_signal_matrix(&lines);
        let example: &SignalMat = &EXAMPLE_INPUT;
        assert_eq!(created, *example);
    }

    fn str_to_signal_matrix(example_str: &str) -> SignalMat {
        let lines = read_lines_in_memory(example_str).collect::<Vec<_>>();
        construct_signal_matrix(&lines)
    }

    #[test]
    fn first_example_n_unique_antinodes() {
        let example_str = indoc! { "
            ..........
            ..........
            ....a.....
            ..........
            .....a....
            ..........
            ..........
        "};
        let actual = n_unique_antinodes(&str_to_signal_matrix(example_str));
        assert_eq!(actual, 2);
    }

    #[test]
    fn first_example_1_in_bounds_1_out() {
        let example_str = indoc! { "
            ..........
            ....a.....
            ..........
            .....a....
            ..........
            ..........
        "};
        let actual = n_unique_antinodes(&str_to_signal_matrix(example_str));
        assert_eq!(actual, 1);
    }

    #[test]
    fn first_example_both_out_bounds() {
        let example_str = indoc! { "
            ..........
            ....a.....
            ..........
            .....a....
            ..........
        "};
        let actual = n_unique_antinodes(&str_to_signal_matrix(example_str));
        assert_eq!(actual, 0);
    }

    #[test]
    fn second_eaxmple_multiple_antennas_one_source() {
        let example_str = indoc! {"
            ..........
            ..........
            ....a.....
            ........a.
            .....a....
            ..........
            ..........
        "};
        let actual = n_unique_antinodes(&str_to_signal_matrix(example_str));
        assert_eq!(actual, 4);
    }

    #[test]
    fn pt1_soln_example() {
        let example: &SignalMat = &EXAMPLE_INPUT;

        let expected = vec![
            Coordinate { row: 0, col: 6 },
            Coordinate { row: 0, col: 11 },
            Coordinate { row: 1, col: 3 },
            Coordinate { row: 2, col: 4 },
            Coordinate { row: 2, col: 10 },
            Coordinate { row: 3, col: 2 },
            Coordinate { row: 4, col: 9 },
            Coordinate { row: 5, col: 1 },
            Coordinate { row: 5, col: 6 },
            Coordinate { row: 6, col: 3 },
            Coordinate { row: 7, col: 0 },
            Coordinate { row: 7, col: 7 },
            Coordinate { row: 10, col: 10 },
            Coordinate { row: 11, col: 10 },
        ];

        if n_unique_antinodes(example) != expected.len() as u64 {
            let actual = build_full_signals(example)
                .iter()
                .enumerate()
                .flat_map(|(i, row)| {
                    row.iter().enumerate().flat_map(move |(j, states)| {
                        states.iter().flat_map(move |s| match s {
                            State::Antinote {
                                antenna: antenna,
                                ant_loc: ant_loc,
                            } => {
                                if i == 1 && j == 3 {
                                    println!(
                                        "DEBUG ({i},{j}): source antenna: {antenna} @ {ant_loc}"
                                    );
                                }
                                Some(Coordinate { row: i, col: j })
                            }
                            _ => None,
                        })
                    })
                })
                .collect::<Vec<_>>();

            match check_same_elements(&actual, &expected) {
                CheckSameElements::Equal => println!("SUCCESS! found {} and expected {}", actual.len(), expected.len()),
                CheckSameElements::ActualExtra(extra) => assert!(false, "[Extra] Expected {} elements: {},\nbut actual has {} extra: {}", expected.len(), pp(expected), actual.len(), pp(extra)),
                CheckSameElements::ActualMissing(missing) => assert!(false, "[Missing] Expected {} elements: {},\nbut actual has {} and is missing {}: {}", expected.len(), pp(expected), actual.len(), missing.len(), pp(missing)),
                CheckSameElements::ActualProblems { missing, extra } => assert!(false, "[Problems] Expected {} elements: {}\nbut actual has {} -\n\tit has {} extra: {}\n\tand is missing {}: {}\n-- actual: {}", expected.len(), pp(expected), actual.len(), extra.len(), pp(extra), missing.len(), pp(missing), pp(actual)),
            }
        } else {
            println!("SUCCESS!");
        }
    }

    fn pp<T: Display>(xs: Vec<T>) -> String {
        let elements_fmt = xs
            .iter()
            .map(|t| format!("{}", *t))
            .collect::<Vec<_>>()
            .join(",");
        format!("[{elements_fmt}]")
    }

    #[test]
    fn pt2_soln_example() {
        let example_str = indoc! {"
            T.........
            ...T......
            .T........
            ..........
            ..........
            ..........
            ..........
            ..........
            ..........
            ..........
        "};
        let lines = read_lines_in_memory(example_str).collect::<Vec<_>>();
        let signals = construct_signal_matrix(&lines);
        let actual = n_unique_antinodes_resonant_harmonics(&signals);
        assert_eq!(actual, 9);
    }
}
