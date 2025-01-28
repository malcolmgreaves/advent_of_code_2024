use crate::{
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

type SignalMat = Matrix<Vec<State>>;

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

fn convert_line_to_row(line: &String) -> Vec<Vec<State>> {
    line.trim()
        .chars()
        .map(|c| match c {
            '.' => vec![State::Nothing],
            'a'..='z' | 'A'..='Z' | '0'..='9' => vec![State::Antenna(c)],
            _ => panic!("Unrecognized character! Must be a letter or number, not: '{c}'"),
        })
        .collect()
}

fn n_unique_antinodes(signals: &SignalMat) -> u64 {
    // In particular, an antinode occurs at any point that is perfectly in line with two antennas of the same frequency -
    // but only when one of the antennas is twice as far away as the other. This means that for any pair of antennas with
    // the same frequency, there are two antinodes, one on either side of them.
    todo!()
    // dictionary: antenna type (char) -> locations ([]Coordinate)
    // for each key:
    //      for each pair of locations -> a,b:
    //          loc = antinode_location(a,b)
    //          signals[loc] += antinode(key antenna type)
    // n = 0
    // for x in signals:
    //      for y in x:
    //          if y is antinode:
    //              n += 1
    // return n
}

///////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////

pub fn solution_pt2() -> u64 {
    let lines = io_help::read_lines("./inputs/8").collect::<Vec<String>>();
    let signals = construct_signal_matrix(&lines);
    todo!()
}

///////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod test {
    use indoc::indoc;
    use io_help::read_lines_in_memory;
    use lazy_static::lazy_static;

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

    lazy_static! {
        static ref EXAMPLE_INPUT: SignalMat = vec![
            // ............
            vec![
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing]
            ],
            // ........0...
            vec![
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Antenna('0')],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing]
            ],
            // .....0......
            vec![
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Antenna('0')],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing]
            ],
            vec![
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Antenna('0')],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
            ],
            // ....0.......
            vec![
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Antenna('0')],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing]
            ],
            // ......A.....
            vec![
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Antenna('A')],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing]
            ],
            // ............
            vec![
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing]
            ],
            // ............
            vec![
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing]
            ],
            // ........A...
            vec![
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Antenna('A')],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing]
            ],
            // .........A..
            vec![
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Antenna('A')],
                vec![State::Nothing],
                vec![State::Nothing]
            ],
            // ............
            vec![
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing]
            ],
            // ............
            vec![
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing],
                vec![State::Nothing]
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

    #[test]
    fn pt1_soln_example() {
        panic!()
    }

    #[test]
    fn pt2_soln_example() {
        panic!()
    }
}
