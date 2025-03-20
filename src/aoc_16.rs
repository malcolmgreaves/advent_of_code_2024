use crate::{io_help, matrix::Matrix, utils::collect_results};

///////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Debug, PartialEq, Eq)]
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
    Err(format!("part 1 is unimplemented!"))
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

    use crate::io_help::read_lines_in_memory;

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
            vec![Tile::Wall,],
            vec![Tile::Wall,],
            vec![Tile::Wall,],
            vec![Tile::Wall,],
            vec![Tile::Wall,],
            vec![Tile::Wall,],
            vec![Tile::Wall,],
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

    lazy_static! {
        static ref EXAMPLE_EXPECTED_2: Puzzle = vec![vec![],];
    }

    ///////////////////////////////////////////////

    #[test]
    fn construction() {
        panic!();
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
