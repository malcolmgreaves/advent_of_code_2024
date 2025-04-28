use std::{
    collections::{HashSet, hash_set},
    fmt::Display,
};

use crate::{io_help, matrix::Matrix, utils::collect_results};

///////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Color {
    White,
    Blue,
    Black,
    Red,
    Green,
}

impl Color {
    fn code(&self) -> char {
        match self {
            Self::White => 'w',
            Self::Blue => 'u',
            Self::Black => 'b',
            Self::Red => 'r',
            Self::Green => 'g',
        }
    }

    fn parse_color(c: char) -> Result<Self, String> {
        match c {
            'w' => Ok(Self::White),
            'u' => Ok(Self::Blue),
            'b' => Ok(Self::Black),
            'r' => Ok(Self::Red),
            'g' => Ok(Self::Green),
            _ => Err(format!(
                "unrecognized color character '{c}': not a valid color"
            )),
        }
    }
}

impl Display for Color {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.code())
    }
}

// #[allow(dead_code)]
// struct VecColor<'a>(&'a Vec<&'a Color>);

// impl <'a> Display for VecColor<'a> {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         write!(f, "[{}]", self.0.iter().fold(String::new(), |mut a, x| {
//             if a.len() > 0 {
//                 a.push(',');
//             }
//             a.push(x.code());
//             a
//         }))
//     }
// }

#[allow(dead_code)]
struct VecDesign<'a>(&'a [&'a Design]);

impl<'a> Display for VecDesign<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // let as_str = self.0.iter().fold(
        //     String::with_capacity(
        //         // each color code, of each design
        //         // +
        //         // the ", " in-between
        //         self.0.iter().fold(0, |s, x| s + x.len()) + (self.0.len()-1)*2
        //     ),
        //     |mut a, x| {
        //         if a.len() > 0 {
        //             a.push(',');
        //             a.push(' ');
        //         }
        //         for d in x.iter() {
        //             a.push(d.code());
        //         }
        //         a
        //     },
        // );
        // write!(f, "[{}]", as_str)
        if self.0.len() == 0 {
            write!(f, "{}", "[]")
        } else {
            let mut first = true;
            for design in self.0.iter() {
                if !first {
                    match write!(f, "{}", ',') {
                        Ok(_) => (),
                        error => return error,
                    }
                } else {
                    match write!(f, "{}", '[') {
                        Ok(_) => (),
                        error => return error,
                    }
                    first = false;
                }
                match write!(f, "{}", VecColor(*design)) {
                    Ok(_) => (),
                    error => return error,
                }
            }
            write!(f, "{}", ']')
        }
    }
}

#[allow(dead_code)]
struct VecColor<'a>(&'a [Color]);

impl<'a> Display for VecColor<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // let as_str = self
        //     .0
        //     .iter()
        //     .fold(String::with_capacity(self.0.len()), |mut a, x| {
        //         a.push(x.code());
        //         a
        //     });
        // write!(f, "{}", as_str)
        for x in self.0.iter() {
            match write!(f, "{}", x.code()) {
                Ok(_) => (),
                error => return error,
            }
        }
        Ok(())
    }
}

type Design = Vec<Color>;

type Towel = Vec<Color>;

#[derive(Debug, Clone, PartialEq, Eq)]
struct Puzzle {
    designs: Vec<Design>,
    towels: Vec<Towel>,
}

fn construct(lines: impl Iterator<Item = String>) -> Result<Puzzle, String> {
    let ls = lines.collect::<Vec<_>>();
    if ls.len() < 3 {
        return Err(format!(
            "need at least 3 lines from input to make puzzle, not {}",
            ls.len()
        ));
    }
    let designs = match ls.get(0) {
        Some(s) => parse_designs(s)?,
        None => return Err(format!("no first line in input!")),
    };

    let towels = parse_towels(&ls.as_slice()[2..ls.len()])?;
    Ok(Puzzle { designs, towels })
}

fn parse_designs(s: &str) -> Result<Vec<Design>, String> {
    let (designs, errors) = collect_results(s.split(",").map(parse_design));
    if errors.len() > 0 {
        return Err(format!(
            "encountered {} errors when parsing '{}' as designs:\n\t{}",
            errors.len(),
            s,
            errors.join("\n\t")
        ));
    }
    Ok(designs)
}

fn parse_design(s: &str) -> Result<Design, String> {
    parse_colors(s)
}

fn parse_towels<'a>(s: &[String]) -> Result<Vec<Towel>, String> {
    let s_len = s.len();
    let (towels, errors) = collect_results(s.into_iter().map(|x| parse_towel(x.as_str())));
    if errors.len() > 0 {
        return Err(format!(
            "encountered {} errors when parsing {} lines as towels:\n\t{}",
            errors.len(),
            s_len,
            errors.join("\n\t")
        ));
    }
    Ok(towels)
}

fn parse_towel(s: &str) -> Result<Towel, String> {
    parse_colors(s)
}

fn parse_colors(s: &str) -> Result<Vec<Color>, String> {
    let (colors, errors) = collect_results(s.trim().chars().map(Color::parse_color));
    if errors.len() > 0 {
        return Err(format!(
            "[towel] failed to parse '{s}' into valid colors: found {} errors:\n\t{}",
            errors.len(),
            errors.join("\n\t")
        ));
    }
    Ok(colors)
}

type Solution<'a> = Vec<&'a Design>;

///////////////////////////////////////////////////////////////////////////////////////////////////

pub fn solution_pt1() -> Result<u64, String> {
    /*

    Structure this as the knapscak problem.

    For each `Design`, construct a matrix:

                pattern 1, pattern 2, ..., pattern M
        char 0
        char 1
        ...                          <# times that pattern is used to make up the design UP to this character>
        char N

    The answer is going to be at the last possible matrix entry.
    If it is zero, then it's not possible.

     */
    let lines = io_help::read_lines("./inputs/19");
    let puzzle = construct(lines)?;
    /*


    - at position i in the towel pattern string (e.g. i=2 of "bwurrg" is 'u')
    - we want to know which patterns start with char @ i


    => c @ towel[i] and (previous patterns)
        -> for each pattern, does pattern[0] == c?
            if so, recurse to towel[i+1] and pattern[0+1]

     */

    let n_design_solution_exists = puzzle
        .towels
        .iter()
        // .map(|t| solve_dfs_design(&puzzle.designs, t))
        .map(|t| solve_dp_design(&puzzle.designs, t))
        .filter(|s| s.is_some())
        .count();
    Ok(n_design_solution_exists as u64)
}

fn solve_dp_design<'a>(designs: &'a [Design], towel: &[Color]) -> Option<Solution<'a>> {
    /*

        brwrr
        bggr
        gbbr
        rrbgbr
        ubwu
        bwurrg
        brgr
        bbrgwb

                                          < DESIGN >
    < end    |  'r'  |  'wr' |  'b'  |  'g'  | 'bwu' |  'rb' |  'gb' |  'br' |
    position |   0   |   1   |   2   |   3   |   4   |   5   |   6   |   7   |   [k]
    -----------------------------------------------------------------------------------
       0     |   1       1       1       1       1       1       1       1
       1     |   0       0       1       0       0       0       0       0
       2     |   1       0       0       0       0       0       0       1
       3     |   0       0       0       0       0       0       0       0
       4     |   1       1       0       0       0       0       0       0
       5     |   1       0       0       0       0       0       0       0
      ...    |   0       0       0       0       0       0       0       0
       m     |   0       0       0       0       0       0       0       0

     */

    let table = fill_dp_table(designs, towel);
    // println!("towel: {}", VecColor(towel));
    // print_table(designs, &table);
    backtrack_solution(&table, designs)
}

fn fill_dp_table(designs: &[Design], towel: &[Color]) -> Matrix<bool> {
    let mut table = vec![vec![false; designs.len()]; towel.len() + 1];
    for i in 0..designs.len() {
        table[0][i] = true;
    }

    // table[i,j] => true if pattern j *COULD* end at position i+1 (i is exclusive bound) in the string
    //            => false if pattern j cannot end here

    for end in 1..=towel.len() {
        for (j, d) in designs.iter().enumerate() {
            match end.checked_sub(d.len()) {
                Some(start) => {
                    if towel[start..end] == *d {
                        table[end][j] = true;
                    }
                }
                None => (),
            }
        }
    }

    table
}

fn candidates(table: &[Vec<bool>], end: usize) -> impl Iterator<Item = usize> {
    table[end]
        .iter()
        .enumerate()
        .filter(|(_, could_end_here)| **could_end_here)
        .map(|(i, _)| i)
}

#[allow(dead_code)]
fn print_table(designs: &[Design], table: &Matrix<bool>) {
    assert!(table.len() > 0);
    assert_eq!(table[0].len(), designs.len());

    for (j, d) in designs.iter().enumerate() {
        println!(" [{j}]: {}", VecColor(d));
    }
    println!("----------------\n");

    print!(" | ");
    for j in 0..designs.len() {
        print!("{j} ");
    }
    println!("");
    for i in 0..table.len() {
        print!("{ }| ", i);
        for j in 0..designs.len() {
            if table[i][j] {
                print!("x ");
            } else {
                print!(". ");
            }
        }
        println!("");
    }
    println!("-----------------------------------------\n");
}

fn backtrack_solution<'a>(table: &Matrix<bool>, designs: &'a [Design]) -> Option<Solution<'a>> {
    let end = table.len() - 1;

    candidates(table, end)
        // flat-map: first non-empty solution will be what we
        //           compute & return since we call .next() below
        .flat_map(|idx_design| {
            let mut accum = Vec::new();
            if backtrack(table, designs, &mut accum, end, idx_design) {
                accum.push(idx_design);
                Some(accum)
            } else {
                None
            }
        })
        .next()
        .map(|accum| accum.into_iter().map(|d| &designs[d]).collect::<Vec<_>>())
}

fn backtrack(
    table: &Matrix<bool>,
    designs: &[Design],
    accum: &mut Vec<usize>,
    end: usize,
    idx_design: usize,
) -> bool {
    if end == 0 {
        true
    } else if !table[end][idx_design] {
        false
    } else if table[end][idx_design] && end == 1 {
        true
    } else {
        let d_len = designs[idx_design].len();
        match end.checked_sub(d_len) {
            Some(potential_preceeding_end) => candidates(table, potential_preceeding_end)
                .flat_map(|idx_preceeding| {
                    if backtrack(
                        table,
                        designs,
                        accum,
                        potential_preceeding_end,
                        idx_preceeding,
                    ) {
                        accum.push(idx_preceeding);
                        Some(true)
                    } else {
                        None
                    }
                })
                .next()
                .unwrap_or(false),
            None => false,
        }
    }
}

#[allow(dead_code)]
fn solve_dfs_design<'a>(designs: &'a [Design], towel: &'a [Color]) -> Option<Solution<'a>> {
    let mut rev_soln = vec![];
    if solve_dfs(designs, towel, &mut rev_soln, 0) {
        rev_soln.reverse();
        Some(rev_soln)
    } else {
        None
    }
}

#[allow(dead_code)]
fn solve_dfs<'a>(
    designs: &'a [Design],
    towel: &'a [Color],
    rev_soln: &mut Vec<&'a Design>,
    start: usize,
) -> bool {
    if start >= towel.len() {
        return true;
    }
    let remaining = &towel[start..towel.len()];
    for d in designs.iter() {
        if remaining.starts_with(d) {
            let is_solved_with_this_choice = solve_dfs(designs, towel, rev_soln, start + d.len());
            if is_solved_with_this_choice {
                rev_soln.push(d);
                return true;
            }
        }
    }
    return false;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

pub fn solution_pt2() -> Result<u64, String> {
    let lines = io_help::read_lines("./inputs/19");
    let puzzle = construct(lines)?;
    Ok(count_solutions(&puzzle))
}

fn count_solutions(puzzle: &Puzzle) -> u64 {
    puzzle
        .towels
        .iter()
        .map(|towel| {
            let solutions: Vec<Solution<'_>> = solve_full(&puzzle.designs, towel);
            solutions.len() as u64
        })
        .sum()
}

fn solve_full<'a>(designs: &'a [Design], towel: &'a [Color]) -> Vec<Solution<'a>> {
    let table = fill_dp_table(designs, towel);
    backtrack_full_solution(&table, &designs)
}

#[derive(Debug, PartialEq, Eq, Clone)]
enum CacheEntry<'a> {
    Unvisisted,
    NotViable,
    Viable(HashSet<Solution<'a>>),
}

impl<'a> CacheEntry<'a> {
    fn update(&mut self, viable: Solution<'a>) {
        match self {
            Self::Unvisisted => {
                let mut h = HashSet::new();
                h.insert(viable);
                *self = Self::Viable(h);
            }
            Self::Viable(existing) => {
                existing.insert(viable);
            }
            Self::NotViable => {
                panic!("already marked as not-viable! cannot push viable solution: {viable:?}")
            }
        }
    }
}

fn backtrack_full_solution<'a>(table: &Matrix<bool>, designs: &'a [Design]) -> Vec<Solution<'a>> {
    let end = table.len() - 1;

    // NOTE: table's .len() is already towel.len() + 1
    let mut cache: Matrix<CacheEntry<'a>> =
        vec![vec![CacheEntry::Unvisisted; designs.len()]; table.len()];

    candidates(table, end)
        // flat-map: first non-empty solution will be what we
        //           compute & return since we call .next() below
        .flat_map(|idx_design| {
            let mut complete = HashSet::new();
            if backtrack_full(
                table,
                designs,
                &mut cache,
                &mut complete,
                &mut vec![],
                end,
                idx_design,
            ) {
                println!(
                    "Success INITIAL backtrack from d={} from end={end} --> {complete:?}",
                    VecColor(&designs[idx_design]),
                    // VecDesign(&complete),
                );
            } else {
                println!(
                    "Failed INITIAL backtrack from d={} from end={end} --> {complete:?}",
                    VecColor(&designs[idx_design]),
                    // VecDesign(&complete),
                );
            }
            complete
            // match &cache[end][idx_design] {
            //     CacheEntry::Viable(fwds) => fwds.iter().map(|x| x.clone()).collect(),
            //     _ => vec![],
            // }
        })
        .collect::<HashSet<_>>()
        .into_iter()
        .collect()
}

fn append<T: Clone>(accum: &Vec<T>, next: T) -> Vec<T> {
    if accum.len() == 0 {
        vec![next]
    } else {
        let mut ax = accum.clone();
        ax.push(next);
        ax
    }
}

fn extend<T: Clone>(accum: &Vec<T>, next: impl Iterator<Item = T>) -> Vec<T> {
    let mut ax = accum.clone();
    ax.extend(next);
    ax
}

fn backtrack_full<'a>(
    table: &Matrix<bool>,
    designs: &'a [Design],
    cache: &mut Matrix<CacheEntry<'a>>,
    complete: &mut HashSet<Vec<&'a Design>>,
    accum: &Vec<usize>,
    end: usize,
    idx_design: usize,
) -> bool {
    match &cache[end][idx_design] {
        CacheEntry::Viable(forward_computed) => {
            if end == table.len() - 1 {
                println!("\t\t\t\t\t++short-circut!");
                for f in forward_computed.iter() {
                    complete.insert(f.clone());
                }
                return true;
            }
        }
        CacheEntry::NotViable => {
            println!("\t\t\t\t\t--short-circut!");
            return false;
        }
        _ => (),
    }

    let result = if end == 0 {
        println!(
            "\t\t\tSUCCESS (bottom): [{end}][d:{} -- {}] --> {:?}",
            idx_design,
            VecColor(&designs[idx_design]),
            append(accum, idx_design)
        );

        let finished = {
            // let mut ax = append(accum, idx_design);
            let mut ax = accum.clone();
            ax.reverse();
            ax.into_iter().map(|d| &designs[d]).collect::<Vec<_>>()
        };
        complete.insert(finished.clone());

        // cache[end][idx_design].update(vec![&designs[idx_design]]);

        true
    } else if !table[end][idx_design] {
        println!(
            "\t\t\tFAILED: [{end}][{}] --> cannot end here!",
            VecColor(&designs[idx_design])
        );

        cache[end][idx_design] = CacheEntry::NotViable;
        false
    } else if table[end][idx_design] && end == 1 {
        println!(
            "\t\t\tSUCCESS: made it to beginning, first design is: [{idx_design}] {} --> {:?}",
            VecColor(&designs[idx_design]),
            append(accum, idx_design),
        );

        let finished = {
            let mut ax = append(accum, idx_design);
            // let mut ax = accum.clone();
            ax.reverse();
            ax.into_iter().map(|d| &designs[d]).collect::<Vec<_>>()
        };
        complete.insert(finished.clone());

        cache[end][idx_design].update(vec![&designs[idx_design]]);

        true
    } else {
        println!(
            "<end: {end} idx_design: {idx_design} = {}>",
            VecColor(&designs[idx_design])
        );
        let d_len = designs[idx_design].len();
        match end.checked_sub(d_len) {
            Some(potential_preceeding_end) => candidates(table, potential_preceeding_end)
                .flat_map(|idx_preceeding| {
                    let attempt_accum = append(accum, idx_design);

                    match &cache[potential_preceeding_end][idx_preceeding] {
                        // CacheEntry::NotViable => {
                        //     println!("\t\t\t\tshort circut => failed path detected");
                        //     None
                        // },
                        // CacheEntry::Viable(fwds_from_here) => {
                        //     let xs = fwds_from_here.iter().map(|f| append(f, &designs[idx_design])).collect::<HashSet<_>>();
                        //     println!("\t\t\t\tshort circut => viable path detected: [{potential_preceeding_end}][{idx_preceeding}]--> {xs:?}");
                        //     cache[end][idx_design] = CacheEntry::Viable(xs);
                        //     Some(true)
                        // },
                        // CacheEntry::Unvisisted => {
                        _ => {
                            if backtrack_full(
                                table,
                                designs,
                                cache,
                                complete,
                                &attempt_accum,
                                potential_preceeding_end,
                                idx_preceeding,
                            ) {
                                let new_entires = match &cache[end][idx_preceeding] {
                                    CacheEntry::NotViable => panic!("CANNOT BE INVALID HERE!"),
                                    CacheEntry::Unvisisted => {
                                        println!("WARNING: unvisisted @ [{end}][{idx_preceeding}]");
                                        let mut h = HashSet::new();
                                        h.insert(vec![&designs[idx_design]]);
                                        h
                                    },
                                    CacheEntry::Viable(forwards) => {
                                        if forwards.is_empty() {
                                            let mut h = HashSet::new();
                                            h.insert(vec![&designs[idx_design]]);
                                            h

                                        } else {
                                            forwards
                                                .iter()
                                                .map(|fwd| append(fwd, &designs[idx_design]))
                                                .collect::<HashSet<_>>()
                                        }
                                    }
                                };
                                cache[end][idx_design] = CacheEntry::Viable(new_entires);
                                println!(
                                    "\t\tSUCCESS backtrack from d={} [{}, {end}) to new={} (end@: {potential_preceeding_end}) --> accum: {} -> CACHE design: {:?}",
                                    VecColor(&designs[idx_design]),
                                    potential_preceeding_end + 1,
                                    VecColor(&designs[idx_preceeding]),
                                    VecDesign(&accum.iter().map(|i| &designs[*i]).collect::<Vec<_>>()),
                                    // VecColor(&designs[idx_design])
                                    cache[end][idx_preceeding],
                                );
                                Some(true)
                            } else {
                                println!(
                                    "\t\tFailed backtrack from d={} [{}, {end}) to new={} (end@: {potential_preceeding_end})",
                                    VecColor(&designs[idx_design]),
                                    potential_preceeding_end + 1,
                                    VecColor(&designs[idx_preceeding]),
                                );
                                cache[end][idx_preceeding] = CacheEntry::NotViable;
                                None
                            }
                        }
                    }
                })
                .into_iter()
                .fold(false, |at_least_one_ok, check| at_least_one_ok || check),
            None => {
                println!("\t\t\tFAILED out of bounds: {end} and design: {}", VecColor(&designs[idx_design]));
                cache[end][idx_design] = CacheEntry::NotViable;
                false
            }
        }
    };
    result
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
        r, wr, b, g, bwu, rb, gb, br

        brwrr
        bggr
        gbbr
        rrbgbr
        ubwu
        bwurrg
        brgr
        bbrgwb
    "};

    lazy_static! {
        static ref EXAMPLE_EXPECTED: Puzzle = Puzzle {
            designs: vec![
                vec![Color::Red],
                vec![Color::White, Color::Red],
                vec![Color::Black],
                vec![Color::Green],
                vec![Color::Black, Color::White, Color::Blue],
                vec![Color::Red, Color::Black],
                vec![Color::Green, Color::Black],
                vec![Color::Black, Color::Red],
            ],
            towels: vec![
                vec![
                    Color::Black,
                    Color::Red,
                    Color::White,
                    Color::Red,
                    Color::Red
                ],
                vec![Color::Black, Color::Green, Color::Green, Color::Red],
                vec![Color::Green, Color::Black, Color::Black, Color::Red],
                vec![
                    Color::Red,
                    Color::Red,
                    Color::Black,
                    Color::Green,
                    Color::Black,
                    Color::Red
                ],
                vec![Color::Blue, Color::Black, Color::White, Color::Blue],
                vec![
                    Color::Black,
                    Color::White,
                    Color::Blue,
                    Color::Red,
                    Color::Red,
                    Color::Green
                ],
                vec![Color::Black, Color::Red, Color::Green, Color::Red],
                vec![
                    Color::Black,
                    Color::Black,
                    Color::Red,
                    Color::Green,
                    Color::White,
                    Color::Black
                ],
            ],
        };
    }

    ///////////////////////////////////////////////

    #[test]
    fn construction() {
        let expected: &Puzzle = &EXAMPLE_EXPECTED;
        let puzzle = construct(read_lines_in_memory(EXAMPLE_INPUT_STR)).unwrap();
        assert_eq!(*expected, puzzle)
    }

    ///////////////////////////////////////////////

    fn solve_test<'a, 'b, F>(solve: F, index: usize, expected: bool)
    where
        F: Fn(&'a [Design], &'b [Color]) -> Option<Solution<'a>>,
    {
        let puzzle: &Puzzle = &EXAMPLE_EXPECTED;
        match solve(&puzzle.designs, &puzzle.towels[index]) {
            Some(soln) => {
                if expected {
                    println!("solution: {}", VecDesign(&soln))
                } else {
                    assert!(false, "FAIL: not expecting solution: {}", VecDesign(&soln))
                }
            }
            None => {
                if expected {
                    assert!(false, "expecting solution for index={index}")
                } else {
                    println!("OK: not expecting solution for index={index}")
                }
            }
        }
    }

    fn solve_dfs_test(index: usize, expected: bool) {
        solve_test(solve_dfs_design, index, expected)
    }

    #[test]
    fn solve_dfs_test_0() {
        solve_dfs_test(0, true)
    }

    #[test]
    fn solve_dfs_test_1() {
        solve_dfs_test(1, true)
    }

    #[test]
    fn solve_dfs_test_2() {
        solve_dfs_test(2, true)
    }

    #[test]
    fn solve_dfs_test_3() {
        solve_dfs_test(3, true)
    }

    #[test]
    fn solve_dfs_test_4() {
        solve_dfs_test(4, false)
    }

    #[test]
    fn solve_dfs_test_5() {
        solve_dfs_test(5, true)
    }

    #[test]
    fn solve_dfs_test_6() {
        solve_dfs_test(6, true)
    }

    #[test]
    fn solve_dfs_test_7() {
        solve_dfs_test(7, false)
    }

    fn solve_dp_test(index: usize, expected: bool) {
        solve_test(solve_dp_design, index, expected)
    }

    #[test]
    fn solve_dp_test_0() {
        solve_dp_test(0, true)
    }

    #[test]
    fn solve_dp_test_1() {
        solve_dp_test(1, true)
    }

    #[test]
    fn solve_dp_test_2() {
        solve_dp_test(2, true)
    }

    #[test]
    fn solve_dp_test_3() {
        solve_dp_test(3, true)
    }

    #[test]
    fn solve_dp_test_4() {
        solve_dp_test(4, false)
    }

    #[test]
    fn solve_dp_test_5() {
        solve_dp_test(5, true)
    }

    #[test]
    fn solve_dp_test_6() {
        solve_dp_test(6, true)
    }

    #[test]
    fn solve_dp_test_7() {
        solve_dp_test(7, false)
    }

    #[test]
    fn pt1_soln_example() {
        assert_eq!(solution_pt1().unwrap(), 353);
    }

    ///////////////////////////////////////////////
    fn full_test<'a>(index: usize, expected_values: Vec<Vec<&str>>) {
        let expected: HashSet<Vec<Design>> = HashSet::from_iter(expected_values.iter().map(|v| {
            v.iter()
                .map(|c| parse_design(c).unwrap())
                .into_iter()
                .collect::<Vec<_>>()
        }));

        let puzzle: &Puzzle = &EXAMPLE_EXPECTED;
        let actual = solve_full(&puzzle.designs, &puzzle.towels[index]);
        println!("-----------------------");
        for s in actual.iter() {
            println!("{}", VecDesign(s));
        }
        println!("-----------------------");
        assert_eq!(
            expected,
            actual
                .into_iter()
                .map(|x| x.iter().map(|d| (*d).clone()).collect::<Vec<_>>())
                .collect::<HashSet<_>>(),
        )
    }

    #[test]
    fn solve_full_0() {
        full_test(0, vec![vec!["b", "r", "wr", "r"], vec!["br", "wr", "r"]])
    }

    #[test]
    fn solve_full_1() {
        full_test(1, vec![vec!["b", "g", "g", "r"]])
    }

    #[test]
    fn solve_full_2() {
        full_test(
            2,
            vec![
                vec!["g", "b", "b", "r"],
                vec!["g", "b", "br"],
                vec!["gb", "b", "r"],
                vec!["gb", "br"],
            ],
        )
    }

    #[test]
    fn solve_full_3() {
        full_test(
            3,
            vec![
                vec!["r", "r", "b", "g", "b", "r"],
                vec!["r", "r", "b", "g", "br"],
                vec!["r", "r", "b", "gb", "r"],
                vec!["r", "rb", "g", "b", "r"],
                vec!["r", "rb", "g", "br"],
                vec!["r", "rb", "gb", "r"],
            ],
        )
    }

    #[test]
    fn solve_full_4() {
        full_test(4, vec![])
    }

    #[test]
    fn solve_full_5() {
        full_test(5, vec![vec!["bwu", "r", "r", "g"]])
    }

    #[test]
    fn solve_full_6() {
        full_test(6, vec![vec!["b", "r", "g", "r"], vec!["br", "g", "r"]])
    }

    #[test]
    fn solve_full_7() {
        full_test(7, vec![])
    }

    #[test]
    fn solve_pt2_example() {
        let puzzle = &EXAMPLE_EXPECTED;
        let actual: u64 = count_solutions(&puzzle);
        assert_eq!(actual, 16);
    }

    #[ignore]
    #[test]
    fn pt2_soln_example() {
        panic!();
    }
}
