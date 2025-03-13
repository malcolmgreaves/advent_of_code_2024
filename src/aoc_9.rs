use std::ops::Range;

use crate::io_help;

///////////////////////////////////////////////////////////////////////////////////////////////////

pub fn solution_pt1() -> u64 {
    // How many unique locations within the bounds of the map contain an antinode?
    let lines = io_help::read_lines("./inputs/9").collect::<Vec<String>>();
    solution(defrag_inplace, &lines)
}

fn solution(defragment: fn(&mut ExpandedLine) -> (), lines: &[String]) -> u64 {
    let disk_lines: Vec<DiskLine> = construct_disk_lines(lines).unwrap();
    let block_lines: Vec<BlockLine> = disk_lines
        .into_iter()
        .map(|dl| dl.into_blocks())
        .collect::<Vec<_>>();
    let defraged_disks: Vec<ExpandedLine> = block_lines
        .iter()
        .map(expand_as_indexed_blocks)
        .map(|mut expanded| {
            defragment(&mut expanded);
            expanded
        })
        .collect::<Vec<_>>();
    defraged_disks.iter().map(|x| checksum(x)).sum()
}

fn construct_disk_lines(lines: &[String]) -> Result<Vec<DiskLine>, String> {
    let (disk_lines, fails) = lines.iter().map(|l| DiskLine::parse(l)).fold(
        (Vec::new(), Vec::new()),
        |(mut disk_lines, mut fails), maybe_dl| {
            match maybe_dl {
                Ok(dl) => disk_lines.push(dl),
                Err(e) => fails.push(e),
            }
            (disk_lines, fails)
        },
    );
    if fails.len() == 0 {
        Ok(disk_lines)
    } else {
        Err(format!(
            "Constructed {} disk lines, but found {} failures:\n\t{}",
            disk_lines.len(),
            fails.len(),
            fails.join("\n\t")
        ))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DiskLine {
    elements: Vec<u8>,
}

impl DiskLine {
    fn new(elements: Vec<u8>) -> Result<Self, String> {
        if elements.len() == 0 {
            Err("Elements length must be non-empty".into())
        } else {
            Ok(DiskLine { elements })
        }
    }

    fn parse(line: &str) -> Result<Self, String> {
        let (digits, extras) = line
            .trim()
            .chars()
            .enumerate()
            .map(|(i, c)| match c.to_digit(10) {
                Some(d) => TryInto::<u8>::try_into(d)
                    .map_err(|e| format!("Failed to convert digit {} into u8: {}", d, e)),
                None => Err(format!("[{}]: '{}'", i, c)),
            })
            .fold(
                (Vec::new(), Vec::new()),
                |(mut digits, mut extras), maybe_digit| {
                    match maybe_digit {
                        Ok(d) => digits.push(d),
                        Err(e) => extras.push(e),
                    }
                    (digits, extras)
                },
            );
        if extras.len() == 0 {
            DiskLine::new(digits)
        } else {
            Err(format!(
                "Found {} digits and {} extra invalid non-digits:\n\t{}",
                digits.len(),
                extras.len(),
                extras.join("\n\t")
            ))
        }
    }

    fn into_blocks(self) -> BlockLine {
        self.elements
            .into_iter()
            .enumerate()
            .flat_map(|(i, d)| match i % 2 == 0 {
                // even means file
                true => Some(Block::File(d)),
                // odd means free space
                false => {
                    if d > 0 {
                        Some(Block::Free(d))
                    } else {
                        None
                    }
                }
            })
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Block {
    Free(u8),
    File(u8),
}

type BlockLine = Vec<Block>;

type ExpandedLine = Vec<Option<usize>>;

/// Expands blocks of Files and Free space into their memory contents.
///
/// Some(x) means a spot of memory taken up by file x.
/// None means it is free space.
fn expand_as_indexed_blocks(blocks: &BlockLine) -> ExpandedLine {
    let mut index: usize = 0;
    blocks
        .iter()
        .flat_map(|b| match b {
            Block::File(d) => {
                let contents = vec![Some(index); *d as usize];
                index += 1;
                contents
            }
            Block::Free(d) => vec![None; *d as usize],
        })
        .collect()
}

fn first_free(after: Option<usize>, expanded: &[Option<usize>]) -> Option<usize> {
    let a = after.map(|a| a + 1).unwrap_or(0);
    for (i, x) in expanded[a..expanded.len()].iter().enumerate() {
        if x.is_none() {
            return Some(i + a);
        }
    }
    None
}

fn last_occupied(before: Option<usize>, expanded: &[Option<usize>]) -> Option<usize> {
    let b = before.unwrap_or(expanded.len());
    for (i, x) in expanded[0..b].iter().enumerate().rev() {
        if x.is_some() {
            return Some(i);
        }
    }
    None
}

#[allow(dead_code)]
fn all_frees(expanded: &[Option<usize>]) -> Vec<usize> {
    let mut elements = Vec::new();
    let mut free = first_free(None, expanded);
    loop {
        free = match free {
            Some(x) => {
                elements.push(x);
                first_free(Some(x), expanded)
            }
            None => break,
        };
    }
    elements
}

#[allow(dead_code)]
fn all_occupied(expanded: &[Option<usize>]) -> Vec<usize> {
    let mut elements = Vec::new();
    let mut focus = last_occupied(None, expanded);
    loop {
        match focus {
            Some(x) => {
                elements.push(x);
                focus = last_occupied(focus, expanded);
            }
            None => break,
        }
    }
    elements.reverse();
    elements
}

pub fn free_ranges(expanded: &[Option<usize>]) -> Vec<Range<usize>> {
    let mut elements = Vec::new();
    let mut start_last_free = None;
    for (i, e) in expanded.iter().enumerate() {
        match e {
            Some(_) => push_reset(&mut elements, &mut start_last_free, i),
            None => match start_last_free {
                // in free -> in free
                Some(_) => (),
                // not in free -> in free
                None => start_last_free = Some(i),
            },
        }
    }
    // cleanup
    push_reset(&mut elements, &mut start_last_free, expanded.len());
    elements
}

pub fn occupied_ranges(expanded: &[Option<usize>]) -> Vec<Range<usize>> {
    let mut elements = Vec::new();
    let mut start_last_occupied = None;
    let mut id_last_occupied = None;

    let set = |start_last: &mut Option<usize>,
               new_start: Option<usize>,
               id_last: &mut Option<usize>,
               new_id: Option<usize>| {
        *start_last = new_start;
        *id_last = new_id;
    };

    for (i, e) in expanded.iter().enumerate() {
        match e {
            Some(id) => match start_last_occupied {
                Some(_) => {
                    let last_id = id_last_occupied
                        .expect("INVALID STATE: {start,id}_last_occupied **MUST** be in-sync!");

                    if last_id != *id {
                        // this is the same situation as encountering None for the position!
                        // it's the start of a new file
                        push_reset(&mut elements, &mut start_last_occupied, i);
                        set(
                            &mut start_last_occupied,
                            Some(i),
                            &mut id_last_occupied,
                            Some(*id),
                        );
                    }
                    // otherwise, we are still in the same file and we're ok to advance!
                }
                None => set(
                    &mut start_last_occupied,
                    Some(i),
                    &mut id_last_occupied,
                    Some(*id),
                ),
            },
            None => {
                push_reset(&mut elements, &mut start_last_occupied, i);
                set(&mut start_last_occupied, None, &mut id_last_occupied, None);
            }
        }
    }
    // cleanup
    push_reset(&mut elements, &mut start_last_occupied, expanded.len());
    elements
}

fn push_reset(elements: &mut Vec<Range<usize>>, tracking_last: &mut Option<usize>, i: usize) {
    match tracking_last {
        Some(f) => elements.push(Range {
            start: f.clone(),
            end: i,
        }),
        None => (),
    }
    *tracking_last = None;
}

fn defrag_inplace(expanded: &mut ExpandedLine) {
    if expanded.len() <= 1 {
        // empty or one thing => nothing to defrag
        return;
    }

    // goes right-to-left (BACKWARDS)
    // tracks the current file position to move
    let mut focus = match last_occupied(None, &expanded) {
        Some(f) => f,
        // no files => empty! => nothing to defrag
        None => return,
    };

    // goes left-to-right (FORWARDS)
    // tracks the current free position to move into
    let mut free: usize = match first_free(None, &expanded) {
        Some(f) => f,
        // no free space => full! => nothing to defrag
        None => return,
    };

    let space_left = |focus: usize, free: usize| -> bool { free < focus && focus > 0 };

    let swap = |focus: usize, free: usize, expanded: &mut ExpandedLine| {
        assert_eq!(
            expanded[free], None,
            "INVALID STATE: free index is pointing to file contents!"
        );
        expanded[free] = expanded[focus];
        expanded[focus] = None;
    };

    let advance = |focus: &mut usize, free: &mut usize, expanded: &ExpandedLine| -> bool {
        assert_ne!(
            *focus, 0,
            "INVALID STATE: cannot decrement focus index as is at the beginning!"
        );
        match (
            last_occupied(Some(*focus), &expanded),
            first_free(Some(*free), &expanded),
        ) {
            (Some(new_focus), Some(new_free)) => {
                *focus = new_focus;
                *free = new_free;
                true
            }
            _ => {
                *focus = 0;
                *free = expanded.len() - 1;
                false
            }
        }
    };

    // spaces_left
    while space_left(focus, free) {
        swap(focus, free, expanded);
        let ok = advance(&mut focus, &mut free, &expanded);
        if !ok {
            break;
        }
    }
}

fn checksum(expanded: &[Option<usize>]) -> u64 {
    expanded.iter().enumerate().fold(0, |s, (i, x)| match x {
        Some(id) => s + ((i as u64) * (*id as u64)),
        None => s,
    })
}

fn defrag_whole_files_inplace(expanded: &mut ExpandedLine) {
    if expanded.len() <= 1 {
        // empty or one thing => nothing to defrag
        return;
    }

    // goes right-to-left (BACKWARDS)
    // tracks the current file position to move
    let focuses = {
        let mut f = occupied_ranges(&expanded);
        f.reverse();
        f
    };
    // we REVERSE this so it's ok to count from 0 upwards
    // let mut i_focus = focuses.len() - 1;
    let mut i_focus = 0;

    // goes left-to-right (FORWARDS)
    // tracks the current free position to move into
    let mut frees = free_ranges(&expanded);

    let try_swap =
        |focus: &Range<usize>, frees: &mut Vec<Range<usize>>, expanded: &mut ExpandedLine| {
            // {
            //     let id_focus = {
            //         let y = expanded[focus.start];
            //         for x in expanded[focus.start..focus.end].iter() {
            //             assert_eq!(*x, y);
            //         }
            //         y.unwrap()
            //     };
            //     println!("new focus: {focus:?} -> ID: {id_focus}");
            // };
            for i_free in 0..frees.len() {
                let free = &frees[i_free];
                expanded[free.start..free.end].iter().for_each(|e| {
                    assert_eq!(
                        *e, None,
                        "INVALID STATE: free index is pointing to file contents!"
                    )
                });

                if free.end > focus.start {
                    continue;
                }

                let free_space = free.end - free.start;
                let file_size = focus.end - focus.start;

                // println!(
                //     "\tfree: {free:?} | focus: {focus:?}  -> will fit?: {}",
                //     (free_space as i64) - (file_size as i64)
                // );

                if free_space >= file_size {
                    // print!(
                    //     "\t\tSWAPPING ({file_size})!!! before ({free_space}): {:?} | ",
                    //     &expanded[free.start..free.end]
                    // );
                    for (free_, focus_) in (free.start..free.end).zip(focus.start..focus.end) {
                        expanded[free_] = expanded[focus_];
                        expanded[focus_] = None;
                    }
                    // print!(
                    //     "after ({file_size}): {:?}\n",
                    //     &expanded[free.start..(free.start + file_size)]
                    // );

                    // consume the free position since we successfully swapped a file into it
                    if free_space == file_size {
                        // resize this free space --> 0 --> remove it!
                        frees.remove(i_free);
                    } else {
                        // resize the free space to account for the file that is taking up some of it now
                        assert!(free_space > file_size);
                        frees[i_free] = Range {
                            start: free.start + file_size,
                            end: free.end,
                        };
                    }
                    return;
                }
            }
        };

    let advance = |i_focus: &mut usize, frees: &mut Vec<Range<usize>>| -> bool {
        *i_focus += 1;
        !frees.is_empty() && *i_focus < focuses.len()
    };

    loop {
        try_swap(&focuses[i_focus], &mut frees, expanded);
        if !advance(&mut i_focus, &mut frees) {
            break;
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

pub fn solution_pt2() -> u64 {
    let lines = io_help::read_lines("./inputs/9").collect::<Vec<String>>();
    solution(defrag_whole_files_inplace, &lines)
}

///////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod test {

    use indoc::indoc;
    use lazy_static::lazy_static;

    use crate::io_help::read_lines_in_memory;

    use super::*;

    const EXAMPLE_INPUT_STR: &str = indoc! {"
        12345
        90909
    "};

    lazy_static! {
        static ref EXAMPLE_DISK_LINES: Vec<DiskLine> = vec![
            DiskLine {
                elements: vec![1, 2, 3, 4, 5]
            },
            DiskLine {
                elements: vec![9, 0, 9, 0, 9]
            },
        ];
    }

    lazy_static! {
        static ref EXAMPLE_BLOCK_LINES: Vec<BlockLine> = vec![
            vec![
                Block::File(1),
                Block::Free(2),
                Block::File(3),
                Block::Free(4),
                Block::File(5)
            ],
            vec![Block::File(9), Block::File(9), Block::File(9)],
        ];
    }

    lazy_static! {
        static ref EXAMPLE_EXPANDED_LINES: Vec<ExpandedLine> = vec![
            vec![
                Some(0),
                None,
                None,
                Some(1),
                Some(1),
                Some(1),
                None,
                None,
                None,
                None,
                Some(2),
                Some(2),
                Some(2),
                Some(2),
                Some(2)
            ],
            vec![
                Some(0),
                Some(0),
                Some(0),
                Some(0),
                Some(0),
                Some(0),
                Some(0),
                Some(0),
                Some(0),
                Some(1),
                Some(1),
                Some(1),
                Some(1),
                Some(1),
                Some(1),
                Some(1),
                Some(1),
                Some(1),
                Some(2),
                Some(2),
                Some(2),
                Some(2),
                Some(2),
                Some(2),
                Some(2),
                Some(2),
                Some(2),
            ],
        ];
    }

    #[test]
    fn construct() {
        let lines = &read_lines_in_memory(EXAMPLE_INPUT_STR).collect::<Vec<_>>();
        let expected: &Vec<DiskLine> = &EXAMPLE_DISK_LINES;
        let actual = construct_disk_lines(lines).unwrap();
        assert_eq!(actual, *expected, "actual != expected");
    }

    #[test]
    fn blocks() {
        let expected: &Vec<BlockLine> = &EXAMPLE_BLOCK_LINES;
        let actual = EXAMPLE_DISK_LINES
            .clone()
            .into_iter()
            .map(DiskLine::into_blocks)
            .collect::<Vec<_>>();
        assert_eq!(actual, *expected, "actual != expected");
    }

    #[test]
    fn expand() {
        let expected: &Vec<ExpandedLine> = &EXAMPLE_EXPANDED_LINES;
        let actual = EXAMPLE_BLOCK_LINES
            .iter()
            .map(expand_as_indexed_blocks)
            .collect::<Vec<_>>();
        assert_eq!(actual, *expected, "actual != expected");
    }

    #[test]
    fn find_frees_occupied() {
        let expanded: &ExpandedLine = &EXAMPLE_EXPANDED_LINES[0];
        let expected_frees = vec![1, 2, 6, 7, 8, 9];
        let expected_occupied = vec![0, 3, 4, 5, 10, 11, 12, 13, 14];
        let actual_frees = all_frees(expanded);
        let actual_occupied = all_occupied(expanded);
        assert_eq!(
            actual_frees, expected_frees,
            "actual != expected free positions"
        );
        assert_eq!(
            actual_occupied, expected_occupied,
            "actual != expected occupied positions"
        );
    }

    #[test]
    fn defrag_example() {
        // first example can be defragged!
        let expanded: &ExpandedLine = &EXAMPLE_EXPANDED_LINES[0];

        let defrag = |xs: &ExpandedLine| -> ExpandedLine {
            let mut a = xs.clone();
            defrag_inplace(&mut a);
            a
        };

        let expected: ExpandedLine = vec![
            Some(0),
            Some(2),
            Some(2),
            Some(1),
            Some(1),
            Some(1),
            Some(2),
            Some(2),
            Some(2),
            None,
            None,
            None,
            None,
            None,
            None,
        ];
        let actual: ExpandedLine = defrag(expanded);
        assert_eq!(actual, expected, "actual != expected");

        // 2nd example is full!
        let expanded: &ExpandedLine = &EXAMPLE_EXPANDED_LINES[1];
        let actual: ExpandedLine = defrag(expanded);
        assert_eq!(actual, *expanded, "actual != original input");
    }

    #[test]
    fn checksum_example() {
        let actual: u64 = checksum(&expand_direct("0099811188827773336446555566.............."));
        assert_eq!(actual, 1928);
        let actual = checksum(&expand_direct("00992111777.44.333....5555.6666.....8888.."));
        assert_eq!(actual, 2858);
    }

    fn expand_direct(line: &str) -> ExpandedLine {
        line.chars()
            .map(|x| match x {
                '.' => None,
                x => x.to_digit(10).map(|x| x as usize),
            })
            .collect::<Vec<_>>()
    }

    #[test]
    fn pt1_soln_example() {
        let lines = read_lines_in_memory(EXAMPLE_INPUT_STR).collect::<Vec<_>>();
        let part_1 = |ls: &[String]| solution(defrag_inplace, ls);
        ////////////////////////////////////////////////////////
        let actual = part_1(&[lines[0].clone()]);
        assert_eq!(actual, 60);
        ////////////////////////////////////////////////////////
        let actual = part_1(&[lines[1].clone()]);
        assert_eq!(actual, 513);
        ////////////////////////////////////////////////////////
        let actual = part_1(&["2333133121414131402".to_string()]);
        assert_eq!(actual, 1928);
    }

    #[allow(non_camel_case_types, dead_code)]
    #[derive(PartialEq, Debug, Eq)]
    enum PrintKind {
        full,
        dot,
        off,
    }

    fn print_expanded(frees: PrintKind, files: PrintKind, expanded: &[Option<usize>]) {
        expanded.iter().enumerate().for_each(|(i, x)| {
            let s: String = match x {
                Some(file_id) => match files {
                    PrintKind::dot => ".".to_string(),
                    PrintKind::full => {
                        if frees != PrintKind::off {
                            format!("{i} ({file_id})")
                        } else {
                            format!("{i}")
                        }
                    }
                    PrintKind::off => "".to_string(),
                },
                None => match frees {
                    PrintKind::dot => ".".to_string(),
                    PrintKind::full => {
                        if files != PrintKind::off {
                            format!("{i} (.)")
                        } else {
                            format!("{i}")
                        }
                    }
                    PrintKind::off => "".to_string(),
                },
            };
            if s.len() != 0 {
                println!("{s}");
            }
        });
    }

    fn print_ranges(rs: &[Range<usize>]) {
        for Range { start, end } in rs {
            println!("range: [{start}, {end})");
        }
    }

    fn expanded_example() -> Vec<Option<usize>> {
        "00...111...2...333.44.5555.6666.777.888899"
            .chars()
            .map(|d| d.to_digit(10).map(|x| x as usize))
            .collect()
    }

    #[test]
    fn free() {
        let expanded = expanded_example();
        let actual = free_ranges(&expanded);
        let expected = vec![
            Range { start: 2, end: 5 },
            Range { start: 8, end: 11 },
            Range { start: 12, end: 15 },
            Range { start: 18, end: 19 },
            Range { start: 21, end: 22 },
            Range { start: 26, end: 27 },
            Range { start: 31, end: 32 },
            Range { start: 35, end: 36 },
        ];
        print_expanded(PrintKind::full, PrintKind::off, &expanded);
        print_ranges(&actual);
        assert_eq!(actual, expected);
    }

    #[test]
    fn occupied() {
        let expanded = expanded_example();
        let actual = occupied_ranges(&expanded);
        let expected = vec![
            Range { start: 0, end: 2 },
            Range { start: 5, end: 8 },
            Range { start: 11, end: 12 },
            Range { start: 15, end: 18 },
            Range { start: 19, end: 21 },
            Range { start: 22, end: 26 },
            Range { start: 27, end: 31 },
            Range { start: 32, end: 35 },
            Range { start: 36, end: 40 },
            Range { start: 40, end: 42 },
        ];
        print_expanded(PrintKind::off, PrintKind::full, &expanded);
        print_ranges(&actual);
        assert_eq!(actual, expected);
    }

    fn simple_stingify(expanded: &[Option<usize>]) -> String {
        expanded
            .iter()
            .map(|e| match e {
                Some(x) => {
                    assert!(*x <= 9); // NOTE: since x is unsinged, 0 <= *x  is always true
                    x.to_string()
                }
                None => ".".to_string(),
            })
            .collect::<Vec<_>>()
            .join("")
    }

    fn testing_defrag_whole_files_inplace(input: &str, expected_direct: &str) {
        let (actual, original) = {
            let block_line = construct_disk_lines(&[input.to_string()])
                .unwrap()
                .remove(0)
                .into_blocks();
            let mut expanded = expand_as_indexed_blocks(&block_line);
            let original = expanded.clone();
            defrag_whole_files_inplace(&mut expanded);
            (expanded, original)
        };
        let expected = expand_direct(expected_direct);
        println!(
            "\noriginal: {}\nactual:   {}\nexpected: {}\n",
            simple_stingify(&original),
            simple_stingify(&actual),
            simple_stingify(&expected)
        );
        assert_eq!(actual, expected);
    }

    #[test]
    fn defrag_whole_files_inplace_example() {
        testing_defrag_whole_files_inplace(
            "2333133121414131402",
            "00992111777.44.333....5555.6666.....8888..",
        );
        testing_defrag_whole_files_inplace("12345", "0..111....22222");
    }

    #[test]
    fn pt2_soln_example() {
        // let lines = read_lines_in_memory(EXAMPLE_INPUT_STR).collect::<Vec<_>>();
        let part_2 = |ls: &[String]| solution(defrag_whole_files_inplace, ls);
        ////////////////////////////////////////////////////////
        let actual = part_2(&["2333133121414131402".to_string()]);
        assert_eq!(actual, 2858);
        //////////////////////////////////////////////////////
        let actual = part_2(&["12345".to_string()]);
        assert_eq!(actual, 132);
    }
}
