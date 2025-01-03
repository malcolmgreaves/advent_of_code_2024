use crate::io_help;

pub fn solution() -> i32 {
    // https://adventofcode.com/2024/day/2
    count_ok_levels(io_help::read_lines_as_ints(" ", "./inputs/2").to_vec())
}

fn count_ok_levels(levels: Vec<Vec<i32>>) -> i32 {
    levels
        .iter()
        .map(|level| check_level(level))
        .map(|x| if x { 1 } else { 0 })
        .sum::<i32>()
}

fn check_level(level: &[i32]) -> bool {
    // True iff there are at least 2 elements, they're all ascending or descending, and the diff between any two is in [1,3].
    if level.len() < 2 {
        return false;
    }

    // The levels are either all increasing or all decreasing.
    let diffs: Vec<i32> = level
        .windows(2)
        .map(|w| {
            let (a, b) = (w[0], w[1]);
            a - b
        })
        .collect();

    let is_descending = diffs.iter().all(|x| x >= &0);
    // println!("is_descending: {is_descending:?}");
    let is_ascending = diffs.iter().all(|x| x <= &0);
    // println!("is_ascending: {is_ascending:?}");
    let ascending_or_descending = is_ascending || is_descending;

    // level[1..level.len()].iter().fold(level[0], |prev, x| prev - x);

    // Any two adjacent levels differ by at least one and at most three.
    let diff_check = level
        .windows(2)
        .map(|w| {
            let (a, b) = (w[0], w[1]);
            let diff = if a < b { b - a } else { a - b };
            let ok_diff = diff >= 1 && diff <= 3;
            ok_diff
        })
        .fold(true, |acc, x| acc && x);

    // println!("diff_check: {diff_check:?}");
    return ascending_or_descending && diff_check;
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn example_solution() {
        let levels = [
            [7, 6, 4, 2, 1].to_vec(),
            [1, 2, 7, 8, 9].to_vec(),
            [9, 7, 6, 2, 1].to_vec(),
            [1, 3, 2, 4, 5].to_vec(),
            [8, 6, 4, 4, 1].to_vec(),
            [1, 3, 6, 7, 9].to_vec(),
        ]
        .to_vec();

        let result = count_ok_levels(levels);
        assert!(result == 2);
    }
}
