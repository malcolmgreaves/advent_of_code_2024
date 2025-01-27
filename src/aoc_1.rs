use std::collections::HashMap;

use crate::io_help;

// https://adventofcode.com/2024/day/1

fn convert_to_int_lists(nums: Vec<Vec<i32>>) -> (Vec<i32>, Vec<i32>) {
    nums.into_iter()
        .map(|x| {
            assert!(x.len() == 2);
            (x[0], x[1])
        })
        .unzip::<i32, i32, Vec<i32>, Vec<i32>>()
}

pub fn solution_pt1() -> u64 {
    let (l1, l2) = convert_to_int_lists(io_help::read_lines_as_ints("   ", "./inputs/1"));
    sum_sorted_distances(&l1, &l2)
}

fn sum_sorted_distances(list1: &[i32], list2: &[i32]) -> u64 {
    let mut list1 = list1.to_vec();
    list1.sort();
    let mut list2 = list2.to_vec();
    list2.sort();
    let s: i32 = list1
        .iter()
        .zip(list2.iter())
        .map(|(l1, l2)| if l1 < l2 { l2 - l1 } else { l1 - l2 })
        .sum();
    assert!(s > 0);
    s as u64
}

pub fn solution_pt2() -> u64 {
    let (l1, l2) = convert_to_int_lists(io_help::read_lines_as_ints("   ", "./inputs/1"));
    similarity_score(&l1, &l2)
}

fn similarity_score(list1: &[i32], list2: &[i32]) -> u64 {
    let l2_counts: HashMap<i32, i32> = list2.iter().fold(HashMap::new(), |mut accum, x| {
        accum.insert(*x, accum.get(x).unwrap_or(&0) + 1);
        accum
    });

    let s: i32 = list1
        .iter()
        .map(|x| x * l2_counts.get(x).unwrap_or(&0))
        .sum();
    assert!(s > 0);
    s as u64
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn example_solution_pt1() {
        let result = sum_sorted_distances(&[3, 4, 2, 1, 3, 3], &[4, 3, 5, 3, 9, 3]);
        assert_eq!(result, 11);
    }

    #[test]
    fn example_solution_pt2() {
        let result = similarity_score(&[3, 4, 2, 1, 3, 3], &[4, 3, 5, 3, 9, 3]);
        assert_eq!(result, 31);
    }
}
