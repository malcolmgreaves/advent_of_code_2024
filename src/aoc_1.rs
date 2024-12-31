use crate::io_help;

pub fn solution() -> i32 {
    // https://adventofcode.com/2024/day/1
    let (l1, l2) = io_help::read_lines_as_ints("   ", "./inputs/1")
        .into_iter()
        .map(|x| {
            assert!(x.len() == 2);
            (x[0], x[1])
        })
        .unzip::<i32, i32, Vec<i32>, Vec<i32>>();
    sum_sorted_distances(&l1, &l2)
}

fn sum_sorted_distances(list1: &[i32], list2: &[i32]) -> i32 {
    let mut list1 = list1.to_vec();
    list1.sort();
    let mut list2 = list2.to_vec();
    list2.sort();
    list1
        .iter()
        .zip(list2.iter())
        .map(|(l1, l2)| if l1 < l2 { l2 - l1 } else { l1 - l2 })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn example_solution() {
        let result = sum_sorted_distances(&[3, 4, 2, 1, 3, 3], &[4, 3, 5, 3, 9, 3]);
        assert_eq!(result, 11);
    }
}
