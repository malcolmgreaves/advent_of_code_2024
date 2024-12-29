pub fn solution() -> i32 {
    // https://adventofcode.com/2024/day/1
    sum_sorted_distances(&[3,4,2,1,3,3], &[4,3,5,3,9,3])
}

fn sum_sorted_distances(list1: &[i32], list2: &[i32]) -> i32 {
    let mut list1 = list1.to_vec();
    list1.sort();
    let mut list2 = list2.to_vec();
    list2.sort();
    list1.iter().zip(list2.iter()).map(|(l1,l2)| if l1 < l2 { l2 - l1 } else { l1 - l2 }).sum()
}
