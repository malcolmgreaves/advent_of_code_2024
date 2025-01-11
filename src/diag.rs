use std::{
    cmp::{max, min},
    path::Display,
};

use crate::utils;

fn diag_incr(x: (i32, i32)) -> (i32, i32) {
    (x.0 + 1, x.1 + 1)
}

#[test]
pub fn tryit() {
    let chars = [
        ['X', 'M', 'A', 'S'],
        ['S', 'X', 'M', 'A'],
        ['A', 'S', 'X', 'M'],
        ['M', 'A', 'S', 'X'],
        // ['X', 'M', 'A', 'S'],
    ];

    let x = [
        [(0, 0), (0, 1), (0, 2), (0, 3)],
        [(1, 0), (1, 1), (1, 2), (1, 3)],
        [(2, 0), (2, 1), (2, 2), (2, 3)],
        [(3, 0), (3, 1), (3, 2), (3, 3)],
        // [(4,0), (4,1), (4,2), (4,3)],
    ];

    // let p_chars = (0..chars.len()).map(|i| {
    //     format!("{:?}\n", chars[i])
    // }).collect::<String>();
    // println!("original!\n{p_chars}");
    println!(
        "original!\n{}",
        utils::pretty_matrix(&utils::array_to_matrix(chars))
    );

    for diag in utils::diagonal_coordinates(x.len() as i32, x[0].len() as i32) {
        let dstr = diag.iter().map(|(i, j)| chars[*i][*j]).collect::<String>();
        // println!("diag: {dstr} --> {diag:?})");
        println!("diag: {dstr}");
    }

    let flipped_x = utils::fliplr(&utils::array_to_matrix(chars));
    // println!("flipped!!\n{flipped_x:?}");
    println!("flipped!!\n{}", utils::pretty_matrix(&flipped_x));

    let expected_flipped = [
        ['S', 'A', 'M', 'X'],
        ['A', 'M', 'X', 'S'],
        ['M', 'X', 'S', 'A'],
        ['X', 'S', 'A', 'M'],
        // ['S', 'A', 'M', 'X']
    ];

    assert_eq!(flipped_x, utils::array_to_matrix(expected_flipped));
    /*

    */

    for diag in utils::diagonal_coordinates(x.len() as i32, x[0].len() as i32) {
        let dstr = diag
            .iter()
            .map(|(i, j)| flipped_x[*i][*j])
            .collect::<String>();
        println!("flipped: {dstr}");
    }

    // let R = x.len();
    // assert_eq!(R, 5);
    // let C = x[0].len();
    // assert_eq!(C, 4);

    // for col_offset in 0..C {
    //     for d in 0..R {
    //         let x = (d, col_offset);

    //     }
    // }
}

/*

let x = [
    [(0,0), (0,1), (0,2), (0,3)],
    [(1,0), (1,1), (1,2), (1,3)],
    [(2,0), (2,1), (2,2), (2,3)],
    [(3,0), (3,1), (3,2), (3,3)],
    [(4,0), (4,1), (4,2), (4,3)],
]

R=5, C=4

d=0
x=d
while x < C;

    (0,0), (1,1), (2, 2), (3,3)

d=1



(3,0)
(2,0), (3,1)


*/
