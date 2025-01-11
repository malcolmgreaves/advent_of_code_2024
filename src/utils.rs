use std::{
    cmp::{max, min},
    fmt::Debug,
};

pub type Matrix<T> = Vec<Vec<T>>;

pub fn transpose<T: Copy + Default>(
    R: usize,
    C: usize,
    m: &Matrix<T>,
    // m: &[[T; C]; R],
) -> Matrix<T> {
    // ) -> [[T; R]; C] {
    // return the transpose of the matrix
    // let mut result: [[T; R]; C] = [[Default::default(); R]; C];
    assert_eq!(m.len(), R, "T: matrix's rows != expected");
    let mut result: Matrix<T> = vec![vec![Default::default(); R]; C];
    for i in 0..R {
        assert_eq!(m[i].len(), C, "T: matrix's cols != expected");
        for j in 0..C {
            result[j][i] = m[i][j];
        }
    }
    result
}

pub fn fliplr<T: Copy + Default>(mat: &Matrix<T>) -> Matrix<T> {
    if mat.len() == 0 {
        return Vec::new();
    }
    let rows = mat.len();
    let cols = mat[0].len();

    let mut flipped = vec![vec![Default::default(); cols]; rows];

    for i in 0..rows {
        for j in 0..cols {
            let access_j = cols - 1 - j;
            flipped[i][j] = mat[i][access_j];
        }
    }

    flipped
}

pub fn flipud<T: Copy + Default>(mat: &Matrix<T>) -> Matrix<T> {
    if mat.len() == 0 {
        return Vec::new();
    }
    let rows = mat.len();
    let cols = mat[0].len();

    let mut flipped = vec![vec![Default::default(); cols]; rows];

    for i in 0..rows {
        let access_i = rows - 1 - i;
        for j in 0..cols {
            flipped[i][j] = mat[access_i][j];
        }
    }

    flipped
}

pub fn reverse_string(x: String) -> String {
    x.chars().rev().collect::<String>()
}

// pub fn abc() -> Box<[Box<[char]>]> {
//     vec![
//         vec!['a','b','c'].into_boxed_slice(),
//         vec!['d','e','f'].into_boxed_slice(),
//     ].into_boxed_slice()
// }

// macro_rules! box_array {
//     ($val:expr ; $len:expr) => {{
//         // Use a generic function so that the pointer cast remains type-safe
//         fn vec_to_boxed_array<T>(vec: Vec<T>) -> Box<[T; $len]> {
//             let boxed_slice = vec.into_boxed_slice();

//             let ptr = ::std::boxed::Box::into_raw(boxed_slice) as *mut [T; $len];

//             unsafe { Box::from_raw(ptr) }
//         }

//         vec_to_boxed_array(vec![$val; $len])
//     }};
// }

// pub fn example(size: usize) -> Box<i32; size> {
//     const X: usize = 10_000_000;
//     let huge_heap_array = box_array![1; X];
//     huge_heap_array
// }

pub fn convert_to_char_matrix(ROWS: usize, COLS: usize, lines: &[String]) -> Matrix<char> {
    assert_eq!(
        lines.len(),
        ROWS,
        "C: number of strings != expected char array rows"
    );
    let mut result = vec![vec![Default::default(); COLS]; ROWS];

    let mut line_iter = lines.iter();

    for i in 0..ROWS {
        let x = line_iter.next().unwrap();
        let y = string_to_char_array(COLS, x.as_str());
        result[i] = y
    }
    result
}

// pub fn convert_to_char_matrix<const ROWS: usize, const COLS: usize>(
//     lines: &[String],
// ) -> [[char; COLS]; ROWS] {
//     assert_eq!(lines.len(), ROWS);
//     let mut line_iter = lines.iter();

//     [(); ROWS].map(|_| {
//         let x = line_iter.next().unwrap();
//         string_to_char_array::<COLS>(x.as_str())
//     })
// }

pub fn string_to_char_array(N: usize, s: &str) -> Vec<char> {
    assert_eq!(s.len(), N, "C: string length != expected char cols");
    s.chars().collect::<Vec<_>>()
}
// pub fn string_to_char_array<const N: usize>(s: &str) -> [char; N] {
//     assert_eq!(s.len(), N);
//     let mut chars = s.chars();
//     [(); N].map(|_| chars.next().unwrap())
// }

pub fn char_array_to_lines<const R: usize, const C: usize>(chars: [[char; C]; R]) -> Vec<String> {
    (0..R)
        .map(|i| String::from_iter(chars[i]))
        .collect::<Vec<_>>()
}

pub fn char_matrix_to_lines(m: Matrix<char>) -> Vec<String> {
    let (C, R) = (m[0].len(), m.len());
    (0..R)
        .map(|i| m[i].to_owned().into_iter().collect::<String>())
        .collect::<Vec<_>>()
}

pub fn array_to_matrix<const R: usize, const C: usize, T: Default + Copy>(
    arr: [[T; C]; R],
) -> Matrix<T> {
    let mut mat = vec![vec![Default::default(); C]; R];
    for i in 0..R {
        for j in 0..C {
            mat[i][j] = arr[i][j];
        }
    }
    mat
}

pub fn pretty_matrix<T: Debug>(m: &Matrix<T>) -> String {
    (0..m.len())
        .map(|i| format!("{:?}\n", m[i]))
        .collect::<String>()
}

pub fn diagonal_coordinates(n: i32, m: i32) -> Vec<Vec<(usize, usize)>> {
    (0..(n + m - 1))
        .map(|d| {
            // let right_of_diagonal = max(0, d - m + 1);
            (max(0, d - m + 1)..min(n, d + 1))
                .map(|x| {
                    // (x, d-x)
                    (x as usize, (d - x) as usize)
                    // (
                    //     TryInto::<usize>::try_into(x).unwrap(),
                    //     TryInto::<usize>::try_into(d-x).unwrap(),
                    // )
                })
                .collect::<Vec<(_, _)>>()
        })
        .collect::<Vec<_>>()
}
