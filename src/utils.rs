pub type Matrix<T> = Vec<Vec<T>>;

pub fn transpose<T: Copy + Default>(
    R: usize,
    C: usize,
    m: Matrix<T>,
    // m: &[[T; C]; R],
) -> Matrix<T> {
    // ) -> [[T; R]; C] {
    // return the transpose of the matrix
    // let mut result: [[T; R]; C] = [[Default::default(); R]; C];
    let mut result: Matrix<T> = vec![vec![Default::default(); R]; C];
    for i in 0..R {
        for j in 0..C {
            result[j][i] = m[i][j];
        }
    }
    result
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
    assert_eq!(lines.len(), ROWS);
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
    assert_eq!(s.len(), N);
    s.chars().collect::<Vec<_>>()
}
// pub fn string_to_char_array<const N: usize>(s: &str) -> [char; N] {
//     assert_eq!(s.len(), N);
//     let mut chars = s.chars();
//     [(); N].map(|_| chars.next().unwrap())
// }
