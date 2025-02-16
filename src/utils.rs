use std::{
    cmp::{max, min, Ordering},
    collections::{HashMap, HashSet},
    error::Error,
    fmt::{Debug, Display},
    hash::Hash,
    ops::{AddAssign, SubAssign},
};

pub fn group_by<K, V>(key: fn(&V) -> K, values: Vec<V>) -> HashMap<K, Vec<V>>
where
    K: Hash + Eq,
{
    values.into_iter().fold(HashMap::new(), |mut m, v| {
        let v_key = key(&v);
        match m.get_mut(&v_key) {
            Some(existing) => _ = existing.push(v),
            None => _ = m.insert(v_key, vec![v]),
        };
        m
    })
}

pub fn sorted_keys<K, _V>(m: &HashMap<K, _V>) -> Vec<K>
where
    K: Hash + Clone + Ord,
{
    let mut ks = m.keys().map(|x| x.clone()).collect::<Vec<_>>();
    ks.sort();
    ks
}

pub fn pairs<T>(elements: &[T]) -> impl Iterator<Item = (&T, &T)> {
    assert!(
        elements.len() > 1,
        "elements.len() > 1 VIOLATED because length is: {}",
        elements.len()
    );
    (0..(elements.len() - 1)).map(|i| (&elements[i], &elements[i + 1]))
}

pub fn increment<K, V>(m: &mut HashMap<K, V>, key: K, val: V)
where
    K: Hash + Eq,
    V: AddAssign,
{
    match m.get_mut(&key) {
        Some(existing) => *existing += val,
        None => _ = m.insert(key, val),
    }
}

pub fn decrement<K, V>(m: &mut HashMap<K, V>, key: K, val: V)
where
    K: Hash + Eq,
    V: SubAssign,
{
    match m.get_mut(&key) {
        Some(existing) => *existing -= val,
        None => _ = m.insert(key, val),
    }
}

// heap-allocated a rectangular 2D array with runtime-determined size
pub type Matrix<T> = Vec<Vec<T>>;

pub fn cardinal_neighbors<T>(
    mat: &Matrix<T>,
    row: usize,
    col: usize,
) -> Vec<(Option<usize>, Option<usize>)> {
    /*
                  (row-1, col)
                 ---------------
    (row, col-1)| (row,  col) |  (row, col+1)
                 ---------------
                  (row+1, col)
    */
    vec![
        // these {sub,add}_{row,col} functions ensure we're in-bounds
        (sub_row(row), Some(col)),
        (Some(row), sub_col(col)),
        (Some(row), add_col(mat, col)),
        (add_row(mat, row), Some(col)),
    ]
}

pub fn sub_row(x: usize) -> Option<usize> {
    if x > 0 {
        Some(x - 1)
    } else {
        None
    }
}

pub fn add_row<T>(m: &Matrix<T>, x: usize) -> Option<usize> {
    if x + 1 < m.len() {
        Some(x + 1)
    } else {
        None
    }
}

pub fn sub_col(x: usize) -> Option<usize> {
    if x > 0 {
        Some(x - 1)
    } else {
        None
    }
}

pub fn add_col<T>(m: &Matrix<T>, x: usize) -> Option<usize> {
    if m.len() == 0 {
        return None;
    }
    if x + 1 < m[0].len() {
        Some(x + 1)
    } else {
        None
    }
}

pub fn exterior_perimiter<T>(mat: &Matrix<T>, members: &[Coordinate]) -> u64 {
    // create set: (row,col) => in members?
    let membership = members.iter().collect::<HashSet<&Coordinate>>();
    // for each one
    //      ask if each of its immediate neighbors is in members
    //      for each of these 4, increment perimiter iff it's not also in members
    members
        .iter()
        .fold(0, |perimiter, Coordinate { row, col }| {
            cardinal_neighbors(mat, *row, *col)
                .iter()
                .fold(perimiter, |p, x| match x {
                    // only count the side (r,c) as part of the perimiter
                    //      (1) if it is in-bounds and (r,c) is *not* in members
                    //      (2) it is out of bounds
                    (Some(r), Some(c)) => {
                        if !membership.contains(&Coordinate::new(*r, *c)) {
                            // (1) not in members -> borders another region!
                            p + 1
                        } else {
                            // it is already in members, so this is an interior side!
                            p
                        }
                    }
                    // (2) out of bounds => this side borders the edge
                    _ => p + 1,
                })
        })
}

pub fn trace_perimiter<T>(mat: &Matrix<T>, members: &[Coordinate]) -> Vec<Coordinate> {
    // create set: (row,col) => in members?
    let membership = members.iter().collect::<HashSet<&Coordinate>>();
    // for each one
    //      ask if each of its immediate neighbors is in members
    //      for each of these 4, increment perimiter iff it's not also in members
    members
        .iter()
        .filter(|coordinate| {
            cardinal_neighbors(mat, coordinate.row, coordinate.col)
                .iter()
                .any(|x| match x {
                    // only count the side (r,c) as part of the perimiter
                    //      (1) if it is in-bounds and (r,c) is *not* in members
                    //      (2) it is out of bounds
                    (Some(r), Some(c)) => {
                        // (1) not in members -> borders another region!
                        !membership.contains(&Coordinate::new(*r, *c))
                        // it is already in members, so this is an interior side!
                    }
                    // (2) out of bounds => this side borders the edge
                    _ => true,
                })
        })
        .map(|c| c.clone())
        .collect()
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, PartialOrd)]
pub struct Coordinate {
    pub row: usize,
    pub col: usize,
}

impl Coordinate {
    pub fn new(row: usize, col: usize) -> Self {
        Self { row, col }
    }
}

impl Display for Coordinate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({},{})", self.row, self.col)
    }
}

impl Ord for Coordinate {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.row.cmp(&other.row) {
            Ordering::Equal => self.col.cmp(&other.col),
            order => order,
        }
    }
}

pub struct Coords<'a>(pub &'a [Coordinate]);

impl Display for Coords<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        match self.0.len() {
            0 => (),
            _ => {
                write!(f, "{}", &self.0[0])?;
                for x in self.0[1..self.0.len()].iter() {
                    write!(f, ",{x}")?;
                }
            }
        }
        write!(f, "]")
    }
}

#[derive(Debug, PartialEq)]
pub enum InvalidShape {
    RowCount {
        actual_rows: usize,
    },
    Coordinate {
        row_index: usize,
        actual_cols: usize,
    },
}

pub fn _is_square<T>(dimension: usize, mat: &Matrix<T>) -> Option<Vec<InvalidShape>> {
    is_rectangular(dimension, dimension, mat)
}

pub fn is_rectangular<T>(rows: usize, cols: usize, mat: &Matrix<T>) -> Option<Vec<InvalidShape>> {
    let mut invalids: Vec<InvalidShape> = Vec::new();
    if mat.len() != rows {
        invalids.push(InvalidShape::RowCount {
            actual_rows: mat.len(),
        });
    }
    for (i, row) in mat.iter().enumerate() {
        if row.len() != cols {
            invalids.push(InvalidShape::Coordinate {
                row_index: i,
                actual_cols: row.len(),
            });
        }
    }
    if invalids.len() == 0 {
        None
    } else {
        Some(invalids)
    }
}

#[allow(dead_code)]
pub fn transpose<T: Copy + Default>(
    max_rows: usize,
    max_cols: usize,
    m: &Matrix<T>,
    // m: &[[T; C]; R],
) -> Matrix<T> {
    // ) -> [[T; R]; C] {
    // return the transpose of the matrix
    // let mut result: [[T; R]; C] = [[Default::default(); R]; C];
    assert_eq!(m.len(), max_rows, "T: matrix's rows != expected");
    let mut result: Matrix<T> = vec![vec![Default::default(); max_rows]; max_cols];
    for i in 0..max_rows {
        assert_eq!(m[i].len(), max_cols, "T: matrix's cols != expected");
        for j in 0..max_cols {
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

pub fn _flipud<T: Copy + Default>(mat: &Matrix<T>) -> Matrix<T> {
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

pub fn convert_to_char_matrix(max_rows: usize, max_cols: usize, lines: &[String]) -> Matrix<char> {
    assert_eq!(
        lines.len(),
        max_rows,
        "C: number of strings != expected char array rows"
    );
    let mut result = vec![vec![Default::default(); max_cols]; max_rows];

    let mut line_iter = lines.iter();

    for i in 0..max_rows {
        let x = line_iter.next().unwrap();
        let y = string_to_char_array(max_cols, x.as_str());
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

pub fn string_to_char_array(max_rows: usize, s: &str) -> Vec<char> {
    assert_eq!(s.len(), max_rows, "C: string length != expected char cols");
    s.chars().collect::<Vec<_>>()
}
// pub fn string_to_char_array<const N: usize>(s: &str) -> [char; N] {
//     assert_eq!(s.len(), N);
//     let mut chars = s.chars();
//     [(); N].map(|_| chars.next().unwrap())
// }

pub fn _char_array_to_lines<const R: usize, const C: usize>(chars: [[char; C]; R]) -> Vec<String> {
    (0..R)
        .map(|i| String::from_iter(chars[i]))
        .collect::<Vec<_>>()
}

pub fn _char_matrix_to_lines(m: Matrix<char>) -> Vec<String> {
    let (_, max_rows) = (m[0].len(), m.len());
    (0..max_rows)
        .map(|i| m[i].to_owned().into_iter().collect::<String>())
        .collect::<Vec<_>>()
}

pub fn _array_to_matrix<const R: usize, const C: usize, T: Default + Copy>(
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

pub fn _pretty_matrix<T: Debug>(m: &Matrix<T>) -> String {
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

pub type Res<T> = Result<T, Box<dyn Error>>;

pub fn proc_elements_result<A, B>(process: fn(&A) -> Res<B>, elements: &[A]) -> Res<Vec<B>> {
    let mut collected: Vec<B> = Vec::new();
    for x in elements.iter() {
        match process(x) {
            Ok(result) => collected.push(result),
            Err(error) => {
                return Err(error);
            }
        }
    }
    Ok(collected)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn try_flip_diagonal_coordinates() {
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
        println!("original!\n{}", _pretty_matrix(&_array_to_matrix(chars)));

        for diag in diagonal_coordinates(x.len() as i32, x[0].len() as i32) {
            let dstr = diag.iter().map(|(i, j)| chars[*i][*j]).collect::<String>();
            // println!("diag: {dstr} --> {diag:?})");
            println!("diag: {dstr}");
        }

        let flipped_x = fliplr(&_array_to_matrix(chars));
        // println!("flipped!!\n{flipped_x:?}");
        println!("flipped!!\n{}", _pretty_matrix(&flipped_x));

        let expected_flipped = [
            ['S', 'A', 'M', 'X'],
            ['A', 'M', 'X', 'S'],
            ['M', 'X', 'S', 'A'],
            ['X', 'S', 'A', 'M'],
            // ['S', 'A', 'M', 'X']
        ];

        assert_eq!(flipped_x, _array_to_matrix(expected_flipped));
        /*

        */

        for diag in diagonal_coordinates(x.len() as i32, x[0].len() as i32) {
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
}
