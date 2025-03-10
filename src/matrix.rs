use std::{
    cmp::{Ordering, max, min},
    collections::HashSet,
    fmt::{Debug, Display},
};

// heap-allocated a rectangular 2D array with runtime-determined size
pub type Matrix<T> = Vec<Vec<T>>;

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub enum Direction {
    Up,
    Down,
    Left,
    Right,
}

impl Direction {
    pub fn clockwise(&self) -> Direction {
        match self {
            Direction::Up => Direction::Right,
            Direction::Right => Direction::Down,
            Direction::Down => Direction::Left,
            Direction::Left => Direction::Up,
        }
    }

    // [0,3] the # of turns to get from self to other, rotating clockwise
    #[allow(unused)]
    pub fn calculate_clockwise_turns(&self, other: &Direction) -> u8 {
        let mut d = self.clone();
        let mut n = 0;
        while d != *other {
            d = d.clockwise();
            n += 1;
        }
        n
    }

    pub fn opposite(&self) -> Direction {
        match self {
            Direction::Up => Direction::Down,
            Direction::Right => Direction::Left,
            Direction::Down => Direction::Up,
            Direction::Left => Direction::Right,
        }
    }
}

pub struct GridMovement {
    max_row: usize,
    max_col: usize,
}

impl GridMovement {
    pub fn new<T>(matrix: &Matrix<T>) -> GridMovement {
        GridMovement {
            max_row: matrix.len(),
            max_col: matrix[0].len(),
        }
    }

    pub fn full_neighborhood(&self, row: usize, col: usize) -> [(Option<usize>, Option<usize>); 9] {
        [
            (self.sub(row), self.sub(col)),
            (self.sub(row), Some(col)),
            (self.sub(row), self.add_col(col)),
            (Some(row), self.sub(col)),
            (Some(row), Some(col)),
            (Some(row), self.add_col(col)),
            (self.add_row(row), self.sub(col)),
            (self.add_row(row), Some(col)),
            (self.add_row(row), self.add_col(col)),
        ]
    }

    pub fn sub(&self, x: usize) -> Option<usize> {
        if x > 0 { Some(x - 1) } else { None }
    }
    pub fn add_row(&self, x: usize) -> Option<usize> {
        let y = x + 1;
        if y < self.max_row { Some(y) } else { None }
    }
    pub fn add_col(&self, x: usize) -> Option<usize> {
        let y = x + 1;
        if y < self.max_col { Some(y) } else { None }
    }

    /// Obtain up to 4 positions around loc, ordered by clockwise rotation starting from facing.
    pub fn clockwise_rotation(
        &self,
        loc: &Coordinate,
        facing: &Direction,
    ) -> Vec<(Coordinate, Direction)> {
        let mut d = facing.clone();
        let mut positions = Vec::new();
        for _ in 0..4 {
            // println!("\t\t\t[trying] {d:?}");
            match self.next_advance(loc, &d) {
                Some(next) => positions.push((next, d.clone())),
                None => {
                    // println!("\t\t\t[fail] {d:?} failed because next step is out of bounds");
                }
            };
            d = d.clockwise();
        }
        positions
    }

    pub fn next_advance(&self, loc: &Coordinate, current: &Direction) -> Option<Coordinate> {
        match current {
            Direction::Up => self.next_up(loc),
            Direction::Right => self.next_right(loc),
            Direction::Down => self.next_down(loc),
            Direction::Left => self.next_left(loc),
        }
    }

    pub fn next_right(&self, loc: &Coordinate) -> Option<Coordinate> {
        let col = loc.col + 1;
        if col < self.max_col {
            Some(Coordinate { row: loc.row, col })
        } else {
            None
        }
    }

    pub fn next_down(&self, loc: &Coordinate) -> Option<Coordinate> {
        let row = loc.row + 1;
        if row < self.max_row {
            Some(Coordinate { row, col: loc.col })
        } else {
            None
        }
    }

    pub fn next_left(&self, loc: &Coordinate) -> Option<Coordinate> {
        if loc.col > 0 {
            let col = loc.col - 1;
            Some(Coordinate { row: loc.row, col })
        } else {
            None
        }
    }

    pub fn next_up(&self, loc: &Coordinate) -> Option<Coordinate> {
        if loc.row > 0 {
            let row = loc.row - 1;
            Some(Coordinate { row, col: loc.col })
        } else {
            None
        }
    }
}

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
    if x > 0 { Some(x - 1) } else { None }
}

pub fn add_row<T>(m: &Matrix<T>, x: usize) -> Option<usize> {
    if x + 1 < m.len() { Some(x + 1) } else { None }
}

pub fn sub_col(x: usize) -> Option<usize> {
    if x > 0 { Some(x - 1) } else { None }
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
    let g = GridMovement::new(mat);
    // create set: (row,col) => in members?
    let membership = members.iter().collect::<HashSet<&Coordinate>>();
    // for each one
    //      ask if each of its immediate neighbors is in members
    //      for each of these 4, increment perimiter iff it's not also in members
    members
        .iter()
        .filter(|coordinate| {
            // keep the coordinate if and only if it is the border
            // ==> what's a border? it has at least ONE neighbor that is:
            //          - in a different region
            //          - OR out of bounds
            // cardinal_neighbors(mat, coordinate.row, coordinate.col)
            g.full_neighborhood(coordinate.row, coordinate.col)
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

pub fn display_matrix<T: Display>(
    mat: &Matrix<T>,
    c_sep: &str,
    r_sep: &str,
    whitespace: &str,
) -> String {
    mat.iter()
        .map(|row| {
            row.iter()
                .map(|c| {
                    let s = format!("{c}").trim().to_string();
                    if s.len() == 0 {
                        whitespace.to_string()
                    } else {
                        s
                    }
                })
                .collect::<Vec<_>>()
                .join(c_sep)
        })
        .collect::<Vec<_>>()
        .join(r_sep)
}

pub fn print_matrix<T: Display>(m: &Matrix<T>) {
    println!("{}", display_matrix(m, ",", "\n", "_"));
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
