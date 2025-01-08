pub fn transpose<const R: usize, const C: usize, T: Copy + Default>(
    m: &[[T; C]; R],
) -> [[T; R]; C] {
    // return the transpose of the matrix
    let mut result: [[T; R]; C] = [[Default::default(); R]; C];
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

pub fn convert_to_char_matrix<const ROWS: usize, const COLS: usize>(
    lines: &[String],
) -> [[char; COLS]; ROWS] {
    assert_eq!(lines.len(), ROWS);
    let mut line_iter = lines.iter();

    [(); ROWS].map(|_| {
        let x = line_iter.next().unwrap();
        string_to_char_array::<COLS>(x.as_str())
    })
}

pub fn string_to_char_array<const N: usize>(s: &str) -> [char; N] {
    assert_eq!(s.len(), N);
    let mut chars = s.chars();
    [(); N].map(|_| chars.next().unwrap())
}
