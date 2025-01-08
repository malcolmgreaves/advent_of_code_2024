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
