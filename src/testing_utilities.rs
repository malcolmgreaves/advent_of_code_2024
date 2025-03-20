use std::fmt::Debug;

use crate::matrix::Matrix;

pub fn check_matrices<T: Eq + Debug>(
    actual: &Matrix<T>,
    expected: &Matrix<T>,
) -> Result<(), String> {
    if actual.len() != expected.len() {
        return Err(format!(
            "incorrect # of rows: actual={} vs. expected={}",
            actual.len(),
            expected.len()
        ));
    }

    let errors = actual
        .iter()
        .zip(expected.iter())
        .enumerate()
        .flat_map(|(row, (a, e))| {
            if a.len() != e.len() {
                Some(format!("row {} has mismatched column lengths", row + 1))
            } else if a != e {
                let row_num = row+1;
                Some(format!("row {row_num} has differences: actual[{row_num}]={a:?} vs. expected[{row_num}]={e:?}"))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    if errors.len() > 0 {
        Err(format!(
            "Found {} errors:\n\t{}",
            errors.len(),
            errors.join("\n\t")
        ))
    } else {
        Ok(())
    }
}
