#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Binary {
    Zero,
    One,
}

pub type BinaryNumber = Vec<Binary>;

pub fn binary_to_string(binrep: BinaryNumber) -> String {
    binrep
        .iter()
        .map(|x| match x {
            Binary::Zero => '0',
            Binary::One => '1',
        })
        .collect()
}

pub fn binary_representation(max_len: usize, x: usize) -> BinaryNumber {
    let base: usize = 2;
    if base.pow(max_len.try_into().unwrap()) < x {
        panic!("need more space for value: {x} as this is MORE than {base}^{max_len}");
    }
    let mut result = Vec::new();
    let mut quotient = x;
    let mut i = max_len;
    // compute binary representation: find # of times its disible by 2,
    // noting when there's a remainder to denote a 0 instead of a 1 in the position
    while i > 0 {
        i -= 1;
        let remainder = quotient % 2 == 0;
        quotient /= 2;
        result.push(if remainder { Binary::Zero } else { Binary::One });
        if quotient == 0 {
            break;
        }
    }
    // add leading zeros, if necessary
    while i > 0 {
        i -= 1;
        result.push(Binary::Zero);
    }
    // our representation is *backwards* !! flip it around before returning
    result.reverse();
    result
}

pub fn binary_enumeration(max_len: usize) -> Vec<BinaryNumber> {
    let maximum = 2_u32.pow(max_len as u32);
    (0..maximum)
        .map(|x| binary_representation(max_len, x as usize))
        .collect()
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Ternary {
    Zero,
    One,
    Two,
}

pub type TernaryNumber = Vec<Ternary>;

pub fn ternary_to_string(ternrep: TernaryNumber) -> String {
    ternrep
        .iter()
        .map(|x| match x {
            Ternary::Zero => '0',
            Ternary::One => '1',
            Ternary::Two => '2',
        })
        .collect()
}

pub fn ternary_representation(max_len: usize, x: usize) -> TernaryNumber {
    let base: usize = 3;
    if base.pow(max_len.try_into().unwrap()) < x {
        panic!("need more space for value: {x} as this is MORE than {base}^{max_len}");
    }
    let mut result = Vec::new();
    let mut quotient = x;
    let mut i = max_len;
    // compute ternary representation: find # of times its disible by 3,
    // noting when there's a remainder: denote the right # for this
    while i > 0 {
        i -= 1;
        let remainder = quotient % 3;
        quotient /= 3;
        result.push(match remainder {
            0 => Ternary::Zero,
            1 => Ternary::One,
            2 => Ternary::Two,
            unexpected => panic!("expecting [0,1,2] not {unexpected}"),
        });
        if quotient == 0 {
            break;
        }
    }
    // add leading zeros, if necessary
    while i > 0 {
        i -= 1;
        result.push(Ternary::Zero);
    }
    // our representation is *backwards* !! flip it around before returning
    result.reverse();
    result
}

pub fn ternary_enumeration(max_len: usize) -> Vec<TernaryNumber> {
    let maximum = 3_u32.pow(max_len as u32);
    (0..maximum)
        .map(|x| ternary_representation(max_len, x as usize))
        .collect()
}

#[cfg(test)]
mod test {

    use super::*;

    #[test]
    fn known_binrep() {
        let x: usize = 5;
        let expected = format!("{x:08b}");
        let binrep: BinaryNumber = binary_representation(8, x);
        let actual = binary_to_string(binrep);
        assert_eq!(actual, expected);
    }

    #[test]
    fn known_binenum() {
        let len: usize = 4;
        let expecteds = [
            "0000", // 0
            "0001", // 1
            "0010", // 2
            "0011", // 3
            "0100", // 4
            "0101", // 5
            "0110", // 6
            "0111", // 7
            "1000", // 8
            "1001", // 9
            "1010", // 10
            "1011", // 11
            "1100", // 12
            "1101", // 13
            "1110", // 14
            "1111", // 15
        ];
        binary_enumeration(len)
            .into_iter()
            .enumerate()
            .for_each(|(i, binrep)| {
                let binstr = binary_to_string(binrep);
                let e = expecteds[i];
                assert_eq!(binstr, e);
            });
    }

    #[test]
    fn known_ternrep() {
        let x = 10 as usize;
        let expected = "101".to_string();
        let ternrep = ternary_representation(3, x);
        let actual = ternary_to_string(ternrep);
        assert_eq!(actual, expected);
    }

    #[test]
    fn known_ternnum() {
        let len: usize = 3;
        let expecteds = [
            "000", // 0
            "001", // 1
            "002", // 2
            "010", // 3
            "011", // 4
            "012", // 5
            "020", // 6
            "021", // 7
            "022", // 8
            "100", // 9
            "101", // 10
            "102", // 11
            "110", // 12
            "111", // 13
            "112", // 14
            "120", // 15
            "121", // 16
            "122", // 17
            "200", // 18
            "201", // 19
            "202", // 20
            "210", // 21
            "211", // 22
            "212", // 23
            "220", // 24
            "221", // 25
            "222", // 26
        ];
        ternary_enumeration(len)
            .into_iter()
            .enumerate()
            .for_each(|(i, ternrep)| {
                let ternstr = ternary_to_string(ternrep);
                let e = expecteds[i];
                assert_eq!(ternstr, e);
            });
    }
}
