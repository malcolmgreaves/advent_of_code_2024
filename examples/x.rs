use regex::Regex;

pub fn main() {
    let re = Regex::new(r"p=(\d+),(\d+)").unwrap();
    let s = "p=10,2";
    match re.captures(s) {
        Some(caps) => {
            println!("{caps:?}");
            let x = caps.iter().collect::<Vec<_>>();
            println!("{x:?}");
            let (_, actual_captures) = caps.extract::<2>();
            println!("first: {}", actual_captures[0]);
            println!("seccond: {}", actual_captures[1]);
        }
        None => println!("NO CAPTURES!: s={s}, re={}", re.as_str()),
    };
}
