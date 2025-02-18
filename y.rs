fn foo() -> impl Fn(i32) -> i32 {
    |x| x + 1
}

fn bar(f: &impl Fn(i32) -> i32) -> i32 {
   f(10)
}

fn baz(f: &dyn Fn(i32) -> i32) -> i32 {
  f(10)
}

pub fn main() {
  let f = foo();
  println!("foo()(10):       {}", f(10));
  println!("from bar: f(10): {}", bar(&f));
  println!("from baz: f(10): {}", baz(&f));
}
