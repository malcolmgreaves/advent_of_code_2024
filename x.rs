
trait One {
    fn one(&self) -> i32;
}

trait Two {
    fn two(&self) -> i32;
}

trait Combined : One + Two {}

trait Extended : Combined {
    
    fn three(&self) -> i32;

    fn four(&self) -> i32 {
        self.one() + self.two() + self.three()
    }
}

struct O {}

impl One for O {
    fn one(&self) -> i32 { 
        1
    }
}

struct T {}

impl Two for T {
    fn two(&self) -> i32 {
        20
    }   
}

struct C(O, T);

//  ** DOES NOT COMPILE **
// impl Combined for C {
//     fn one(&self) -> i32 {
//         self.0.one()
//     }
//     fn two(&self) -> i32 {
//         self.1.two()
//     }
// }

impl One for C {
    fn one(&self) -> i32 {
        self.0.one()
    }
}

impl Two for C {
    fn two(&self) -> i32 {
        self.1.two()
    }
}

// impl Combined for C {

// }

// struct E {o: O, t: T}

// impl Extended for E {
//     fn one(&self) -> i32 {
//         self.o.one()
//     }

//     fn two(&self) -> i32 {
//         self.t.two()
//     }

//     fn three(&self) -> i32 {
//         300
//     }
// }


fn needs_combined(c: dyn Combined) {
    println!("c.one(): {} | c.two(): {}", c.one(), c.two());
}


fn main() {

    let c = C(O{}, T{});



}
