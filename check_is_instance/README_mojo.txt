Int, FLot32

let c: type = b       immutable
var c: type = b       mutable

fn name(p1: t1, ...) -> rt:
    ...


struct AStruct:
    var f1: int
    var f2: int

    fn __init__(inout self, first: Int, second: Int):
        self.first = first
        self.second = second

    fn __lt__(self, rhs: MyPair) -> Bool:
        return self.first < rhs.first or
              (self.first == rhs.first and
               self.second < rhs.second)

