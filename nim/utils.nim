iterator zip*(a: any, b: any): array =
    # Check seqs length
    if a.len != b.len:
        var e = new IndexError
        e.msg = "Each seqs must be the same length."
        raise e
        
    for i in 0..a.len-1:
        yield [a[i], b[i]]


proc main(): void =
    let a = @[0, 1, 2, 3, 4]
    let b = [5, 6, 7, 8, 9]

    for i in zip(a, b):
        echo i[0], ' ', i[1]


if isMainModule:
    main()