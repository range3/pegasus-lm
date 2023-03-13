from multiprocessing import Pool

def f(a, b):
    return a + b

def main():
    with Pool(10) as p:
        result = p.starmap(f, zip(range(100, 110), range(10)))

    print(result)

if __name__ == "__main__":
    main()
