import argparse
import random

def demonstrate_predictability():
    print("\nDemonstrating predictability of pseudo-random numbers:")
    print("Using seed = 42 for first sequence:")
    random.seed(42)
    sequence1 = [random.randint(1, 100) for _ in range(5)]
    print("First sequence:", sequence1)

    print("\nGenerating some other random numbers in between...")
    random.randint(1, 100)
    random.randint(1, 100)

    print("\nUsing seed = 42 again for second sequence:")
    random.seed(42)
    sequence2 = [random.randint(1, 100) for _ in range(5)]
    print("Second sequence:", sequence2)
    
    print("\nAs you can see, both sequences are identical when using the same seed!")
    print("This demonstrates that classic random numbers are predictable when you know t" \
    "he seed.")

def main():
    parser = argparse.ArgumentParser(description='Classic RNG CLI')
    parser.add_argument('--min', type=int, default=0)
    parser.add_argument('--max', type=int, default=100)
    parser.add_argument('--count', type=int, default=5)
    parser.add_argument('--test', action='store_true', help='Demonstrate predictability of pseudo-random numbers')
    args = parser.parse_args()

    if args.test:
        demonstrate_predictability()
    else:
        for _ in range(args.count):
            print(random.randint(args.min, args.max))

if __name__ == "__main__":
    main()