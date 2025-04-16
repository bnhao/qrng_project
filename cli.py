import argparse
from qrng_core import QuantumRandomNumberGenerator, analyze_randomness
import random
import numpy as np
from scipy import stats

def demonstrate_fair_comparison():
    sample_size = 1000
    min_val = 1
    max_val = 100
    
    # Generate numbers using both methods
    qrng = QuantumRandomNumberGenerator()
    quantum_nums = [qrng.generate_number(min_val, max_val) for _ in range(sample_size)]
    classical_nums = [random.randint(min_val, max_val) for _ in range(sample_size)]
    
    # Basic statistics
    print("\n2. Basic Statistics:")
    print(f"{'':15} {'Quantum':>10} {'Classical':>10}")
    print("-" * 35)
    print(f"{'Mean':15} {np.mean(quantum_nums):10.2f} {np.mean(classical_nums):10.2f}")
    print(f"{'Std Dev':15} {np.std(quantum_nums):10.2f} {np.std(classical_nums):10.2f}")
    print(f"{'Min':15} {min(quantum_nums):10d} {min(classical_nums):10d}")
    print(f"{'Max':15} {max(quantum_nums):10d} {max(classical_nums):10d}")
    
    # Chi-square test for uniformity
    print("\n3. Uniformity Test (Chi-square):")
    q_observed = np.bincount(quantum_nums, minlength=max_val+1)[min_val:]
    c_observed = np.bincount(classical_nums, minlength=max_val+1)[min_val:]
    expected = np.ones_like(q_observed) * sample_size / len(q_observed)
    
    q_chi2, q_pvalue = stats.chisquare(q_observed, expected)
    c_chi2, c_pvalue = stats.chisquare(c_observed, expected)
    
    print(f"{'':15} {'p-value':>10}")
    print("-" * 25)
    print(f"{'Quantum':15} {q_pvalue:10.4f}")
    print(f"{'Classical':15} {c_pvalue:10.4f}")
    
    # Demonstrate predictability
    print("\n4. Predictability Test:")
    print("\nClassical RNG (with same seed):")
    random.seed(70)
    classical_seq1 = [random.randint(min_val, max_val) for _ in range(5)]
    random.seed(70)
    classical_seq2 = [random.randint(min_val, max_val) for _ in range(5)]
    print("Sequence 1:", classical_seq1)
    print("Sequence 2:", classical_seq2)
    
    print("\nQuantum RNG:")
    quantum_seq1 = [qrng.generate_number(min_val, max_val) for _ in range(5)]
    quantum_seq2 = [qrng.generate_number(min_val, max_val) for _ in range(5)]
    print("Sequence 1:", quantum_seq1)
    print("Sequence 2:", quantum_seq2)
    
    # Visual analysis
    print("\n5. Visual Analysis:")
    analyze_randomness(quantum_nums, classical_nums, "Randomness Analysis")

def main():
    parser = argparse.ArgumentParser(description='Quantum vs Classical RNG CLI')
    parser.add_argument('--min', type=int, default=0)
    parser.add_argument('--max', type=int, default=10)
    parser.add_argument('--count', type=int, default=1)
    parser.add_argument('--test', action='store_true', help='Compare quantum vs classical randomness')
    parser.add_argument('--quantum', action='store_true', help='Use quantum random number generator')
    args = parser.parse_args()

    if args.test:
        demonstrate_fair_comparison()
    else:
        if args.quantum:
            qrng = QuantumRandomNumberGenerator()
            for _ in range(args.count):
                print(qrng.generate_number(args.min, args.max))
        else:
            for _ in range(args.count):
                print(random.randint(args.min, args.max))

if __name__ == "__main__":
    main()