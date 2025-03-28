from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.primitives import Sampler
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pandas as pd

class QRNGError(Exception):
    pass

class QuantumRandomNumberGenerator:
    def __init__(self):
        self.sampler = Sampler()

    def generate_bit(self):
        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.measure(0, 0)
        result = self.sampler.run(qc, shots=1).result()
        return list(result.quasi_dists[0].keys())[0]

    def generate_number(self, min_val, max_val):
        return generate_number_safe(min_val, max_val)

    def test_randomness(self, sample_size=1000, num_bits=4):
        numbers = [generate_random_number(num_bits) for _ in range(sample_size)]
        mean = np.mean(numbers)
        std = np.std(numbers)
        observed = np.bincount(numbers)
        expected = np.ones_like(observed) * sample_size / len(observed)
        chi2, p_value = stats.chisquare(observed, expected)
        print(f"Mean: {mean:.2f}, Std Dev: {std:.2f}, Chi2 p-value: {p_value:.4f}")
        plt.hist(numbers, bins=2**num_bits)
        plt.title('Distribution of Quantum RNG')
        plt.show()

def create_multi_qubit_circuit(num_qubits):
    qc = QuantumCircuit(num_qubits, num_qubits)
    for q in range(num_qubits):
        qc.h(q)
    qc.measure(range(num_qubits), range(num_qubits))
    return qc

def run_multi_qubit_circuit(qc, shots=1):
    sampler = Sampler()
    result = sampler.run(qc, shots=shots).result()
    return result.quasi_dists[0]

def binary_to_decimal(bin_str):
    return int(bin_str, 2)

def generate_random_number(num_bits=4):
    qc = create_multi_qubit_circuit(num_bits)
    result = run_multi_qubit_circuit(qc)
    return list(result.keys())[0]

def generate_number_in_range(min_val, max_val):
    num_bits = (max_val - min_val).bit_length()
    while True:
        num = generate_random_number(num_bits)
        if min_val <= num <= max_val:
            return num

def generate_number_safe(min_val, max_val):
    try:
        if not isinstance(min_val, int) or not isinstance(max_val, int):
            raise QRNGError("Inputs must be integers")
        if min_val >= max_val:
            raise QRNGError("min_val must be less than max_val")
        return generate_number_in_range(min_val, max_val)
    except QRNGError as e:
        print(f"QRNG Error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def compare_with_classical(sample_size=1000, num_bits=4):
    import random
    q_nums = [generate_random_number(num_bits) for _ in range(sample_size)]
    c_nums = [random.randint(0, 2**num_bits - 1) for _ in range(sample_size)]
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.hist(q_nums, bins=2**num_bits)
    plt.title("Quantum RNG")
    plt.subplot(1, 2, 2)
    plt.hist(c_nums, bins=2**num_bits)
    plt.title("Classical RNG")
    plt.show()

def analyze_randomness(quantum_nums, classical_nums, title="Randomness Analysis"):
    """Analyze and compare quantum vs classical random numbers"""
    # Create figure with multiple subplots
    plt.figure(figsize=(15, 10))
    
    # 1. Distribution Comparison
    plt.subplot(2, 2, 1)
    data = pd.DataFrame({
        'Quantum': quantum_nums,
        'Classical': classical_nums
    })
    sns.boxplot(data=data)
    plt.title('Distribution Comparison')
    
    # 2. Sequential Pattern Analysis
    plt.subplot(2, 2, 2)
    plt.plot(quantum_nums[:50], label='Quantum', alpha=0.7)
    plt.plot(classical_nums[:50], label='Classical', alpha=0.7)
    plt.title('Sequential Pattern Analysis (first 50 numbers)')
    plt.legend()
    
    # 3. Frequency Analysis
    plt.subplot(2, 2, 3)
    q_freq = Counter(quantum_nums)
    c_freq = Counter(classical_nums)
    x = list(set(list(q_freq.keys()) + list(c_freq.keys())))
    plt.bar([i-0.2 for i in x], [q_freq[i] for i in x], width=0.4, label='Quantum', alpha=0.7)
    plt.bar([i+0.2 for i in x], [c_freq[i] for i in x], width=0.4, label='Classical', alpha=0.7)
    plt.title('Frequency Distribution')
    plt.legend()
    
    # 4. Autocorrelation Analysis
    plt.subplot(2, 2, 4)
    q_auto = pd.Series(quantum_nums).autocorr()
    c_auto = pd.Series(classical_nums).autocorr()
    plt.bar(['Quantum', 'Classical'], [q_auto, c_auto])
    plt.title('Autocorrelation (higher means more predictable)')
    
    plt.tight_layout()
    plt.suptitle(title, y=1.02, fontsize=16)
    plt.show()

def demonstrate_randomness(sample_size=1000, num_bits=4):
    """Generate and compare quantum vs classical random numbers"""
    import random
    
    # Generate numbers
    q_nums = [generate_random_number(num_bits) for _ in range(sample_size)]
    c_nums = [random.randint(0, 2**num_bits - 1) for _ in range(sample_size)]
    
    # Basic statistics
    print("\nBasic Statistics:")
    print(f"{'':15} {'Quantum':>10} {'Classical':>10}")
    print("-" * 35)
    print(f"{'Mean':15} {np.mean(q_nums):10.2f} {np.mean(c_nums):10.2f}")
    print(f"{'Std Dev':15} {np.std(q_nums):10.2f} {np.std(c_nums):10.2f}")
    print(f"{'Min':15} {min(q_nums):10d} {min(c_nums):10d}")
    print(f"{'Max':15} {max(q_nums):10d} {max(c_nums):10d}")
    
    # Chi-square test for uniformity
    q_observed = np.bincount(q_nums)
    c_observed = np.bincount(c_nums)
    expected = np.ones_like(q_observed) * sample_size / len(q_observed)
    
    q_chi2, q_pvalue = stats.chisquare(q_observed, expected)
    c_chi2, c_pvalue = stats.chisquare(c_observed, expected)
    
    print("\nUniformity Test (Chi-square):")
    print(f"{'':15} {'p-value':>10}")
    print("-" * 25)
    print(f"{'Quantum':15} {q_pvalue:10.4f}")
    print(f"{'Classical':15} {c_pvalue:10.4f}")
    print("\nNote: p-value > 0.05 suggests uniform distribution")
    
    # Visual analysis
    analyze_randomness(q_nums, c_nums)

def test_seed_dependency():
    """Demonstrate the effect of seeds in classical vs quantum RNG"""
    import random
    sample_size = 20
    
    # Classical RNG with same seed
    print("\nClassical RNG with same seed:")
    random.seed(42)
    seq1 = [random.randint(0, 15) for _ in range(sample_size)]
    random.seed(42)
    seq2 = [random.randint(0, 15) for _ in range(sample_size)]
    print("Sequence 1:", seq1)
    print("Sequence 2:", seq2)
    print("Are sequences identical?", seq1 == seq2)
    
    # Quantum RNG
    print("\nQuantum RNG (same parameters):")
    seq1 = [generate_random_number(4) for _ in range(sample_size)]
    seq2 = [generate_random_number(4) for _ in range(sample_size)]
    print("Sequence 1:", seq1)
    print("Sequence 2:", seq2)
    print("Are sequences identical?", seq1 == seq2)

# Add these functions to your existing code and use them like this:
if __name__ == "__main__":
    print("=== Comprehensive Randomness Analysis ===")
    demonstrate_randomness(sample_size=1000, num_bits=4)
    
    print("\n=== Seed Dependency Demonstration ===")
    test_seed_dependency()
