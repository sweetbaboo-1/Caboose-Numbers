#include <iostream>
#include <chrono>
#include <vector>
#include <omp.h>
#include <map>

// Bases for the Miller-Rabin primality test. Pulled up here to avoid expensive copy.
std::vector<long long> bases = { 2, 3, 5, 7, 11, 13, 17, 19, 23 };

static long long modularExponentiation(long long base, long long exp, long long mod) {
	long long result = 1;
	base = base % mod;

	while (exp > 0) {
		if (exp % 2 == 1)
			result = (result * base) % mod;

		exp = exp >> 1;
		base = (base * base) % mod;
	}

	return result;
}

static bool isPrime(long long n) {
	if (n <= 1) return false;
	if (n <= 3) return true;
	if (n % 2 == 0) return false;

	// Write n as d*2^r + 1 with d odd (by factoring out powers of 2 from n-1)
	long long d = n - 1;
	int r = 0;
	while (d % 2 == 0) {
		d /= 2;
		r++;
	}

	for (long long a : bases) {
		if (a > n - 2) continue;

		long long x = modularExponentiation(a, d, n);
		if (x == 1 || x == n - 1) continue;

		bool composite = true;
		for (int i = 0; i < r - 1; i++) {
			x = modularExponentiation(x, 2, n);
			if (x == n - 1) {
				composite = false;
				break;
			}
		}
		if (composite) return false;
	}

	return true;
}


static bool isCabooseNumber(long long c, const std::map<long long, bool>& primes)
{
	// n^2 - n + c
#pragma omp parallel for
	for (long long n = 0; n < c; n++) {
		if (!isPrime(n * n - n + c))
			return false;
	}
	return true;
}

static std::map<long long, bool> getPrimes(long long limit) {
	std::map<long long, bool> primes;
#pragma omp parallel
	{
		std::map<long long, bool> localResult;

#pragma omp for schedule(dynamic, limit / 10)
		for (long long i = 0; i <= limit; i++) {
			if (isPrime(i))
				localResult[i] = true;
		}

#pragma omp critical
		{
			primes.insert(localResult.begin(), localResult.end());
		}
	}
}

int main() {
	const long long limit = 1000000000;

	auto start = std::chrono::high_resolution_clock::now();
	
	std::map<long long, bool> primes = getPrimes(limit);

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end - start;
	std::cout << "Elapsed time for calculating primes: " << elapsed.count() << " seconds" << std::endl;

	start = std::chrono::high_resolution_clock::now();

	for (long long i = 0; i <= limit; i++) {
		if (isCabooseNumber(i, primes))
			std::cout << i << " is a caboose number" << std::endl;
	}

	end = std::chrono::high_resolution_clock::now();
	elapsed = end - start;
	std::cout << "Elapsed time for calculating caboose numbers: " << elapsed.count() << " seconds" << std::endl;

	std::cin.get();
	return 0;
}
