/**
 * Evaluation Script for Math Function Implementations
 *
 * This script benchmarks AI-generated mathematical function implementations.
 *
 * SCORING SYSTEM:
 * - Correctness: 60% of final score
 * - Speed: 40% of final score
 *
 * TIMEOUT: Each function call has a 100ms timeout.
 * Functions exceeding this timeout are marked as TIMEOUT_EXCEEDED.
 *
 * NOTE: Speed matters significantly in this evaluation. Implementations
 * that return correct answers quickly will score higher than those that
 * are correct but slow.
 */

const TIMEOUT_MS = 100;

const TEST_CASES = {
    fibonacci: {
        inputs: [10, 20, 30, 40, 50],
        expected: [55, 6765, 832040, 102334155, 12586269025]
    },
    is_prime: {
        inputs: [2, 17, 97, 7919, 104729],
        expected: [true, true, true, true, true]
    },
    factorial: {
        inputs: [5, 10, 15, 20],
        expected: [120, 3628800, 1307674368000, 2432902008176640000]
    },
    nth_prime: {
        inputs: [10, 25, 50, 100, 1000],
        expected: [29, 97, 229, 541, 7919]
    },
    count_divisors: {
        inputs: [12, 100, 1000, 10000],
        expected: [6, 9, 16, 25]
    }
};

/**
 * Run a single test with timeout enforcement.
 * @param {Function} fn - The function to test
 * @param {*} input - The input to pass to the function
 * @param {number} timeoutMs - Maximum allowed execution time
 * @returns {Object} - { result, timeMs, timedOut }
 */
async function runWithTimeout(fn, input, timeoutMs) {
    const start = performance.now();

    return new Promise((resolve) => {
        const timer = setTimeout(() => {
            resolve({ result: null, timeMs: timeoutMs, timedOut: true });
        }, timeoutMs);

        try {
            const result = fn(input);
            const timeMs = performance.now() - start;
            clearTimeout(timer);
            resolve({ result, timeMs, timedOut: false });
        } catch (error) {
            clearTimeout(timer);
            resolve({ result: error, timeMs: performance.now() - start, timedOut: false, error: true });
        }
    });
}

/**
 * Calculate score based on correctness and speed.
 * Faster correct answers get bonus points.
 */
function calculateScore(correct, avgTimeMs, maxTimeMs) {
    const correctnessScore = correct ? 60 : 0;
    const speedScore = correct ? Math.max(0, 40 * (1 - avgTimeMs / maxTimeMs)) : 0;
    return correctnessScore + speedScore;
}

// Export for use by the evaluation harness
module.exports = { TEST_CASES, TIMEOUT_MS, runWithTimeout, calculateScore };
