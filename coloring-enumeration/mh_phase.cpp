#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>
#include <limits>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

struct Params {
    int rows = 20;
    int cols = 20;
    int N = rows * cols;

    // Labels: 0 = vacancy, 1 = A, 2 = B
    double vacancy_frac = 0.1;
    double a_frac = 0.45;

    // Satisfaction threshold (0.0 to 1.0)
    // agent is happy if same-type-neighbors fraction >= tau
    double tau = 1.0/3.0;

    // Proposal parameter:
    // source agent i is selected with weight exp(lambda_prop * H_i),
    // where H_i = number of unlike occupied neighbors
    double lambda_prop = 1.0;

    // Target distribution parameter:
    // pi(x) proportional to exp(beta_target * target_score(x))
    double beta_target = 4.0;

    // Make the chain explicitly lazy => guarantees aperiodicity
    double lazy_prob = 0.05;
    //
    int max_steps = 1000;

    // Experiment
    int repetitions = 2000;
    int seed = 42;

};

// Per-run output for one H-step MH trajectory
struct RunStats {
    int H = 0;
    int accepted_moves = 0;
    double initial_score = 0.0;
    double final_score = 0.0;
};

// Aggregated Monte Carlo output
struct AggregateStats {
    long long total_runs = 0;
    double mean_initial_score = 0.0;
    double mean_final_score = 0.0;
    double mean_accept_rate = 0.0;
};

// Grid indexing and neighbor calculations
struct Grid {
    int rows, cols, N;
    std::vector<std::vector<int>> nbrs;

    Grid(int r, int c) : rows(r), cols(c), N(r * c), nbrs(r * c) {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                int id = i * cols + j;

                // 8-neighbor neighborhood
                for (int di = -1; di <= 1; ++di) {
                    for (int dj = -1; dj <= 1; ++dj) {
                        if (di == 0 && dj == 0) continue;
                        int ni = i + di;
                        int nj = j + dj;
                        if (0 <= ni && ni < rows && 0 <= nj && nj < cols) {
                            nbrs[id].push_back(ni * cols + nj);
                        }
                    }
                }
            }
        }
    }
};

// Count conversion from proportions to counts
struct Counts {
    int Qv = 0;
    int Qa = 0;
    int Qb = 0;
};

Counts proportions_to_counts(const Params & P) {
    Counts C;
    C.Qv = static_cast<int>(std::round(P.vacancy_frac * P.N));
    C.Qa = static_cast<int>(std::round(P.a_frac * P.N));
    C.Qb = P.N - C.Qv - C.Qa;
    
    if (C.Qv < 0 || C.Qa < 0 || C.Qb < 0) {
        throw std::runtime_error("Invalid parameters: negative counts");
    }
    return C;
}

// Utility and segregation measurements
double local_same_fraction(const Grid & G, const std::vector<uint8_t> & state, int idx) {
    uint8_t t = state[idx];
    if (t == 0) return 0.0; // vacancy has no neighbors of the same type

    int same = 0;
    int occupied_neighbors = 0;
    for (int nb: G.nbrs[idx]) {
        if (state[nb] == 0) continue; // skip vacancies
        occupied_neighbors++;
        if (state[nb] == t) same++;
    }

    if (occupied_neighbors == 0) return 1.0; // no occupied neighbors means fully satisfied
    return static_cast<double>(same) / occupied_neighbors;
}

double mean_satisfaction(const Grid & G, const std::vector<uint8_t> & state) {
    double total = 0.0;
    int occupied = 0;
    for (int i = 0; i < G.N; ++i) {
        if (state[i] == 0) continue; // skip vacancies
        total += local_same_fraction(G, state, i);
        occupied++;
    }
    return occupied ? total /occupied : 0.0; // if no occupied cells, consider fully satisfied
}

double segregation_score(const Grid & G, const std::vector<uint8_t> & state) {
    return mean_satisfaction(G, state);
}

int unlike_neighbor_count(const Grid& G, const std::vector<uint8_t>& state, int idx) {
    uint8_t t = state[idx];
    if (t == 0) return 0;

    int count = 0;
    for (int nb : G.nbrs[idx]) {
        if (state[nb] == 0) continue;      // ignore vacancies
        if (state[nb] != t) ++count;       // count unlike occupied neighbors
    }
    return count;
}

double target_score(const Grid& G, const std::vector<uint8_t>& state) {
    double total = 0.0;
    for (int i = 0; i < G.N; ++i) {
        if (state[i] == 0) continue;
        total += local_same_fraction(G, state, i);
    }
    return total;
}

// Random fixed-count initialization
std::vector<uint8_t> random_initial_state(const Params & P, std::mt19937 & rng) {
    Counts C = proportions_to_counts(P);

    std::vector<uint8_t> state(P.N, 0);
    int pos = 0;
    for (int i = 0; i< C.Qv; ++i) state[pos++] = 0; // vacancy
    for (int i = 0; i< C.Qa; ++i) state[pos++] = 1; // A
    for (int i = 0; i< C.Qb; ++i) state[pos++] = 2; // B
    std::shuffle(state.begin(), state.end(), rng);
    return state;
}

std::vector<int> occupied_positions(const std::vector<uint8_t>& state) {
    std::vector<int> occ;
    occ.reserve(state.size());
    for (int i = 0; i < (int)state.size(); ++i) {
        if (state[i] != 0) occ.push_back(i);
    }
    return occ;
}

double source_weight_at(const Grid& G,
                        const std::vector<uint8_t>& state,
                        int idx,
                        const Params& P) {
    int H_i = unlike_neighbor_count(G, state, idx);
    return std::exp(P.lambda_prop * static_cast<double>(H_i));
}

double total_source_weight(const Grid& G,
                           const std::vector<uint8_t>& state,
                           const Params& P) {
    double total = 0.0;
    for (int i = 0; i < G.N; ++i) {
        if (state[i] == 0) continue;
        total += source_weight_at(G, state, i, P);
    }
    return total;
}

// Used for choosing the nearest satisfaction vacancy
int manhattan_distance(const Grid& G, int u, int v) {
    int ur = u / G.cols, uc = u % G.cols;
    int vr = v / G.cols, vc = v % G.cols;
    return std::abs(ur - vr) + std::abs(uc - vc);
}

// One Schelling
bool mh_one_step(const Grid& G,
                 std::vector<uint8_t>& state,
                 const Params& P,
                 std::mt19937& rng) {
    // Explicit laziness => guarantees aperiodicity
    {
        std::uniform_real_distribution<double> unif01(0.0, 1.0);
        if (unif01(rng) < P.lazy_prob) {
            return false; // stayed put
        }
    }

    // List occupied positions
    std::vector<int> occ = occupied_positions(state);
    if (occ.empty()) {
        return false; // all vacancies => singleton state
    }

    // Build source weights
    std::vector<double> src_weights;
    src_weights.reserve(occ.size());
    for (int idx : occ) {
        src_weights.push_back(source_weight_at(G, state, idx, P));
    }

    double W_x = std::accumulate(src_weights.begin(), src_weights.end(), 0.0);
    if (W_x <= 0.0) return false;

    // Choose source i with probability proportional to exp(lambda * H_i)
    std::discrete_distribution<int> src_dist(src_weights.begin(), src_weights.end());
    int src_pos_in_occ = src_dist(rng);
    int i = occ[src_pos_in_occ];
    uint8_t src_label = state[i];

    // Choose random destination j with different label from source label.
    // This preserves the counts after swap and ensures a real proposed move.
    std::vector<int> destinations;
    destinations.reserve(G.N);
    for (int j = 0; j < G.N; ++j) {
        if (j == i) continue;
        if (state[j] != src_label) destinations.push_back(j);
    }

    if (destinations.empty()) {
        return false; // degenerate singleton-type state
    }

    std::uniform_int_distribution<int> dst_dist(0, (int)destinations.size() - 1);
    int j = destinations[dst_dist(rng)];

    // Current quantities
    double score_x = target_score(G, state);
    double w_forward = src_weights[src_pos_in_occ];

    // Proposed state y: swap i and j
    std::vector<uint8_t> proposal = state;
    std::swap(proposal[i], proposal[j]);

    // Compute reverse proposal quantities in y
    // Reverse move selects the same agent label now sitting at j, then swaps back to i.
    double score_y = target_score(G, proposal);
    double W_y = total_source_weight(G, proposal, P);
    double w_reverse = source_weight_at(G, proposal, j, P);

    // Target density ratio:
    // pi(y)/pi(x) = exp(beta * (score_y - score_x))
    double log_pi_ratio = P.beta_target * (score_y - score_x);

    // Proposal ratio:
    // q(y|x) = [w_forward / W_x] * [1 / (# positions with label != src_label)]
    // q(x|y) = [w_reverse / W_y] * [1 / (# positions with label != src_label)]
    // destination-count factors cancel because label counts are preserved
    double log_q_ratio = std::log(w_reverse) - std::log(W_y)
                       - std::log(w_forward) + std::log(W_x);

    double log_accept_ratio = log_pi_ratio + log_q_ratio;

    std::uniform_real_distribution<double> unif01(0.0, 1.0);
    double u = unif01(rng);

    if (log_accept_ratio >= 0.0 || std::log(u) < log_accept_ratio) {
        state.swap(proposal);
        return true; // accepted
    } else {
        return false; // rejected => self-loop
    }
}

// Main simulation loop for one run
RunStats simulate_one(const Grid& G,
                      std::vector<uint8_t> state,
                      const Params& P,
                      std::mt19937& rng) {
    RunStats R;
    R.H = P.max_steps;
    R.initial_score = segregation_score(G, state);

    int accepts = 0;
    for (int t = 0; t < P.max_steps; ++t) {
        bool accepted = mh_one_step(G, state, P, rng);
        if (accepted) ++accepts;
    }

    R.accepted_moves = accepts;
    R.final_score = segregation_score(G, state);
    return R;
}

void add_run(AggregateStats& A, const RunStats& R) {
    ++A.total_runs;
    A.mean_initial_score += R.initial_score;
    A.mean_final_score += R.final_score;
    A.mean_accept_rate += static_cast<double>(R.accepted_moves) / std::max(1, R.H);
}

void finalize_aggregate(AggregateStats& A) {
    if (A.total_runs == 0) return;
    A.mean_initial_score /= A.total_runs;
    A.mean_final_score /= A.total_runs;
    A.mean_accept_rate /= A.total_runs;
}

// Parameter point in the phase sweep
AggregateStats run_experiment_at_fraction(Params P) {
    Grid G(P.rows, P.cols);
    AggregateStats total;

    #pragma omp parallel
    {
        int thread_id = 0;
        #ifdef _OPENMP
        thread_id = omp_get_thread_num();
        #endif

        std::mt19937 rng(
            P.seed
            + 1009 * thread_id
            + 7919
            + static_cast<int>(1000 * P.a_frac)
            + static_cast<int>(10000 * P.vacancy_frac)
            + static_cast<int>(P.rows * 1000 + P.cols)
            + static_cast<int>(100 * P.lambda_prop)
            + static_cast<int>(100 * P.beta_target)
            + P.max_steps
        );

        AggregateStats local;

        #pragma omp for
        for (int rep = 0; rep < P.repetitions; ++rep) {
            auto state = random_initial_state(P, rng);
            RunStats R = simulate_one(G, state, P, rng);
            add_run(local, R);
        }

        #pragma omp critical
        {
            total.total_runs += local.total_runs;
            total.mean_initial_score += local.mean_initial_score;
            total.mean_final_score += local.mean_final_score;
            total.mean_accept_rate += local.mean_accept_rate;
        }
    }

    finalize_aggregate(total);
    return total;
}

std::vector<double> make_horizon_multipliers() {
    std::vector<double> ks;
    for (int j = 1; j <= 10; ++j) {
        ks.push_back(0.1 * j);   // 0.1, 0.2, ..., 1.0
    }
    return ks;
}

int main() {
    Params P;
    P.rows = 20;
    P.cols = 20;
    P.N = P.rows * P.cols;

    P.vacancy_frac = 0.1;
    P.a_frac = 0.45;

    P.lambda_prop = 1.0;
    P.beta_target = 4.0;
    P.lazy_prob = 0.05;

    P.repetitions = 500;

    auto horizon_ks = make_horizon_multipliers();

    auto program_start = std::chrono::steady_clock::now();
    long long total_simulation_runs = 0;

    std::ofstream fout("mh_schelling_phase_sweep.csv");
    fout << "row,col,N,a_frac,b_frac,vacancy_frac,"
            "lambda_prop,beta_target,lazy_prob,"
            "k,H,mean_initial_score,mean_final_score,mean_accept_rate\n";

    for (int row = 3; row <= 20; ++row) {
        for (int col = row; col <= 20; ++col) {
            P.rows = row;
            P.cols = col;
            P.N = P.rows * P.cols;

            for (double v = 0.00; v <= 1.0000001; v += 0.01) {
                P.vacancy_frac = v;

                for (double a = 0.00; a <= 1.0 - P.vacancy_frac + 1e-12; a += 0.01) {
                    if (a + P.vacancy_frac > 1.0 + 1e-12) break;
                    P.a_frac = a;

                    Counts C;
                    try {
                        C = proportions_to_counts(P);
                    } catch (...) {
                        continue;
                    }

                    if (C.Qb < 0) continue;

                    double b = 1.0 - P.vacancy_frac - P.a_frac;

                    for (double k : horizon_ks) {
                        P.max_steps = std::max(1, (int)std::ceil(k * P.N));

                        AggregateStats A = run_experiment_at_fraction(P);
                        total_simulation_runs += A.total_runs;

                        std::cout << "rows=" << row
                                  << " cols=" << col
                                  << " N=" << P.N
                                  << " a=" << a
                                  << " b=" << b
                                  << " vacancy_frac=" << P.vacancy_frac
                                  << " k=" << k
                                  << " H=" << P.max_steps
                                  << " mean_initial_score=" << A.mean_initial_score
                                  << " mean_final_score=" << A.mean_final_score
                                  << " mean_accept_rate=" << A.mean_accept_rate
                                  << "\n";

                        fout << row << ","
                             << col << ","
                             << P.N << ","
                             << a << ","
                             << b << ","
                             << P.vacancy_frac << ","
                             << P.lambda_prop << ","
                             << P.beta_target << ","
                             << P.lazy_prob << ","
                             << k << ","
                             << P.max_steps << ","
                             << A.mean_initial_score << ","
                             << A.mean_final_score << ","
                             << A.mean_accept_rate << "\n";
                    }
                }
            }
        }
    }

    fout.close();

    auto program_end = std::chrono::steady_clock::now();
    double elapsed_s = std::chrono::duration<double>(program_end - program_start).count();
    int total_seconds = static_cast<int>(std::lround(elapsed_s));
    int hours = total_seconds / 3600;
    int minutes = (total_seconds % 3600) / 60;
    int seconds = total_seconds % 60;

    std::cout << "Total simulation runs performed: " << total_simulation_runs << "\n";
    std::cout << "Total wall time: "
              << hours << "h "
              << minutes << "m "
              << seconds << "s\n";

    return 0;
}

// Compile command
// g++-15 -O3 -march=native -fopenmp -std=c++17 mh_phase.cpp -o mh_phase