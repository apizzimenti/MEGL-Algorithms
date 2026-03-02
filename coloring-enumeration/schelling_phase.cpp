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

    // Dynamics
    bool random_agent_order = true;

    //
    int max_steps = 200000;

    // Experiment
    int repetitions = 2000;
    int seed = 42;

};

struct RunStats {
    bool absorbed_by_H = false; // indicator {T_abs <= H}
    int T_abs = -1; // exact absorption time (number of steps until absorption, or -1 if not absorbed)
    int tau_H = 0; // min(T_abs, H)
    double S_stop_H = 0.0; // segregation index at time tau_H
};

struct AggregateStats {
    long long total_runs = 0;
    
    // During accumulation these are sums;
    // after finalize_aggregate() they become means.
    double p_abs_H = 0.0;
    double tau_H = 0.0;
    double s_H = 0.0;
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

bool is_happy(const Grid & G, const std::vector<uint8_t> & state, int idx, double tau) {
    if (state[idx] == 0) return true; // vacancy is always "happy"
    return local_same_fraction(G, state, idx) >= tau;
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

// Used for choosing the nearest satisfaction vacancy
int manhattan_distance(const Grid& G, int u, int v) {
    int ur = u / G.cols, uc = u % G.cols;
    int vr = v / G.cols, vc = v % G.cols;
    return std::abs(ur - vr) + std::abs(uc - vc);
}

// One Schelling
bool do_one_step(const Grid& G, std::vector<uint8_t> & state, const Params& P, std::mt19937& rng) {
    std::vector<int> unhappy_agents;
    std::vector<int> vacancies;

    unhappy_agents.reserve(G.N);
    vacancies.reserve(G.N);

    for (int i=0; i < G.N; ++i) {
        if (state[i] == 0) {
            vacancies.push_back(i);
        } else if (!is_happy(G, state, i, P.tau)) {
            unhappy_agents.push_back(i);
        }
    }
    if (unhappy_agents.empty()) return false; // already absorbed
    if (vacancies.empty()) return false; // no place to move

    if (P.random_agent_order) {
        std::shuffle(unhappy_agents.begin(), unhappy_agents.end(), rng);
    }
    for (int agent_idx : unhappy_agents) {
        int best_dist = std::numeric_limits<int>::max();
        std::vector<int> best_vacancies;
        best_vacancies.reserve(8);

        for (int v : vacancies) {
            // try move agent to vacancy v
            std::swap(state[agent_idx], state[v]);
            bool satisfactory = is_happy(G, state, v, P.tau);
            std::swap(state[agent_idx], state[v]); // swap back

            if (!satisfactory) continue;
            int d = manhattan_distance(G, agent_idx, v);
            if (d < best_dist) {
                best_dist = d;
                best_vacancies.clear();
                best_vacancies.push_back(v);
            } else if (d == best_dist) {
                best_vacancies.push_back(v);
            }
        }
        if (!best_vacancies.empty()) {
            std::uniform_int_distribution<int> dist(0, static_cast<int>(best_vacancies.size()) - 1);
            int chosen_v = best_vacancies[dist(rng)];
            std::swap(state[agent_idx], state[chosen_v]);
            return true; // one move made
        }
    }
    return false; // no unhappy agent found any legal improving move
}

bool is_absorbing_state(const Grid& G,
                        const std::vector<uint8_t>& state,
                        const Params& P) {
    std::vector<uint8_t> probe = state;
    std::mt19937 dummy_rng(123456789); // deterministic probe RNG
    return !do_one_step(G, probe, P, dummy_rng);
}
// Main simulation loop for one run, with cycle detection
RunStats simulate_one(const Grid& G,
                      std::vector<uint8_t> state,
                      const Params& P,
                      std::mt19937& rng) {
    RunStats R;
    const int H = P.max_steps;

    // Interpretation:
    // X_0 = initial state
    // after 1 successful move -> X_1
    // after 2 successful moves -> X_2
    // ...
    // after k successful moves -> X_k

    for (int t = 0; t < H; ++t) {
        bool moved = do_one_step(G, state, P, rng);

        if (!moved) {
            // Current state is absorbing, so T_abs = t
            R.absorbed_by_H = true;
            R.T_abs = t;
            R.tau_H = t;
            R.S_stop_H = segregation_score(G, state);
            return R;
        }
    }

    // We have simulated exactly up to X_H.
    // Need to check whether absorption occurs exactly at time H.
    if (is_absorbing_state(G, state, P)) {
        R.absorbed_by_H = true;
        R.T_abs = H;
    } else {
        R.absorbed_by_H = false;
        R.T_abs = -1; // means T_abs > H in the truncated sense
    }

    R.tau_H = H;
    R.S_stop_H = segregation_score(G, state);
    return R;
}

void add_run(AggregateStats& A, const RunStats& R) {
    ++A.total_runs;
    A.p_abs_H += (R.absorbed_by_H ? 1.0 : 0.0);
    A.tau_H += static_cast<double>(R.tau_H);
    A.s_H += R.S_stop_H;
}

void finalize_aggregate(AggregateStats& A) {
    if (A.total_runs == 0) return;
    A.p_abs_H /= A.total_runs;
    A.tau_H   /= A.total_runs;
    A.s_H     /= A.total_runs;
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

        std::mt19937 rng(P.seed + 1009 * thread_id + 7919 + static_cast<int>(1000 * P.a_frac) + static_cast<int>(10000 * P.vacancy_frac)+ static_cast<int>(100000 * P.rows *1000 + P.cols)); // unique seed per thread and parameter point
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
            total.p_abs_H += local.p_abs_H;
            total.tau_H += local.tau_H;
            total.s_H += local.s_H;
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
    P.tau = 1.0 / 3.0;
    P.repetitions = 2000;

    auto horizon_ks = make_horizon_multipliers();

    auto program_start = std::chrono::steady_clock::now();
    long long total_simulation_runs = 0;

    std::ofstream fout("phase_sweep.csv");
    fout << "row,col,N,a_frac,b_frac,vacancy_frac,k,H,p_abs_H,tau_H,s_H\n";

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

                    // New sweep over k, where H = ceil(kN)
                    for (double k : horizon_ks) {
                        P.max_steps = std::max(1, static_cast<int>(std::ceil(k * P.N)));

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
                                  << " p_abs_H=" << A.p_abs_H
                                  << " tau_H=" << A.tau_H
                                  << " s_H=" << A.s_H
                                  << "\n";

                        fout << row << ","
                             << col << ","
                             << P.N << ","
                             << a << ","
                             << b << ","
                             << P.vacancy_frac << ","
                             << k << ","
                             << P.max_steps << ","
                             << A.p_abs_H << ","
                             << A.tau_H << ","
                             << A.s_H << "\n";
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
// g++ -O3 -march=native -fopenmp -std=c++17 schelling_phase.cpp -o schelling_phase