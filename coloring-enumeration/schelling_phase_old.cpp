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
    int max_steps = 200000;

    // Experiment
    int repetitions = 2000;
    int seed = 42;

    // Diagnostics for "works well"
    int min_good_steps = 50;
    double min_good_segregation = 0.75;
};

struct RunStats {
    bool absorbed = false;
    bool cycle_detected = false;
    int steps = 0;
    double initial_satisfaction = 0.0;
    double final_satisfaction = 0.0;
    double initial_segregation = 0.0;
    double final_segregation = 0.0;
};

struct AggregateStats {
    long long total_runs = 0;
    long long absorbed_runs = 0;
    long long cycle_runs = 0;
    long long good_runs = 0;
    double mean_steps = 0.0;
    double mean_final_satisfaction = 0.0;
    double mean_final_segregation = 0.0;
};

// Grid indexing and neighbor calculations
struct Grid {
    int rows, cols, N;
    std::vector<std::vector<int>> nbrs; // neighbors for each cell

    Grid(int r, int c) : rows(r), cols(c), N(r*c), nbrs(r*c) {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                int id = i * cols + j;
                if (i > 0) nbrs[id].push_back((i-1)*cols + j); // up
                if (i < rows-1) nbrs[id].push_back((i+1)*cols + j); // down
                if (j > 0) nbrs[id].push_back(i*cols + (j-1)); // left
                if (j < cols-1) nbrs[id].push_back(i*cols + (j+1)); // right
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

double segregation_index(const Grid & G, const std::vector<uint8_t> & state) {
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

// Hashing states for cycle detection
uint64_t hash_state(const std::vector<uint8_t> & state) {
    uint64_t h = 1099511628211ULL; // FNV-1a 64-bit offset basis
    for (uint8_t x : state) {
        h ^= static_cast<uint64_t>(x + 1); // +1 to avoid zero values
        h *= 1099511628211ULL; // FNV-1a prime
    }
    return h;
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
        double old_u = local_same_fraction(G, state, agent_idx);

        std::vector<int> improving_vacancies;
        improving_vacancies.reserve(vacancies.size());

        uint8_t t = state[agent_idx];
        for (int v : vacancies) {
            if (v == agent_idx) continue; // skip current location

            // try move agent to vacancy v
            std::swap(state[agent_idx], state[v]);
            double new_u = local_same_fraction(G, state, v);
            std::swap(state[agent_idx], state[v]); // swap back

            if (new_u > old_u) {
                    improving_vacancies.push_back(v);
            }
        }
        if (!improving_vacancies.empty()) {
            std::uniform_int_distribution<int> dist(0, (int)improving_vacancies.size() - 1);
            int chosen_v = improving_vacancies[dist(rng)];
            std::swap(state[agent_idx], state[chosen_v]);
            return true; // one move made 
        }    
    }
    return false; // no unhappy agent found any legal improving move
}

// Main simulation loop for one run, with cycle detection
RunStats simulate_one(const Grid& G, std::vector<uint8_t> state, const Params& P, std::mt19937& rng) {
    RunStats R;
    R.initial_satisfaction = mean_satisfaction(G, state);
    R.initial_segregation = segregation_index(G, state);

    std::unordered_set<uint64_t> seen;
    seen.reserve((size_t)std::min(P.max_steps, 50000)); // limit memory usage for cycle detection
    for (int step = 0; step < P.max_steps; ++step) {
        uint64_t h = hash_state(state); // hash before move
        if (seen.find(h) != seen.end()) {
            R.cycle_detected = true;
            R.steps = step;
            R.final_satisfaction = mean_satisfaction(G, state);
            R.final_segregation = segregation_index(G, state);
            return R;
        }
        seen.insert(h);

        bool moved = do_one_step(G, state, P, rng);
        if (!moved) {
            R.absorbed = true;
            R.steps = step;
            R.final_satisfaction = mean_satisfaction(G, state);
            R.final_segregation = segregation_index(G, state);
            return R;
        }
    }
    // Reached step cap without absorption or cycle
    R.steps = P.max_steps;
    R.final_satisfaction = mean_satisfaction(G, state);
    R.final_segregation = segregation_index(G, state);
    return R;
}

// Aggregate results and define ``work well'' criteria ? (this is somewhat arbitrary, but we can say a run "works well" if it reaches a good level of segregation within a reasonable number of steps without getting stuck in a cycle)
bool is_good_outcome(const RunStats & R, const Params & P) {
    return R.absorbed && R.steps >= P.min_good_steps && R.final_segregation >= P.min_good_segregation;
}

void add_run(AggregateStats& A, const RunStats& R, const Params& P) {
    A.total_runs++;
    if (R.absorbed) A.absorbed_runs++;
    if (R.cycle_detected) A.cycle_runs++;
    if (is_good_outcome(R, P)) A.good_runs++;

    // Update means using incremental formula
    A.mean_steps += R.steps;
    A.mean_final_satisfaction += R.final_satisfaction;
    A.mean_final_segregation += R.final_segregation;
}

void finalize_aggregate(AggregateStats& A) {
    if (A.total_runs == 0) return;
    A.mean_steps /= A.total_runs;
    A.mean_final_satisfaction /= A.total_runs;
    A.mean_final_segregation /= A.total_runs;
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

        std::mt19937 rng(P.seed + 1009 * thread_id + 7919 + (int)(1000 * P.a_frac) + (int)(10000 * P.vacancy_frac)); // unique seed per thread and parameter point
        AggregateStats local;
    
        #pragma omp for
        for (int rep = 0; rep < P.repetitions; ++rep) {
            auto state = random_initial_state(P, rng);
            RunStats R = simulate_one(G, state, P, rng);
            add_run(local, R, P);
        }

        #pragma omp critical
        {
            total.total_runs += local.total_runs;
            total.absorbed_runs += local.absorbed_runs;
            total.cycle_runs += local.cycle_runs;
            total.good_runs += local.good_runs;
            total.mean_steps += local.mean_steps;
            total.mean_final_satisfaction += local.mean_final_satisfaction; 
            total.mean_final_segregation += local.mean_final_segregation; 
        }
    }
    finalize_aggregate(total);
    return total;
}

int main() {
    Params P;
    P.rows = 20;
    P.cols = 20;
    P.N = P.rows * P.cols;
    P.vacancy_frac = 0.1;
    P.tau = 1.0/3.0;
    P.repetitions = 2000;
    P.max_steps = 100000;

    std::ofstream fout("phase_sweep.csv");
    fout << "a_frac, b_frac, vacancy_frac, absorbed_prob, cycle_prob, good_prob, "
            "mean_steps, mean_final_satisfaction, mean_final_segregation\n";
    for (double v = 0.00; v <= 1; v += 0.01){
        P.vacancy_frac = v;
        for (double a = 0.00; a<= 1 - P.vacancy_frac; a += 0.01) {
            if (a+ P.vacancy_frac > 1.0) break; // ensure valid parameters
            P.a_frac = a;

            Counts C = proportions_to_counts(P);
            if (C.Qb < 0) continue; // skip invalid parameter points

            double b = 1.0 - P.vacancy_frac - P.a_frac;
            auto A = run_experiment_at_fraction(P);

            double absorbed_prob = (double)A.absorbed_runs / A.total_runs;
            double cycle_prob = (double)A.cycle_runs / A.total_runs;
            double good_prob = (double)A.good_runs / A.total_runs;

            std::cout << "a=" << a
                    << "b=" << b
                    << "vacancy_frac=" << P.vacancy_frac
                    << "absorbed_prob=" << absorbed_prob
                    << "good_prob=" << good_prob
                    << "mean_steps=" << A.mean_steps
                    << "\n";
            fout << a << "," 
                << b << "," 
                << P.vacancy_frac << ","
                << absorbed_prob << ","
                << cycle_prob << ","
                << good_prob << ","
                << A.mean_steps << ","
                << A.mean_final_satisfaction << ","
                << A.mean_final_segregation << "\n";
        }
    }
    fout.close();
    return 0;
}

// Compile command
// g++ -O3 -march=native -fopenmp -std=c++17 schelling_phase.cpp -o schelling_phase