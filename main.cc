#include <algorithm>
#include <atomic>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <iomanip>
#include <ios>
#include <numbers>
#include <fstream>
#include <sstream>
#include <tuple>
#include <vector>
#include <libbear/core/coordinates.h>
#include <libbear/core/range.h>
#include <libbear/core/system.h>
#include <libbear/ea/elements.h>
#include <libbear/ea/evolution.h>
#include <libbear/ea/fitness.h>
#include <libbear/ea/genotype.h>
#include <libbear/ea/population.h>
#include <libbear/ea/variation.h>

using namespace libbear;

namespace {

  const std::string pp{"B.pbesol-n-kjpaw_psl.0.1.UPF"};

  std::string pwx_fixed(double d) {
    std::ostringstream oss{};
    oss << std::fixed << std::setprecision(9) << d;
    return oss.str();
  }

  std::string pwx_scientific(double d) {
    std::ostringstream oss{};
    oss << std::scientific << std::setprecision(9) << d;
    std::string s{oss.str()};
    return s.replace(s.find('e'), 1, "D");
  }

  std::string pwx_control(const std::string& outdir_postfix) {
    std::ostringstream oss{};
    oss << "&CONTROL\n"
        << "calculation = 'scf'\n"
        << "prefix = 'dft'\n"
        << "pseudo_dir = './'\n"
        << "outdir = 'results-" << outdir_postfix << "'\n"
        << "/\n";
    return oss.str();
  }

  std::string pwx_system(int nat,
                         int ntyp,
                         double tot_charge,
                         double degauss,
                         double ecutwfc) {
    std::ostringstream oss{};
    oss << "&SYSTEM\n"
        << "ibrav = 0\n"
        << "nat = " << nat << "\n"
        << "ntyp = " << ntyp << "\n"
        << "tot_charge = " << pwx_scientific(tot_charge) << "\n"
        << "occupations = 'smearing'\n"
        << "smearing = 'methfessel-paxton'\n"
        << "degauss = " << pwx_scientific(degauss) << "\n"
        << "ecutwfc = " << pwx_scientific(ecutwfc) << "\n"
        << "/\n";
    return oss.str();
  }

  std::string pwx_electrons(int electron_maxstep, double mixing_beta) {
    std::ostringstream oss{};
    oss << "&ELECTRONS\n"
        << "electron_maxstep = " << electron_maxstep << "\n"
        << "mixing_beta = " << pwx_scientific(mixing_beta) << "\n"
        << "/\n";
    return oss.str();
  }

  std::string pwx_cell_parameters_diag(double x, double y, double z) {
    std::ostringstream oss{};
    oss << "CELL_PARAMETERS angstrom\n"
        << pwx_fixed(x) << " " << pwx_fixed(0) << " " << pwx_fixed(0) << "\n"
        << pwx_fixed(0) << " " << pwx_fixed(y) << " " << pwx_fixed(0) << "\n"
        << pwx_fixed(0) << " " << pwx_fixed(0) << " " << pwx_fixed(z) << "\n"
        << "\n";
    return oss.str();
  }

  struct atom {
    std::string symbol;
    double mass;
    std::string pp;
  };
  
  std::string pwx_atomic_species(const std::vector<atom>& v) {
    std::ostringstream oss{};
    oss << "ATOMIC_SPECIES\n";
    for (const auto& a : v) {
      oss << a.symbol << " " << pwx_fixed(a.mass) << " " << a.pp << "\n";
    }
    oss << "\n";
    return oss.str();
  }

  struct position {
    std::string symbol;
    double x;
    double y;
    double z;
    double distance(const position& p) const {
      return std::hypot(x - p.x, y - p.y, z - p.z);
    }
  };

  std::string pwx_atomic_positions(const std::vector<position>& v) {
    std::ostringstream oss{};
    oss << "ATOMIC_POSITIONS angstrom\n";
    for (auto p : v) {
      oss << p.symbol << " "
          << pwx_fixed(p.x) << " "
          << pwx_fixed(p.y) << " "
          << pwx_fixed(p.z) << "\n";
    }
    oss << "\n";
    return oss.str();
  }

  std::string pwx_k_points(int k) {
    std::ostringstream oss{};
    oss << "K_POINTS automatic\n"
        << k << " 1 1 1 1 0\n";
    return oss.str();
  }

  std::size_t number_of_atoms(const genotype& g) {
    return 1 + g.size() / 3;
  }

  std::vector<position> adjust_positions(const std::vector<position>& v) {
    std::vector<position> res{};
    const auto min_x =
      std::ranges::min_element(v, [](auto a, auto b) { return a.x < b.x; })->x;
    const auto min_y =
      std::ranges::min_element(v, [](auto a, auto b) { return a.y < b.y; })->y;
    std::ranges::transform(v, std::back_inserter(res),
                           [min_x, min_y](const position& p) {
                             return position{p.symbol,
                                             p.x - min_x, p.y - min_y, p.z};
                           });
    return res;
  }

  template<std::floating_point T>
  std::tuple<std::vector<position>, double> geometry(const genotype& g) {
    // n = number_of_atoms(g) i.e. number of atoms in unit cell:
    // a) n > 0: dz_n
    // b) n > 1: dz_n, rho_1, dz_1
    // c) n > 2: dz_n, rho_1, dz_1, (rho_i, phi_i, dz_i) for i = 2, ..., n - 1
    // Note: g.size() == 1 && n == 1 || g.size() == 3 * (n - 1) && n > 1
    std::size_t i = 0;
    const T dz_n = g.at(i++)->value<T>();
    T z = 0.;
    std::vector<position> res{position{"B11", 0., 0., z}};  // 0
    if (number_of_atoms(g) > 1) {                           // 1
      const auto [x, y] = polar2cart(g.at(i++)->value<T>(), 0.);
      z += g.at(i++)->value<T>();
      res.push_back(position{"B11", x, y, z});
    }
    while (i < g.size()) {                                  // 2, ..., n - 1
      const auto rho = g.at(i++)->value<T>();
      const auto [x, y] = polar2cart(rho, g.at(i++)->value<T>());
      z += g.at(i++)->value<T>();
      res.push_back(position{"B11", x, y, z});
    }
    assert(i == g.size() && res.size() == number_of_atoms(g));
    return std::tuple<std::vector<position>, double>{adjust_positions(res),
                                                     z + dz_n};
  }

  template<std::floating_point T>
  std::vector<position> geometry_pbc(const genotype& g) {
    auto [ps, h] = geometry<T>(g);
    auto p = ps[0];
    p.z += h;
    ps.push_back(p);
    return ps;
  }

  template<std::floating_point T>
  void input_file(const std::string& filename, const genotype& g) {
    std::ofstream file{filename};
    const auto [p, h] = geometry<T>(g);
    const auto max_x =
      std::ranges::max_element(p, [](auto a, auto b) { return a.x < b.x; })->x;
    const auto max_y =
      std::ranges::max_element(p, [](auto a, auto b) { return a.y < b.y; })->y;
    // With electron_maxstep == 25 about 5% of SCF calculations will not finish.
    file << pwx_control(filename)
         << pwx_system(number_of_atoms(g), 1, 0., 5.e-3, 6.e+1)
         << pwx_electrons(25, 7.e-1)
         << pwx_cell_parameters_diag(max_x + 15., max_y + 15., h)
         << pwx_atomic_species({{"B11", 11.009305, pp}})
         << pwx_atomic_positions(p)
         << pwx_k_points(4);
  }

  std::string unique_filename() {
    static std::atomic_size_t i{0};
    std::ostringstream oss{};
    oss << "stripe-" << i++ << ".in";
    return oss.str();
  }

}

int main() {
  using type = double;

  execute("/bin/bash download.sh " + pp);

  const range<type> bond_range{0.5, 2.5}; // Angstrom
  const range<type> rho_range{0., 2 * bond_range.max()};
  const range<type> phi_range{0.,
                              std::nextafter(2 * std::numbers::pi_v<type>, 0.)};
  const range<type> dz_range{0., bond_range.max()};

  const genotype g{gene{dz_range},  // dz_3
                   gene{rho_range}, // rho_1
                   gene{dz_range},  // dz_1
                   gene{rho_range}, // rho_2
                   gene{phi_range}, // phi_2
                   gene{dz_range}}; // dz_2

  const genotype_constraints cs = [bond_range](const genotype& g) -> bool {
    // returns true for valid genotype
    const auto ps = geometry_pbc<type>(g);
    for (std::size_t i = 0; i < ps.size(); ++i) {
      for (std::size_t j = i + 1; j < ps.size(); ++j) {
        if (ps[i].distance(ps[j]) < bond_range.min()) {
          return false;
        }
      }
    }
    for (std::size_t i = 0; i < ps.size(); ++i) {
      bool res = false;
      for (std::size_t j = 0; j < ps.size(); ++j) {
        if (i != j && ps[i].distance(ps[j]) <= bond_range.max()) {
          res = true;
        }
      }
      if (!res) {
        return res;
      }
    }
    return true;
  };

  const auto f = [](const genotype& g) -> fitness {
    const std::string input_filename{unique_filename()};
    input_file<type>(input_filename, g);
    const auto [o, e] = execute("/bin/bash calc.sh " + input_filename);
    return o == "Calculations failed.\n"? incalculable : -std::stod(o);
  };
  const fitness_function ff{f, cs};

  const auto first_generation_creator = random_population{g, cs};
  const auto parents_selection =
    roulette_wheel_selection{fitness_proportional_selection{ff}};
  const auto survivor_selection =
    adapter(roulette_wheel_selection{fitness_proportional_selection{ff}});

  const populate_fns p{first_generation_creator,
                       parents_selection,
                       survivor_selection};

  const type sigma{.02};
  const variation v{Gaussian_mutation<type>{sigma},
                    arithmetic_recombination<type>};
  const std::size_t generation_sz{1000};
  const std::size_t parents_sz{42};
  const generation_creator::options o{v, generation_sz, parents_sz};
  const generation_creator gc{p, o};
  const auto tc = max_fitness_improvement_termination(ff, 10, 0.05);
  const evolution e{gc, tc};

  std::ofstream file{"evolution.dat"};
  for (std::size_t i = 0; const auto& x : e()) {
    for (const auto& xx : x) {
      file << i << ' ';
      for (std::size_t i = 0; i < xx.size(); ++i) {
        file << std::scientific << std::setprecision(9) << xx[i]->value<type>()
             << ' ';
      }
      file << std::scientific << std::setprecision(9) << ff(xx) << '\n';
    }
    ++i;
  }
}
