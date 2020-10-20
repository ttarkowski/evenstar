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
#include <libbear/core/coordinates.h>
#include <libbear/core/range.h>
#include <libbear/core/system.h>
#include <libbear/ea/elements.h>
#include <libbear/ea/evolution.h>
#include <libbear/ea/fitness.h>
#include <libbear/ea/genotype.h>
#include <libbear/ea/population.h>
#include <libbear/ea/variation.h>
#include "src/pwx.h"

using namespace libbear;
using namespace evenstar;

namespace {

  std::size_t number_of_atoms(const genotype& g) {
    return 1 + g.size() / 3;
  }

  template<typename... Ts>
  genotype append(const genotype& g, const gene<Ts>&... gs) {
    genotype res{g};
    (res.push_back(gs), ...);
    return res;
  }

  template<std::floating_point T>
  genotype nanowire_genotype(std::size_t n,
                             const range<T>& dz,
                             const range<T>& rho = range<T>{},
                             const range<T>& phi = range<T>{}) {
    assert(n > 0);
    if (n == 1) {
      return genotype{gene{dz}};
    } else if (n == 2) {
      return genotype{gene{dz}, gene{rho}, gene{dz}};
    } else {
      return append(nanowire_genotype<T>(n - 1, dz, rho, phi),
                    gene{rho}, gene{phi}, gene{dz});
    }
  }

  pwx_positions
  adjust_positions(const pwx_positions& ps) {
    pwx_positions res{};
    const auto min_x =
      std::ranges::min_element(ps, [](auto a, auto b) { return a.x < b.x; })->x;
    const auto min_y =
      std::ranges::min_element(ps, [](auto a, auto b) { return a.y < b.y; })->y;
    std::ranges::transform(ps, std::back_inserter(res),
                           [min_x, min_y](const pwx_position& p) {
                             return pwx_position{p.symbol,
                                                 p.x - min_x, p.y - min_y, p.z};
                           });
    return res;
  }

  template<std::floating_point T>
  std::tuple<pwx_positions, double> geometry(const genotype& g,
                                             const std::string& atom_symbol) {
    // n = number_of_atoms(g) i.e. number of atoms in unit cell:
    // a) n > 0: dz_n
    // b) n > 1: dz_n, rho_1, dz_1
    // c) n > 2: dz_n, rho_1, dz_1, (rho_i, phi_i, dz_i) for i = 2, ..., n - 1
    // Note: g.size() == 1 && n == 1 || g.size() == 3 * (n - 1) && n > 1
    std::size_t i = 0;
    const T dz_n = g.at(i++)->value<T>();
    T z = 0.;
    pwx_positions res{pwx_position{atom_symbol, 0., 0., z}}; // 0
    if (number_of_atoms(g) > 1) { // 1
      const auto [x, y] = polar2cart(g.at(i++)->value<T>(), 0.);
      z += g.at(i++)->value<T>();
      res.push_back(pwx_position{atom_symbol, x, y, z});
    }
    while (i < g.size()) { // 2, ..., n - 1
      const auto rho = g.at(i++)->value<T>();
      const auto [x, y] = polar2cart(rho, g.at(i++)->value<T>());
      z += g.at(i++)->value<T>();
      res.push_back(pwx_position{atom_symbol, x, y, z});
    }
    assert(i == g.size() && res.size() == number_of_atoms(g));
    return std::tuple<pwx_positions, double>{adjust_positions(res), z + dz_n};
  }

  template<std::floating_point T>
  pwx_positions geometry_pbc(const genotype& g, const std::string& atom_symbol) {
    auto [ps, h] = geometry<T>(g, atom_symbol);
    auto p = ps[0];
    p.z += h;
    ps.push_back(p);
    return ps;
  }

  template<std::floating_point T>
  void input_file(const std::string& filename,
                  const genotype& g,
                  const pwx_atom& atom) {
    std::ofstream file{filename};
    const auto [p, h] = geometry<T>(g, atom.symbol);
    const auto max_x =
      std::ranges::max_element(p, [](auto a, auto b) { return a.x < b.x; })->x;
    const auto max_y =
      std::ranges::max_element(p, [](auto a, auto b) { return a.y < b.y; })->y;
    // With electron_maxstep == 25 about 5% of SCF calculations will not finish.
    file << pwx_control(filename)
         << pwx_system(number_of_atoms(g), 1, 0., 5.e-3, 6.e+1)
         << pwx_electrons(25, 7.e-1)
         << pwx_cell_parameters_diag(max_x + 15., max_y + 15., h)
         << pwx_atomic_species({atom})
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
  const pwx_atom atom{"B11", 11.009305, "B.pbesol-n-kjpaw_psl.0.1.UPF"};
  execute("/bin/bash download.sh " + atom.pp);

  const std::size_t cell_atoms = 3;
  const range<type> bond_range{0.5, 2.5}; // Angstrom
  const range<type> rho_range{0., (cell_atoms - 1) * bond_range.max()};
  const range<type> phi_range{0.,
                              std::nextafter(2 * std::numbers::pi_v<type>, 0.)};
  const range<type> dz_range{0., bond_range.max()};
  const genotype g{nanowire_genotype<type>(cell_atoms,
                                           dz_range, rho_range, phi_range)};

  const genotype_constraints cs = [=](const genotype& g) -> bool {
    // returns true for valid genotype
    const auto ps = geometry_pbc<type>(g, atom.symbol);
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

  const auto f = [atom](const genotype& g) -> fitness {
    const std::string input_filename{unique_filename()};
    input_file<type>(input_filename, g, atom);
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
