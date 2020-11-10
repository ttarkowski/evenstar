#include <algorithm>
#include <concepts>
#include <cstddef>
#include <iomanip>
#include <ios>
#include <fstream>
#include <unordered_map>
#include <libbear/core/coordinates.h>
#include <libbear/core/range.h>
#include <libbear/core/system.h>
#include <libbear/ea/elements.h>
#include <libbear/ea/evolution.h>
#include <libbear/ea/fitness.h>
#include <libbear/ea/genotype.h>
#include <libbear/ea/population.h>
#include <libbear/ea/variation.h>
#include "src/nanowire.h"
#include "src/pwx.h"

using namespace libbear;
using namespace evenstar;

namespace {

  std::unordered_map<genotype, std::string> file_db{};

  template<std::floating_point T>
  void input_file(const std::string& filename,
                  const genotype& g,
                  const pwx_atom& atom,
                  bool flat) {
    file_db[g] = filename;
    std::ofstream file{filename};
    const auto [p, h] = geometry<T>(g, atom.symbol, flat);
    const auto max_x = std::ranges::max_element(p, {}, &pwx_position::x)->x;
    const auto max_y = std::ranges::max_element(p, {}, &pwx_position::y)->y;
    const T free_space = 10.;
    file << pwx_control(filename)
         << pwx_system(number_of_atoms(g, flat), 1, 0., 1.e-2, 6.e+1)
         << pwx_electrons(100, 7.e-1)
         << pwx_cell_parameters_diag(max_x + free_space, max_y + free_space, h)
         << pwx_atomic_species({atom})
         << pwx_atomic_positions(p)
         << pwx_k_points(8);
  }

}

int main() {
  using type = double;
  const pwx_atom atom{"B", 10.811, "B.pbe-n-kjpaw_psl.1.0.0.UPF"};
  execute("/bin/bash download.sh " + atom.pp);

  const bool flat = false;
  const std::size_t cell_atoms = 3;
  const range<type> bond_range{0.5, 2.5}; // Angstrom
  const genotype g{nanowire<type>(cell_atoms, bond_range, flat)};

  const genotype_constraints cs = [=](const genotype& g) -> bool {
    // Function returns true for valid genotype.
    const auto ps = geometry_pbc<type>(g, atom.symbol, flat);
    return atoms_not_too_close(ps, bond_range.min())
           && all_atoms_connected(ps, bond_range.max());
  };

  const auto f = [atom](const genotype& g) -> fitness {
    const std::string input_filename{pwx_unique_filename()};
    input_file<type>(input_filename, g, atom, flat);
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
      for (std::size_t j = 0; j < xx.size(); ++j) {
        file << std::scientific << std::setprecision(9) << xx[j]->value<type>()
             << ' ';
      }
      file << std::scientific << std::setprecision(9) << ff(xx) << ' '
           << file_db[xx] << '\n';
    }
    ++i;
  }
}
