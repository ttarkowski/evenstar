#include <atomic>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <initializer_list>
#include <iomanip>
#include <ios>
#include <numbers>
#include <fstream>
#include <sstream>
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
  
  std::string pwx_atomic_species(std::initializer_list<atom> l) {
    std::ostringstream oss{};
    oss << "ATOMIC_SPECIES\n";
    for (auto a : l) {
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
  };

  std::string pwx_atomic_positions(std::initializer_list<position> l) {
    std::ostringstream oss{};
    oss << "ATOMIC_POSITIONS angstrom\n";
    for (auto p : l) {
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

  template<std::floating_point T>
  void input_file(const std::string& filename, T distance, T angle) {
    std::ofstream file{filename};
    T dx = distance * std::sin(angle / 2.);
    T dy = distance * std::cos(angle / 2.);
    // With electron_maxstep == 25 about 5% of SCF calculations will not finish.
    file << pwx_control(filename)
         << pwx_system(2, 1, 0., 5.e-3, 6.e+1)
         << pwx_electrons(25, 7.e-1)
         << pwx_cell_parameters_diag(2 * dx, 15 + dy, 15)
         << pwx_atomic_species({{"B11", 11.009305, pp}})
         << pwx_atomic_positions({{"B11", 0., 0., 0.}, {"B11", dx, dy, 0.}})
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

  // function
  const auto f = [](type distance, type angle) -> fitness {
    const std::string input_filename{unique_filename()};
    input_file(input_filename, distance, angle);
    const auto [o, e] = execute("/bin/bash calc.sh " + input_filename);
    return o == "Calculations failed.\n"? incalculable : -std::stod(o);
  };
  // domain
  const range<type> distance_range{0.5, 2.5}; // Angstrom
  // Min angle can be calculated from this equation:
  // 2 * distance * sin(angle / 2.) >= min bond length == 0.5
  const range<type> angle_range{.25, std::numbers::pi_v<type>}; // rad

  const fitness_function ff{
    [&](const genotype& g) {
      return f(g[0]->value<type>(), g[1]->value<type>());
    }
  };

  const auto first_generation_creator =
    random_population{genotype{gene{distance_range}, gene{angle_range}}};
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
      file << i << ' '
           << xx[0]->value<type>() << ' '
           << xx[1]->value<type>() << ' '
           << ff(xx) << '\n';
    }
    ++i;
  }
}
