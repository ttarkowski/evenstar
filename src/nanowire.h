#ifndef EVENSTAR_SRC_NANOWIRE_H
#define EVENSTAR_SRC_NANOWIRE_H

#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <numbers>
#include <ranges>
#include <string>
#include <tuple>
#include <vector>
#include <libbear/core/coordinates.h>
#include <libbear/core/range.h>
#include <libbear/ea/genotype.h>
#include "pwx.h"

namespace evenstar {

  inline std::size_t number_of_atoms(const libbear::genotype& g, bool flat)
  { return flat? (1 + g.size()) / 2 : 1 + g.size() / 3; }

  namespace detail {

    template<std::floating_point T>
    libbear::genotype
    nanowire_flat(std::size_t n, const libbear::range<T>& bond) {
      const T y_max = (n - 1) * bond.max();
      const libbear::range<T> y{-y_max, y_max};
      const libbear::range<T> dz{0., bond.max()};
      libbear::genotype res{libbear::gene{dz}};
      for (std::size_t i = 1; i < n; ++i) {
        res = libbear::merge(res, libbear::gene{y}, libbear::gene{dz});
      }
      return res;
    }

    template<std::floating_point T>
    libbear::genotype
    nanowire_buckled(std::size_t n, const libbear::range<T>& bond) {
      const libbear::range<T> rho{0., (n - 1) * bond.max()};
      const libbear::range<T> phi{0.,
                                  std::nextafter(2 * std::numbers::pi_v<T>, 0.)};
      const libbear::range<T> dz{0., bond.max()};
      libbear::genotype res{libbear::gene{dz}};
      if (n > 1) {
        res = libbear::merge(res, libbear::gene{rho}, libbear::gene{dz});
      }
      for (std::size_t i = 2; i < n; ++i) {
        res = libbear::merge(res,
          libbear::gene{rho}, libbear::gene{phi}, libbear::gene{dz});
      }
      return res;
    }

  } // namespace detail

  template<std::floating_point T>
  libbear::genotype
  nanowire(std::size_t n, const libbear::range<T>& bond, bool flat) {
    assert(n > 0);
    const libbear::genotype res{flat
      ? detail::nanowire_flat<T>(n, bond)
      : detail::nanowire_buckled<T>(n, bond)};
    assert(n == number_of_atoms(res, flat));
    return res;
  }

  namespace detail {

    // Mystic Rose is a complete graph with vertices placed on the points of
    // a regular polygon. This function returns edges needed to construct this
    // graph, e.g. for (0, 1, 2) it returns ((0, 1), (0, 2), (1, 2)).
    template<typename T, template<typename> typename C>
    std::vector<std::tuple<T, T>> mystic_rose_edges(const C<T>& c) {
      std::vector<std::tuple<T, T>> res{};
      for (std::size_t i = 0; const auto& x : c) {
        for (const auto& y : c | std::views::drop(++i)) {
          res.push_back(std::make_tuple(x, y));
        }
      }
      return res;
    }

  } // namespace detail

  // This function template checks if all atoms are separated from each other.
  template<std::floating_point T>
  bool atoms_not_too_close(const pwx_positions& ps, T min_distance) {
    return std::ranges::
      all_of(detail::mystic_rose_edges(ps),
             [=](const auto& t) { return pwx_distance(t) > min_distance; });
  }

  // This function template checks if every atom has at least one neighbor.
  template<std::floating_point T>
  bool atoms_not_alone(const pwx_positions& ps, T max_distance) {
    return std::ranges::
      all_of(ps,
             [&ps, max_distance](const auto& x) {
               return std::ranges::
                 any_of(ps | std::views::filter(x.different()),
                        [&x, max_distance](const auto& y) {
                          return x.distance(y) <= max_distance;
                        });
             });
  }

  // This function template checks if all atoms are connected, i.e. form a wire.
  template<std::floating_point T>
  bool all_atoms_connected(const pwx_positions& ps, T max_distance) {
    pwx_positions connected{};
    for (const auto& x : ps) {
      if (std::ranges::any_of(connected,
                              [&x, max_distance](const auto& y) {
                                return x.distance(y) <= max_distance;
                              }) || connected.size() == 0) {
        connected.push_back(x);
      } else {
        return false;
      }
    }
    assert(connected.size() == ps.size());
    return true;
  }

  pwx_positions adjust_positions(const pwx_positions& ps);

  namespace detail {

    template<std::floating_point T>
    std::tuple<pwx_positions, double>
    geometry_flat(const libbear::genotype& g,
                  const std::string& atom_symbol) {
      // n = number_of_atoms(g) i.e. number of atoms in unit cell:
      // a) n > 0: dz_n
      // b) n > 1: dz_n, (y_i, dz_i) for i = 1, ..., n - 1
      // Note: g.size() == 2 * n - 1
      std::size_t i = 0;
      const T dz_n = g.at(i++)->value<T>();
      T z = 0.;
      pwx_positions res{pwx_position{atom_symbol, 0., 0., z}}; // 0
      while (i < g.size()) { // 1, ..., n - 1
        const auto y = g.at(i++)->value<T>();
        z += g.at(i++)->value<T>();
        res.push_back(pwx_position{atom_symbol, 0., y, z});
      }
      assert(i == g.size() && res.size() == number_of_atoms(g, true));
      return std::tuple<pwx_positions, double>{adjust_positions(res), z + dz_n};
    }

    template<std::floating_point T>
    std::tuple<pwx_positions, double>
    geometry_buckled(const libbear::genotype& g,
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
      if (number_of_atoms(g, false) > 1) { // 1
        const auto [x, y] = libbear::polar2cart(g.at(i++)->value<T>(), 0.);
        z += g.at(i++)->value<T>();
        res.push_back(pwx_position{atom_symbol, x, y, z});
      }
      while (i < g.size()) { // 2, ..., n - 1
        const auto rho = g.at(i++)->value<T>();
        const auto [x, y] = libbear::polar2cart(rho, g.at(i++)->value<T>());
        z += g.at(i++)->value<T>();
        res.push_back(pwx_position{atom_symbol, x, y, z});
      }
      assert(i == g.size() && res.size() == number_of_atoms(g, false));
      return std::tuple<pwx_positions, double>{adjust_positions(res), z + dz_n};
    }

  } // namespace detail

  template<std::floating_point T>
  std::tuple<pwx_positions, double>
  geometry(const libbear::genotype& g,
           const std::string& atom_symbol,
           bool flat) {
    return flat
      ? detail::geometry_flat<T>(g, atom_symbol)
      : detail::geometry_buckled<T>(g, atom_symbol);
  }

  template<std::floating_point T>
  pwx_positions geometry_pbc(const libbear::genotype& g,
                             const std::string& atom_symbol,
                             bool flat) {
    auto [ps, h] = geometry<T>(g, atom_symbol, flat);
    ps.push_back(pwx_position{ps[0].symbol, ps[0].x, ps[0].y, ps[0].z + h});
    return ps;
  }

} // namespace evenstar

#endif // EVENSTAR_SRC_NANOWIRE_H
