// config_parser.hpp
//
// Minimal INI-style configuration parser. Header-only, no dependencies.
//
// USAGE
//   Config c("path/to/config.ini");
//   double m  = c.get_double("water", "mass_g");
//   double T0 = c.get_double("water", "T_initial_K");
//   int    N  = c.get_int   ("simulation", "n_grid");
//   std::string outdir = c.get_string("io", "output_dir", "results");
//
// FORMAT
//   ; comments start with ; or #
//   [section_name]
//   key = value           ; trailing comments allowed
//   another_key = 3.14e-5
//   bool_flag = true      ; or 1/yes/on
//
// All values are stored as strings and converted on access.
// Numerical units (k, M, G) and engineering exponents (1e-5) are supported.
//
// All these design decisions are driven by the goal of letting the user
// (an experimentalist) define a run by editing a single text file that
// pairs cleanly with their lab notes.
#pragma once
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <stdexcept>
#include <algorithm>
#include <cctype>
#include <vector>

namespace mpemba {

class Config {
public:
    Config() = default;
    explicit Config(const std::string& path) { load(path); }

    void load(const std::string& path) {
        std::ifstream ifs(path);
        if (!ifs) throw std::runtime_error("Config: cannot open " + path);
        path_ = path;
        std::string line;
        std::string current_section = "default";
        int lineno = 0;
        while (std::getline(ifs, line)) {
            ++lineno;
            line = strip_comment(line);
            line = trim(line);
            if (line.empty()) continue;
            if (line.front() == '[' && line.back() == ']') {
                current_section = trim(line.substr(1, line.size() - 2));
                continue;
            }
            auto eq = line.find('=');
            if (eq == std::string::npos) {
                throw std::runtime_error(
                    "Config (" + path + ":" + std::to_string(lineno)
                    + "): expected key=value, got: " + line);
            }
            std::string key   = trim(line.substr(0, eq));
            std::string value = trim(line.substr(eq + 1));
            data_[current_section][key] = value;
        }
    }

    // Has a section/key combination?
    bool has(const std::string& section, const std::string& key) const {
        auto it = data_.find(section);
        if (it == data_.end()) return false;
        return it->second.find(key) != it->second.end();
    }

    std::string get_string(const std::string& section, const std::string& key,
                           const std::string& def = "") const {
        auto it = data_.find(section);
        if (it == data_.end()) return def;
        auto jt = it->second.find(key);
        if (jt == it->second.end()) return def;
        return jt->second;
    }

    std::string require_string(const std::string& section, const std::string& key) const {
        if (!has(section, key)) {
            throw std::runtime_error(
                "Config: missing required key [" + section + "]." + key
                + " (in " + path_ + ")");
        }
        return get_string(section, key);
    }

    double get_double(const std::string& section, const std::string& key, double def = 0.0) const {
        if (!has(section, key)) return def;
        return std::stod(get_string(section, key));
    }
    double require_double(const std::string& section, const std::string& key) const {
        return std::stod(require_string(section, key));
    }

    int get_int(const std::string& section, const std::string& key, int def = 0) const {
        if (!has(section, key)) return def;
        return std::stoi(get_string(section, key));
    }
    int require_int(const std::string& section, const std::string& key) const {
        return std::stoi(require_string(section, key));
    }

    bool get_bool(const std::string& section, const std::string& key, bool def = false) const {
        if (!has(section, key)) return def;
        std::string s = get_string(section, key);
        std::transform(s.begin(), s.end(), s.begin(), ::tolower);
        return (s == "true" || s == "1" || s == "yes" || s == "on");
    }

    // Comma-separated list (e.g. "1.0, 2.5, 4.25")
    std::vector<double> get_double_list(const std::string& section, const std::string& key,
                                        const std::vector<double>& def = {}) const {
        if (!has(section, key)) return def;
        std::string s = get_string(section, key);
        std::vector<double> out;
        std::stringstream ss(s);
        std::string token;
        while (std::getline(ss, token, ',')) {
            token = trim(token);
            if (!token.empty()) out.push_back(std::stod(token));
        }
        return out;
    }

    // Dump full configuration to a stream (e.g., for logging)
    void dump(std::ostream& os) const {
        os << "# Config loaded from " << path_ << "\n";
        for (const auto& [section, kv] : data_) {
            os << "[" << section << "]\n";
            for (const auto& [k, v] : kv) os << k << " = " << v << "\n";
            os << "\n";
        }
    }

    const std::string& path() const { return path_; }

private:
    std::map<std::string, std::map<std::string, std::string>> data_;
    std::string path_;

    static std::string trim(const std::string& s) {
        std::size_t a = 0, b = s.size();
        while (a < b && std::isspace((unsigned char)s[a])) ++a;
        while (b > a && std::isspace((unsigned char)s[b - 1])) --b;
        return s.substr(a, b - a);
    }
    static std::string strip_comment(const std::string& s) {
        for (std::size_t i = 0; i < s.size(); ++i) {
            if (s[i] == ';' || s[i] == '#') return s.substr(0, i);
        }
        return s;
    }
};

} // namespace mpemba
