// csv_io.hpp
// Minimal CSV writer. No deps. Buffered.
#pragma once
#include <fstream>
#include <string>
#include <vector>
#include <stdexcept>

namespace mpemba {

class CSVWriter {
public:
    explicit CSVWriter(const std::string& path) : ofs_(path) {
        if (!ofs_) throw std::runtime_error("CSVWriter: cannot open " + path);
        ofs_.precision(10);
    }
    void header(const std::vector<std::string>& cols) {
        for (std::size_t i = 0; i < cols.size(); ++i) {
            if (i) ofs_ << ',';
            ofs_ << cols[i];
        }
        ofs_ << '\n';
    }
    void row(const std::vector<double>& vals) {
        for (std::size_t i = 0; i < vals.size(); ++i) {
            if (i) ofs_ << ',';
            ofs_ << vals[i];
        }
        ofs_ << '\n';
    }
    void flush() { ofs_.flush(); }
private:
    std::ofstream ofs_;
};

} // namespace mpemba
