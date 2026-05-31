#include "itensor/all.h"
#include "itensor/util/print_macro.h"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <complex> 
#include <cmath>

constexpr double PI{3.1415926535897};
constexpr std::complex<double> II(0.0, 1.0);
using namespace itensor;

std::vector<std::vector<double>> readCSV(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<std::vector<double>> data;
    std::string line;

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return data;
    }

    while (std::getline(file, line)) {
        std::vector<double> row;
        std::stringstream lineStream(line);
        std::string cell;
        
        while (std::getline(lineStream, cell, ',')) {
            row.push_back(std::stod(cell));  // Convert string to double
        }
        
        data.push_back(row);
    }

    file.close();
    return data;
}

std::vector<std::vector<double>> transpose(const std::vector<std::vector<double>>& data) {
    if (data.empty()) return {};

    std::vector<std::vector<double>> transposed(data[0].size(), std::vector<double>(data.size()));
    for (size_t i = 0; i < data.size(); ++i) {
        for (size_t j = 0; j < data[i].size(); ++j) {
            transposed[j][i] = data[i][j];
        }
    }
    return transposed;
}

ITensor createGate(const MPS& psi, int site1, int site2, const std::vector<std::vector<double>>& uq, int locdim){
  auto gate = ITensor(siteIndex(psi, site1), siteIndex(psi, site2),
                      prime(siteIndex(psi, site1)), prime(siteIndex(psi, site2)));
  for (int i = 0; i < locdim * locdim; ++i){
    for (int j = 0; j < locdim * locdim; ++j){
      int a1 = i / locdim + 1;
      int a2 = i % locdim + 1;
      int b1 = j / locdim + 1;
      int b2 = j % locdim + 1;
      gate.set(a1, a2, b1, b2, uq[i][j]);
    }
  }
  return gate;
}

void createGates(std::vector<ITensor>& gates, const MPS& psi, int L, const std::vector<std::vector<double>>& uq, int locdim){
  for (int k = 1; k <= L / 2; ++k){
    gates.push_back(createGate(psi, 2 * k - 1, 2 * k, uq, locdim));
  }
  for (int k = 1; k <= L / 2; ++k){
    gates.push_back(createGate(psi, 2 * k, ((2*k)%L+1), uq, locdim));
  }
}

template <typename T>
void initializeBoundary(MPS& bnd, int locdim, int NA, int L, const std::vector<double>& Fup, const std::vector<T>& Fdn){
  auto setter_first = ITensor(siteIndex(bnd, 1), rightLinkIndex(bnd, 1));
  for (int n = 1; n <= locdim; ++n) {
    if (NA==1) {
      setter_first.set(n, 1, Fdn[n - 1]);
    } else {
      setter_first.set(n, 1, Fup[n - 1]);
    }
  }
  bnd.set(1, setter_first);
  for (int k = 2; k <= L-1; ++k) {
    auto setter = ITensor(leftLinkIndex(bnd, k), siteIndex(bnd, k), rightLinkIndex(bnd, k));
    for (int n = 1; n <= locdim; ++n) {
      if (k==NA) {
        setter.set(1, n, 1, Fdn[n - 1]);
      } else {
        setter.set(1, n, 1, Fup[n - 1]);
      }
    }
    bnd.set(k, setter);
  }
  auto setter_last = ITensor(siteIndex(bnd, L), leftLinkIndex(bnd, L));
  for (int n = 1; n <= locdim; ++n) {
    setter_last.set(n, 1, Fup[n - 1]);
  }
  bnd.set(L, setter_last);
}

void initializePsi(MPS& psi, int locdim, int L, const std::vector<double>& Theta){
  auto setter_first = ITensor(siteIndex(psi, 1), rightLinkIndex(psi, 1));
  for (int n = 1; n <= locdim; ++n) {
    setter_first.set(n, 1, Theta[n - 1]);
  }
  psi.set(1, setter_first);
  for (int k = 2; k <= L - 1; ++k) {
    auto setter = ITensor(leftLinkIndex(psi, k), siteIndex(psi, k), rightLinkIndex(psi, k));
    for (int n = 1; n <= locdim; ++n) {
      setter.set(1, n, 1, Theta[n - 1]);
    }
    psi.set(k, setter);
  }
  auto setter_last = ITensor(siteIndex(psi, L), leftLinkIndex(psi, L));
  for (int n = 1; n <= locdim; ++n) {
    setter_last.set(n, 1, Theta[n - 1]);
  }
  psi.set(L, setter_last);
}

void applyGateAndSVD(MPS& psi, const std::vector<ITensor>& gates, int site1, int site2, int gateIndex, double CUTOFF, int MAX_DIM){
  psi.position(site1);
  auto wf = psi(site1) * psi(site2);
  wf *= gates[gateIndex];
  wf.noPrime();
  auto [U, S, V] = svd(wf, inds(psi(site1)), {"Cutoff=", CUTOFF, "MaxDim=", MAX_DIM});
  psi.set(site1, U);
  psi.set(site2, S * V);
}

int main(int argc, char* argv[]){
  
  
  const int locdim= 6;
  const int L{std::stoi(argv[1])}; 
  const double th{std::stod(argv[2])};
  double costh2 = cos(th/2.0)*cos(th/2.0);
  double sinth2 = sin(th/2.0)*sin(th/2.0);
  const int tmax = 120;
  const int NA=L/2;

  auto sites = CustomSpin(L,{"S=", 2.5,  "ConserveQNs=",false});
  auto init = InitState(sites);
  const std::vector<double> Fup ={1,1,1,0,0,1};
  const std::vector<double> Sup ={0,0,0,1,0,0};
  const std::vector<double> Theta ={costh2*costh2,costh2*sinth2,costh2*sinth2,costh2*sinth2,costh2*sinth2,sinth2*sinth2};

  auto psi = MPS(init);
  auto bnd = MPS(init);

  initializePsi(bnd, locdim, L, Theta);
  initializeBoundary(psi, locdim, NA, L, Fup, Sup);

  std::string filename = "../T2_q1.csv";
  std::vector<std::vector<double>> data = readCSV(filename);
  std::vector<std::vector<double>> uq = transpose(data);
  std::vector<ITensor> gates;
  createGates(gates,psi,L,uq,locdim);

  
  for (int i=1; i<tmax; i++){
    double pur_a = inner(bnd,psi);
    std::cout<< L << "," << i << ","<< th << ","<< pur_a << "," << maxLinkDim(psi) << std::endl;
    for (int k = 1; k <= L / 2; ++k) {
      int site1 = 2 * k - 1;
      int site2 = 2 * k;
      int gateIndex = k - 1;
      applyGateAndSVD(psi, gates, site1, site2, gateIndex, 1E-12, 1300);
    }  
    for (int k = 1; k < L / 2; ++k) {
      int site1 = 2 * k;
      int site2 = (2 * k) % L + 1;
      int gateIndex = k + L / 2 - 1;
      applyGateAndSVD(psi, gates, site1, site2, gateIndex, 1E-12, 1300);
    }
  }

  return 0;

}

