#include <cmath>
#include <iostream>
#include <vector>
#include <fstream>
#include <omp.h>

// Compile via: g++ mls.cpp -o bn -fopenmp -O3 -march=native
// Consult README.md and the jupyter notebook mls_plots.ipynb for further details

// Struct used to store symmetric 3x3 matrices:
struct Sym3
{
    double a11;
    double a21, a22;
    double a31, a32, a33;
};

// Struct used to store 3D vectors:
struct Vec3
{
  double v1, v2, v3;
};


inline Vec3 inv3x3(const Sym3& A) {
// This function calculates the first column auf A^{-1}, if A is a symmetric 3x3 matrix.
// It is used in MLS to calculate lambda(x) = A(x)^{-1} w_0;
// Note that w_0 = e_1 in case a monomial basis is used.

  // Cofactors of first column
  const double c11 = A.a22 * A.a33 - A.a32 * A.a32;
  const double c21 = A.a31 * A.a32 - A.a21 * A.a33;
  const double c31 = A.a21 * A.a32 - A.a31 * A.a22;

  // Determinant
  const double det = A.a11 * c11 + A.a21 * c21 + A.a31 * c31;
  const double inv_det = 1.0 / det;

  return {
        c11 * inv_det,
        c21 * inv_det,
        c31 * inv_det
    };
}

int vector_sum_int(const std::vector<int> &v){
// Simple helper function summing the values stored in an std::vector<int> container.
  int sum = 0;

  for(int i = 0; i < v.size(); i++){
    sum += v[i];
  }

  return sum;
}

void write_csv(const std::vector<double>& error_sl, const std::vector<double>& error_ml, const std::string& filename) {
// Writes the output into a .csv file
// error_sl: vector of doubles containing the errors of the singlelevel scheme; Entry j contains the error on the (j+1)'st grid
// error_ml: vector of doubles containing the errors of the multilevel  scheme; Entry j contains the error on the (j+1)'st grid

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: could not open file\n";
        return;
    }

    // First row: error_sl
    for (size_t i = 0; i < error_sl.size(); ++i) {
        file << error_sl[i];
        if (i + 1 < error_sl.size())
            file << ",";
    }
    file << "\n";

    // Second row: error_ml
    for (size_t i = 0; i < error_sl.size(); ++i) {
        file << error_ml[i];
        if (i + 1 < error_ml.size())
            file << ",";
    }
    file << "\n";
}


struct GridLevel {
// GridLevel is a two dimensinal data structure for double values, which allows for arbitrary index ranges.
// Given a GridLevel called "g", the (i,j)'th entry of g can be stored  in  a double variable  c  by  calling "c = g(i,j)";
// Given a GridLevel called "g", the (i,j)'th entry of g can be overwritten by a double variable c by calling "g(i,j) = c";
// i_min and i_max denote the (possibly negative) minimum/maximum coordinates in the first dimension,
// j_min and j_max are the minumum/maximum coordinates in the second dimension.

    int i_min, i_max;
    int j_min, j_max;
    int nx, ny;

    std::vector<double> values;

    GridLevel(int i_min_, int i_max_, int j_min_, int j_max_) 
        : i_min(i_min_), i_max(i_max_), j_min(j_min_),j_max(j_max_)
    {
        nx = i_max - i_min + 1;
        ny = j_max - j_min + 1;

        values.resize(static_cast<std::size_t>(nx) * ny);
    }

    inline std::size_t index(int i, int j) const {
        return static_cast<std::size_t>(j - j_min) * nx
             + static_cast<std::size_t>(i - i_min);
    }

    inline double& operator()(int i, int j) {
        return values[index(i, j)];
    }

    inline const double& operator()(int i, int j) const {
        return values[index(i, j)];
    }
};

inline double phi(double r){
  // Wendland kernel phi_{2,3}
  double t = 1-r; t = t*t; t = t*t; t=t*t; 
  return t*(1.0 + r*(8.0 + r*(25.0 + 32.0*r)));
}

inline double f_example_1(double x, double y){
  // Target function from Example 4.1
  return std::pow(std::sqrt((x-0.5)*(x-0.5) + (y-0.5)*(y-0.5)), 4.01);
}

inline double f_example_2(double x, double y){
  // Target function from Example 4.2
  return std::sin(x*x+y*y)*(1 + std::cos(x*x+y*y));
}

inline double custom_f(double x, double y){
  // Custom target function to be edited by the user
  return std::sin(x*x)*y;
}

GridLevel init_grid(int target, int N, int padding, double h){
// Assumes "N" (= 1/h) and "padding" to be positive integers and generates a GridLevel of size (N + padding)^2.
// The indices in both dimensions range from -padding to N + padding.
// Semantically, all coordinates (i,j) are identified with the grid points h(i,j) = (i,j)/N.
// They are initialized by evaluating the target function there.
// This corresponds to "Level 0" of the multilevel scheme, as \mathcal{E}_0 f = f. 
// The choice of target function depends on "target".

// All (i,j) with 0 <= i,j <= N are mapped to points in the unit square S = [0,1]^2.
// Other entries correspond to points lying ouside S;
// These outliers are necessary to evaluate multilevel MLS inside S.

  int i_min = -padding;
  int j_min = -padding;

  int i_max = N + padding;
  int j_max = N + padding;

  GridLevel grid(i_min, i_max, j_min, j_max);

  if(target == 1){

    #pragma omp parallel for collapse(2)
    for(int i = grid.i_min; i <= grid.i_max; i++){
      for(int j = grid.j_min; j <= grid.j_max; j++){
        grid(i,j) = f_example_1(h*static_cast<double>(i), h*static_cast<double>(j));
      }
    }

  } else if(target == 2) {

    #pragma omp parallel for collapse(2)
    for(int i = grid.i_min; i <= grid.i_max; i++){
      for(int j = grid.j_min; j <= grid.j_max; j++){
        grid(i,j) = f_example_2(h*static_cast<double>(i), h*static_cast<double>(j));
      }
    }

  } else {

    #pragma omp parallel for collapse(2)
    for(int i = grid.i_min; i <= grid.i_max; i++){
      for(int j = grid.j_min; j <= grid.j_max; j++){
        grid(i,j) = custom_f(h*static_cast<double>(i), h*static_cast<double>(j));
      }
    }
  }

  return grid;
}


std::vector<int> init_lfactor(int L, int last_refinement){
// This function calculates an std::vector<int> called "lfactor" of length L (= number of levels);
// If N_j is the number of points per dimension after j refinements of X_{h_1}, then N_j * lfactor[j] is the number of points per dimension on the finest grid.
// As described in section 4 of the paper, the last error is evaluated on a grid 4 times as the previous one, that is, last_refinement = 4;

// For example, if h_0 = 1/4, mu = 0.5 and L = 5, we have lfactor = [2*2*2*2*4, 2*2*2*4, 2*2*4, 2*4, 4] = [64, 32, 16, 8, 4]
// In particular, \mathcal{E}_0 f, \mathcal{E}_1 f, ..., \mathcal{E}_L f are all evaluated on a (64/h_1)*(64/h_1) = 512x512 grid covering the unit square.

  std::vector<int> lfactor(L, 2);

  lfactor[L-1] = last_refinement;

  for(int i = L-1; i > 0; i--){
    lfactor[i-1] *= lfactor[i];
  }

  return lfactor;
}

std::vector<double> multilvl_mls(int target, int N, double nu, int L, int r, int last_refinement){

  // err[j] stores max(\mathcal{E}_{j+1} f) over Y_L (ref: Section 4).
  std::vector<double> err(L); 

  // Refinement factors to reach the finest grid from each level.
  std::vector<int> lfactor = init_lfactor(L, last_refinement);

  // We use the convention h_0 = 1/N;
  // The number of points contained within S = [0,1]^2 on the finest grid is given by N_fine.
  int    N_fine = 2*N*lfactor[0]; 
  double h_fine = 1.0/static_cast<double>(N_fine);
  double delta  = nu/static_cast<double>(2*N); // = delta_1


  // We work exclusively on the finest grid, mapping index (i,j) to h_fine*(i,j).
  //
  // Domain Padding: To evaluate MLS anywhere inside S, we must extend S by 
  // \delta_1 + ... + \delta_L (ref: Section 3).
  // Expressed in fine-grid index steps (\delta_j / h_fine = lfactor[j] * nu), 
  // the total required grid padding per dimension is:
  int padding = std::ceil(vector_sum_int(lfactor) * nu);

  
  // Initialize finest grid. 'grid' and 'update' swap buffer pointers each iteration.
  GridLevel grid = init_grid(target, N_fine, padding, h_fine);
  GridLevel update = GridLevel(grid.i_min, grid.i_max, grid.j_min, grid.j_max);

  int shift   = 0;

  if(r == 0){ // Polynomial reproduction degree 0

    double p_sum;
    double sup;

    for(int lvl = 1; lvl <= L; lvl++){

      const int lf     = lfactor[lvl-1];
      const int radius = lf*std::floor(nu+1);
      const int lfnu2  = lf*lf*nu*nu;
      
      // Shift evaluation domain inward by \delta_{lvl} (in fine-grid units)
      shift += std::floor(lf*nu);

      #pragma omp parallel for collapse(2) schedule(dynamic)
      for(int i = grid.i_min + shift; i <= grid.i_max - shift; i++){
        for(int j = grid.j_min + shift; j <= grid.j_max - shift; j++){

          // Project fine-grid index (i,j) to near coarse point in X_{h_lvl}
          const int proj_i = i - (i % lf);
          const int proj_j = j - (j % lf);

          p_sum = 0.0;
          sup   = 0.0; 

          // Accumulate operator Q_{h_lvl} over nodes in X_{h_lvl} \cap B_{\delta_{lvl}}(x)
          for(int di = -radius; di <= radius; di += lf){

            const int curr_i  = proj_i + di;
            const int dist_i  = i-curr_i;
            const int dist_i2 = dist_i*dist_i;

            const double x   = (h_fine * static_cast<double>(dist_i))/delta;

            for(int dj = -radius; dj <= radius; dj += lf){

              const int curr_j = proj_j + dj;
              const int dist_j = j-curr_j;

              if(dist_i2 + dist_j*dist_j < lfnu2)
                {
                  const double y   = (h_fine * static_cast<double>(dist_j))/delta;
                  const double p   = phi(std::sqrt(x*x + y*y));

                  p_sum += p;
                  sup   += p*grid(curr_i, curr_j);
                }
            }
          }

          // Residual update: \mathcal{E}_{lvl} f = \mathcal{E}_{lvl-1} f - Q_{h_lvl} \mathcal{E}_{lvl-1} f
          update(i,j) = grid(i,j) - sup/p_sum; 
        }
      }

      // swap old values with updated values
      std::swap(grid.values, update.values);
      delta = delta/2.0;

      double max_val = -1.0;

      // Track max residual \mathcal{E}_{lvl} f over the unit square S (0 <= i, j <= N_fine)
      #pragma omp parallel for collapse(2) reduction(max:max_val)
      for (int i = 0; i <= N_fine; i++) {
        for (int j = 0; j <= N_fine; j++) {
          max_val = std::max(max_val, std::abs(grid(i, j)));
        }
      }

      err[lvl-1] = max_val;
    }
  }

  if(r == 1){ // Polynomial reproduction degree 1
              // Works analogously to the case r=0, only the MLS update within the deepest loop needs to be adjusted.

    for(int lvl = 1; lvl <= L; lvl++){

      const int lf     = lfactor[lvl-1];
      const int radius = lf * std::floor(nu+1);
      const int lfnu2  = lf*lf*nu*nu;
      
      shift += std::floor(lf*nu);

      #pragma omp parallel for collapse(2) schedule(dynamic)
      for(int i = grid.i_min + shift; i <= grid.i_max - shift; i++){
        for(int j = grid.j_min + shift; j <= grid.j_max - shift; j++){
          const int proj_i = i - (i % lf);
          const int proj_j = j - (j % lf);

          Sym3 mat{};
          Vec3 sup{};

          for(int di = -radius; di <= radius; di += lf){

            const int curr_i  = proj_i + di;
            const int dist_i  = i-curr_i;
            const int dist_i2 = dist_i*dist_i;
            
            const double x   = (h_fine * static_cast<double>(dist_i))/delta;
            const double x2  = x*x;

            for(int dj = -radius; dj <= radius; dj += lf){
              
              const int curr_j = proj_j + dj;
              const int dist_j = j-curr_j;

              if(dist_i2 + dist_j*dist_j < lfnu2)
                {
                  const double y   = (h_fine * static_cast<double>(dist_j))/delta;
                  const double y2  = y*y;

                  const double p   = phi(std::sqrt(x2 + y2));
                  const double val = grid(curr_i, curr_j);
                  const double pval = p*val;

                  mat.a11 += p;                      
                  mat.a21 += p * x; mat.a22 += p * x2;  
                  mat.a31 += p * y; mat.a32 += p * x * y; mat.a33 += p * y2;

                  sup.v1 += pval; sup.v2 += pval * x; sup.v3 += pval * y;
                }
            }
          }

          Vec3 lambda = inv3x3(mat);

          update(i,j) = grid(i,j) - lambda.v1 * sup.v1 - lambda.v2 * sup.v2 - lambda.v3 * sup.v3;
        }
      }

      std::swap(grid.values, update.values);
      delta = delta/2.0;

      double max_val = -1.0;

      #pragma omp parallel for collapse(2) reduction(max:max_val)
      for (int i = 0; i <= N_fine; i++) {
        for (int j = 0; j <= N_fine; j++) {
          max_val = std::max(max_val, std::abs(grid(i, j)));
        }
      }

      err[lvl-1] = max_val;
    }
  } 

  return err;
}

std::vector<double> singlelvl_mls(int target, int N, double nu, int L, int r, int last_refinement){

  // The singlelevel method can be seen as a special case of our multilevel algorithm.

  std::vector<double> error_sl(L);

  int reciprocal_of_h_lvl  = N;
  int refinement_factor    = last_refinement;

  for(int i = 1; i < L; i++){refinement_factor *= 2;} // corresponds to lfactor[0]

  for(int i = 0; i < L; i++){
    error_sl[i] = multilvl_mls(target, reciprocal_of_h_lvl, nu, 1, r, refinement_factor)[0];
    reciprocal_of_h_lvl *= 2; // reciprocal_of_h_{lvl} --> reciprocal_of_h_{lvl+1}
    refinement_factor   /= 2; // lfactor[i] --> lfactor[i+1]
  }

  return error_sl;
}

int main(int argc, char **argv){

  // Relevant Parameters are read from the command line or replaced by defaults;
  // A more detailed description is given the jupyter notebook mls_plots.ipynb.
  int N      = argc >= 2 ? atoi(argv[1]) : 4;    // Initial mesh size h_0 = 1/N
  int L      = argc >= 3 ? atoi(argv[2]) : 5;    // Number of levels
  double nu  = argc >= 4 ? atof(argv[3]) : 4.1;  // Support-to-meshsize ratio
  int r      = argc >= 5 ? atoi(argv[4]) : 1;    // Polynomial reproduction degree (only r=0,1 implemendted)
  int target = argc >= 6 ? atoi(argv[5]) : 2;    // Choice of target function
  
  int last_refinement = 4; // used to refine the L'th grid one more time for error evaluation.

  std::vector<double> error_ml =  multilvl_mls(target, N, nu, L, r, last_refinement);
  std::vector<double> error_sl = singlelvl_mls(target, N, nu, L, r, last_refinement);
  
  write_csv(error_sl, error_ml, "errors.csv");

  return 0;
}

