#ifndef COMMON_HPP
#define COMMON_HPP

/** Returns a pointer to a function double -> double used for query execution time prediction (sigmoid basis). */
double (*initialize_basis_function(const char *name))(double);

#endif /* COMMON_HPP */
