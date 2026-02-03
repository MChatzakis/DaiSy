#include "common.hpp"
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>

/* Sigmoid parameters for query execution time prediction (set by initialize_basis_function). */
static double SIG_m = 0.0, SIG_M = 0.0, SIG_b = 0.0, SIG_c = 0.0, SIG_d = 0.0;

static double sigmoid_parameterized_chatzakis(double x)
{
    return SIG_m + (SIG_M - SIG_m) / (1.0 + std::exp(-SIG_b * (x - SIG_c))) + SIG_d;
}

double (*initialize_basis_function(const char *name))(double)
{
    if (std::strcmp(name, "default") == 0)
    {
        SIG_m = 64.2275;
        SIG_M = 1278.0273;
        SIG_b = 28.6739;
        SIG_c = 0.0772;
        SIG_d = 49.3181;
    }
    else if (std::strcmp(name, "seismic") == 0)
    {
        SIG_m = 64.2275;
        SIG_M = 1278.0273;
        SIG_b = 28.6739;
        SIG_c = 0.0772;
        SIG_d = 49.3181;
    }
    else if (std::strcmp(name, "astro") == 0)
    {
        SIG_m = -283.53068;
        SIG_M = 2996.2312;
        SIG_b = 3.52898;
        SIG_c = 0.12639;
        SIG_d = 10.39833;
    }
    else if (std::strcmp(name, "deep") == 0)
    {
        SIG_m = 9335.5998;
        SIG_M = 923.5349;
        SIG_b = -6.457;
        SIG_c = 18.6772;
        SIG_d = -96.6434;
    }
    else if (std::strcmp(name, "random") == 0)
    {
        SIG_m = 13.19602;
        SIG_M = 1311.974;
        SIG_b = 38.177;
        SIG_c = 0.033796;
        SIG_d = -1.4822;
    }
    else if (std::strcmp(name, "yandex") == 0)
    {
        SIG_m = 3104.5000;
        SIG_M = 3104.4999;
        SIG_b = 2157.8090;
        SIG_c = 0.000243;
        SIG_d = 0.076929;
    }
    else if (std::strcmp(name, "sift") == 0)
    {
        SIG_m = 1.0;
        SIG_M = 2186.020202;
        SIG_b = 1.0;
        SIG_c = 1.0;
        SIG_d = 1.0;
    }
    else
    {
        fprintf(stderr, "Invalid name for priority queue size prediction basis: %s!\n", name);
        std::exit(EXIT_FAILURE);
    }
    return sigmoid_parameterized_chatzakis;
}
