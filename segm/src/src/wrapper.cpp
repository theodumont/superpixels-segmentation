#include <boost/python.hpp>
#include <boost/python/numeric.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include "fastmarching.h"

using namespace boost::python;

BOOST_PYTHON_MODULE(libfmm){

    class_<Fast_marching>("Fast_marching", init<int, int, int,
    list&, list&, list&, list&, bool>())
        .def("iterate", &Fast_marching::iterate)
        .def("run", &Fast_marching::run)
        .def("get_results", &Fast_marching::get_results)
        .def("add_seeds", &Fast_marching::add_seeds)
    ;
}
