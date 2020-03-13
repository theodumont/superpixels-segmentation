#ifndef FASTMARCHING_H_
#define FASTMARCHING_H_

#include <vector>
#include <iostream>
#include<math.h>
#include <boost/python.hpp>
#include <boost/python/numeric.hpp>

#include "heap.h"

using namespace std;


/**
\class Fast_marching

\brief 

This class provides an implementation of a fast marching algorithm used to solve Eikonal equation. The
implementation allows to keep track of the boundary from which originates the wave corresponding to the
first arrival time through the use of labels.

The fast marching method operates on a 2D voxel lattice. The size of this lattice is specified to be NX by NY. 
In our implementation, the arrival time (resp. the label),at each location in the lattice is stored in a 2D vector.
The algorithm is initialized from a subset of voxels, which will be referred to as seeds, which are associated an initial 
time and a label.

*/

class Fast_marching{

    int NX, NY;		              // Lattice size
    int NC;                           // Number of texture channels
    Heap narrow_band;	              // Heap used to store the points in the narrow band

    vector<vector<double> > img;      // LAB channels of the processed image
    vector<vector<double> > texture;  // Texture channels of the processed image

    vector<vector<double> > seeds;    // LAB seeds
    vector<vector<double> > t_seeds;  // Texture seeds

    vector<double> vec2dist;	      // Vector used to store the arrival times
    vector<int> vec2lab;              // Vector used to store the labels
    vector<double> vec2strength;      // Vector used to store the strength of the boundary
    vector<bool> vec2frozen;          // Vector storing the location of the frozen pixels

    int idx;                          // Number of elements in the narrow band
    double w0, w1, w2;                // Weights used for the distance computation

    bool update;                      // Indicates if the seeds are updated when elements are frozen
    vector<double> lb_count;          // Count the number of pixels associated to each cell 
                

    /*!
    \brief 
    Set the distance at voxel (x, y)
	
    \param x, y : Coordinates of the voxel
    \param lb : Propagation label for which the calculation is performed. 
    */
    void set_distance(const int & x, const int & y, const int & lb);

    /*!
    \brief 
    Compute the velocity at voxel (x, y) for the specified label
	
    \param x, y : Coordinates of the voxel
    \param lb : Propagation label for which the calculation is performed. 

    \return : Velocity at voxel (x, y)
    */
    double compute_velocity(const int & x, const int & y, const int & lb);

    /*!
    \brief 
    Update the seeds when a voxel is frozen
    \param x, y : Coordinate of the point to add to the segment
    \param lb: Label of the segment
    */
    void update_seeds(int x, int y, int lb);


  public:

    /*!
    \brief 
    Constructor

    \param NX, NY : Lattice size
    \param NC: Number of texture channels
    \param img_lab: LAB channels of the processed image
    \param img_texture: Texture channels of the processed image
    \param coordinates: Coordinates of the initial nuclei
    \param weights: Distance weights
    */
    Fast_marching(const int NX, const int NY, const int NC,
       boost::python::list & img,
       boost::python::list & texture,
       boost::python::list & coordinates, 
       boost::python::list & weights,
       bool update
    );

    /*!
    \brief 
    Iterate
    */
    void iterate();

    /*!
    \brief 
    Run the fast marching algorithm
    */
    void run();

    /*!
    \brief
    Add new seeds
    */
    void add_seeds(boost::python::list coordinates);

    /*!
    \brief 
    Return results
    \return : Python list containing the labels, the distances and the boundary strength
    */
    boost::python::list get_results();

    /*!
    \brief 
    Return the binary heap containing the narrow band
    \return : Binary heap describing the narrow band
    */
    Heap get_narrow_band();
		
};

#endif /*FASTMARCHING_H_*/
