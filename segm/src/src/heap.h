#ifndef HEAP_H_
#define HEAP_H_

#include <vector>
#include <iostream>
#include <math.h>
#include <boost/python.hpp>
#include <boost/python/numeric.hpp>

using namespace std;

/*!
\struct Element
\brief 
Structure containing an element from the narrow band
*/

struct Element{

    vector<int> c;  // Coordinates
    double dist;    // Distance
    int lab; 	    // Label
};


/*!
\class Heap

\brief 

Binary min-heap data structure supporting the fast marching algorithm implementation.

A min heap is a list where the smallest element is always at the first position. 
The fast marching method relies on this data structure to store the elements from the narrow band. 
At each iteration, the fast marching algorithm is searching for the element in the narrow band with minimal 
distance from the propagation front.

For each element of the narrow band, we need to store the following information:
- the coordinate of the voxel
- the distance associated to the voxel

The distance associated to a specific voxel can potentially be modified during the fast marching algorithm run. 
Therefore, for the algorithm to run efficiently, it is necessary to associate a label to each voxel entering the narrow band.
The label at each position is stored in an array structure and is used to efficiently recover the location of a given element 
of the narrow band.
*/

class Heap{

    vector<double> distance;		        // Vector containing the distance of each element in the binary heap
    vector<vector<int> > coordinates; 		// Vector containing the voxel coordinates of each element in the binary heap  
    int nb; 				        // Number of elements in the binary heap
    vector<vector<int> > indexes;	        // Vector structure storing the label associated to each position in the binary heap

    /*!
    \brief 
    Swap two elements from the Heap structure. 

    \param idx1 : Index of the first element
    \param idx2 : Index of the second element
    */	
    void swap(const int idx1, const int idx2);

    /*!
    \brief 
    Swap a child with its parent if its associated distance is smaller. This method is used to sort 
    the Heap structure. 

    \param idx : Index of the child node
    */	
    int swap_up(const int idx);

    /*!
    \brief 
    Swap a parent with one of its children if its associated distance is larger. This method is used to sort 
    the Heap structure. 

    \param idx : Index of the parent node
    */	
    int swap_down(const int idx);	


  public:

    /*!
    \brief 
    Constructor of the class binary heap. 

    \param coordinates: Boost python list containing the coordinates of the elements in the binary heap
    \param NX: X-Size of the simulation volume
    \param NY: Y-Size of the simulation volume 
    */	
    Heap(boost::python::list & coordinates, const int & NX, const int & NY); 

    /*!
    \brief 
    Add an element to the Heap structure. 

    \param dist : Distance associated to the new element
    \param c : Coordinates of the new element
    \param l : Label associated to the new element
    */	
    void push(const double &dist, const vector<int> &c, const int &l); 

    /*!
    \brief 
    Modify the distance of an element in the Heap structure. 

    \param dist : New distance associated to the element
    \param c : Coordinates
    */
    void set(const double &dist, const vector<int> &c);

    /*!
    \brief 
    Return and remove the root of the binary heap structure. 
    */
    Element pop();

    /*!
    \brief 
    Display the content of the binary heap. 
    */
    void display();

    /*!
    \brief 
    Returns a vector containing the distance of each element of the binary heap container
    \return : Vector containing the distance of each element of the binary heap container
    */
    vector<double> get_distance();

    /*!
    \brief 
    Returns a vector containing the coordinates of each element of the binary heap container
    \return : Vector containing the coordinates of each element of the binary heap container
    */
    vector<vector<int> > get_coordinates();

    /*!
    \brief 
    Returns the number of elements contained in the binary heap container
    \return : number of elements contained in the binary heap container
    */
    int get_nb();

};

#endif /*HEAP_H_*/
