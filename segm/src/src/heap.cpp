#include "heap.h"

using namespace std;

// Swap two elements from the Heap structure. 
void Heap::swap(const int idx1, const int idx2){

    // Swap distances
    double d = this->distance[idx1];
    this->distance[idx1] = this->distance[idx2];
    this->distance[idx2] = d;
    
    // Swap coordinates
    vector<int> c1 = this->coordinates[idx1];
    vector<int> c2 = this->coordinates[idx2];
    this->coordinates[idx1] = c2;
    this->coordinates[idx2] = c1;
		
    // Update indexes
    this->indexes[c1[0]][c1[1]] = idx2;
    this->indexes[c2[0]][c2[1]] = idx1;
}


// Swap a child with its parent element if its distance is smaller.
int Heap::swap_up(const int idx){

    // Calculate the index of the parenting node
    int p_idx = floor((idx - 1)/2);
    int out = 0;

    if(p_idx >= 0 and this->distance[p_idx] > this->distance[idx]){
        swap(p_idx, idx);
        out = p_idx;
    }
    return out;	
}

// Swap a parent with its children elements if its distance is larger.
int Heap::swap_down(const int idx){

    // Calculate the index of the children node
    int c_idx = 2*idx + 1;
    int out = 0;

    if(c_idx < this->nb and c_idx + 1 < this->nb and this->distance[c_idx] > this->distance[c_idx + 1] ){
        c_idx++;
    }
	
    if(c_idx < this->nb and this->distance[c_idx] < this->distance[idx]){

        // Swap distances
        swap(c_idx, idx);
        out = c_idx;
    }
    return out;	
}

// Class constructor
Heap::Heap(boost::python::list & coordinates, const int & NX, const int & NY):
indexes(NX, vector<int> (NY, 0))
{
    this->nb = boost::python::len(coordinates)/2;
    this->coordinates.resize(this->nb);

    for(int n = 0; n < this->nb; n++){
        this->coordinates[n].resize(2);
        this->coordinates[n][0] = boost::python::extract<int> (coordinates[2*n]);
        this->coordinates[n][1] = boost::python::extract<int> (coordinates[2*n + 1]);
    }

    this->distance.resize(this->nb);
    for(int n = 0; n < this->nb; n++){
        vector<int> c = this->coordinates[n]; 
        this->indexes[c[0]][c[1]] = n;
        this->distance[n] = 0.;
    }
}

// Add an element to the Heap structure and re-sort the Heap structure.
void Heap::push(const double & dist, const vector<int> & c, const int & l){

    // Adds the new element to the binary heap
    this->distance.push_back(dist);
    this->coordinates.push_back(c);
    this->indexes[c[0]][c[1]] = l;
	
    // Heap sort
    int next_idx;
    int idx = this->nb;
    do{
        next_idx = swap_up(idx);
        if(next_idx != 0){
            idx = next_idx;
	}
    }while(next_idx != 0);

    // Update the number of elements
    this->nb++;

}

// Modify the distance of an element in the Heap structure and re-sort the Heap structure. 
void Heap::set(const double & dist, const vector<int> & c){

    // Update distance
    int idx = this->indexes[c[0]][c[1]];
    this->distance[idx] = dist;

    // Heap sort
    int next_idx;
    if(dist < this->distance[floor((idx - 1)/2)]){

        do{
            next_idx = swap_up(idx);
            if(next_idx != 0){
                idx = next_idx;
            }
        }while(next_idx != 0);
    }
    else if((2*idx + 1 < this->nb and dist > this->distance[2*idx + 1]) or 
     (2*idx + 2 < this->nb and dist > this->distance[2*idx + 2])){
        do{
            next_idx = swap_down(idx);
            if(next_idx != 0){
                idx = next_idx;
            }
        }while(next_idx != 0);
    }
}

// Return and remove the root of the binary heap structure. 
Element Heap::pop(){

    // Replace the root of the heap with the last element from the last level of the binary heap
    swap(0, this->nb - 1);
    
    int idx = 0;

    // Creates an instance of structure Element to store the output
    Element e;
    e.dist = this->distance.back();
    e.c = this->coordinates.back();

    // Update
    this->coordinates.pop_back();
    this->distance.pop_back();
    this->nb -= 1;

    // Heap sort
    int next_idx;
	
    do{
        next_idx = swap_down(idx);
        if(next_idx != 0){
            idx = next_idx;
        }
    }while(next_idx != 0);
	
    // Returns the extracted element
    return e;
}

// Display the content of a binary heap on screen
void Heap::display(){
	
    cout << "SIZE: " << this->distance.size() << endl;
	
    // Display indexes
    cout << "Displaying Distances:" << endl;
    for(int n = 0; n < this->nb; n++){
        cout << this->distance[n] << endl;
    }

    // Display indexes
    cout << "Displaying Coordinates:" << endl;
    for(int n = 0; n < this->nb; n++){
        cout << "(" << this->coordinates[n][0]
        << ", " << this->coordinates[n][1] << ")" << endl;
    }
}

// Returns a copy of the distances
vector<double> Heap::get_distance(){
    return this->distance;
}

// Returns a copy of the coordinates
vector<vector<int> > Heap::get_coordinates(){
    return this->coordinates;
}

// Returns the number of elements in the binary heap
int Heap::get_nb(){
    return this->nb;
}


