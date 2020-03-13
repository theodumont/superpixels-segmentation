#include "fastmarching.h"

#define VMAX 10e2


/* PRIVATE METHODS */

// Calculates the arrival time at the specified voxel
void Fast_marching::set_distance(const int & x, const int & y, const int & lb){

    // STEP 1: Initialization

    // Load the neighbors of the specified pixel
    int x_p = x - 1;
    int x_n = x + 1;
    int y_p = y - 1;
    int y_n = y + 1;
     
    // STEP 2: Processing neighbors along each direction

    double d1 = 0.;   // Distance in the x-direction
    double d2 = 0.;   // Distance in the y-direction
    int count = 0;    // Count the number of neighbor pixels with an associated distance

    // x-direction

    int label_1 = 0;
    int label_2 = 0;
    bool frozen_1;
    bool frozen_2;

    // Get the label of the neighbors along the specified direction 
    if(x_p > -1){
        label_1 = this->vec2lab[this->NY*x_p + y];
        frozen_1 = this->vec2frozen[this->NY*x_p + y];
    }
    if(x_n < this->NX){
        label_2 = this->vec2lab[this->NY*x_n + y];
        frozen_2 = this->vec2frozen[this->NY*x_n + y];
    }

    // Determines the neighbor with minimal distance along each direction
    if(label_1 == lb and label_2 == lb and frozen_1 and frozen_2){ 
        d1 = min(this->vec2dist[this->NY*x_p + y], 
            this->vec2dist[this->NY*x_n + y]);
        count++;
    }
    else if(label_1 == lb and frozen_1){
        d1 = this->vec2dist[this->NY*x_p + y];
        count++;
    }
    else if(label_2 == lb and frozen_2){
        d1 = this->vec2dist[this->NY*x_n + y];
        count++;
    }

    // y-direction

    label_1 = 0;
    label_2 = 0;
    frozen_1 = false;
    frozen_2 = false;
 
    // Get the label of the neighbors along the specified direction 
    if(y_p > -1){
        label_1 = this->vec2lab[this->NY*x + y_p];
        frozen_1 = this->vec2frozen[this->NY*x + y_p];
    }
    if(y_n < this->NY){
        label_2 = this->vec2lab[this->NY*x + y_n];
        frozen_2 = this->vec2frozen[this->NY*x + y_n];
    }

    // Determines the neighbor with minimal distance along each direction
    if(label_1 == lb and label_2 == lb and frozen_1 and frozen_2){
        d2 = min(this->vec2dist[this->NY*x + y_p], 
            this->vec2dist[this->NY*x + y_n]);
        count++;
    }
    else if(label_1 == lb and frozen_1){
        d2 = this->vec2dist[this->NY*x + y_p];
        count++;
    }
    else if(label_2 == lb and frozen_2){
        d2 = this->vec2dist[this->NY*x + y_n];
        count++;
    }

    // STEP 3: Solve for the upwind difference finite scheme
    double v = this->compute_velocity(x, y, lb);
    double delta = 4*(pow(d1 + d2, 2) - count*(pow(d1, 2) + pow(d2, 2) - pow(v, 2)));
    double d = (2*(d1 + d2) + sqrt(delta))/(2.*count);

    // STEP 4: Update
    if(this->vec2dist[this->NY*x + y] > d){

        // Case 1: The selected pixel is visited for the first time. 
        // It is added to the binary heap

        if(this->vec2lab[this->NY*x + y] < 1){

            // Update distance and label
            this->vec2dist[this->NY*x + y] = d;
            vector<int> c(2);
            c[0] = x;
            c[1] = y;
            this->vec2lab[this->NY*x + y] = lb;

            // The element is added to the binary heap
            this->narrow_band.push(d, c, this->idx);
            this->idx++;
        }

        // Case 2: The selected pixel has already been visited. 
        // The associated time is updated in the binary heap.
	
        else{

            // Update distance and label
            this->vec2strength[this->NY*x + y] += this->vec2dist[this->NY*x + y] - d;
            this->vec2dist[this->NY*x + y] = d;
            vector<int> c(2);
            c[0] = x;
            c[1] = y;
            this->vec2lab[this->NY*x + y] = lb;

            // In this case the element is in the binary heap but with a larger distance
            this->narrow_band.set(d, c);
        }	
    }
}

// Compute the velocity at voxel (x, y) for the specified label
double Fast_marching::compute_velocity(const int & x, const int & y, const int & lb){

    vector<double> lab = this->img[this->NY*x + y];
    vector<double> texture = this->texture[this->NY*x + y];
 
    vector<double> s_lab = this->seeds[lb - 1];
    vector<double> s_texture = this->t_seeds[lb - 1];

    // Texture distance
    double t_dist = 0;
    for(int n = 0; n < texture.size(); n++){
        t_dist += pow(texture[n] - s_texture[n], 2);
    }
    t_dist /= (double)texture.size();
    t_dist = sqrt(t_dist);

    // LAB Distance
    double lab_dist = sqrt((pow(lab[0] - s_lab[0], 2) + pow(lab[1] - s_lab[1], 2) 
      + pow(lab[2] - s_lab[2], 2))/3.);

    // Eikonal velocity
    //double out = this->w0 + this->w1*lab_dist;
    double out = exp(this->w0*lab_dist + this->w1*t_dist);
    return out;


}


// Update seed
void Fast_marching::update_seeds(int x, int y, int lb){

    double c = this->lb_count[lb - 1];

    // Update LAB seed
    for(int p = 0; p < 3; p++){
        this->seeds[lb - 1][p] = (c*this->seeds[lb - 1][p] + this->img[y + x*this->NY][p])/(c + 1);
    }

    // Update texture seed
    for(int p = 0; p < this->NC; p++){
        this->t_seeds[lb - 1][p] = (c*this->t_seeds[lb - 1][p] + this->texture[y + x*this->NY][p])/(c + 1);
    }

    // Update count
    this->lb_count[lb - 1] += 1;  
}


/* PUBLIC METHODS */

// Constructor
Fast_marching::Fast_marching(const int NX, const int NY, const int NC,
boost::python::list & img,
boost::python::list & texture,
boost::python::list & coordinates,
boost::python::list & weights,
bool update):
NX(NX),
NY(NY),
NC(NC),
update(update),
vec2dist(NX*NY, DBL_MAX),
vec2lab(NX*NY, 0),
vec2strength(NX*NY, 0.),
vec2frozen(NX*NY, false),
narrow_band(coordinates, NX, NY){

    // LAB and Texture channels
    this->img.resize(NX*NY);
    for(int i = 0; i < NX*NY; i++){
        this->img[i].resize(3);
        this->img[i][0] = boost::python::extract<double> (img[3*i]);
        this->img[i][1] = boost::python::extract<double> (img[3*i + 1]);
        this->img[i][2] = boost::python::extract<double> (img[3*i + 2]);
    }

    this->texture.resize(NX*NY);
    for(int i = 0; i < NX*NY; i++){
        this->texture[i].resize(this->NC);
        for(int j = 0; j < this->NC; j++){
            this->texture[i][j] = boost::python::extract<double> (texture[this->NC*i + j]);
        }
    }

    // Initializes the distance and the label arrays
    this->idx = boost::python::len(coordinates)/2;
    this->lb_count.resize(this->idx);
    this->seeds.resize(this->idx);
    this->t_seeds.resize(this->idx);

    for(int n = 0; n < this->idx; n++){

        this->lb_count[n] = 1.;

        int x = boost::python::extract<int> (coordinates[2*n]);
        int y = boost::python::extract<int> (coordinates[2*n + 1]);
        this->vec2lab[y + x*this->NY] = n + 1;
        this->vec2dist[y + x*this->NY] = 0.;

        this->seeds[n].resize(3);
        for(int p = 0; p < 3; p++){
            this->seeds[n][p] = this->img[y + x*this->NY][p];
        }

        this->t_seeds[n].resize(this->NC);
        for(int p = 0; p < this->NC; p++){
            this->t_seeds[n][p] = this->texture[y + x*this->NY][p];
        }
    }

    // Initializes the weights
    this->w0 = boost::python::extract<double> (weights[0]);
    this->w1 = boost::python::extract<double> (weights[1]);
    this->w2 = boost::python::extract<double> (weights[2]);
}

// Iterate
void Fast_marching::iterate(){
	
    // Extract the pixel in the narrow band with minimal distance
    Element e = this->narrow_band.pop();
    int x = e.c[0];
    int y = e.c[1];
    this->vec2frozen[y + x*this->NY] = true;
    this->idx--;

    // Get the label of the pixel
    int lb = this->vec2lab[y + x*this->NY];

    // Update seeds
    if(this->update){
        update_seeds(x, y, lb);
    }

    // Process all neighbors
    int x_p = x - 1;
    if(x_p > -1 and !this->vec2frozen[y + x_p*this->NY]){
        set_distance(x_p, y, lb);
    }
    int x_n = x + 1;  
    if(x_n < this->NX and !this->vec2frozen[y + x_n*this->NY]){
        set_distance(x_n, y, lb);
    }
    int y_p = y - 1;
    if(y_p > -1 and !this->vec2frozen[y_p + x*this->NY]){
        set_distance(x, y_p, lb);
    }
    int y_n = y + 1;
    if(y_n < this->NY and !this->vec2frozen[y_n + x*this->NY]){
        set_distance(x, y_n, lb);
    }
    
}


// Run the algorithm
void Fast_marching::run(){

    int idx = 0;
    while(this->narrow_band.get_nb() > 0){
        iterate();
        idx++;
    }
}

// Add new seeds
void Fast_marching::add_seeds(boost::python::list coordinates){

    // Reset simulation field
    for(int i = 0; i < NX*NY; i++){
        this->vec2frozen[i] = false;
        this->vec2dist[i] = DBL_MAX;
        this->vec2lab[i] = 0;
        this->vec2strength[i] = 0.;
    } 

    // Initializes the distance and the label arrays
    this->idx = boost::python::len(coordinates)/2;
    this->lb_count.resize(this->idx);
    this->seeds.resize(this->idx);
    this->t_seeds.resize(this->idx);

    for(int n = 0; n < this->idx; n++){

        this->lb_count[n] = 1.;

        int x = boost::python::extract<int> (coordinates[2*n]);
        int y = boost::python::extract<int> (coordinates[2*n + 1]);
        this->vec2lab[y + x*this->NY] = n + 1;
        this->vec2dist[y + x*this->NY] = 0.;

        this->seeds[n].resize(3);
        for(int p = 0; p < 3; p++){
            this->seeds[n][p] = this->img[y + x*this->NY][p];
        }

        this->t_seeds[n].resize(this->NC);
        for(int p = 0; p < this->NC; p++){
            this->t_seeds[n][p] = this->texture[y + x*this->NY][p];
        }

        vector<int> c(2);
        c[0] = x;
        c[1] = y;
        this->narrow_band.push(0., c, n);  
    }    
}


// Returns the labels/distances/boundary strength map
boost::python::list Fast_marching::get_results(){

    // Convert results
    boost::python::list l2lb;
    for(int i = 0; i < this->NX*this->NY; i++){
	l2lb.append(this->vec2lab[i]);
    }

    boost::python::list l2dist;
    for(int i = 0; i < this->NX*this->NY; i++){
	l2dist.append(this->vec2dist[i]);
    }

    boost::python::list l2strength;
    for(int i = 0; i < this->NX*this->NY; i++){
	l2strength.append(this->vec2strength[i]);
    }

    boost::python::list output;
    output.append(l2lb);
    output.append(l2dist);
    output.append(l2strength);
    return output;
}

// Returns the binary heap used to describe the narrow band
Heap Fast_marching::get_narrow_band(){
    return this->narrow_band;
}

