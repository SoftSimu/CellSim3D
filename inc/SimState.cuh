#ifndef SIMSTATE_CUH
#define SIMSTATE_CUH
#include <vector>
#include "Types.cuh"
#include "SimParams.cuh"
#include "SimList.cuh"
#include "globals.cuh"
#include "VectorFunctions.hpp"
#include "State.cuh"
 
struct SimState{ // contains state on gpu and cpu side
    SimStatePtrs devPtrs; // state to be given to the gpu

    // all of the below must be synchronized manually and separately
    SimList3D<real> posP;
    SimList3D<real> pos;
    SimList3D<real> posM;

    SimList3D<real> vel;
    SimList3D<real> conForce;
    SimList3D<real> disForce;
    SimList3D<real> ranForce;
    SimList3D<real> totForce;

    SimList3D<real> cellCOMs;
    SimList1D<real> vol;
    SimList1D<real> areas;
    SimList1D<char> cellShouldDiv;
    SimList1D<real> boundingBoxes;
    SimList1D<real> pressures;
    SimList1D<int> C180_nn;
    SimList1D<int> C180_sign;
    SimList1D<int> C180_56; 

    SimList3D<real> mins; 
    SimList3D<real> maxs; 

    SimList1D<int> nnList;
    SimList1D<int> numOfNNList; 

    SimList1D<real> bondStiffness;

    SimList1D<real> R0;

    SimList1D<int> resetIndices;    

    long int no_of_cells;
    long int no_new_cells; 

    std::vector<int> numDivisions;

    SimList1D<angles3> theta0;

    real3 sysCM; 

    sim_params_struct* sm;

    bool growthDone; 

    long int frameCount;
    
    SimState(long int _no_of_cells,
             sim_params_struct& sim_params):
        no_of_cells(_no_of_cells),
        
        posP(sim_params.core_params.max_no_of_nodes), 
        pos(sim_params.core_params.max_no_of_nodes), 
        posM(sim_params.core_params.max_no_of_nodes),
        
        vel(sim_params.core_params.max_no_of_nodes),
        
        conForce(sim_params.core_params.max_no_of_nodes), 
        disForce(sim_params.core_params.max_no_of_nodes), 
        ranForce(sim_params.core_params.max_no_of_nodes),
        totForce(sim_params.core_params.max_no_of_nodes),
        mins(1024),
        maxs(1024),
        cellCOMs(sim_params.core_params.max_no_of_cells),
        vol(sim_params.core_params.max_no_of_cells),
        areas(sim_params.core_params.max_no_of_cells),
        cellShouldDiv(sim_params.core_params.max_no_of_cells),
        boundingBoxes(sim_params.core_params.max_no_of_cells*6),
        pressures(sim_params.core_params.max_no_of_cells),        
        C180_nn(192*3),
        C180_sign(180),
        C180_56(92*7),
        nnList(MAX_NN*sim_params.core_params.max_no_of_cells),
        numOfNNList(sim_params.core_params.max_no_of_cells),
        bondStiffness(sim_params.core_params.max_no_of_cells),
        numDivisions(sim_params.core_params.max_no_of_cells, 0),
        R0(192*3),
        theta0(180, real3_to_angles3(make_real3(0,0,0))),
        resetIndices(sim_params.core_params.max_no_of_cells),
        no_new_cells(0),
        growthDone(false),
        frameCount(0){

        devPtrs.posP          = posP.devPtrs; 
        devPtrs.pos           = pos.devPtrs; 
        devPtrs.posM          = posM.devPtrs;

        devPtrs.vel           = vel.devPtrs;

        devPtrs.conForce      = conForce.devPtrs;
        devPtrs.disForce      = disForce.devPtrs;
        devPtrs.ranForce      = ranForce.devPtrs;
        devPtrs.totForce      = totForce.devPtrs;

        devPtrs.cellCOMs      = cellCOMs.devPtrs;
        devPtrs.vol           = vol.devPtr;
        devPtrs.areas         = areas.devPtr;
        devPtrs.cellShouldDiv = cellShouldDiv.devPtr;

        devPtrs.boundingBoxes = boundingBoxes.devPtr;

        devPtrs.pressures     = pressures.devPtr;

        devPtrs.C180_nn       = C180_nn.devPtr;
        devPtrs.C180_sign     = C180_sign.devPtr;
        devPtrs.C180_56       = C180_56.devPtr;

        devPtrs.mins          = mins.devPtrs;
        devPtrs.maxs          = maxs.devPtrs;

        devPtrs.nnList        = nnList.devPtr;
        devPtrs.numOfNNList   = numOfNNList.devPtr;

        devPtrs.bondStiffness     = bondStiffness.devPtr;

        devPtrs.R0            = R0.devPtr;
        devPtrs.theta0 = theta0.devPtr;

        devPtrs.resetIndices = resetIndices.devPtr;

        devPtrs.no_of_cells = no_of_cells;
        devPtrs.no_new_cells = no_new_cells; 
        
        sm = &sim_params; 
    }
};


#endif // SIMSTATE_CUH
