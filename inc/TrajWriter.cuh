#ifndef TRAJ_WRITER_HPP
#define TRAJ_WRITER_HPP
#include<hdf5.h>
#include<string>
#include<exception>
#include<functional>
#include "Types.cuh"
#include "SimParams.cuh"
#include "SimState.cuh"

#define VAR_NAME(v) #v
#define WRITE_SIM_PARAM(v) WriteSimParam(v, VAR_NAME(v));
#define NUM_NODES 192

enum TrajDetailLevel{
    MIN_DETAIL, // upto positions
    MED_DETAIL, // upto velocities
    MAX_DETAIL, // upto forces
    EVERYTHING
};

class TrajException: public std::exception{
    virtual const char* what() const throw()
        {
            return "Trajectory Exception!";
        }
};



class TrajWriter{
private:
    hid_t file_id;
    hid_t ds_id;
    hid_t att_id;
    hid_t frame_id;
    hid_t dspace_id;
    hid_t dset_id;
    hid_t cell_id;
    const sim_params_struct* sm;
    hid_t traj_id;
    TrajDetailLevel detailLevel;
    uint write_int;
    void init(const sim_params_struct& sm);
    void WriteSimParam(void *buf, hid_t mem_type,
                       const char* name, hid_t par_id);

    void WriteSimParam(void *buf, hid_t mem_type,
                       const char* name, hid_t par_id, int ndim, hsize_t *dims);

    void WriteSimParam(const real var, const char* name);
    void WriteSimParam(const real3 var, const char* name);
    void WriteSimParam(const uint var, const char* name);
    void WriteSimParam(const long int var, const char* name);
    void WriteSimParam(const bool var, const char* name);
    void WriteSimParam(const char* var, const char* name);
    void WriteSimParam(const char var, const char* name);
    void WriteSimParam(const angles3 *var, const hsize_t n,  const char* name);
    void WriteReals(void *buf, hid_t cell_id,
                    hid_t dspace_id, std::string name);
    void WriteR3Nptrs(R3Nptrs s, std::string name, hid_t group_id,
                      hid_t dspace_id, size_t offset);
    void WriteStateVar(const R3Nptrs s, std::string name,  long int cellIdx,
                       uint nNodes, hid_t frame_id);
    void WriteStateVar(real* s, std::string name, uint num_cells,
                       hid_t frame_id);

public:
    TrajWriter(const sim_params_struct& _sm);
    ~TrajWriter();
    herr_t status;
    std::string fileName ;
    TrajDetailLevel getDetailLevel();
    void WriteState(const SimState& state);
    uint frame;
    uint lastFrameWritten;
};




#endif // TRAJ_WRITER_HPP
