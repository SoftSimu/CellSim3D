#include<hdf5.h>
#include<string>

#include "Types.cuh"
#include "SimParams.cuh"
#include "TrajWriter.cuh"


#ifdef REAL_TYPE_F32
#define TRAJ_REAL_TYPE H5T_NATIVE_FLOAT
#endif

#ifdef REAL_TYPE_F64
#define TRAJ_REAL_TYPE H5T_NATIVE_DOUBLE
#endif

void TrajWriter::init(const sim_params_struct& _sm){
    std::cout << "Initializing trajectory file... " << fileName << std::endl;
    std::cout << "WARNING: THIS PROGRAM OVERWRITES EXISTING TRAJECTORIES"
              << std::endl;

    //turn off hdf5 error reporting
    //probably a really bad idea, but what the hay
    H5Eset_auto(0, NULL, NULL);


    sm = &_sm;

    file_id = H5Fcreate(sm->core_params.traj_file_name, H5F_ACC_TRUNC,
                        H5P_DEFAULT, H5P_DEFAULT);

    write_int = sm->core_params.traj_write_int;

    if (file_id < 0){ // H5create returns neg value if it fails
        std::cerr << "ERROR: HDF5 trajectory creation failed." << std::endl;
        status = -1;
        throw new TrajException();
    }

    // write the 10 billion parameters
    traj_id = H5Gopen(file_id, "/", H5P_DEFAULT);


    { // version
        const char *version = sm->version;
        WRITE_SIM_PARAM(version);
    }


    { // core_params
        const core_params_struct* core_params = &(sm->core_params);

        WRITE_SIM_PARAM(core_params->max_no_of_cells);
        WRITE_SIM_PARAM(core_params->node_mass);
        WRITE_SIM_PARAM(core_params->eq_bond_len);
        WRITE_SIM_PARAM(core_params->rep_range);
        WRITE_SIM_PARAM(core_params->attr_range);
        WRITE_SIM_PARAM(core_params->bond_stiff);
        WRITE_SIM_PARAM(core_params->rep_stiff);
        WRITE_SIM_PARAM(core_params->attr_stiff);
        WRITE_SIM_PARAM(core_params->stiff_factor1);
        WRITE_SIM_PARAM(core_params->inter_membr_fric);
        WRITE_SIM_PARAM(core_params->internal_damping);
        WRITE_SIM_PARAM(core_params->gamma_visc);
        WRITE_SIM_PARAM(core_params->division_vol);
        WRITE_SIM_PARAM(core_params->random_z_offset);
        WRITE_SIM_PARAM(core_params->z_offset);
        WRITE_SIM_PARAM(core_params->div_time_steps);
        WRITE_SIM_PARAM(core_params->delta_t);
        WRITE_SIM_PARAM(core_params->restart);
        WRITE_SIM_PARAM(core_params->traj_write_int);
        WRITE_SIM_PARAM(core_params->non_div_time_steps);
        WRITE_SIM_PARAM(core_params->traj_file_name);
        WRITE_SIM_PARAM(core_params->max_pressure);
        WRITE_SIM_PARAM(core_params->min_pressure);
        WRITE_SIM_PARAM(core_params->growth_rate);
        WRITE_SIM_PARAM(core_params->check_sphericity);
        WRITE_SIM_PARAM(core_params->angle_pot);
        WRITE_SIM_PARAM(core_params->dom_len);

        uint tot_frames = core_params->div_time_steps + core_params->non_div_time_steps;
        WRITE_SIM_PARAM(tot_frames);
    }

    { // angle_params
        angle_params_struct angle_params = sm->angle_params;
        WRITE_SIM_PARAM(angle_params.angle_stiffness);

        // special for array of angles3
        WriteSimParam(angle_params.theta0, 180, VAR_NAME(angle_params.theta0));
    }

    { // adaptive_params
        adaptive_params_struct adaptive_params = sm->adaptive_params;
        WRITE_SIM_PARAM(adaptive_params.do_adaptive_dt);
        WRITE_SIM_PARAM(adaptive_params.dt_max);
        WRITE_SIM_PARAM(adaptive_params.dt_tol);
    }

    { // counting_params
        counting_params_struct counting_params = sm->counting_params;
        WRITE_SIM_PARAM(counting_params.count_cells);
        WRITE_SIM_PARAM(counting_params.mit_index_file_name);
        WRITE_SIM_PARAM(counting_params.count_int_cells_only);
        WRITE_SIM_PARAM(counting_params.radius_cutoff);
        WRITE_SIM_PARAM(counting_params.overwrite_mit_index_file);
        WRITE_SIM_PARAM(counting_params.cell_count_int);
    }

    { // pop_params
        pop_params_struct pop_params = sm->pop_params;
        WRITE_SIM_PARAM(pop_params.do_pop_model);
        WRITE_SIM_PARAM(pop_params.total_food);
        WRITE_SIM_PARAM(pop_params.regular_consumption);
        WRITE_SIM_PARAM(pop_params.division_consumption);
        WRITE_SIM_PARAM(pop_params.death_release_food);
        WRITE_SIM_PARAM(pop_params.haylimit);
        WRITE_SIM_PARAM(pop_params.cell_life_time);
    }

    { // Wall parameters
        wall_params_struct wall_params = sm->wall_params;
        WRITE_SIM_PARAM(wall_params.use_walls);
        WRITE_SIM_PARAM(wall_params.perp_axis);
        WRITE_SIM_PARAM(wall_params.d_axis);
        WRITE_SIM_PARAM(wall_params.wall_len);
        WRITE_SIM_PARAM(wall_params.wall_width);
        WRITE_SIM_PARAM(wall_params.thresh_dist);
        WRITE_SIM_PARAM(wall_params.wall1);
        WRITE_SIM_PARAM(wall_params.wall2);
    }

    { // Division parameters
        div_params_struct div_params = sm->div_params;
        WRITE_SIM_PARAM(div_params.use_div_plane_basis);
        WRITE_SIM_PARAM(div_params.div_plane_basis);

    }

    { // Stiffness parameters
        stiffness_params_struct stiff_params = sm->stiff_params;
        WRITE_SIM_PARAM(stiff_params.use_diff_stiff);
        WRITE_SIM_PARAM(stiff_params.soft_stiff_factor);
        WRITE_SIM_PARAM(stiff_params.num_softer_cells);
        WRITE_SIM_PARAM(stiff_params.frac_softer_cells);
        WRITE_SIM_PARAM(stiff_params.during_growth);
        WRITE_SIM_PARAM(stiff_params.daught_same_stiff);
        WRITE_SIM_PARAM(stiff_params.closeness_to_center);
        WRITE_SIM_PARAM(stiff_params.start_at_pop);
        WRITE_SIM_PARAM(stiff_params.rand_cell_ind);

    }

    { // box parameters
        box_params_struct box_params = sm->box_params;
        WRITE_SIM_PARAM(box_params.use_rigid_sim_box);
        WRITE_SIM_PARAM(box_params.use_pbc);
        WRITE_SIM_PARAM(box_params.box_len);
        WRITE_SIM_PARAM(box_params.box_len_x);
        WRITE_SIM_PARAM(box_params.box_len_y);
        WRITE_SIM_PARAM(box_params.box_len_z);
        WRITE_SIM_PARAM(box_params.flatbox);
    }

    detailLevel = EVERYTHING;
    frame = 0;
}


TrajWriter::TrajWriter(const sim_params_struct& _sm){
    init(_sm);
}


TrajWriter::~TrajWriter(){
    WRITE_SIM_PARAM(lastFrameWritten);
    status = H5Gclose(traj_id);
    status = H5Fclose(file_id);
}

TrajDetailLevel TrajWriter::getDetailLevel(){
    return detailLevel;
}

void TrajWriter::WriteSimParam(void *buf, hid_t mem_type,
                               const char* name, hid_t par_id, int ndim,
                               hsize_t *dims){

    if (H5Aexists(par_id, name) > 0){
        return;
    }

    ds_id = H5Screate_simple(ndim, dims, NULL);
    att_id = H5Acreate2(par_id, name, mem_type, ds_id,
                        H5P_DEFAULT, H5P_DEFAULT);

    if (H5Awrite(att_id, mem_type, buf) < 0){
        std::cerr << "ERROR: failed to write simulation parameter "
                  << name << std::endl;
        status = -1;
        throw new TrajException();
    }

    if (H5Aclose(att_id) < 0){
        std::cerr << "ERROR: failed to close simulation parameter"
                  << name << std::endl;
        status = -1;
        throw new TrajException();
    }
}

void TrajWriter::WriteSimParam(void *buf, hid_t mem_type,
                               const char* name, hid_t par_id){
    int ndim = 1;
    hsize_t dims[1] = {1};
    WriteSimParam(buf, mem_type, name, par_id, ndim, dims);
}

void TrajWriter::WriteSimParam(const real var, const char* name){
    WriteSimParam((void *)&var, TRAJ_REAL_TYPE, name, traj_id);
}

void TrajWriter::WriteSimParam(const uint var, const char* name){
    WriteSimParam((void *)&var, H5T_NATIVE_UINT, name, traj_id);
}

void TrajWriter::WriteSimParam(const long int var, const char* name){
    WriteSimParam((void *)&var, H5T_NATIVE_LONG, name, traj_id);
}

void TrajWriter::WriteSimParam(const bool var, const char* name){
    // hdf5 has no bool implementation
    // as it was written in c, so write as char
    char v = 'F';
    if(var) v = 'T';
    WriteSimParam((void *)&v, H5T_NATIVE_CHAR, name, traj_id);
}

void TrajWriter::WriteSimParam(const char *var, const char *name){
    int ndim = 1;
    hsize_t dims[1] = {strlen(var)};
    WriteSimParam((void* )&var, H5T_NATIVE_CHAR, name, traj_id, ndim, dims);
}

void TrajWriter::WriteSimParam(const char var, const char *name){
    WriteSimParam((void* )&var, H5T_NATIVE_CHAR, name, traj_id);
}


void TrajWriter::WriteSimParam(const real3 var, const char* name){
    int ndim = 1;
    hsize_t dims[1] = {3};
    real v[3] = {var.x, var.y, var.z};
    WriteSimParam((void* )&v, TRAJ_REAL_TYPE, name, traj_id, ndim, dims);
}

void TrajWriter::WriteSimParam(const angles3 *var, const hsize_t n,
                               const char* name){
    int ndim = 2;
    hsize_t dims[2] = {3, n};
    real a[3][180];

    for (int i = 0; i< n; i++){
        a[0][i] = var[n].aij;
        a[1][i] = var[n].ajk;
        a[2][i] = var[n].aik;
    }

    WriteSimParam( (void *)&a, TRAJ_REAL_TYPE, name, traj_id, ndim, dims);
}



void TrajWriter::WriteReals(void *buf, hid_t group_id,
                            hid_t dspace_id, std::string name){

    dset_id = H5Dopen(group_id, name.c_str(), H5P_DEFAULT);
    if (dset_id < 0){
        dset_id = H5Dcreate(group_id, name.c_str(), TRAJ_REAL_TYPE, dspace_id,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    }

    if (H5Dwrite(dset_id, TRAJ_REAL_TYPE, dspace_id, H5S_ALL, H5P_DEFAULT, buf)
        < 0){
        std::cerr << "Failed to write " << name  << std::endl;
        throw TrajException();
    }

    if (H5Dclose(dset_id) < 0){
        std::cerr << "Failed to close " << name << "data set."
                  << std::endl;
        throw TrajException();
    }
}


void TrajWriter::WriteR3Nptrs(R3Nptrs s, std::string name, hid_t group_id,
                              hid_t dspace_id, size_t offset){
    try{
        WriteReals((void *)&(s.x[offset]), group_id, dspace_id,
                   name + ".x");

        WriteReals((void *)&(s.y[offset]), group_id, dspace_id,
                   name + ".y");

        WriteReals((void *)&(s.z[offset]), group_id, dspace_id,
                   name + ".z");
    } catch (const TrajException& e){
        std::cerr << "Error while writing R3Nptrs of " << name << std::endl;
        throw e;
    }

}

void TrajWriter::WriteStateVar(R3Nptrs s, std::string name, long int cellIdx,
                               uint num_elements, hid_t frame_id){
    const hsize_t dims[1] = {num_elements};
    std::string cellName = "cell" + std::to_string(cellIdx);

    dspace_id = H5Screate_simple(1, dims, dims);
    size_t offset = 0;

    if (cellIdx >= 0){
        cell_id = H5Gopen(frame_id, cellName.c_str(), H5P_DEFAULT);
        if (cell_id < 0){
            cell_id = H5Gcreate(frame_id, cellName.c_str(), H5P_DEFAULT,
                                H5P_DEFAULT, H5P_DEFAULT);
        }

        if (cell_id < 0 || dspace_id < 0){
            std::cerr << "Failed to initialize data of cell " << cellIdx
                      << std::endl;
            throw TrajException();
        }
        offset = cellIdx*num_elements;
    }

    try{
        if(cellIdx >= 0){
            WriteR3Nptrs(s, name, cell_id, dspace_id, offset);
        } else {
            WriteR3Nptrs(s, name, frame_id, dspace_id, offset);
        }

    } catch (const TrajException& e){
        std::cerr << "Cell state data write exception for cell " << cellIdx
                  << std::endl;
        throw e;
    }

    if (cellIdx >= 0){
        if (H5Gclose(cell_id) < 0){
            std::cerr << "Failed to close cell " << cellIdx << " data group"
                      << std::endl;
            throw TrajException();
        }
    }
}

void TrajWriter::WriteStateVar(real* s, std::string name, uint num_cells,
                               hid_t frame_id){
    const hsize_t dims[1] = {num_cells};

    dspace_id = H5Screate_simple(1, dims, dims);
    try{

        WriteReals((void *)s, frame_id, dspace_id, name);

    } catch (const TrajException& e){
        std::cerr << "Real list data write exception for data" << name
                  << std::endl;
        throw e;
    }
}

void TrajWriter::WriteState(state_struct state){
    std::string frameName = "frame" + std::to_string(frame);

    size_t size_hint = sizeof(real)*3*NUM_NODES*state.no_of_cells;

    frame_id = H5Gcreate(traj_id, frameName.c_str(), H5P_DEFAULT,
                         H5P_DEFAULT, H5P_DEFAULT);

    if (frame_id < 0){
        std::cerr << "Failed to load " << frameName << std::endl;
        throw TrajException();
    }

    try{
        for (long int i = 0; i<state.no_of_cells; ++i){
            WriteStateVar(state.pos, "pos", i, NUM_NODES, frame_id);

            if (detailLevel > MIN_DETAIL){
                WriteStateVar(state.vel, "vels", i, NUM_NODES, frame_id);
            }

            if (detailLevel > MED_DETAIL){
                WriteStateVar(state.conForce, "forces", i, NUM_NODES, frame_id);
            }
        }

        if (detailLevel == EVERYTHING){
            //WriteStateVar(state.cellCoMs, "CoMs", -1, state.no_of_cells, frame_id);
            WriteStateVar(state.vol, "volumes", state.no_of_cells, frame_id);
        }
    } catch (const TrajException& e){
        std::cerr << "Trajectory exception in " << frameName
                  << std::endl;
        throw e;
    }

    if (H5Gclose(frame_id) < 0){
        std::cerr << "Failed to close " << frameName << std::endl;
        throw TrajException();
    }

    lastFrameWritten = frame;
    frame += write_int;
}

