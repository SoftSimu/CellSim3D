#ifndef SIMPARAMS_CUH
#define SIMPARAMS_CUH
#include<string>
#include "Types.cuh"

#define PARAM_VERSION 000
struct core_params_struct{
    long int max_no_of_cells;
    int num_nodes;
    size_t max_no_of_nodes;
    real node_mass;
    real eq_bond_len; 
    real rep_range;
    real attr_range;
    real bond_stiff;
    real rep_stiff;
    real attr_stiff;
    real stiff_factor1;
    real inter_membr_fric;
    real internal_damping;
    real gamma_visc;
    real division_vol;
    bool random_z_offset;
    real z_offset;
    long int div_time_steps;
    real delta_t;
    bool restart;
    long int traj_write_int;
    real non_div_time_steps;
    char traj_file_name[256];
    real max_pressure;
    real min_pressure;
    real growth_rate;
    bool check_sphericity;
    bool angle_pot;
    real dom_len;
    long int phase_count;
    bool correct_com;
    long int init_num_cells; 
};

struct angle_params_struct{
    bool angle_pot;
    real angle_stiffness;
    angles3*  theta0;
};

struct adaptive_params_struct{
    bool do_adaptive_dt;
    real dt_max;
    real dt_tol;
};

struct counting_params_struct{
    bool count_cells;
    char mit_index_file_name[256];
    bool count_int_cells_only;
    real radius_cutoff;
    bool overwrite_mit_index_file;
    long int cell_count_int;
};


struct pop_params_struct{
    bool do_pop_model;
    real total_food;
    real regular_consumption;
    real division_consumption;
    real death_release_food;
    long int haylimit;
    long int cell_life_time;
};


struct wall_params_struct{
    bool use_walls;
    char perp_axis;
    real d_axis;
    real wall_len;
    real wall_width;
    real thresh_dist;
    real wall1;
    real wall2;
};

struct div_params_struct{
    bool use_div_plane_basis;
    real3 div_plane_basis;
};

struct stiffness_params_struct{
    bool use_diff_stiff;
    real soft_stiff_factor;
    long int num_softer_cells;
    real frac_softer_cells;
    bool during_growth;
    bool daught_same_stiff;
    real closeness_to_center;
    long int start_at_pop;
    bool rand_cell_ind;
};

struct box_params_struct{
    bool use_rigid_sim_box;
    bool use_pbc;
    real box_len;
    real box_len_x;
    real box_len_y;
    real box_len_z;
    real3 box_max; 
    bool flatbox;
    real thresh_dist;
};

struct rand_params_struct{
    bool add_rands;
    long int rand_seed;
    int rand_dist;
    real rand_scale_factor; // something like temperature.
};

typedef struct core_params_struct core_params_struct;
typedef struct angle_params_struct angle_params_struct; 
typedef struct adaptive_params_struct adaptive_params_struct;
typedef struct counting_params_struct counting_params_struct;
typedef struct pop_params_struct pop_params_struct;
typedef struct wall_params_struct wall_params_struct;
typedef struct div_params_struct div_params_struct;
typedef struct stiffness_params_struct stiff_params_struct;
typedef struct box_params_struct box_params_struct;
typedef struct rand_params_struct rand_params_struct;

struct sim_params_struct{
    char version[10];
    core_params_struct core_params;
    angle_params_struct angle_params; 
    adaptive_params_struct adaptive_params;
    counting_params_struct counting_params;
    pop_params_struct pop_params;
    wall_params_struct wall_params;
    div_params_struct div_params;
    stiffness_params_struct stiff_params;
    box_params_struct box_params;
    rand_params_struct rand_params;
};

int ReadSimParams(sim_params_struct& sim_params, const char* fileName);

typedef struct sim_params_struct sim_params_struct;

#endif // SIMPARAMS_CUH
