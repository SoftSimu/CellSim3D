#include <iostream>
#include <fstream>
#include <cstring>
//#include <string.h>
#include "Types.cuh"
#include "VectorFunctions.hpp"
#include "SimParams.cuh"
#include "json/json.h"
#include "ErrorCodes.h"

// Code with function that reads input parameters

int ReadSimParams(sim_params_struct& sim_params, const char* fileName){

    Json::Value inpRoot;
    Json::Reader inpReader;

    std::ifstream inpStream(fileName);
    std::string inpString((std::istreambuf_iterator<char>(inpStream)),
                          std::istreambuf_iterator<char>());

    bool parseSucceeded = inpReader.parse(inpString, inpRoot);

    if (!parseSucceeded){
        std::cerr << "ERROR: Failed to parse " << fileName << std::endl;
        std::cerr << inpReader.getFormattedErrorMessages();
        return SIM_PARAM_READ_ERROR;
    }

      
    if ( strcmp("0.1", inpRoot["version"].asString().c_str()) != 0 ){
        std::cout << "Incompatible input file version" << std::endl;
        return PARAM_INVALID_VERSION; 
    }


    std::cout<< fileName << " parsed successfully" << std::endl
             << "Parameters:" << std::endl;

    //std::cout << inpRoot.toStyledString();


    std::strcpy(sim_params.version,
                inpRoot["version"].asString().c_str());
        

    // begin detailed parameter extraction

    Json::Value coreParams = inpRoot.get("core_params", Json::nullValue);

    if (coreParams == Json::nullValue){
        std::cerr << "ERROR: Could not load core simulation parameters"
                  << std::endl << "Exiting..." << std::endl;
        return CORE_SIM_PARAM_ERROR;
    }

    // Load core simulation parameters

    sim_params.core_params.max_no_of_cells =
        coreParams["max_no_of_cells"].asInt64();
    
    sim_params.core_params.max_no_of_nodes =
        192*sim_params.core_params.max_no_of_cells; 

    sim_params.core_params.node_mass =
        coreParams["node_mass"].asDouble();

    sim_params.core_params.eq_bond_len =
        coreParams["eq_bond_len"].asDouble(); 

    sim_params.core_params.rep_range =
        coreParams["rep_range"].asDouble();

    sim_params.core_params.attr_range =
        coreParams["attr_range"].asDouble();

    sim_params.core_params.rep_stiff =
        coreParams["rep_stiff"].asDouble();

    sim_params.core_params.attr_stiff =
        coreParams["attr_stiff"].asDouble();

    sim_params.core_params.bond_stiff =
        coreParams["bond_stiff"].asDouble();

    sim_params.core_params.stiff_factor1 =
        coreParams["stiff_factor1"].asDouble();

    sim_params.core_params.inter_membr_fric =
        coreParams["inter_membr_fric"].asDouble();

    sim_params.core_params.internal_damping =
        coreParams["internal_damping"].asDouble();

    sim_params.core_params.gamma_visc =
        coreParams["gamma_visc"].asDouble();

    sim_params.core_params.division_vol =
        coreParams["division_vol"].asDouble();

    sim_params.core_params.random_z_offset =
        coreParams["random_z_offset"].asBool();

    sim_params.core_params.z_offset =
        coreParams["z_offset"].asDouble();

    sim_params.core_params.div_time_steps =
        coreParams["div_time_steps"].asInt64();

    sim_params.core_params.delta_t =
        coreParams["delta_t"].asDouble();

    sim_params.core_params.restart =
        coreParams["restart"].asBool();

    sim_params.core_params.traj_write_int =
        coreParams["traj_write_int"].asInt64();

    sim_params.core_params.non_div_time_steps =
        coreParams["non_div_time_steps"].asInt64();


    std::strcpy(sim_params.core_params.traj_file_name,
                coreParams["traj_file_name"].asString().c_str());

    sim_params.core_params.max_pressure =
        coreParams["max_pressure"].asDouble();

    sim_params.core_params.min_pressure =
        coreParams["min_pressure"].asDouble();

    sim_params.core_params.growth_rate =
        coreParams["growth_rate"].asDouble();

    sim_params.core_params.check_sphericity =
        coreParams["check_sphericity"].asBool();

    sim_params.core_params.angle_pot =
        coreParams["angle_pot"].asBool();

    sim_params.core_params.correct_com =
        coreParams["correct_com"].asBool();



    // Angle parameters
    Json::Value angleParams = inpRoot.get("angle_params",
                                          Json::nullValue);
    if (angleParams == Json::nullValue){
        std::cerr << "ERROR: Could not load angle parameters." << std::endl;
        return ADAPTIVE_TIME_PARAM_ERROR;
    }

    sim_params.angle_params.angle_stiffness =
        angleParams["angle_stiffness"].asDouble();

    sim_params.angle_params.angle_pot =
        angleParams["angle_pot"].asBool();

    // Adaptive time step parameters
    Json::Value adaptiveParams = inpRoot.get("adaptive_params",
                                             Json::nullValue);

    if (adaptiveParams == Json::nullValue){
        std::cerr << "ERROR: Cannot load adaptive time parameters." << std::endl;
        return ADAPTIVE_TIME_PARAM_ERROR;
    }

    sim_params.adaptive_params.do_adaptive_dt =
        adaptiveParams["do_adaptive_dt"].asBool();

    sim_params.adaptive_params.dt_max =
        adaptiveParams["dt_max"].asDouble();

    sim_params.adaptive_params.dt_tol =
        adaptiveParams["dt_tol"].asDouble();

    // Cell counting params
    Json::Value countingParams = inpRoot.get("counting_params",
                                             Json::nullValue);

    if (countingParams == Json::nullValue){
        std::cerr << "ERROR: Cannot load counting parameters." << std::endl;
        return COUNTING_PARAM_ERROR;
    }

    // I'm not sure why this variable is a thing...
    // -Pranav
    sim_params.counting_params.count_cells =
        countingParams["count_cells"].asBool();

    std::strcpy(sim_params.counting_params.mit_index_file_name,
                countingParams["mit_index_file_name"].asString().c_str());

    sim_params.counting_params.count_int_cells_only =
        countingParams["count_int_cells_only"].asBool();

    sim_params.counting_params.radius_cutoff =
        countingParams["radius_cutoff"].asDouble();

    sim_params.counting_params.overwrite_mit_index_file =
        countingParams["overwrite_mit_index_file"].asBool();

    sim_params.counting_params.cell_count_int =
        countingParams["cell_count_int"].asInt64();

    // This section is currently unused.
    // population params
    Json::Value popParams = inpRoot.get("pop_params", Json::nullValue);

    if (popParams == Json::nullValue){
        std::cout << "WARNING: Something wrong with population params."
                  << std::endl;
        sim_params.pop_params.do_pop_model = false;
    } else {

        sim_params.pop_params.do_pop_model =
            popParams["do_pop_model"].asBool();

        sim_params.pop_params.total_food =
            popParams["total_food"].asDouble();

        sim_params.pop_params.regular_consumption =
            popParams["regular_consumption"].asDouble();

        sim_params.pop_params.death_release_food =
            popParams["death_release_food"].asDouble();
    }


    // wall params

    Json::Value wallParams = inpRoot.get("wall_params", Json::nullValue);

    if (wallParams == Json::nullValue){
        std::cerr << "ERROR: Could not read wall parameters" << std::endl;
        return WALL_PARAM_ERROR;
    }

    sim_params.wall_params.use_walls =
        wallParams["use_walls"].asBool();

    sim_params.wall_params.perp_axis = 'z';

    sim_params.wall_params.d_axis =
        wallParams["d_axis"].asDouble();

    sim_params.wall_params.wall_len =
        wallParams["wall_len"].asDouble();

    sim_params.wall_params.wall_width =
        wallParams["wall_width"].asDouble();

    sim_params.wall_params.thresh_dist =
        wallParams["thresh_dist"].asDouble();


    // division parameters

    Json::Value divParams = inpRoot.get("div_params", Json::nullValue);

    if (divParams == Json::nullValue){
        std::cout << "ERROR: Could not read division plane parameters."
                  << std::endl;

        return DIV_PARAM_ERROR;
    }


    sim_params.div_params.use_div_plane_basis =
        divParams["use_div_plane_basis"].asBool();

    sim_params.div_params.div_plane_basis =
        make_real3(divParams["div_plane_basis_x"].asDouble(),
                   divParams["div_plane_basis_y"].asDouble(),
                   divParams["div_plane_basis_z"].asDouble());

    // daughter stiffness parameters
    Json::Value stiffParams = inpRoot.get("stiffness_params", Json::nullValue);

    if (stiffParams == Json::nullValue){
        std::cout << "ERROR: Could not load daughter stiffness parameters"
                  << std::endl;

        return STIFF_PARAM_ERROR;
    }


    sim_params.stiff_params.use_diff_stiff =
        stiffParams["use_diff_stiff"].asBool();

    sim_params.stiff_params.soft_stiff_factor =
        stiffParams["soft_stiff_factor"].asDouble();

    sim_params.stiff_params.num_softer_cells =
        stiffParams["num_softer_cells"].asInt64();

    sim_params.stiff_params.frac_softer_cells =
        stiffParams["frac_softer_cells"].asDouble();

    sim_params.stiff_params.during_growth =
        stiffParams["during_growth"].asBool();

    sim_params.stiff_params.daught_same_stiff =
        stiffParams["daught_same_stiff"].asBool();

    sim_params.stiff_params.closeness_to_center =
        stiffParams["closeness_to_center"].asDouble();

    sim_params.stiff_params.start_at_pop =
        stiffParams["start_at_pop"].asInt64();

    sim_params.stiff_params.rand_cell_ind =
        stiffParams["rand_cell_ind"].asBool();

    // Read box parameters

    Json::Value boxParams = inpRoot.get("box_params", Json::nullValue);

    if (boxParams == Json::nullValue){
        std::cout << "ERROR: Could not load box parameters." << std::endl;
        return BOX_PARAM_ERROR;
    }

    sim_params.box_params.use_rigid_sim_box =
        boxParams["use_rigid_sim_box"].asBool();

    sim_params.box_params.use_pbc =
        boxParams["use_pbc"].asBool();

    sim_params.box_params.box_len =
        boxParams["box_len"].asDouble();
    
    sim_params.box_params.box_len_x =
        boxParams["box_len_x"].asDouble();

    sim_params.box_params.box_len_y =
        boxParams["box_len_y"].asDouble();

    sim_params.box_params.box_len_z =
        boxParams["box_len_z"].asDouble();

    sim_params.box_params.box_max = make_real3(sim_params.box_params.box_len_x,
                                               sim_params.box_params.box_len_y,
                                               sim_params.box_params.box_len_z);

    sim_params.box_params.flatbox =
        boxParams["flatbox"].asBool();

    sim_params.box_params.dom_len =
        boxParams["dom_len"].asDouble();

    sim_params.box_params.thresh_dist =
        boxParams["thresh_dist"].asDouble();

    sim_params.box_params.rand_pos =
        boxParams["rand_pos"].asBool();        


    // Readin the random number parameters

    Json::Value randParams = inpRoot.get("box_params", Json::nullValue);

    if (randParams == Json::nullValue){
        std::cout << "ERROR: Could not load box parameters." << std::endl;
        return BOX_PARAM_ERROR;
    }

    sim_params.rand_params.add_rands =
        randParams["add_rands"].asBool();

    sim_params.rand_params.rand_seed =
        randParams["rand_seed"].asInt();

    sim_params.rand_params.rand_dist =
        randParams["rand_dist"].asInt();

    sim_params.rand_params.rand_scale_factor =
        randParams["rand_scale_factor"].asDouble();
        
        

    // Now the weird parameter checks
    bool invalidParam = false;
    
    {
        // core
        const core_params_struct core = sim_params.core_params;
        uint total = core.div_time_steps + core.non_div_time_steps;
        if ( total % core.traj_write_int > 0){
            std::cout << "ERROR: Trajectory write interval is invalid"
                      << std::endl;
            std::cout << "       " << total << " % " << core.traj_write_int
                      << " = " << total%core.traj_write_int << std::endl;
            invalidParam = true;
        }

        if (core.node_mass < 0){
            std::cout << "ERROR: Invalid node mass" << std::endl
                      << "       mass cannot be " << core.node_mass
                      << std::endl;
            invalidParam = true;
        }

        if (core.rep_range > core.attr_range){
            std::cout << "ERROR: Invalid repulsion and attraction ranges."
                      << std::endl
                      << "       Repulsion range = " << core.rep_range << ", "
                      << "Attraction range = " << core.attr_range << std::endl;
            invalidParam = true;
        }

        if (core.stiff_factor1 > 1 || core.stiff_factor1 < 0){
            std::cout << "ERROR: Invalid stiffness factor." << std::endl
                      << "       stiff_factor1 is " << core.stiff_factor1
                      << std::endl;
            invalidParam = true;
        }

        if (core.min_pressure > core.max_pressure){
            std::cout<< "ERROR: Invalid pressure parameters." << std::endl
                     << "       Min. Pressure is " << core.min_pressure
                     << " Max. Pressure is " << core.max_pressure << std::endl;
            invalidParam = true;
        }



        // Below are not necessarily errors
        if (core.bond_stiff < 500){

            std::cout << "WARNING: Bond stiffness may be too low." << std::endl
                      << "       Stifness is " << core.bond_stiff << std::endl;
        }

        if (core.rep_stiff <= 5*core.attr_stiff){ // repulsive springs should be really stiff
            std::cout << "WARNING: Weird attractive and repulsive spring values."
                      << std::endl
                      <<"        Attractive spring stiffness is "
                      << core.attr_stiff <<" repulsive spring stiffness is "
                      << core.rep_stiff;
        }

        if (core.division_vol != 2.9f){
            std::cout << "WARNING: Set division volume may cause undefined"
                      << " behaviour." << std::endl
                      << "       Division volume set to " << core.division_vol
                      << std::endl;

        }

        if (core.random_z_offset == true){
            std::cout << "WARNING: Setting of random z offset is not implemented"
                      << std::endl;
        }

        if (core.z_offset < 0){
            std::cout << "WARNING: Negative z_offset value"
                      << ", may screw up box settings. " << std::endl
                      << "       z offset is " << core.z_offset << std::endl;

        }

        if (core.div_time_steps == 0){
            std::cout << "WARNING: No growth will happen." << std::endl
                      << "       Division time steps set to "
                      << core.div_time_steps << std::endl;

        }

        if (core.dom_len != 2.9f){
            std::cout<< "WARNING: Untested domain length " << core.dom_len
                     << "." << std::endl;
        }
            
    }

    { // angle_params
        angle_params_struct ap = sim_params.angle_params;

        if (ap.angle_stiffness != 1000.f){
            std::cout << "WARNING: Untested angle stiffness value of "<< ap.angle_stiffness
                      << std::endl;
        }
    }

    { // adaptive time params
        adaptive_params_struct at = sim_params.adaptive_params;
        if (at.do_adaptive_dt == true){
            std::cout << "WARNING: Adaptive time simulaitons unsupported."<< std::endl;
            std::cout << "         ingoring adaptive time parameters..." << std::endl;
        }
    }
    
    {
        pop_params_struct pp = sim_params.pop_params;
        
        if (pp.do_pop_model == true){ 
            std::cout << "ERROR: Population modeling not currently supported."
                      << std::endl;
            std::cout << "       Disable population modeling." << std::endl;

            invalidParam = true; 
        }

    }

    { // wall_params
        std::cout << "WARNING: Setting of perpendicular axis not supported." << std::endl
                  << "         defaulting to z."<< std::endl; 

        sim_params.wall_params.perp_axis = 'z';
    }


    if (invalidParam)
        return INVALID_PARAM_ERROR;

    //success
    return CELLDIV_SUCCESS;
}
