// This file will contain definitions of error codes

#ifndef ERR_CODES_HPP
#define ERR_CODES_HPP

typedef int celldiv_err;

#define CELLDIV_SUCCESS            0
#define CELLDIV_ERROR              1
#define FATAL_ERROR                72

// All errors are > 0, with a special prefix

// Simulation param read error prefix is 1
#define SIM_PARAM_READ_ERROR      10
#define PARAM_INVALID_VERSION     11
#define CORE_SIM_PARAM_ERROR      12
#define ANGLE_PARAM_ERROR         13
#define ADAPTIVE_TIME_PARAM_ERROR 14
#define COUNTING_PARAM_ERROR      15
#define WALL_PARAM_ERROR          16
#define DIV_PARAM_ERROR           17
#define STIFF_PARAM_ERROR         18
#define BOX_PARAM_ERROR           19
#define INVALID_PARAM_ERROR       110

#define TRAJ_ERROR                20
#define TRAJ_STATE_ERROR          21
#endif
