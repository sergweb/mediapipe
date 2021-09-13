#ifndef __GPS_HELPER_H__
#define __GPS_HELPER_H__

#include <string>
#include <functional>
#include <gps.h>

using namespace std;

void gps_helper_register_logger_callback(function<void(string)> func);

string gps_helper_status_to_string(int status);

string gps_helper_mode_to_string(int mode);

int gps_helper_open();

int gps_helper_close();

gps_data_t*  gps_helper_read();

#endif //__GPS_HELPER_H__
