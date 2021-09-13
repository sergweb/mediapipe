#include <stdio.h>
#include <string.h>
#include <stddef.h>
#include <errno.h>
#include <math.h>

#include <iostream>
#include <functional>

using namespace std;

#include <gps.h>

#include "gps_helper.h"


int                  gps_rc = 0;
struct gps_data_t    gps_data_rec;

function<void(string)> logger_func; 

void gps_helper_register_logger_callback(function<void(string)> func) {
    logger_func = func;
} 

void logger(string msg) {
    if(logger_func)
        logger_func(msg);
    else
        cout<<msg<<endl;
}

void check_gps_error() {
    if(gps_rc==-1)
        logger(string("GPS Error: ") + string(gps_errstr(errno)));
} 

string gps_helper_status_to_string(int status) {
    return STATUS_NO_FIX   == status ? "STATUS_NO_FIX"   :
           STATUS_FIX      == status ? "STATUS_FIX"      :
           STATUS_DGPS_FIX == status ? "STATUS_DGPS_FIX" : "STATUS_UNKNOWN";
}

string gps_helper_mode_to_string(int mode) {
    return MODE_NOT_SEEN == mode ? "MODE_NOT_SEEN" :
           MODE_NO_FIX   == mode ? "MODE_NO_FIX"   :
           MODE_2D       == mode ? "MODE_2D"       :
           MODE_3D       == mode ? "MODE_3D"       : "MODE_UNKNOWN";
}

int gps_helper_open() {
    gps_rc = gps_open("localhost", "2947", &gps_data_rec);
    check_gps_error();
    if(gps_rc) return gps_rc;
    logger("GPS opened");
    gps_rc = gps_stream(&gps_data_rec, WATCH_ENABLE | WATCH_JSON, NULL);
    check_gps_error();
    if(gps_rc) return gps_rc;
    logger("GPS streaming started");
    return 0;
}

int gps_helper_close() {
    gps_rc = gps_stream(&gps_data_rec, WATCH_DISABLE, NULL);
    check_gps_error();
    if(gps_rc) return gps_rc;
    logger("GPS streaming stopped");
    gps_rc = gps_close (&gps_data_rec);
    check_gps_error();
    if(gps_rc) return gps_rc;
    logger("GPS closed");
    return 0;
}


gps_data_t*  gps_helper_read() {
    gps_rc = 0;
    if(gps_waiting(&gps_data_rec, 1)) {
        errno = 0;
        gps_rc = gps_read(&gps_data_rec);
        if(gps_rc>0) gps_rc=0;
        check_gps_error();
    }
    if(gps_rc==-1)
        return nullptr;
    else
        return &gps_data_rec;
}

/*
int gps_helper_status() {
    if(gps_attr.set|STATUS_SET)
        return gps_attr.status;
    else
        return -1;
}

// Returns speed in MPH 
int gps_helper_speed() {
    if(gps_helper_status() <= 0)
        return -1;
    if(gps_attr.set|SPEED_SET)
        return round(gps_attr.fix.speed*MPS_TO_MPH);
    else
        return -1;
}

gps_fix_t gps_helper_fix() {
    return gps_attr.fix;
}


int gps_helper_close() {
    gps_rc = gps_stream(&gps_attr, WATCH_DISABLE, NULL);
    if(gps_rc) return gps_rc;
    gps_rc = gps_close (&gps_attr);
    return gps_rc;
}

int read_gps_speed() {
    int speed = -1;
    {
        unique_lock<mutex> lk(gps_mutex);
        speed = gps_speed;
    }
    return speed;
}

void write_gps_speed() {
    unique_lock<mutex> lk(gps_mutex);
    gps_speed = gps_helper_speed();    
}

int gps_helper_thread_func() {
    int rc = 0;
    gps_data_t gps_data;
    while(true) {
        {   unique_lock<mutex> lk(gps_thread_mutex);
            if(!gps_thread_keep_running) break;
        }
        //
        if(gps_waiting(&gps_data, 2e6)) {
            errno = 0;
            rc = gps_read(&gps_data);
            if(rc>0) rc=0;
        }
        if(!rc) {
            unique_lock<mutex> lk(gps_mutex);
            memcpy(&gps_attr, &gps_data, sizeof(gps_data));
        }
        //
        this_thread::sleep_for(milliseconds(200));
        //cout << "GPS thread keeps running" << endl;
    }
    return rc;
}

void gps_helper_stop_thread() {
    unique_lock<mutex> lk(gps_thread_mutex);
    gps_thread_keep_running = false;
}
*/
