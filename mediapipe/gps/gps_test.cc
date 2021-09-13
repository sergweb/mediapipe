#include <math.h>
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <thread>
#include <iomanip>
#include <ctime>

#include "gps_helper.h"

using namespace std;
using namespace std::chrono;

void custom_logger(string msg) {
    time_t t = time(nullptr);
    tm tm = *localtime(&t);
    cout<<put_time(&tm, "%c %Z")<<" "<<msg<<endl;
} 


int main() {
    int rc = 0;

    gps_helper_register_logger_callback(custom_logger);

    rc = gps_helper_open();
    if(rc) return rc;

    gps_data_t* gps_data_ptr = nullptr;

    while(true) {
        gps_data_ptr = gps_helper_read();
        if(gps_data_ptr) {
            custom_logger(gps_helper_status_to_string(gps_data_ptr->set|STATUS_SET ?  gps_data_ptr->status : -1));
            custom_logger(gps_helper_mode_to_string(gps_data_ptr->set|MODE_SET ?  gps_data_ptr->fix.mode : -1));
            //custom_logger(to_string(gps_data_ptr->set|SPEED_SET ? (long)round(gps_data_ptr->fix.speed*MPS_TO_MPH) : -1L));
            custom_logger(string("speed      : ") + to_string(gps_data_ptr->fix.speed*MPS_TO_MPH));
            custom_logger(string("eps        : ") + to_string(gps_data_ptr->fix.eps*MPS_TO_MPH));
            custom_logger(string("latitude   : ") + to_string(gps_data_ptr->fix.latitude));
            custom_logger(string("longtitude : ") + to_string(gps_data_ptr->fix.longitude));
            const time_t fix_time_sec = (time_t)gps_data_ptr->fix.time;
            custom_logger(string("fix time   : ") + string(ctime(&fix_time_sec)));
        }
        this_thread::sleep_for(milliseconds(1000));
    }
/*
    unique_ptr<thread> gps_thread(nullptr);

    bool use_gps_speed_flag = true;

    if(use_gps_speed_flag) {
        gps_thread = make_unique<thread>(gps_helper_thread_func);
        cout << "GPS helper thread started"<<endl;
    }

    for(int i=0;i<100;i++) {
        cout<<"status: "<<gps_helper_status()<<endl;
        cout<<"speed: "<<read_gps_speed()<<endl;
        this_thread::sleep_for(milliseconds(1000));
    }
    
    if(use_gps_speed_flag) {
        gps_helper_stop_thread();
        gps_thread->join();
        cout << "GPS helper thread stopped"<<endl;
    }

*/

    rc = gps_helper_close();
    if(rc) return rc;

    return 0;
}
